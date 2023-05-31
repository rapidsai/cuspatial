/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuproj/ellipsoid.hpp>
#include <cuproj/error.hpp>
#include <cuproj/transform.cuh>

#include <cuspatial/geometry/vec_2d.hpp>

#include <cuspatial_test/vector_equality.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/tabulate.h>

#include <proj.h>

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>

#if 1
constexpr double DEG_TO_RAD = 0.017453292519943295769236907684886;
constexpr double M_TWOPI    = 6.283185307179586476925286766559005;
constexpr double EPS_LAT    = 1e-12;
constexpr int ETMERC_ORDER  = 6;

struct PoderEngsager {
  double Qn;     /* Merid. quad., scaled to the projection */
  double Zb;     /* Radius vector in polar coord. systems  */
  double cgb[6]; /* Constants for Gauss -> Geo lat */
  double cbg[6]; /* Constants for Geo lat -> Gauss */
  double utg[6]; /* Constants for transv. merc. -> geo */
  double gtu[6]; /* Constants for geo -> transv. merc. */
};

struct ellipsoid {
  double a;      // semi-major axis
  double b;      // semi-minor axis
  double e;      // first eccentricity
  double es;     // first eccentricity squared
  double alpha;  // angular eccentricity
  double f;      // flattening
  double n;      // third flattening
};

template <typename T>
struct transverse_mercator {
  PoderEngsager constants;

  cuproj::ellipsoid<T> ellps;

  double k0;          // scaling
  double lam0 = 0.0;  // central meridian
  double phi0 = 0.0;  // central parallel
  double x0   = 0.0;  // false easting
  double y0   = 0.0;  // false northing
};

inline static double gatg(const double* p1, int len_p1, double B, double cos_2B, double sin_2B)
{
  double h = 0, h1, h2 = 0;

  const double two_cos_2B = 2 * cos_2B;
  const double* p         = p1 + len_p1;
  h1                      = *--p;
  while (p - p1) {
    h  = -h2 + two_cos_2B * h1 + *--p;
    h2 = h1;
    h1 = h;
  }
  return (B + h * sin_2B);
}

/* Complex Clenshaw summation */
inline static double clenS(const double* a,
                           int size,
                           double sin_arg_r,
                           double cos_arg_r,
                           double sinh_arg_i,
                           double cosh_arg_i,
                           double* R,
                           double* I)
{
  double r, i, hr, hr1, hr2, hi, hi1, hi2;

  /* arguments */
  const double* p = a + size;
  r               = 2 * cos_arg_r * cosh_arg_i;
  i               = -2 * sin_arg_r * sinh_arg_i;

  /* summation loop */
  hi1 = hr1 = hi = 0;
  hr             = *--p;
  for (; a - p;) {
    hr2 = hr1;
    hi2 = hi1;
    hr1 = hr;
    hi1 = hi;
    hr  = -hr2 + r * hr1 - i * hi1 + *--p;
    hi  = -hi2 + i * hr1 + r * hi1;
  }

  r  = sin_arg_r * cosh_arg_i;
  i  = cos_arg_r * sinh_arg_i;
  *R = r * hr - i * hi;
  *I = r * hi + i * hr;
  return *R;
}

/* Real Clenshaw summation */
static double clens(const double* a, int size, double arg_r)
{
  double r, hr, hr1, hr2, cos_arg_r;

  const double* p = a + size;
  cos_arg_r       = cos(arg_r);
  r               = 2 * cos_arg_r;

  /* summation loop */
  hr1 = 0;
  hr  = *--p;
  for (; a - p;) {
    hr2 = hr1;
    hr1 = hr;
    hr  = -hr2 + r * hr1 + *--p;
  }
  return sin(arg_r) * hr;
}

template <typename T>
static PJ* setup_tmerc_exact(transverse_mercator<T>& tmerc,
                             cuproj::ellipsoid<T> const& ellps,
                             PJ* P)
{
  assert(ellps.es > 0);

  tmerc.ellps = ellps;

  tmerc.y0 = 10000000;
  tmerc.x0 = 500000.;

  auto zone = 56;
  if (zone > 0 && zone <= 60) --zone;
  tmerc.lam0 = (zone + .5) * M_PI / 30. - M_PI;
  tmerc.k0   = 0.9996;
  tmerc.phi0 = 0.;

  /* third flattening */
  const double n = ellps.n;
  double np      = n;

  /* COEF. OF TRIG SERIES GEO <-> GAUSS */
  /* cgb := Gaussian -> Geodetic, KW p190 - 191 (61) - (62) */
  /* cbg := Geodetic -> Gaussian, KW p186 - 187 (51) - (52) */
  /* PROJ_ETMERC_ORDER = 6th degree : Engsager and Poder: ICC2007 */

  auto& pe = tmerc.constants;

  pe.cgb[0] =
    n * (2 + n * (-2 / 3.0 + n * (-2 + n * (116 / 45.0 + n * (26 / 45.0 + n * (-2854 / 675.0))))));
  pe.cbg[0] =
    n *
    (-2 + n * (2 / 3.0 + n * (4 / 3.0 + n * (-82 / 45.0 + n * (32 / 45.0 + n * (4642 / 4725.0))))));
  np *= n;
  pe.cgb[1] =
    np * (7 / 3.0 + n * (-8 / 5.0 + n * (-227 / 45.0 + n * (2704 / 315.0 + n * (2323 / 945.0)))));
  pe.cbg[1] =
    np * (5 / 3.0 + n * (-16 / 15.0 + n * (-13 / 9.0 + n * (904 / 315.0 + n * (-1522 / 945.0)))));
  np *= n;
  /* n^5 coeff corrected from 1262/105 -> -1262/105 */
  pe.cgb[2] = np * (56 / 15.0 + n * (-136 / 35.0 + n * (-1262 / 105.0 + n * (73814 / 2835.0))));
  pe.cbg[2] = np * (-26 / 15.0 + n * (34 / 21.0 + n * (8 / 5.0 + n * (-12686 / 2835.0))));
  np *= n;
  /* n^5 coeff corrected from 322/35 -> 332/35 */
  pe.cgb[3] = np * (4279 / 630.0 + n * (-332 / 35.0 + n * (-399572 / 14175.0)));
  pe.cbg[3] = np * (1237 / 630.0 + n * (-12 / 5.0 + n * (-24832 / 14175.0)));
  np *= n;
  pe.cgb[4] = np * (4174 / 315.0 + n * (-144838 / 6237.0));
  pe.cbg[4] = np * (-734 / 315.0 + n * (109598 / 31185.0));
  np *= n;
  pe.cgb[5] = np * (601676 / 22275.0);
  pe.cbg[5] = np * (444337 / 155925.0);

  /* Constants of the projections */
  /* Transverse Mercator (UTM, ITM, etc) */
  np = n * n;
  /* Norm. mer. quad, K&W p.50 (96), p.19 (38b), p.5 (2) */
  pe.Qn = tmerc.k0 / (1 + n) * (1 + np * (1 / 4.0 + np * (1 / 64.0 + np / 256.0)));
  /* coef of trig series */
  /* utg := ell. N, E -> sph. N, E,  KW p194 (65) */
  /* gtu := sph. N, E -> ell. N, E,  KW p196 (69) */
  pe.utg[0] =
    n * (-0.5 + n * (2 / 3.0 + n * (-37 / 96.0 +
                                    n * (1 / 360.0 + n * (81 / 512.0 + n * (-96199 / 604800.0))))));
  pe.gtu[0] =
    n * (0.5 + n * (-2 / 3.0 +
                    n * (5 / 16.0 + n * (41 / 180.0 + n * (-127 / 288.0 + n * (7891 / 37800.0))))));
  pe.utg[1] =
    np * (-1 / 48.0 +
          n * (-1 / 15.0 + n * (437 / 1440.0 + n * (-46 / 105.0 + n * (1118711 / 3870720.0)))));
  pe.gtu[1] =
    np * (13 / 48.0 +
          n * (-3 / 5.0 + n * (557 / 1440.0 + n * (281 / 630.0 + n * (-1983433 / 1935360.0)))));
  np *= n;
  pe.utg[2] = np * (-17 / 480.0 + n * (37 / 840.0 + n * (209 / 4480.0 + n * (-5569 / 90720.0))));
  pe.gtu[2] =
    np * (61 / 240.0 + n * (-103 / 140.0 + n * (15061 / 26880.0 + n * (167603 / 181440.0))));
  np *= n;
  pe.utg[3] = np * (-4397 / 161280.0 + n * (11 / 504.0 + n * (830251 / 7257600.0)));
  pe.gtu[3] = np * (49561 / 161280.0 + n * (-179 / 168.0 + n * (6601661 / 7257600.0)));
  np *= n;
  pe.utg[4] = np * (-4583 / 161280.0 + n * (108847 / 3991680.0));
  pe.gtu[4] = np * (34729 / 80640.0 + n * (-3418889 / 1995840.0));
  np *= n;
  pe.utg[5] = np * (-20648693 / 638668800.0);
  pe.gtu[5] = np * (212378941 / 319334400.0);

  /* Gaussian latitude value of the origin latitude */
  auto phi0      = tmerc.phi0;
  const double Z = gatg(pe.cbg, ETMERC_ORDER, phi0, cos(2 * phi0), sin(2 * phi0));

  /* Origin northing minus true northing at the origin latitude */
  /* i.e. true northing = N - P->Zb                         */
  pe.Zb = -pe.Qn * (Z + clens(pe.gtu, ETMERC_ORDER, 2 * Z));

  return P;
}

void swap_xy_4d(PJ_COORD& coo, PJ*) { std::swap(coo.xyzt.x, coo.xyzt.y); }

double adjlon(double longitude)
{
  /* Let longitude slightly overshoot, to avoid spurious sign switching at the
   * date line */
  if (fabs(longitude) < M_PI + 1e-12) return longitude;

  /* adjust to 0..2pi range */
  longitude += M_PI;

  /* remove integral # of 'revolutions'*/
  longitude -= M_TWOPI * floor(longitude / M_TWOPI);

  /* adjust back to -pi..pi range */
  longitude -= M_PI;

  return longitude;
}

static PJ_XY unitconvert_forward(PJ_LP lp, PJ* P)
{
  /************************************************************************
      Forward unit conversions in the plane
  ************************************************************************/
  PJ_COORD point = {{0, 0, 0, 0}};
  point.lp       = lp;

  point.xy.x *= DEG_TO_RAD;
  point.xy.y *= DEG_TO_RAD;

  return point.xy;
}

template <typename T>
void prepare_utm(PJ_COORD& coo, PJ* P, transverse_mercator<T> const& tmerc)
{
  double t;

  // check for latitude or longitude over-range
  t = (coo.lp.phi < 0 ? -coo.lp.phi : coo.lp.phi) - M_PI_2;
  CUPROJ_EXPECTS(t <= EPS_LAT, "Invalid latitude");
  CUPROJ_EXPECTS(coo.lp.lam <= 10 || coo.lp.lam >= -10, "Invalid longitude");

  /* Clamp latitude to -90..90 degree range */
  coo.lp.phi = std::clamp(coo.lp.phi, -M_PI_2, M_PI_2);

  // Ensure longitude is in the -pi:pi range
  coo.lp.lam = adjlon(coo.lp.lam);

  // Distance from central meridian, taking system zero meridian into account
  double const from_greenwich = 0.;
  double const lam0           = tmerc.lam0;
  coo.lp.lam                  = (coo.lp.lam - from_greenwich) - lam0;

  // Ensure longitude is in the -pi:pi range
  coo.lp.lam = adjlon(coo.lp.lam);
}

template <typename T>
static PJ_XY tmerc_exact_e_fwd(PJ_LP lp, PJ* P, transverse_mercator<T> const& tmerc)
{
  PJ_XY xy = {0.0, 0.0};

  /* ell. LAT, LNG -> Gaussian LAT, LNG */
  double Cn = gatg(tmerc.constants.cbg, ETMERC_ORDER, lp.phi, cos(2 * lp.phi), sin(2 * lp.phi));
  /* Gaussian LAT, LNG -> compl. sph. LAT */
  const double sin_Cn = sin(Cn);
  const double cos_Cn = cos(Cn);
  const double sin_Ce = sin(lp.lam);
  const double cos_Ce = cos(lp.lam);

  const double cos_Cn_cos_Ce = cos_Cn * cos_Ce;
  Cn                         = atan2(sin_Cn, cos_Cn_cos_Ce);

  const double inv_denom_tan_Ce = 1. / hypot(sin_Cn, cos_Cn_cos_Ce);
  const double tan_Ce           = sin_Ce * cos_Cn * inv_denom_tan_Ce;
#if 0
    // Variant of the above: found not to be measurably faster
    const double sin_Ce_cos_Cn = sin_Ce*cos_Cn;
    const double denom = sqrt(1 - sin_Ce_cos_Cn * sin_Ce_cos_Cn);
    const double tan_Ce = sin_Ce_cos_Cn / denom;
#endif

  /* compl. sph. N, E -> ell. norm. N, E */
  double Ce = asinh(tan_Ce); /* Replaces: Ce  = log(tan(FORTPI + Ce*0.5)); */

  /*
   *  Non-optimized version:
   *  const double sin_arg_r  = sin(2*Cn);
   *  const double cos_arg_r  = cos(2*Cn);
   *
   *  Given:
   *      sin(2 * Cn) = 2 sin(Cn) cos(Cn)
   *          sin(atan(y)) = y / sqrt(1 + y^2)
   *          cos(atan(y)) = 1 / sqrt(1 + y^2)
   *      ==> sin(2 * Cn) = 2 tan_Cn / (1 + tan_Cn^2)
   *
   *      cos(2 * Cn) = 2cos^2(Cn) - 1
   *                  = 2 / (1 + tan_Cn^2) - 1
   */
  const double two_inv_denom_tan_Ce        = 2 * inv_denom_tan_Ce;
  const double two_inv_denom_tan_Ce_square = two_inv_denom_tan_Ce * inv_denom_tan_Ce;
  const double tmp_r                       = cos_Cn_cos_Ce * two_inv_denom_tan_Ce_square;
  const double sin_arg_r                   = sin_Cn * tmp_r;
  const double cos_arg_r                   = cos_Cn_cos_Ce * tmp_r - 1;

  /*
   *  Non-optimized version:
   *  const double sinh_arg_i = sinh(2*Ce);
   *  const double cosh_arg_i = cosh(2*Ce);
   *
   *  Given
   *      sinh(2 * Ce) = 2 sinh(Ce) cosh(Ce)
   *          sinh(asinh(y)) = y
   *          cosh(asinh(y)) = sqrt(1 + y^2)
   *      ==> sinh(2 * Ce) = 2 tan_Ce sqrt(1 + tan_Ce^2)
   *
   *      cosh(2 * Ce) = 2cosh^2(Ce) - 1
   *                   = 2 * (1 + tan_Ce^2) - 1
   *
   * and 1+tan_Ce^2 = 1 + sin_Ce^2 * cos_Cn^2 / (sin_Cn^2 + cos_Cn^2 *
   * cos_Ce^2) = (sin_Cn^2 + cos_Cn^2 * cos_Ce^2 + sin_Ce^2 * cos_Cn^2) /
   * (sin_Cn^2 + cos_Cn^2 * cos_Ce^2) = 1. / (sin_Cn^2 + cos_Cn^2 * cos_Ce^2)
   * = inv_denom_tan_Ce^2
   *
   */
  const double sinh_arg_i = tan_Ce * two_inv_denom_tan_Ce;
  const double cosh_arg_i = two_inv_denom_tan_Ce_square - 1;

  double dCn, dCe;
  Cn += clenS(
    tmerc.constants.gtu, ETMERC_ORDER, sin_arg_r, cos_arg_r, sinh_arg_i, cosh_arg_i, &dCn, &dCe);
  Ce += dCe;
  CUPROJ_EXPECTS(fabs(Ce) <= 2.623395162778, "Coordinate transform outside projection domain");
  xy.y = tmerc.constants.Qn * Cn + tmerc.constants.Zb; /* Northing */
  xy.x = tmerc.constants.Qn * Ce;                      /* Easting  */

  return xy;
}

template <typename T>
void my_trans(PJ* P, PJ_DIRECTION direction, PJ_COORD& point, transverse_mercator<T> const& tmerc)
{
  swap_xy_4d(point, P);
  point.xy = unitconvert_forward(point.lp, P);
  prepare_utm<T>(point, P, tmerc);
  point.xy = tmerc_exact_e_fwd<T>(point.lp, P, tmerc);

  // units of semimajor axis
  point.xy.x *= tmerc.ellps.a;
  point.xy.y *= tmerc.ellps.a;

  point.xyz.x = /*P->fr_meter **/ (point.xyz.x + tmerc.x0);
  point.xyz.y = /*P->fr_meter **/ (point.xyz.y + tmerc.y0);

  point.xyz.z = 0;
}
#endif

template <typename T>
struct ProjectionTest : public ::testing::Test {};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(ProjectionTest, TestTypes);

template <typename T>
using coordinate = typename cuspatial::vec_2d<T>;

TYPED_TEST(ProjectionTest, Test_one)
{
  PJ_CONTEXT* C;
  PJ* P;

  C = proj_context_create();

  P = proj_create_crs_to_crs(C, "EPSG:4326", "EPSG:32756", NULL);

  double ellps_a{};
  double ellps_b{};
  int is_semi_minor_computed{};
  double ellps_inv_flattening{};
  auto wgs84   = proj_create(C, "EPSG:4326");
  PJ* pj_ellps = proj_get_ellipsoid(C, wgs84);
  proj_ellipsoid_get_parameters(
    C, pj_ellps, &ellps_a, &ellps_b, &is_semi_minor_computed, &ellps_inv_flattening);
  proj_destroy(pj_ellps);

  using T = TypeParam;

  std::vector<PJ_COORD> input_coords{{-28.667003, 153.090959, 0, 0}};
  std::vector<PJ_COORD> expected_coords(input_coords);

  proj_trans_array(P, PJ_FWD, expected_coords.size(), expected_coords.data());

  /* Clean up */
  proj_destroy(P);
  proj_context_destroy(C);  // may be omitted in the single threaded case

  // semimajor and inverse flattening
  cuproj::ellipsoid<T> ellps{static_cast<T>(ellps_a), static_cast<T>(ellps_inv_flattening)};

  transverse_mercator<T> tmerc{};
  std::vector<PJ_COORD> output_coords(input_coords);

  setup_tmerc_exact<T>(tmerc, ellps, P);

  my_trans<TypeParam>(P, PJ_FWD, output_coords[0], tmerc);

  std::cout << "output after " << std::setprecision(20) << output_coords[0].xy.x << " "
            << output_coords[0].xy.y << std::endl;

  std::vector<coordinate<T>> input(input_coords.size());
  std::vector<coordinate<T>> expected(expected_coords.size());
  std::transform(input_coords.begin(), input_coords.end(), input.begin(), [](auto const& c) {
    return coordinate<T>{static_cast<T>(c.xy.x), static_cast<T>(c.xy.y)};
  });
  std::transform(
    expected_coords.begin(), expected_coords.end(), expected.begin(), [](auto const& c) {
      return coordinate<T>{static_cast<T>(c.xy.x), static_cast<T>(c.xy.y)};
    });

  thrust::device_vector<coordinate<T>> d_in = input;
  thrust::device_vector<coordinate<T>> d_out(d_in.size());
  thrust::device_vector<coordinate<T>> d_expected = expected;

  cuproj::projection<T> tmerc_proj{ellps, 56, 0, 0};

  cuproj::transform(
    tmerc_proj, d_in.begin(), d_in.end(), d_out.begin(), cuproj::direction::DIR_FWD);

  std::cout << "expected " << std::setprecision(20) << expected_coords[0].xy.x << " "
            << expected_coords[0].xy.y << std::endl;
  coordinate<T> c_out = d_out[0];
  std::cout << "Device: " << std::setprecision(20) << c_out.x << " " << c_out.y << std::endl;

  // We can expect nanometer accuracy with double precision. The precision ratio of
  // double to single precision is 2^53 / 2^24 == 2^29 ~= 10^9, then we should
  // expect meter (10^9 nanometer) accuracy with single precision.
  if constexpr (std::is_same_v<T, float>) {
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_expected, d_out, T{1.0});  // within 1 meter
  } else {
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_expected, d_out);  // just use normal 1-ulp comparison
  }
}

template <typename T>
struct grid_generator {
  coordinate<T> min_corner{};
  coordinate<T> max_corner{};
  coordinate<T> spacing{};
  int num_points_x{};
  int num_points_y{};

  grid_generator(coordinate<T> min_corner,
                 coordinate<T> max_corner,
                 int num_points_x,
                 int num_points_y)
    : min_corner(min_corner),
      max_corner(max_corner),
      num_points_x(num_points_x),
      num_points_y(num_points_y)
  {
    spacing = coordinate<T>{(max_corner.x - min_corner.x) / num_points_x,
                            (max_corner.y - min_corner.y) / num_points_y};
  }

  __device__ coordinate<T> operator()(int i) const
  {
    return min_corner +
           coordinate<T>{(i % num_points_x) * spacing.x, (i / num_points_x) * spacing.y};
  }
};

TYPED_TEST(ProjectionTest, Test_many)
{
  using T = TypeParam;
  // generate (lat, lon) points on a grid between -60 and 60 degrees longitude and
  // -40 and 80 degrees latitude
  int num_points_x         = 100;
  int num_points_y         = 100;
  coordinate<T> min_corner = {-26.5, -152.5};
  coordinate<T> max_corner = {-25.5, -153.5};

  auto gen = grid_generator<T>(min_corner, max_corner, num_points_x, num_points_y);

  thrust::device_vector<coordinate<T>> input(num_points_x * num_points_y);

  thrust::tabulate(rmm::exec_policy(), input.begin(), input.end(), gen);

  // create PROJ context
  PJ_CONTEXT* C = proj_context_create();

  double ellps_a{};
  double ellps_b{};
  int is_semi_minor_computed{};
  double ellps_inv_flattening{};
  auto wgs84   = proj_create(C, "EPSG:4326");
  PJ* pj_ellps = proj_get_ellipsoid(C, wgs84);
  proj_ellipsoid_get_parameters(
    C, pj_ellps, &ellps_a, &ellps_b, &is_semi_minor_computed, &ellps_inv_flattening);
  proj_destroy(pj_ellps);

  // semimajor and inverse flattening
  cuproj::ellipsoid<T> ellps{static_cast<T>(ellps_a), static_cast<T>(ellps_inv_flattening)};

  // create a projection object
  cuproj::projection<T> tmerc_proj{ellps, 56};
  // create a vector of output points
  thrust::device_vector<coordinate<T>> output(input.size());
  // transform the input points to output points
  cuproj::transform(
    tmerc_proj, input.begin(), input.end(), output.begin(), cuproj::direction::DIR_FWD);

  using T = TypeParam;

  thrust::host_vector<PJ_COORD> input_coords{input.size()};
  std::transform(input.begin(), input.end(), input_coords.begin(), [](coordinate<T> const& c) {
    return PJ_COORD{c.x, c.y, 0, 0};
  });
  thrust::host_vector<PJ_COORD> expected_coords(input_coords);

  PJ* P = proj_create_crs_to_crs(C, "EPSG:4326", "EPSG:32756", NULL);

  proj_trans_array(P, PJ_FWD, expected_coords.size(), expected_coords.data());

  proj_destroy(P);
  proj_context_destroy(C);

  std::vector<coordinate<T>> expected(expected_coords.size());
  std::transform(
    expected_coords.begin(), expected_coords.end(), expected.begin(), [](auto const& c) {
      return coordinate<T>{static_cast<T>(c.xy.x), static_cast<T>(c.xy.y)};
    });

  thrust::device_vector<coordinate<T>> d_expected = expected;

  std::cout << "expected " << std::setprecision(20) << expected_coords[0].xy.x << " "
            << expected_coords[0].xy.y << std::endl;
  coordinate<T> c_out = output[0];
  std::cout << "Device: " << std::setprecision(20) << c_out.x << " " << c_out.y << std::endl;

  // Assumption: we can expect 5 nanometer (5e-9m) accuracy with double precision. The precision
  // ratio of double to single precision is 2^53 / 2^24 == 2^29 ~= 10^9, so we should expect 5 meter
  // (5e9 nanometer) accuracy with single precision. However we are seeing 10 meter accuracy
  // relative to PROJ with single precision (which uses double precision internally)

  // TODO: can we use double precision for key parts of the algorithm for accuracy while
  // using single precision for the rest for performance?
  if constexpr (std::is_same_v<T, float>) {
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_expected, output, T{10.0});  // within 5m
  } else {
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_expected, output, T{5e-9});  // within 5nm
  }
}
