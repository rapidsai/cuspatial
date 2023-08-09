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

// Code in this file is originally from the [OSGeo/PROJ project](https://github.com/OSGeo/PROJ)
// and has been modified to run on the GPU using CUDA.
//
// PROJ License from https://github.com/OSGeo/PROJ/blob/9.2/COPYING:
//   Note however that the file it is taken from did not have a copyright notice.
/*
 Copyright information can be found in source files.

 --------------

 Permission is hereby granted, free of charge, to any person obtaining a
 copy of this software and associated documentation files (the "Software"),
 to deal in the Software without restriction, including without limitation
 the rights to use, copy, modify, merge, publish, distribute, sublicense,
 and/or sell copies of the Software, and to permit persons to whom the
 Software is furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included
 in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 DEALINGS IN THE SOFTWARE.
*/

// The following text is taken from the PROJ file
// [tmerc.cpp](https://github.com/OSGeo/PROJ/blob/9.2/src/projections/tmerc.cpp)
// see the file LICENSE_PROJ in the cuspatial repository root directory for the original PROJ
// license text.
/*****************************************************************************/
//
//                  Exact Transverse Mercator functions
//
//
// The code in this file is largly based upon procedures:
//
// Written by: Knud Poder and Karsten Engsager
//
// Based on math from: R.Koenig and K.H. Weise, "Mathematische
// Grundlagen der hoeheren Geodaesie und Kartographie,
// Springer-Verlag, Berlin/Goettingen" Heidelberg, 1951.
//
// Modified and used here by permission of Reference Networks
// Division, Kort og Matrikelstyrelsen (KMS), Copenhagen, Denmark
//
/*****************************************************************************/

#pragma once

#include <cuproj/detail/utility/cuda.hpp>
#include <cuproj/detail/wrap_to_pi.hpp>
#include <cuproj/ellipsoid.hpp>
#include <cuproj/operation/operation.cuh>
#include <cuproj/projection_parameters.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <assert.h>

namespace cuproj {

/**
 * @addtogroup operations
 * @{
 */

namespace detail {

/**
 * @brief Evaluate Gaussian<-->Geographic trigonometric series
 *
 * @tparam T data type
 * @param p1  pointer to the first element of the array
 * @param len_p1  length of the array
 * @param B The argument to the trigonometric series
 * @param cos_2B precomputed cos(2*B)
 * @param sin_2B precomputed sin(2*B)
 * @return The value of the trigonometric series at B
 */
template <typename T>
inline static CUPROJ_HOST_DEVICE T gatg(T const* p1, int len_p1, T B, T cos_2B, T sin_2B)
{
  T h = 0, h1, h2 = 0;

  T const two_cos_2B = 2 * cos_2B;
  T const* p         = p1 + len_p1;
  h1                 = *--p;
  while (p - p1) {
    h  = -h2 + two_cos_2B * h1 + *--p;
    h2 = h1;
    h1 = h;
  }
  return (B + h * sin_2B);
}

/**
 * @brief  Clenshaw summation for complex numbers
 *
 * @see https://en.wikipedia.org/wiki/Clenshaw_algorithm
 *
 * @tparam T data type
 * @param a coefficients
 * @param size size of coefficients
 * @param sin_arg_r precomputed sin(arg_r)
 * @param cos_arg_r precomputed cos(arg_r)
 * @param sinh_arg_i precomputed sinh(arg_i)
 * @param cosh_arg_i precomputed cosh(arg_i)
 * @param R real part of the summation
 * @param I imaginary part of the summation
 * @return The real part of the summation at arg_r
 */
template <typename T>
inline static CUPROJ_HOST_DEVICE T clenshaw_complex(
  T const* a, int size, T sin_arg_r, T cos_arg_r, T sinh_arg_i, T cosh_arg_i, T* R, T* I)
{
  T r, i, hr, hr1, hr2, hi, hi1, hi2;

  // arguments
  T const* p = a + size;
  r          = 2 * cos_arg_r * cosh_arg_i;
  i          = -2 * sin_arg_r * sinh_arg_i;

  // summation loop
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

/**
 * @brief Clenshaw summation for real numbers
 *
 * @see https://en.wikipedia.org/wiki/Clenshaw_algorithm
 *
 * @tparam T data type
 * @param a pointer to array of coefficients
 * @param size number of coefficients
 * @param arg_r argument value
 * @return the summation at arg_r
 */
template <typename T>
static CUPROJ_HOST_DEVICE T clenshaw_real(T const* a, int size, T arg_r)
{
  T r, hr, hr1, hr2, cos_arg_r;

  T const* p = a + size;
  cos_arg_r  = cos(arg_r);
  r          = 2 * cos_arg_r;

  // summation loop
  hr1 = 0;
  hr  = *--p;
  for (; a - p;) {
    hr2 = hr1;
    hr1 = hr;
    hr  = -hr2 + r * hr1 + *--p;
  }
  return sin(arg_r) * hr;
}
}  // namespace detail

template <typename Coordinate, typename T = typename Coordinate::value_type>
class transverse_mercator : operation<Coordinate> {
 public:
  static constexpr int ETMERC_ORDER = 6;  ///< 6th order series expansion

  /**
   * @brief construct a transverse mercator projection
   *
   * @param params projection parameters
   */
  CUPROJ_HOST_DEVICE transverse_mercator(projection_parameters<T> const& params) : params_(params)
  {
  }

  /**
   * @brief Perform UTM projection for a single coordinate
   *
   * @param coord the coordinate to project
   * @param dir direction of projection
   * @return projected coordinate
   */
  CUPROJ_HOST_DEVICE Coordinate operator()(Coordinate const& coord, direction dir) const
  {
    if (dir == direction::FORWARD)
      return forward(coord);
    else
      return inverse(coord);
  }

  static constexpr T utm_central_meridian = T{500000};  // false easting center of UTM zone
  // false northing center of UTM zone
  static constexpr T utm_central_parallel(hemisphere h)
  {
    return (h == hemisphere::NORTH) ? T{0} : T{10000000};
  }

  /**
   * @brief Set up the projection parameters for transverse mercator projection
   *
   * @param input_params projection parameters
   * @return projection parameters modified for transverse mercator projection
   */
  projection_parameters<T> setup(projection_parameters<T> const& input_params)
  {
    params_ = input_params;

    // so we don't have to qualify the class name everywhere.
    auto& params       = params_;
    auto& tmerc_params = params_.tmerc_params_;
    auto& ellipsoid    = params_.ellipsoid_;

    assert(ellipsoid.es > 0);

    params.x0 = utm_central_meridian;
    params.y0 = utm_central_parallel(params.utm_hemisphere_);

    if (params.utm_zone_ > 0 && params.utm_zone_ <= 60) {
      --params.utm_zone_;
    } else {
      params.utm_zone_ = lround((floor((detail::wrap_to_pi(params.lam0_) + M_PI) * 30. / M_PI)));
      params.utm_zone_ = std::clamp(params.utm_zone_, 0, 59);
    }

    params.lam0_ = (params.utm_zone_ + .5) * M_PI / 30. - M_PI;
    params.k0    = T{0.9996};
    params.phi0  = T{0};

    // third flattening
    T const n = ellipsoid.n;
    T np      = n;

    // COEFFICIENTS OF TRIG SERIES GEO <-> GAUSS
    // cgb := Gaussian -> Geodetic, KW p190 - 191 (61) - (62)
    // cbg := Geodetic -> Gaussian, KW p186 - 187 (51) - (52)
    // ETMERC_ORDER = 6th degree : Engsager and Poder: ICC2007

    tmerc_params.cgb[0] =
      n *
      (2 + n * (-2 / 3.0 + n * (-2 + n * (116 / 45.0 + n * (26 / 45.0 + n * (-2854 / 675.0))))));
    tmerc_params.cbg[0] =
      n * (-2 + n * (2 / 3.0 +
                     n * (4 / 3.0 + n * (-82 / 45.0 + n * (32 / 45.0 + n * (4642 / 4725.0))))));
    np *= n;
    tmerc_params.cgb[1] =
      np * (7 / 3.0 + n * (-8 / 5.0 + n * (-227 / 45.0 + n * (2704 / 315.0 + n * (2323 / 945.0)))));
    tmerc_params.cbg[1] =
      np * (5 / 3.0 + n * (-16 / 15.0 + n * (-13 / 9.0 + n * (904 / 315.0 + n * (-1522 / 945.0)))));
    np *= n;
    // n^5 coeff corrected from 1262/105 -> -1262/105
    tmerc_params.cgb[2] =
      np * (56 / 15.0 + n * (-136 / 35.0 + n * (-1262 / 105.0 + n * (73814 / 2835.0))));
    tmerc_params.cbg[2] =
      np * (-26 / 15.0 + n * (34 / 21.0 + n * (8 / 5.0 + n * (-12686 / 2835.0))));
    np *= n;
    // n^5 coeff corrected from 322/35 -> 332/35
    tmerc_params.cgb[3] = np * (4279 / 630.0 + n * (-332 / 35.0 + n * (-399572 / 14175.0)));
    tmerc_params.cbg[3] = np * (1237 / 630.0 + n * (-12 / 5.0 + n * (-24832 / 14175.0)));
    np *= n;
    tmerc_params.cgb[4] = np * (4174 / 315.0 + n * (-144838 / 6237.0));
    tmerc_params.cbg[4] = np * (-734 / 315.0 + n * (109598 / 31185.0));
    np *= n;
    tmerc_params.cgb[5] = np * (601676 / 22275.0);
    tmerc_params.cbg[5] = np * (444337 / 155925.0);

    // Constants of the projections
    // Transverse Mercator (UTM, ITM, etc)
    np = n * n;
    // Norm. meridian quadrant, K&W p.50 (96), p.19 (38b), p.5 (2)
    tmerc_params.Qn = params.k0 / (1 + n) * (1 + np * (1 / 4.0 + np * (1 / 64.0 + np / 256.0)));
    // coef of trig series
    // utg := ell. N, E -> sph. N, E,  KW p194 (65)
    // gtu := sph. N, E -> ell. N, E,  KW p196 (69)
    tmerc_params.utg[0] =
      n * (-0.5 +
           n * (2 / 3.0 +
                n * (-37 / 96.0 + n * (1 / 360.0 + n * (81 / 512.0 + n * (-96199 / 604800.0))))));
    tmerc_params.gtu[0] =
      n * (0.5 + n * (-2 / 3.0 + n * (5 / 16.0 + n * (41 / 180.0 +
                                                      n * (-127 / 288.0 + n * (7891 / 37800.0))))));
    tmerc_params.utg[1] =
      np * (-1 / 48.0 +
            n * (-1 / 15.0 + n * (437 / 1440.0 + n * (-46 / 105.0 + n * (1118711 / 3870720.0)))));
    tmerc_params.gtu[1] =
      np * (13 / 48.0 +
            n * (-3 / 5.0 + n * (557 / 1440.0 + n * (281 / 630.0 + n * (-1983433 / 1935360.0)))));
    np *= n;
    tmerc_params.utg[2] =
      np * (-17 / 480.0 + n * (37 / 840.0 + n * (209 / 4480.0 + n * (-5569 / 90720.0))));
    tmerc_params.gtu[2] =
      np * (61 / 240.0 + n * (-103 / 140.0 + n * (15061 / 26880.0 + n * (167603 / 181440.0))));
    np *= n;
    tmerc_params.utg[3] = np * (-4397 / 161280.0 + n * (11 / 504.0 + n * (830251 / 7257600.0)));
    tmerc_params.gtu[3] = np * (49561 / 161280.0 + n * (-179 / 168.0 + n * (6601661 / 7257600.0)));
    np *= n;
    tmerc_params.utg[4] = np * (-4583 / 161280.0 + n * (108847 / 3991680.0));
    tmerc_params.gtu[4] = np * (34729 / 80640.0 + n * (-3418889 / 1995840.0));
    np *= n;
    tmerc_params.utg[5] = np * (-20648693 / 638668800.0);
    tmerc_params.gtu[5] = np * (212378941 / 319334400.0);

    // Gaussian latitude value of the origin latitude
    T const Z = detail::gatg(
      tmerc_params.cbg, ETMERC_ORDER, params.phi0, cos(2 * params.phi0), sin(2 * params.phi0));

    // Origin northing minus true northing at the origin latitude
    // i.e. true northing = N - P->Zb
    tmerc_params.Zb =
      -tmerc_params.Qn * (Z + detail::clenshaw_real(tmerc_params.gtu, ETMERC_ORDER, 2 * Z));

    return params;
  }

 private:
  /**
   * @brief  Forward projection, from geographic to transverse mercator.
   *
   * @param coord Geographic coordinate (lat, lon) in radians.
   * @return Transverse mercator coordinate (x, y) in meters.
   */
  CUPROJ_HOST_DEVICE Coordinate forward(Coordinate const& coord) const
  {
    // so we don't have to qualify the class name everywhere.
    auto& tmerc_params = this->params_.tmerc_params_;
    auto& ellipsoid    = this->params_.ellipsoid_;

    // ell. LAT, LNG -> Gaussian LAT, LNG
    T Cn =
      detail::gatg(tmerc_params.cbg, ETMERC_ORDER, coord.y, cos(2 * coord.y), sin(2 * coord.y));

    // Gaussian LAT, LNG -> compl. sph. LAT
    T const sin_Cn = sin(Cn);
    T const cos_Cn = cos(Cn);
    T const sin_Ce = sin(coord.x);
    T const cos_Ce = cos(coord.x);

    T const cos_Cn_cos_Ce = cos_Cn * cos_Ce;
    Cn                    = atan2(sin_Cn, cos_Cn_cos_Ce);

    T const inv_denom_tan_Ce = 1. / hypot(sin_Cn, cos_Cn_cos_Ce);
    T const tan_Ce           = sin_Ce * cos_Cn * inv_denom_tan_Ce;

    // Variant of the above: found not to be measurably faster
    // T const sin_Ce_cos_Cn = sin_Ce*cos_Cn;
    // T const denom = sqrt(1 - sin_Ce_cos_Cn * sin_Ce_cos_Cn);
    // T const tan_Ce = sin_Ce_cos_Cn / denom;

    // compl. sph. N, E -> ell. norm. N, E
    T Ce = asinh(tan_Ce); /* Replaces: Ce  = log(tan(FORTPI + Ce*0.5)); */

    //  Non-optimized version:
    //  T const sin_arg_r  = sin(2*Cn);
    //  T const cos_arg_r  = cos(2*Cn);
    //
    //  Given:
    //      sin(2 * Cn) = 2 sin(Cn) cos(Cn)
    //          sin(atan(y)) = y / sqrt(1 + y^2)
    //          cos(atan(y)) = 1 / sqrt(1 + y^2)
    //      ==> sin(2 * Cn) = 2 tan_Cn / (1 + tan_Cn^2)
    //
    //      cos(2 * Cn) = 2cos^2(Cn) - 1
    //                  = 2 / (1 + tan_Cn^2) - 1
    //
    T const two_inv_denom_tan_Ce        = 2 * inv_denom_tan_Ce;
    T const two_inv_denom_tan_Ce_square = two_inv_denom_tan_Ce * inv_denom_tan_Ce;
    T const tmp_r                       = cos_Cn_cos_Ce * two_inv_denom_tan_Ce_square;
    T const sin_arg_r                   = sin_Cn * tmp_r;
    T const cos_arg_r                   = cos_Cn_cos_Ce * tmp_r - 1;

    //  Non-optimized version:
    //  T const sinh_arg_i = sinh(2*Ce);
    //  T const cosh_arg_i = cosh(2*Ce);
    //
    //  Given
    //      sinh(2 * Ce) = 2 sinh(Ce) cosh(Ce)
    //          sinh(asinh(y)) = y
    //          cosh(asinh(y)) = sqrt(1 + y^2)
    //      ==> sinh(2 * Ce) = 2 tan_Ce sqrt(1 + tan_Ce^2)
    //
    //      cosh(2 * Ce) = 2cosh^2(Ce) - 1
    //                   = 2 * (1 + tan_Ce^2) - 1
    //
    // and 1+tan_Ce^2 = 1 + sin_Ce^2 * cos_Cn^2 / (sin_Cn^2 + cos_Cn^2 *
    // cos_Ce^2) = (sin_Cn^2 + cos_Cn^2 * cos_Ce^2 + sin_Ce^2 * cos_Cn^2) /
    // (sin_Cn^2 + cos_Cn^2 * cos_Ce^2) = 1. / (sin_Cn^2 + cos_Cn^2 * cos_Ce^2)
    // = inv_denom_tan_Ce^2

    T const sinh_arg_i = tan_Ce * two_inv_denom_tan_Ce;
    T const cosh_arg_i = two_inv_denom_tan_Ce_square - 1;

    T dCn, dCe;
    Cn += detail::clenshaw_complex(
      tmerc_params.gtu, ETMERC_ORDER, sin_arg_r, cos_arg_r, sinh_arg_i, cosh_arg_i, &dCn, &dCe);

    Ce += dCe;

    CUPROJ_HOST_DEVICE_EXPECTS(fabs(Ce) <= 2.623395162778,  // value comes from PROJ
                               "Coordinate transform outside projection domain");
    Coordinate xy{0.0, 0.0};
    xy.y = tmerc_params.Qn * Cn + tmerc_params.Zb;  // Northing
    xy.x = tmerc_params.Qn * Ce;                    // Easting

    return xy;
  }

  /**
   * @brief inverse transform, from projected to geographic coordinate
   * @param coord projected coordinate (x, y) in meters
   * @return geographic coordinate (lon, lat) in radians
   */
  CUPROJ_HOST_DEVICE Coordinate inverse(Coordinate const& coord) const
  {
    // so we don't have to qualify the class name everywhere.
    auto& tmerc_params = this->params_.tmerc_params_;
    auto& ellipsoid    = this->params_.ellipsoid_;

    // normalize N, E
    T Cn = (coord.y - tmerc_params.Zb) / tmerc_params.Qn;
    T Ce = coord.x / tmerc_params.Qn;

    CUPROJ_HOST_DEVICE_EXPECTS(fabs(Ce) <= 2.623395162778,  // value comes from PROJ
                               "Coordinate transform outside projection domain");

    // norm. N, E -> compl. sph. LAT, LNG
    T const sin_arg_r = sin(2 * Cn);
    T const cos_arg_r = cos(2 * Cn);

    // T const sinh_arg_i = sinh(2*Ce);
    // T const cosh_arg_i = cosh(2*Ce);
    T const exp_2_Ce          = exp(2 * Ce);
    T const half_inv_exp_2_Ce = T{0.5} / exp_2_Ce;
    T const sinh_arg_i        = T{0.5} * exp_2_Ce - half_inv_exp_2_Ce;
    T const cosh_arg_i        = T{0.5} * exp_2_Ce + half_inv_exp_2_Ce;

    T dCn_ignored, dCe;
    Cn += detail::clenshaw_complex(tmerc_params.utg,
                                   ETMERC_ORDER,
                                   sin_arg_r,
                                   cos_arg_r,
                                   sinh_arg_i,
                                   cosh_arg_i,
                                   &dCn_ignored,
                                   &dCe);
    Ce += dCe;

    // compl. sph. LAT -> Gaussian LAT, LNG
    T const sin_Cn = sin(Cn);
    T const cos_Cn = cos(Cn);

#if 0
        // Non-optimized version:
        T sin_Ce, cos_Ce;
        Ce = atan (sinh (Ce));  // Replaces: Ce = 2*(atan(exp(Ce)) - FORTPI);
        sin_Ce = sin (Ce);
        cos_Ce = cos (Ce);
        Ce     = atan2 (sin_Ce, cos_Ce*cos_Cn);
        Cn     = atan2 (sin_Cn*cos_Ce,  hypot (sin_Ce, cos_Ce*cos_Cn));
#else
    // One can divide both member of Ce = atan2(...) by cos_Ce, which
    // gives: Ce     = atan2 (tan_Ce, cos_Cn) = atan2(sinh(Ce), cos_Cn)
    //
    // and the same for Cn = atan2(...)
    // Cn     = atan2 (sin_Cn, hypot (sin_Ce, cos_Ce*cos_Cn)/cos_Ce)
    //        = atan2 (sin_Cn, hypot (sin_Ce/cos_Ce, cos_Cn))
    //        = atan2 (sin_Cn, hypot (tan_Ce, cos_Cn))
    //        = atan2 (sin_Cn, hypot (sinhCe, cos_Cn))
    T const sinhCe     = sinh(Ce);
    Ce                 = atan2(sinhCe, cos_Cn);
    T const modulus_Ce = hypot(sinhCe, cos_Cn);
    Cn                 = atan2(sin_Cn, modulus_Ce);
#endif

    // Gaussian LAT, LNG -> ell. LAT, LNG

    // Optimization of the computation of cos(2*Cn) and sin(2*Cn)
    T const tmp      = 2 * modulus_Ce / (sinhCe * sinhCe + 1);
    T const sin_2_Cn = sin_Cn * tmp;
    T const cos_2_Cn = tmp * modulus_Ce - 1.;
    // T const cos_2_Cn = cos(2 * Cn);
    // T const sin_2_Cn = sin(2 * Cn);

    return Coordinate{Ce, detail::gatg(tmerc_params.cgb, ETMERC_ORDER, Cn, cos_2_Cn, sin_2_Cn)};
  }

  /**
   * @brief Get the origin longitude
   *
   * @return the origin longitude in radians
   */
  T lam0() const { return this->params_.lam0_; }

 private:
  projection_parameters<T> params_{};
};

/**
 * @} // end of doxygen group
 */

}  // namespace cuproj
