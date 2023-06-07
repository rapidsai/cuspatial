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

#include <cuproj/axis_swap.cuh>
#include <cuproj/ellipsoid.hpp>
#include <cuproj/prepare_angular_coordinates.cuh>
#include <cuproj/projection.cuh>
#include <cuproj/to_radians.cuh>

#include <thrust/iterator/transform_iterator.h>

#include <assert.h>

namespace cuproj {

template <typename T>
inline static __host__ __device__ T gatg(const T* p1, int len_p1, T B, T cos_2B, T sin_2B)
{
  T h = 0, h1, h2 = 0;

  const T two_cos_2B = 2 * cos_2B;
  const T* p         = p1 + len_p1;
  h1                 = *--p;
  while (p - p1) {
    h  = -h2 + two_cos_2B * h1 + *--p;
    h2 = h1;
    h1 = h;
  }
  return (B + h * sin_2B);
}

/* Complex Clenshaw summation */
template <typename T>
inline static __host__ __device__ T
clenS(const T* a, int size, T sin_arg_r, T cos_arg_r, T sinh_arg_i, T cosh_arg_i, T* R, T* I)
{
  T r, i, hr, hr1, hr2, hi, hi1, hi2;

  /* arguments */
  const T* p = a + size;
  r          = 2 * cos_arg_r * cosh_arg_i;
  i          = -2 * sin_arg_r * sinh_arg_i;

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
template <typename T>
static __host__ __device__ T clens(const T* a, int size, T arg_r)
{
  T r, hr, hr1, hr2, cos_arg_r;

  const T* p = a + size;
  cos_arg_r  = cos(arg_r);
  r          = 2 * cos_arg_r;

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

template <typename Coordinate, typename T = typename Coordinate::value_type>
struct transverse_mercator {
  static constexpr int ETMERC_ORDER = 6;

  transverse_mercator(ellipsoid<T> ell, int zone) : ellipsoid_(ell) { setup(zone); }

  __host__ __device__ Coordinate operator()(Coordinate coord) const
  {
    Coordinate xy{0.0, 0.0};

    /* ell. LAT, LNG -> Gaussian LAT, LNG */
    T Cn = gatg(params_.cbg, ETMERC_ORDER, coord.y, cos(2 * coord.y), sin(2 * coord.y));

    /* Gaussian LAT, LNG -> compl. sph. LAT */
    const T sin_Cn = sin(Cn);
    const T cos_Cn = cos(Cn);
    const T sin_Ce = sin(coord.x);
    const T cos_Ce = cos(coord.x);

    const T cos_Cn_cos_Ce = cos_Cn * cos_Ce;
    Cn                    = atan2(sin_Cn, cos_Cn_cos_Ce);

    const T inv_denom_tan_Ce = 1. / hypot(sin_Cn, cos_Cn_cos_Ce);
    const T tan_Ce           = sin_Ce * cos_Cn * inv_denom_tan_Ce;

#if 0
    // Variant of the above: found not to be measurably faster
    const T sin_Ce_cos_Cn = sin_Ce*cos_Cn;
    const T denom = sqrt(1 - sin_Ce_cos_Cn * sin_Ce_cos_Cn);
    const T tan_Ce = sin_Ce_cos_Cn / denom;
#endif

    /* compl. sph. N, E -> ell. norm. N, E */
    T Ce = asinh(tan_Ce); /* Replaces: Ce  = log(tan(FORTPI + Ce*0.5)); */

    /*
     *  Non-optimized version:
     *  const T sin_arg_r  = sin(2*Cn);
     *  const T cos_arg_r  = cos(2*Cn);
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
    const T two_inv_denom_tan_Ce        = 2 * inv_denom_tan_Ce;
    const T two_inv_denom_tan_Ce_square = two_inv_denom_tan_Ce * inv_denom_tan_Ce;
    const T tmp_r                       = cos_Cn_cos_Ce * two_inv_denom_tan_Ce_square;
    const T sin_arg_r                   = sin_Cn * tmp_r;
    const T cos_arg_r                   = cos_Cn_cos_Ce * tmp_r - 1;

    /*
     *  Non-optimized version:
     *  const T sinh_arg_i = sinh(2*Ce);
     *  const T cosh_arg_i = cosh(2*Ce);
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
    const T sinh_arg_i = tan_Ce * two_inv_denom_tan_Ce;
    const T cosh_arg_i = two_inv_denom_tan_Ce_square - 1;

    T dCn, dCe;
    Cn +=
      clenS(params_.gtu, ETMERC_ORDER, sin_arg_r, cos_arg_r, sinh_arg_i, cosh_arg_i, &dCn, &dCe);
    Ce += dCe;
    // CUPROJ_EXPECTS(fabs(Ce) <= 2.623395162778, "Coordinate transform outside projection domain");
    xy.y = params_.Qn * Cn + params_.Zb;  // Northing
    xy.x = params_.Qn * Ce;               // Easting

    xy.x *= ellipsoid_.a;
    xy.y *= ellipsoid_.a;

    xy.x = /*P->fr_meter **/ (xy.x + params_.x0);
    xy.y = /*P->fr_meter **/ (xy.y + params_.y0);

    return xy;
  }

  T lam0() const { return params_.lam0; }

 private:
  ellipsoid<T> ellipsoid_{};

  struct params {
    T k0;          // scaling
    T lam0 = 0.0;  // central meridian
    T phi0 = 0.0;  // central parallel
    T x0   = 0.0;  // false easting
    T y0   = 0.0;  // false northing

    T Qn;     /* Merid. quad., scaled to the projection */
    T Zb;     /* Radius vector in polar coord. systems  */
    T cgb[6]; /* Constants for Gauss -> Geo lat */
    T cbg[6]; /* Constants for Geo lat -> Gauss */
    T utg[6]; /* Constants for transv. merc. -> geo */
    T gtu[6]; /* Constants for geo -> transv. merc. */
  };

  params params_;

  void setup(int zone)
  {
    assert(ellipsoid_.es > 0);

    params_.x0 = static_cast<T>(500000);
    params_.y0 = static_cast<T>(10000000);

    if (zone > 0 && zone <= 60) --zone;
    params_.lam0 = (zone + .5) * M_PI / 30. - M_PI;
    params_.k0   = T{0.9996};
    params_.phi0 = T{0};

    /* third flattening */
    const T n = ellipsoid_.n;
    T np      = n;

    /* COEF. OF TRIG SERIES GEO <-> GAUSS */
    /* cgb := Gaussian -> Geodetic, KW p190 - 191 (61) - (62) */
    /* cbg := Geodetic -> Gaussian, KW p186 - 187 (51) - (52) */
    /* PROJ_ETMERC_ORDER = 6th degree : Engsager and Poder: ICC2007 */

    params_.cgb[0] =
      n *
      (2 + n * (-2 / 3.0 + n * (-2 + n * (116 / 45.0 + n * (26 / 45.0 + n * (-2854 / 675.0))))));
    params_.cbg[0] =
      n * (-2 + n * (2 / 3.0 +
                     n * (4 / 3.0 + n * (-82 / 45.0 + n * (32 / 45.0 + n * (4642 / 4725.0))))));
    np *= n;
    params_.cgb[1] =
      np * (7 / 3.0 + n * (-8 / 5.0 + n * (-227 / 45.0 + n * (2704 / 315.0 + n * (2323 / 945.0)))));
    params_.cbg[1] =
      np * (5 / 3.0 + n * (-16 / 15.0 + n * (-13 / 9.0 + n * (904 / 315.0 + n * (-1522 / 945.0)))));
    np *= n;
    /* n^5 coeff corrected from 1262/105 -> -1262/105 */
    params_.cgb[2] =
      np * (56 / 15.0 + n * (-136 / 35.0 + n * (-1262 / 105.0 + n * (73814 / 2835.0))));
    params_.cbg[2] = np * (-26 / 15.0 + n * (34 / 21.0 + n * (8 / 5.0 + n * (-12686 / 2835.0))));
    np *= n;
    /* n^5 coeff corrected from 322/35 -> 332/35 */
    params_.cgb[3] = np * (4279 / 630.0 + n * (-332 / 35.0 + n * (-399572 / 14175.0)));
    params_.cbg[3] = np * (1237 / 630.0 + n * (-12 / 5.0 + n * (-24832 / 14175.0)));
    np *= n;
    params_.cgb[4] = np * (4174 / 315.0 + n * (-144838 / 6237.0));
    params_.cbg[4] = np * (-734 / 315.0 + n * (109598 / 31185.0));
    np *= n;
    params_.cgb[5] = np * (601676 / 22275.0);
    params_.cbg[5] = np * (444337 / 155925.0);

    /* Constants of the projections */
    /* Transverse Mercator (UTM, ITM, etc) */
    np = n * n;
    /* Norm. mer. quad, K&W p.50 (96), p.19 (38b), p.5 (2) */
    params_.Qn = params_.k0 / (1 + n) * (1 + np * (1 / 4.0 + np * (1 / 64.0 + np / 256.0)));
    /* coef of trig series */
    /* utg := ell. N, E -> sph. N, E,  KW p194 (65) */
    /* gtu := sph. N, E -> ell. N, E,  KW p196 (69) */
    params_.utg[0] =
      n * (-0.5 +
           n * (2 / 3.0 +
                n * (-37 / 96.0 + n * (1 / 360.0 + n * (81 / 512.0 + n * (-96199 / 604800.0))))));
    params_.gtu[0] =
      n * (0.5 + n * (-2 / 3.0 + n * (5 / 16.0 + n * (41 / 180.0 +
                                                      n * (-127 / 288.0 + n * (7891 / 37800.0))))));
    params_.utg[1] =
      np * (-1 / 48.0 +
            n * (-1 / 15.0 + n * (437 / 1440.0 + n * (-46 / 105.0 + n * (1118711 / 3870720.0)))));
    params_.gtu[1] =
      np * (13 / 48.0 +
            n * (-3 / 5.0 + n * (557 / 1440.0 + n * (281 / 630.0 + n * (-1983433 / 1935360.0)))));
    np *= n;
    params_.utg[2] =
      np * (-17 / 480.0 + n * (37 / 840.0 + n * (209 / 4480.0 + n * (-5569 / 90720.0))));
    params_.gtu[2] =
      np * (61 / 240.0 + n * (-103 / 140.0 + n * (15061 / 26880.0 + n * (167603 / 181440.0))));
    np *= n;
    params_.utg[3] = np * (-4397 / 161280.0 + n * (11 / 504.0 + n * (830251 / 7257600.0)));
    params_.gtu[3] = np * (49561 / 161280.0 + n * (-179 / 168.0 + n * (6601661 / 7257600.0)));
    np *= n;
    params_.utg[4] = np * (-4583 / 161280.0 + n * (108847 / 3991680.0));
    params_.gtu[4] = np * (34729 / 80640.0 + n * (-3418889 / 1995840.0));
    np *= n;
    params_.utg[5] = np * (-20648693 / 638668800.0);
    params_.gtu[5] = np * (212378941 / 319334400.0);

    /* Gaussian latitude value of the origin latitude */
    const T Z =
      gatg(params_.cbg, ETMERC_ORDER, params_.phi0, cos(2 * params_.phi0), sin(2 * params_.phi0));

    /* Origin northing minus true northing at the origin latitude */
    /* i.e. true northing = N - P->Zb                         */
    params_.Zb = -params_.Qn * (Z + clens(params_.gtu, ETMERC_ORDER, 2 * Z));
  }
};

}  // namespace cuproj
