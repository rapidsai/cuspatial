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

#pragma once

#include <cassert>
#include <cmath>

namespace cuproj {

template <typename T>
struct ellipsoid {
  ellipsoid() = default;

  ellipsoid(T a, T inverse_flattening) : a(a)
  {
    assert(inverse_flattening != 0.0);
    b     = a * (1. - 1. / inverse_flattening);
    f     = 1.0 / inverse_flattening;
    es    = 2 * f - f * f;
    e     = sqrt(es);
    alpha = asin(e);
    n     = pow(tan(alpha / 2), 2);
  }

  T a{};      // semi-major axis
  T b{};      // semi-minor axis
  T e{};      // first eccentricity
  T es{};     // first eccentricity squared
  T alpha{};  // angular eccentricity
  T f{};      // flattening
  T n{};      // third flattening
};

}  // namespace cuproj
