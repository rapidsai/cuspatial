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

#include <cmath>

namespace cuproj {

/**
 * @addtogroup constants
 * @{
 */

// TODO, use C++20 numerical constants when we can

/// Pi * 2.0
template <typename T>
static constexpr T M_TWOPI = T{2.0} * M_PI;  // 6.283185307179586476925286766559005l;

/// Epsilon in radians used for hysteresis in wrapping angles to e.g. [-pi,pi]
template <typename T>
static constexpr T EPSILON_RADIANS = T{1e-12};

/// Conversion factor from degrees to radians
template <typename T>
constexpr T DEG_TO_RAD = T{0.017453292519943295769236907684886};

/// Conversion factor from radians to degrees
template <typename T>
constexpr T RAD_TO_DEG = T{57.295779513082320876798154814105};

/**
 * @} // end of doxygen group
 */

}  // namespace cuproj
