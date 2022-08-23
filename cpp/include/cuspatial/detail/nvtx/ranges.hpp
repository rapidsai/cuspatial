/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "nvtx3.hpp"

namespace cuspatial {
/**
 * @brief Tag type for libcudf's NVTX domain.
 */
struct libcuspatial_domain {
  static constexpr char const* name{"libcuspatial"};  ///< Name of the libcudf domain
};

/**
 * @brief Alias for an NVTX range in the libcudf domain.
 */
using thread_range = ::nvtx3::domain_thread_range<libcudf_domain>;

}  // namespace cuspatial

/**
 * @brief Convenience macro for generating an NVTX range in the `libcuspatial` domain
 * from the lifetime of a function.
 *
 * Uses the name of the immediately enclosing function returned by `__func__` to
 * name the range.
 *
 * Example:
 * ```
 * void some_function(){
 *    CUSPATIAL_FUNC_RANGE();
 *    ...
 * }
 * ```
 */
#define CUSPATIAL_FUNC_RANGE() NVTX3_FUNC_RANGE_IN(cuspatial::libcuspatial_domain)
