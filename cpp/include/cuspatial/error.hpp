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

#include <stdexcept>
#include <string>

namespace cuspatial {

/**---------------------------------------------------------------------------*
 * @brief Exception thrown when logical precondition is violated.
 *
 * This exception should not be thrown directly and is instead thrown by the
 * CUSPATIAL_EXPECTS macro.
 *
 *---------------------------------------------------------------------------**/
struct logic_error : public std::logic_error {
  logic_error(char const* const message) : std::logic_error(message) {}
  logic_error(std::string const& message) : std::logic_error(message) {}
};

}  // namespace cuspatial

#define STRINGIFY_DETAIL(x)    #x
#define CUSPATIAL_STRINGIFY(x) STRINGIFY_DETAIL(x)

/**---------------------------------------------------------------------------*
 * @brief Macro for checking (pre-)conditions that throws an exception when
 * a condition is violated.
 *
 * Example usage:
 *
 * @code
 * CUSPATIAL_EXPECTS(lhs->dtype == rhs->dtype, "Column type mismatch");
 * @endcode
 *
 * @param[in] cond Expression that evaluates to true or false
 * @param[in] reason String literal description of the reason that cond is
 * expected to be true
 * @throw cuspatial::logic_error if the condition evaluates to false.
 *---------------------------------------------------------------------------**/
#define CUSPATIAL_EXPECTS(cond, reason)                                       \
  (!!(cond)) ? static_cast<void>(0)                                           \
             : throw cuspatial::logic_error("cuSpatial failure at: " __FILE__ \
                                            ":" CUSPATIAL_STRINGIFY(__LINE__) ": " reason)

/**---------------------------------------------------------------------------*
 * @brief Indicates that an erroneous code path has been taken.
 *
 * In host code, throws a `cuspatial::logic_error`.
 *
 *
 * Example usage:
 * ```
 * CUSPATIAL_FAIL("Non-arithmetic operation is not supported");
 * ```
 *
 * @param[in] reason String literal description of the reason
 *---------------------------------------------------------------------------**/
#define CUSPATIAL_FAIL(reason)                                   \
  throw cuspatial::logic_error("cuSpatial failure at: " __FILE__ \
                               ":" CUSPATIAL_STRINGIFY(__LINE__) ": " reason)

namespace cuspatial {
namespace detail {

}  // namespace detail
}  // namespace cuspatial
