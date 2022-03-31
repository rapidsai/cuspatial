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

/**
 * @brief Exception thrown when a CUDA error is encountered.
 *
 */
struct cuda_error : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

}  // namespace cuspatial

#define CUSPATIAL_STRINGIFY_DETAIL(x) #x
#define CUSPATIAL_STRINGIFY(x)        CUSPATIAL_STRINGIFY_DETAIL(x)

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

/**
 * @brief Error checking macro for CUDA runtime API functions.
 *
 * Invokes a CUDA runtime API function call. If the call does not return
 * `cudaSuccess`, invokes cudaGetLastError() to clear the error and throws an
 * exception detailing the CUDA error that occurred
 *
 * Defaults to throwing `cuspatial::cuda_error`, but a custom exception may also be
 * specified.
 *
 * Example:
 * ```c++
 *
 * // Throws `cuspatial::cuda_error` if `cudaMalloc` fails
 * CUSPATIAL_CUDA_TRY(cudaMalloc(&p, 100));
 *
 * // Throws `std::runtime_error` if `cudaMalloc` fails
 * CUSPATIAL_CUDA_TRY(cudaMalloc(&p, 100), std::runtime_error);
 * ```
 *
 */
#define CUSPATIAL_CUDA_TRY(...)                                                         \
  GET_CUSPATIAL_CUDA_TRY_MACRO(__VA_ARGS__, CUSPATIAL_CUDA_TRY_2, CUSPATIAL_CUDA_TRY_1) \
  (__VA_ARGS__)
#define GET_CUSPATIAL_CUDA_TRY_MACRO(_1, _2, NAME, ...) NAME
#define CUSPATIAL_CUDA_TRY_2(_call, _exception_type)                                               \
  do {                                                                                             \
    cudaError_t const error = (_call);                                                             \
    if (cudaSuccess != error) {                                                                    \
      cudaGetLastError();                                                                          \
      /*NOLINTNEXTLINE(bugprone-macro-parentheses)*/                                               \
      throw _exception_type{std::string{"CUDA error at: "} + __FILE__ + ":" +                      \
                            CUSPATIAL_STRINGIFY(__LINE__) + ": " + cudaGetErrorName(error) + " " + \
                            cudaGetErrorString(error)};                                            \
    }                                                                                              \
  } while (0)
#define CUSPATIAL_CUDA_TRY_1(_call) CUSPATIAL_CUDA_TRY_2(_call, cuspatial::cuda_error)

namespace cuspatial {
namespace detail {

}  // namespace detail
}  // namespace cuspatial
