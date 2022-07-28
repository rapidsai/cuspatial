/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cusparse.h>

#include <stdexcept>

namespace cuspatial {

/**---------------------------------------------------------------------------*
 * @brief Exception thrown when logical precondition is violated.
 *
 * This exception should not be thrown directly and is instead thrown by the
 * CUSPARSE_TRY macro.
 *
 *---------------------------------------------------------------------------**/
struct cusparse_error : public std::runtime_error {
  cusparse_error(std::string const& message) : std::runtime_error(message) {}
};

namespace detail {

inline void throw_cusparse_error(cusparseStatus_t error, const char* file, unsigned int line)
{
  // would be nice to include `cusparseGetErrorName(error)` and
  // `cusparseGetErrorString(error)`, but those aren't introduced until
  // cuda 10.1 (and not in the initial release).
  throw cuspatial::cusparse_error(
    std::string{"CUSPARSE error encountered at: " + std::string{file} + ":" + std::to_string(line) +
                ": " + std::to_string(error)});
}

}  // namespace detail
}  // namespace cuspatial

/**---------------------------------------------------------------------------*
 * @brief Error checking macro for cuSPARSE runtime API functions.
 *
 * Invokes a cuSPARSE runtime API function call, if the call does not return
 * CUSPARSE_STATUS_SUCCESS, throws an exception detailing the cuSPARSE error
 * that occurred.
 *
 *---------------------------------------------------------------------------**/
#define CUSPARSE_TRY(call)                                                 \
  do {                                                                     \
    cusparseStatus_t status = (call);                                      \
    if (CUSPARSE_STATUS_SUCCESS != status) {                               \
      cuspatial::detail::throw_cusparse_error(status, __FILE__, __LINE__); \
    }                                                                      \
  } while (0);
