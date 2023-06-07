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

#include <cuda_runtime.h>

/**
 * @brief `assert`-like macro for device code
 *
 * This is effectively the same as the standard `assert` macro, except it
 * relies on the `__PRETTY_FUNCTION__` macro which is specific to GCC and Clang
 * to produce better assert messages.
 */
#if !defined(NDEBUG) && defined(__CUDA_ARCH__) && (defined(__clang__) || defined(__GNUC__))
#define __ASSERT_STR_HELPER(x) #x
#define cuspatial_assert(e)   \
  ((e) ? static_cast<void>(0) \
       : __assert_fail(__ASSERT_STR_HELPER(e), __FILE__, __LINE__, __PRETTY_FUNCTION__))
#else
#define cuspatial_assert(e) (static_cast<void>(0))
#endif
