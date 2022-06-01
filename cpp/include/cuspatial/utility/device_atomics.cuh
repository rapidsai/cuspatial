/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

namespace cuspatial {
namespace detail {

/**
 * @internal
 * @brief CUDA device atomic minimum for double
 *
 * Atomically computes the min of the value stored in `addr` and `val` and stores it in `addr`,
 * returning the previous value of `*addr`.
 *
 * @note based on https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
 *
 * @param addr The address to atomically compare and update with the minimum.
 * @param val The value to compare
 * @return The old value stored in `addr`.
 */
__device__ inline double atomicMin(double* addr, double val)
{
  unsigned long long int* address_as_ll = reinterpret_cast<unsigned long long int*>(addr);
  unsigned long long int old            = __double_as_longlong(*addr);
  unsigned long long int assumed;

  do {
    assumed = old;
    old     = atomicCAS(
      address_as_ll, assumed, __double_as_longlong(std::min(val, __longlong_as_double(assumed))));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}

/**
 * @internal
 * @brief CUDA device atomic minimum for float
 *
 * Atomically computes the min of the value stored in `addr` and `val` and stores it in `addr`,
 * returning the previous value of `*addr`.
 *
 * @note based on https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
 *
 * @param addr The address to atomically compare and update with the minimum.
 * @param val The value to compare
 * @return The old value stored in `addr`.
 */
__device__ inline float atomicMin(float* addr, float val)
{
  unsigned int* address_as_ui = reinterpret_cast<unsigned int*>(addr);
  unsigned int old            = __float_as_uint(*addr);
  unsigned int assumed;

  do {
    assumed = old;
    old =
      atomicCAS(address_as_ui, assumed, __float_as_uint(std::min(val, __uint_as_float(assumed))));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __uint_as_float(old);
}

/**
 * @internal
 * @brief CUDA device atomic maximum for double
 *
 * Atomically computes the max of the value stored in `addr` and `val` and stores it in `addr`,
 * returning the previous value of `*addr`.
 *
 * @note based on https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
 *
 * @param addr The address to atomically compare and update with the max.
 * @param val The value to compare
 * @return The old value stored in `addr`.
 */
__device__ inline double atomicMax(double* addr, double val)
{
  unsigned long long int* address_as_ll = reinterpret_cast<unsigned long long int*>(addr);
  unsigned long long int old            = __double_as_longlong(*addr);
  unsigned long long int assumed;

  do {
    assumed = old;
    old     = atomicCAS(
      address_as_ll, assumed, __double_as_longlong(std::max(val, __longlong_as_double(assumed))));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}

/**
 * @internal
 * @brief CUDA device atomic maximum for float
 *
 * Atomically computes the max of the value stored in `addr` and `val` and stores it in `addr`,
 * returning the previous value of `*addr`.
 *
 * @note based on https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
 *
 * @param addr The address to atomically compare and update with the max.
 * @param val The value to compare
 * @return The old value stored in `addr`.
 */
__device__ inline float atomicMax(float* addr, float val)
{
  unsigned int* address_as_ui = reinterpret_cast<unsigned int*>(addr);
  unsigned int old            = __float_as_uint(*addr);
  unsigned int assumed;

  do {
    assumed = old;
    old =
      atomicCAS(address_as_ui, assumed, __float_as_uint(std::max(val, __uint_as_float(assumed))));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __uint_as_float(old);
}

}  // namespace detail
}  // namespace cuspatial
