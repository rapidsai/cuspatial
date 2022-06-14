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

#include <algorithm>
#include <type_traits>

namespace {

template <typename T>
const T& __device__ min(const T& a, const T& b)
{
  return std::min(a, b);
}

template <typename T>
const T& __device__ max(const T& a, const T& b)
{
  return std::max(a, b);
}

template <typename T,
          typename RepresentationType,
          typename OpType,
          typename ToRepFuncType,
          typename FromRepFuncType>
__device__ T
atomicOpImpl(T* addr, T val, OpType op, ToRepFuncType toRepFunc, FromRepFuncType fromRepFunc)
{
  RepresentationType* address_as_ll = reinterpret_cast<RepresentationType*>(addr);
  RepresentationType old            = toRepFunc(*addr);
  RepresentationType assumed;

  do {
    assumed = old;
    old     = atomicCAS(address_as_ll, assumed, toRepFunc(op(val, fromRepFunc(assumed))));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return fromRepFunc(old);
}

template <typename T, typename OpType>
__device__ std::enable_if_t<std::is_same_v<T, double>, T> atomicOp(T* addr, T val, OpType op)
{
  return atomicOpImpl<double, unsigned long long int>(
    addr, val, op, __double_as_longlong, __longlong_as_double);
}

template <typename T, typename OpType>
__device__ std::enable_if_t<std::is_same_v<T, float>, T> atomicOp(T* addr, T val, OpType op)
{
  return atomicOpImpl<float, unsigned int>(addr, val, op, __float_as_uint, __uint_as_float);
}

}  // namespace

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
__device__ double atomicMin(double* addr, double val)
{
  return atomicOp<double>(addr, val, min<double>);
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
__device__ float atomicMin(float* addr, float val)
{
  return atomicOp<float>(addr, val, min<float>);
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
__device__ double atomicMax(double* addr, double val)
{
  return atomicOp<double>(addr, val, max<double>);
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
__device__ float atomicMax(float* addr, float val)
{
  return atomicOp<float>(addr, val, max<float>);
}

}  // namespace detail
}  // namespace cuspatial
