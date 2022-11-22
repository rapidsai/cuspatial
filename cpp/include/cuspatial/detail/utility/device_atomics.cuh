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

#include <thrust/detail/raw_reference_cast.h>
#include <thrust/device_ptr.h>
#include <thrust/device_reference.h>

#include <cuda/atomic>

#include <type_traits>

namespace cuspatial {
namespace detail {

/**
 * @internal
 * @brief A helper function to help lookup the correct overload of CUDA intrinsic
 * function min.
 */
template <typename T>
const T __device__ min_(const T a, const T b)
{
  return ::min(a, b);
}

/**
 * @internal
 * @brief A helper function to help lookup the correct overload of CUDA intrinsic
 * function max.
 */
template <typename T>
const T __device__ max_(const T a, const T b)
{
  return ::max(a, b);
}

/**
 * @internal
 * @brief General implementation for atomic ops.
 *
 * Reads the value from `addr`, performs `op(*addr, val)`, and stores the result
 * to `addr` in one atomic transaction.
 *
 * @tparam T The type value to apply atomic operation to
 * @tparam RepresentationType The unsigned integer type that has the same bit width as `T`
 * @tparam OpType The type of the atomic operation
 * @tparam ToRepFuncType The type of function to cast `T` to `RepresentationType`
 * @tparam FromRepFuncType The type of function to cast `RepresentationType` to T
 * @param addr The address where the atomic operation will be performed
 * @param val The right hand side value of the opeartion
 * @param op The atomic operation to perform
 * @param to_rep_func The function to cast `T` to `RepresentationType`, see notes below.
 * @param from_rep_func The function to cast `RepresentationType` to `T`, see notes below.
 * @return
 *
 * @note based on https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
 * @note `T`, `RepresentationType`, `to_rep_func` and `from_rep_func` are correlated. 32-bit floats
 * corresponds to `unsigned int`, `__float_as_uint`, `__uint_as_float` respectively. 64-bit floats
 * corresponds to `unsigned long long int`, `__double_as_longlong`, `__longlong_as_double`
 * respectively.
 */
template <typename T,
          typename RepresentationType,
          typename OpType,
          typename ToRepFuncType,
          typename FromRepFuncType>
__device__ T
atomic_op_impl(T* addr, T val, OpType op, ToRepFuncType to_rep_func, FromRepFuncType from_rep_func)
{
  RepresentationType* address_as_ll = reinterpret_cast<RepresentationType*>(addr);
  RepresentationType old            = to_rep_func(*addr);
  RepresentationType assumed;

  do {
    assumed = old;
    old     = atomicCAS(address_as_ll, assumed, to_rep_func(op(from_rep_func(assumed), val)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return from_rep_func(old);
}

/**
 * @internal
 * @brief `float` specialization for `atomic_op_impl`
 */
template <typename T, typename OpType>
__device__ std::enable_if_t<std::is_same_v<T, double>, T> inline atomicOp(T* addr, T val, OpType op)
{
  return atomic_op_impl<double, unsigned long long int>(
    addr, val, op, __double_as_longlong, __longlong_as_double);
}

/**
 * @internal
 * @brief `double` specialization for `atomic_op_impl`
 */
template <typename T, typename OpType>
__device__ std::enable_if_t<std::is_same_v<T, float>, T> inline atomicOp(T* addr, T val, OpType op)
{
  return atomic_op_impl<float, unsigned int>(addr, val, op, __float_as_uint, __uint_as_float);
}

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
  return atomicOp<double>(addr, val, min_<double>);
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
  return atomicOp<float>(addr, val, min_<float>);
}

/**
 * @internal
 * @brief CUDA device atomic minimum for double
 *
 * Atomically computes the min of the value stored in `ptr` and `val` and stores it in `ptr`,
 * returning the previous value of `*ptr`.
 *
 * @note based on https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
 *
 * @param ptr The thrust device pointer to the address to atomically compare and update with the
 * minimum.
 * @param val The value to compare
 * @return The old value stored in `addr`.
 */
__device__ inline double atomicMin(thrust::device_ptr<double> ptr, double val)
{
  return atomicMin(thrust::raw_pointer_cast(ptr), val);
}

/**
 * @internal
 * @brief CUDA device atomic minimum for float
 *
 * Atomically computes the min of the value stored in `ptr` and `val` and stores it in `ptr`,
 * returning the previous value of `*ptr`.
 *
 * @note based on https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
 *
 * @param ptr The thrust device pointer to address to atomically compare and update with the
 * minimum.
 * @param val The value to compare
 * @return The old value stored in `addr`.
 */
__device__ inline float atomicMin(thrust::device_ptr<float> ptr, float val)
{
  return atomicMin(thrust::raw_pointer_cast(ptr), val);
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
  return atomicOp<double>(addr, val, max_<double>);
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
  return atomicOp<float>(addr, val, max_<float>);
}

/**
 * @internal
 * @brief CUDA device atomic maximum for double
 *
 * Atomically computes the max of the value stored in `addr` and `val` and stores it in `ptr`,
 * returning the previous value of `*ptr`.
 *
 * @note based on https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
 *
 * @param ptr The thrust device pointer to the address atomically compare and update with the
 * maximum.
 * @param val The value to compare
 * @return The old value stored in `addr`.
 */
__device__ inline double atomicMax(thrust::device_ptr<double> ptr, double val)
{
  return atomicMax(thrust::raw_pointer_cast(ptr), val);
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
 * @param addr The thrust device pointer to the address to atomically compare and update with the
 * maximum.
 * @param val The value to compare
 * @return The old value stored in `addr`.
 */
__device__ inline float atomicMax(thrust::device_ptr<float> ptr, float val)
{
  return atomicMax(thrust::raw_pointer_cast(ptr), val);
}

/**
 * @brief Factory function to create atomic_ref from a thrust::device_reference
 */
template <cuda::thread_scope Scope, typename T>
auto __device__ make_atomic_ref(thrust::device_reference<T> ref)
{
  T& raw_ref = thrust::raw_reference_cast(ref);
  return cuda::atomic_ref<T, Scope>{raw_ref};
}

/**
 * @brief Factory function to create atomic_ref from raw reference
 */
template <cuda::thread_scope Scope, typename T>
auto __device__ make_atomic_ref(T& ref)
{
  return cuda::atomic_ref<T, Scope>{ref};
}

}  // namespace detail
}  // namespace cuspatial
