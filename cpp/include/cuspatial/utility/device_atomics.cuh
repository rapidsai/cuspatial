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
__device__ double atomicMin(double* addr, double val);

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
__device__ float atomicMin(float* addr, float val);

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
__device__ double atomicMax(double* addr, double val);

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
__device__ float atomicMax(float* addr, float val);

}  // namespace detail
}  // namespace cuspatial
