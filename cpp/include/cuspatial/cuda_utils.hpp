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
#include <utility>

#ifdef __CUDACC__
#define CUSPATIAL_HOST_DEVICE __host__ __device__
#else
#define CUSPATIAL_HOST_DEVICE
#endif

/**
 * @brief Return kernel launch parameters for 1D grid with total `n` threads.
 *
 * @tparam threads_per_block Number of threads per block
 * @param n Number of threads
 * @return Threads per block and number of blocks
 */
template <std::size_t threads_per_block = 256>
std::pair<std::size_t, std::size_t> constexpr grid_1d(std::size_t const n)
{
  return {threads_per_block, (n + threads_per_block - 1) / threads_per_block};
}
