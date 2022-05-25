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

namespace cuspatial {
namespace detail {

__device__ double atomicMin(double* addr, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)addr;
  unsigned long long int old             = *address_as_ull, assumed;

  do {
    assumed = old;
    old     = atomicCAS(
      address_as_ull, assumed, __double_as_longlong(std::min(val, __longlong_as_double(assumed))));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}

__device__ float atomicMin(float* addr, float val)
{
  unsigned int* address_as_ull = (unsigned int*)addr;
  unsigned int old             = *address_as_ull, assumed;

  do {
    assumed = old;
    old =
      atomicCAS(address_as_ull, assumed, __float_as_uint(std::min(val, __uint_as_float(assumed))));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __uint_as_float(old);
}

__device__ double atomicMax(double* addr, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)addr;
  unsigned long long int old             = *address_as_ull, assumed;

  do {
    assumed = old;
    old     = atomicCAS(
      address_as_ull, assumed, __double_as_longlong(std::max(val, __longlong_as_double(assumed))));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}

__device__ float atomicMax(float* addr, float val)
{
  unsigned int* address_as_ull = (unsigned int*)addr;
  unsigned int old             = *address_as_ull, assumed;

  do {
    assumed = old;
    old =
      atomicCAS(address_as_ull, assumed, __float_as_uint(std::max(val, __uint_as_float(assumed))));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __uint_as_float(old);
}

}  // namespace detail
}  // namespace cuspatial
