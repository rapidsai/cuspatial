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

namespace {

template <typename T>
const T& __device__ min(const T& a, const T& b)
{
  return (a < b) ? a : b;
}

template <typename T>
const T& __device__ max(const T& a, const T& b)
{
  return (a > b) ? a : b;
}

template <typename T,
          typename RepresentationType,
          typename OpType,
          typename ToRepFuncType,
          typename FromRepFuncType>
__device__ T
atomicOp(T* addr, T val, OpType op, ToRepFuncType toRepFunc, FromRepFuncType fromRepFunc)
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

}  // namespace

namespace cuspatial {
namespace detail {

__device__ double atomicMin(double* addr, double val)
{
  atomicOp<double, unsigned long long int>(
    addr, val, min<double>, __double_as_longlong, __longlong_as_double);
}

__device__ float atomicMin(float* addr, float val)
{
  atomicOp<float, unsigned int>(addr, val, min<float>, __float_as_uint, __uint_as_float);
}

__device__ double atomicMax(double* addr, double val)
{
  atomicOp<double, unsigned long long int>(
    addr, val, max<double>, __double_as_longlong, __longlong_as_double);
}

__device__ float atomicMax(float* addr, float val)
{
  atomicOp<float, unsigned int>(addr, val, max<float>, __float_as_uint, __uint_as_float);
}

}  // namespace detail
}  // namespace cuspatial
