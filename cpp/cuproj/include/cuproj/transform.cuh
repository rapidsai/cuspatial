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

#include <cuproj/projection.cuh>
#include <cuproj/transverse_mercator.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>

namespace cuproj {

enum class direction { DIR_FWD, DIR_INV };

template <class CoordIter,
          typename Coordinate = typename CoordIter::value_type,
          typename T          = typename Coordinate::value_type>
void transform(projection<T> proj,
               CoordIter first,
               CoordIter last,
               CoordIter result,
               direction dir,
               rmm::cuda_stream_view stream = rmm::cuda_stream_default)
{
  // currently only supports forward UTM transform from WGS84
  assert(dir == direction::DIR_FWD);

  auto utm     = transverse_mercator<Coordinate>{proj.ellipsoid_, proj.utm_zone_};
  auto swap    = axis_swap<Coordinate>{};
  auto radians = to_radians<Coordinate>{};
  auto prep    = prepare_angular_coordinates<Coordinate>(utm.lam0(), T{0});

  auto pipeline = [=] __device__(auto c) { return utm(prep(radians(swap(c)))); };

  thrust::transform(rmm::exec_policy(stream), first, last, result, pipeline);
}

}  // namespace cuproj
