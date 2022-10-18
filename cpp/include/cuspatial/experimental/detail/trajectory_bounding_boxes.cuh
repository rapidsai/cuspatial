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

#include <cuspatial/traits.hpp>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/zip_function.h>

namespace cuspatial {

namespace detail {

template <typename T>
struct box_minmax {
  using point_tuple = thrust::tuple<cuspatial::vec_2d<T>, cuspatial::vec_2d<T>>;
  __host__ __device__ point_tuple operator()(point_tuple const& a, point_tuple const& b)
  {
    // structured binding doesn't seem to work with thrust::tuple
    vec_2d<T> p1, p2, p3, p4;
    thrust::tie(p1, p2) = a;
    thrust::tie(p3, p4) = b;
    using cuspatial::min;
    return {min(min(p1, p2), p3), max(max(p1, p2), p4)};
  }
};

}  // namespace detail

template <typename IdInputIt, typename PointInputIt, typename PointOutputIt>
std::pair<PointOutputIt, PointOutputIt> trajectory_bounding_boxes(
  IdInputIt ids_first,
  IdInputIt ids_last,
  PointInputIt points_first,
  PointOutputIt bounding_box_minima_first,
  PointOutputIt bounding_box_maxima_first,
  rmm::cuda_stream_view stream)
{
  using T      = iterator_vec_base_type<PointInputIt>;
  using IdType = iterator_value_type<IdInputIt>;

  auto points_zipped_first = thrust::make_zip_iterator(points_first, points_first);

  auto extrema_first =
    thrust::make_zip_iterator(bounding_box_minima_first, bounding_box_maxima_first);

  [[maybe_unused]] auto [_, extrema_last] = thrust::reduce_by_key(rmm::exec_policy(stream),
                                                                  ids_first,
                                                                  ids_last,
                                                                  points_zipped_first,
                                                                  thrust::make_discard_iterator(),
                                                                  extrema_first,
                                                                  thrust::equal_to<IdType>(),
                                                                  detail::box_minmax<T>{});

  auto num_trajectories = std::distance(extrema_first, extrema_last);
  return {bounding_box_minima_first + num_trajectories,
          bounding_box_maxima_first + num_trajectories};
}

}  // namespace cuspatial
