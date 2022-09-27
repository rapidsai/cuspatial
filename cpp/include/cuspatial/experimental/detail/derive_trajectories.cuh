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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <cub/device/device_merge_sort.cuh>

#include <cstdint>

namespace cuspatial {

namespace detail {

template <typename Tuple>
struct trajectory_comparator {
  __device__ bool operator()(Tuple const& lhs, Tuple const& rhs)
  {
    auto lhs_id = thrust::get<0>(lhs);
    auto rhs_id = thrust::get<0>(rhs);
    auto lhs_ts = thrust::get<1>(lhs);
    auto rhs_ts = thrust::get<1>(rhs);
    return (lhs_id < rhs_id) || ((lhs_id == rhs_id) && (lhs_ts < rhs_ts));
  };
};

template <typename IdInputIt,
          typename PointInputIt,
          typename TimestampInputIt,
          typename IdOutputIt,
          typename PointOutputIt,
          typename TimestampOutputIt>
void order_trajectories(IdInputIt ids_first,
                        IdInputIt ids_last,
                        PointInputIt points_first,
                        TimestampInputIt timestamps_first,
                        IdOutputIt ids_out_first,
                        PointOutputIt points_out_first,
                        TimestampOutputIt timestamps_out_first,
                        rmm::cuda_stream_view stream,
                        rmm::mr::device_memory_resource* mr)
{
  using id_type        = iterator_value_type<IdInputIt>;
  using timestamp_type = iterator_value_type<TimestampInputIt>;
  using tuple_type     = thrust::tuple<id_type, timestamp_type>;

  auto keys_first     = thrust::make_zip_iterator(ids_first, timestamps_first);
  auto keys_out_first = thrust::make_zip_iterator(ids_out_first, timestamps_out_first);

  std::size_t temp_storage_bytes = 0;
  cub::DeviceMergeSort::SortPairsCopy(nullptr,
                                      temp_storage_bytes,
                                      keys_first,
                                      points_first,
                                      keys_out_first,
                                      points_out_first,
                                      std::distance(ids_first, ids_last),
                                      trajectory_comparator<tuple_type>{},
                                      stream);

  auto temp_storage = rmm::device_buffer(temp_storage_bytes, stream, mr);

  cub::DeviceMergeSort::SortPairsCopy(temp_storage.data(),
                                      temp_storage_bytes,
                                      keys_first,
                                      points_first,
                                      keys_out_first,
                                      points_out_first,
                                      std::distance(ids_first, ids_last),
                                      trajectory_comparator<tuple_type>{},
                                      stream);

  stream.synchronize();
}

}  // namespace detail

template <typename IdInputIt,
          typename PointInputIt,
          typename TimestampInputIt,
          typename IdOutputIt,
          typename PointOutputIt,
          typename TimestampOutputIt,
          typename OffsetType>
std::unique_ptr<rmm::device_uvector<OffsetType>> derive_trajectories(
  IdInputIt ids_first,
  IdInputIt ids_last,
  PointInputIt points_first,
  TimestampInputIt timestamps_first,
  IdOutputIt ids_out_first,
  PointOutputIt points_out_first,
  TimestampOutputIt timestamps_out_first,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  detail::order_trajectories(ids_first,
                             ids_last,
                             points_first,
                             timestamps_first,
                             ids_out_first,
                             points_out_first,
                             timestamps_out_first,
                             stream,
                             mr);

  auto const num_points = std::distance(ids_first, ids_last);
  auto lengths          = rmm::device_uvector<OffsetType>(num_points, stream);
  auto grouped          = thrust::reduce_by_key(rmm::exec_policy(stream),
                                       ids_out_first,
                                       ids_out_first + num_points,
                                       thrust::make_constant_iterator(1),
                                       thrust::make_discard_iterator(),
                                       lengths.begin());

  auto const num_trajectories = std::distance(lengths.begin(), grouped.second);
  auto offsets = std::make_unique<rmm::device_uvector<OffsetType>>(num_trajectories, stream, mr);

  thrust::exclusive_scan(rmm::exec_policy(stream),
                         lengths.begin(),
                         lengths.begin() + num_trajectories,
                         offsets->begin());

  return offsets;
}

/**
 * @} // end of doxygen group
 */

}  // namespace cuspatial
