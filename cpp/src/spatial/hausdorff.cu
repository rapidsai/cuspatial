/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required point_b_y applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuspatial/detail/hausdorff.cuh>
#include <cuspatial/error.hpp>
#include "utility/scatter_output_iterator.cuh"
#include "utility/size_from_offsets.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>

#include <memory>

namespace cuspatial {
namespace detail {
namespace {

using size_type = cudf::size_type;

template <typename T>
std::unique_ptr<cudf::column> make_column(size_type size,
                                          cudaStream_t stream,
                                          rmm::mr::device_memory_resource* mr)
{
  auto tid = cudf::type_to_id<T>();

  return cudf::make_fixed_width_column(
    cudf::data_type{tid}, size, cudf::mask_state::UNALLOCATED, stream, mr);
}

struct hausdorff_index {
  int32_t space_row;
  int32_t space_col;
  int32_t point_row;
  int32_t point_col;
  int32_t result_idx;
};

/** @brief Traverses a cartesian product of indices.
 *
 * Traverses a cartesian product of indices such that pairs of points within a pair of spaces appear
 * consecutively. This is used in a transform iterator to calculate multiple hausdorff distances
 * simultaneously by producing distances in the correct order for consumption by `reduce_by_key`.
 * The reduce key is the pair of spaces. The pair of points within a pair of spaces appears
 * consecutively.
 *
 */
template <typename SpaceSizeIterator>
struct haus_traversal_functor {
  int32_t num_spaces;
  int32_t num_points;
  size_type const* space_offsets;
  size_type const* space_lookup;
  SpaceSizeIterator const space_sizes;

  hausdorff_index __device__ operator()(int32_t idx)
  {
    int32_t space_a_idx      = space_lookup[idx / num_points];
    int32_t space_a_offset   = space_offsets[space_a_idx];
    int64_t space_a_offset_n = space_a_offset * static_cast<int64_t>(num_points);
    int32_t space_a_size     = space_sizes[space_a_idx];

    int32_t space_b_idx    = space_lookup[(idx - space_a_offset_n) / space_a_size];
    int32_t space_b_offset = space_offsets[space_b_idx];
    int32_t space_b_size   = space_sizes[space_b_idx];

    int32_t space_begin = space_a_offset_n + space_a_size * space_b_offset;
    int32_t cell_idx    = idx - space_begin;
    int32_t cell_col    = cell_idx / space_b_size;

    int64_t source_idx =
      space_a_offset_n + space_b_offset + (num_points - space_b_size) * cell_col + cell_idx;
    int32_t source_col = source_idx / num_points;
    int32_t source_row = source_idx % num_points;

    int32_t destination_idx =
      space_a_offset_n + (space_a_size - 1) * num_points + space_b_offset + (space_b_size - 1);

    auto result_idx = destination_idx == source_idx ? space_a_idx * num_spaces + space_b_idx : -1;

    return {
      space_a_idx,
      space_b_idx,
      source_row,
      source_col,
      result_idx,
    };
  }
};

template <typename T>
struct hausdorff_index_to_acc_functor {
  cudf::column_device_view xs;
  cudf::column_device_view ys;

  hausdorff_acc<T> __device__ operator()(hausdorff_index index)
  {
    auto a_x = xs.element<T>(index.point_row);
    auto a_y = ys.element<T>(index.point_row);
    auto b_x = xs.element<T>(index.point_col);
    auto b_y = ys.element<T>(index.point_col);

    auto distance = hypot(b_x - a_x, b_y - a_y);

    auto key = thrust::make_pair(index.space_row, index.space_col);

    return hausdorff_acc<T>{
      key, index.result_idx, index.point_col, index.point_col, distance, distance, 0};
  }
};

struct hausdorff_functor {
  template <typename T, typename... Args>
  std::enable_if_t<not std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    Args&&...)
  {
    CUSPATIAL_FAIL("Non-floating point operation is not supported");
  }

  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    cudf::column_view const& xs,
    cudf::column_view const& ys,
    cudf::column_view const& space_offsets,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
  {
    size_type num_points  = xs.size();
    size_type num_spaces  = space_offsets.size();
    size_type num_results = num_spaces * num_spaces;

    if (num_results == 0) { return make_column<T>(0, stream, mr); }

    // ===== Make Lookup for Space by Point ========================================================
    // these space lookups could be replaced with a `lower_bound` to reduce temporary memory

    auto temp_space_lookup = rmm::device_vector<size_type>(num_points);

    thrust::scatter(rmm::exec_policy(stream)->on(stream),
                    thrust::make_constant_iterator(1),
                    thrust::make_constant_iterator(1) + num_spaces - 1,
                    space_offsets.begin<size_type>() + 1,
                    temp_space_lookup.begin());

    thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream),
                           temp_space_lookup.cbegin(),
                           temp_space_lookup.cend(),
                           temp_space_lookup.begin());

    // ===== Make Space Size Iterator ==============================================================

    auto d_space_offsets = cudf::column_device_view::create(space_offsets);

    auto space_offset_iterator =
      thrust::make_transform_iterator(thrust::make_counting_iterator<int32_t>(0),
                                      size_from_offsets_functor{*d_space_offsets, xs.size()});

    // ===== Make Hausdorff Accumulator ============================================================

    auto num_distances = static_cast<int32_t>(num_points) * static_cast<int32_t>(num_points);

    auto hausdorff_index_iter = thrust::make_transform_iterator(
      thrust::make_counting_iterator<int32_t>(0),
      haus_traversal_functor<decltype(space_offset_iterator)>{num_spaces,
                                                              num_points,
                                                              space_offsets.data<size_type>(),
                                                              temp_space_lookup.data().get(),
                                                              space_offset_iterator});

    auto d_xs = cudf::column_device_view::create(xs);
    auto d_ys = cudf::column_device_view::create(ys);

    auto hausdorff_acc_iter = thrust::make_transform_iterator(
      hausdorff_index_iter, hausdorff_index_to_acc_functor<T>{*d_xs, *d_ys});

    // ===== Materialize ===========================================================================

    std::unique_ptr<cudf::column> result = make_column<T>(num_results, stream, mr);

    auto result_temp      = rmm::device_buffer(sizeof(hausdorff_acc<T>) * num_results);
    auto result_temp_iter = static_cast<hausdorff_acc<T>*>(result_temp.data());

    auto scatter_map = thrust::make_transform_iterator(
      hausdorff_index_iter, [] __device__(hausdorff_index acc) { return acc.result_idx; });

    // the following output iterator and `inclusive_scan_by_key` could be replaced by a
    // reduce_by_key, if it supported noncommutative operators.

    auto scatter_out = make_scatter_output_iterator(result_temp_iter, scatter_map);

    thrust::inclusive_scan_by_key(rmm::exec_policy(stream)->on(stream),
                                  hausdorff_acc_iter,
                                  hausdorff_acc_iter + num_distances,
                                  hausdorff_acc_iter,
                                  scatter_out,
                                  hausdorff_key_compare<T>{});

    thrust::transform(rmm::exec_policy(stream)->on(stream),
                      result_temp_iter,
                      result_temp_iter + num_results,
                      result->mutable_view().begin<T>(),
                      [] __device__(hausdorff_acc<T> const& a) { return static_cast<T>(a); });

    return result;
  }
};

}  // namespace
}  // namespace detail

std::unique_ptr<cudf::column> directed_hausdorff_distance(cudf::column_view const& xs,
                                                          cudf::column_view const& ys,
                                                          cudf::column_view const& points_per_space,
                                                          rmm::mr::device_memory_resource* mr)
{
  CUSPATIAL_EXPECTS(xs.type() == ys.type(), "Inputs `xs` and `ys` must have same type.");
  CUSPATIAL_EXPECTS(xs.size() == ys.size(), "Inputs `xs` and `ys` must have same length.");

  CUSPATIAL_EXPECTS(not xs.has_nulls() and not ys.has_nulls() and not points_per_space.has_nulls(),
                    "Inputs must not have nulls.");

  CUSPATIAL_EXPECTS(xs.size() >= points_per_space.size(),
                    "At least one point is required for each space");

  cudaStream_t stream = 0;

  return cudf::type_dispatcher(
    xs.type(), detail::hausdorff_functor(), xs, ys, points_per_space, mr, stream);
}

}  // namespace cuspatial
