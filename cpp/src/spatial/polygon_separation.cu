/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cuspatial/detail/cartesian_product_iterator.cuh>
#include <cuspatial/error.hpp>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <limits>
//  #include <thrust/functional.h>
//  #include <thrust/iterator/transform_iterator.h>

//  #include <rmm/device_uvector.hpp>

#include <memory>
#include "thrust/functional.h"
#include "thrust/iterator/discard_iterator.h"

namespace cuspatial {
namespace detail {
namespace {

using size_type = cudf::size_type;

template <typename T>
__device__ inline T distance_line_segment_to_point(
  T const m, T const nx, T const ny, T const tx, T const ty, T const px, T const py)
{
  auto edge_travel = px * tx + py * ty;

  if (edge_travel < 0) { return hypot(px, py); }
  if (edge_travel <= m) { return abs(px * nx + py * ny); }

  return std::numeric_limits<double>::infinity();
}

template <typename T>
__device__ inline T distance_line_segment_to_points(
  T const edge_x, T const edge_y, T const p0_x, T const p0_y, T const p1_x, T const p1_y)
{
  auto const m  = hypot(edge_x, edge_y);
  auto const tx = edge_x / m;
  auto const ty = edge_y / m;
  auto const nx = -ty;
  auto const ny = +tx;

  return min(distance_line_segment_to_point(m, nx, ny, tx, ty, p0_x, p0_y),
             distance_line_segment_to_point(m, nx, ny, tx, ty, p1_x, p1_y));
}

template <typename T>
__device__ inline T distance_line_segments_to_points(T const a0_x,
                                                     T const a0_y,
                                                     T const a1_x,
                                                     T const a1_y,
                                                     T const b0_x,
                                                     T const b0_y,
                                                     T const b1_x,
                                                     T const b1_y)
{
  return min(distance_line_segment_to_points(
               a1_x - a0_x, a1_y - a0_y, b0_x - a0_x, b0_y - a0_y, b1_x - a0_x, b1_y - a0_y),
             distance_line_segment_to_points(
               b1_x - b0_x, b1_y - b0_y, a0_x - b0_x, a0_y - b0_y, a1_x - b0_x, a1_y - b0_y));
}

template <typename T>
struct gcp_to_polygon_separation_functor {
  cudf::column_device_view xs;
  cudf::column_device_view ys;

  T __device__ operator()(cartesian_product_group_index idx)
  {
    auto a_idx_0 = idx.group_a.offset + (idx.element_a_idx);
    auto a_idx_1 = idx.group_a.offset + (idx.element_a_idx + 1) % idx.group_a.size;
    auto b_idx_0 = idx.group_b.offset + (idx.element_b_idx);
    auto b_idx_1 = idx.group_b.offset + (idx.element_b_idx + 1) % idx.group_b.size;

    return distance_line_segments_to_points(xs.element<T>(a_idx_0),
                                            ys.element<T>(a_idx_0),
                                            xs.element<T>(a_idx_1),
                                            ys.element<T>(a_idx_1),
                                            xs.element<T>(b_idx_0),
                                            ys.element<T>(b_idx_0),
                                            xs.element<T>(b_idx_1),
                                            ys.element<T>(b_idx_1));
  }
};

struct polygon_separation_functor {
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

    if (num_results == 0) {
      return cudf::make_empty_column(cudf::data_type{cudf::type_to_id<T>()});
    }

    // ===== Make Separation and Key Iterators =====================================================

    auto gcp_iter = make_grouped_cartesian_product_iterator(
      num_points, num_spaces, space_offsets.begin<cudf::size_type>());

    auto gpc_key_iter =
      thrust::make_transform_iterator(gcp_iter, [] __device__(cartesian_product_group_index idx) {
        return thrust::make_pair(idx.group_a.idx, idx.group_b.idx);
      });

    auto d_xs = cudf::column_device_view::create(xs);
    auto d_ys = cudf::column_device_view::create(ys);

    auto separation_iter =
      thrust::make_transform_iterator(gcp_iter, gcp_to_polygon_separation_functor<T>{*d_xs, *d_ys});

    // ===== Materialize ===========================================================================

    auto result = cudf::make_fixed_width_column(cudf::data_type{cudf::type_to_id<T>()},
                                                num_results,
                                                cudf::mask_state::UNALLOCATED,
                                                stream,
                                                mr);

    auto num_cartesian = num_points * num_points;

    thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream),
                          gpc_key_iter,
                          gpc_key_iter + num_cartesian,
                          separation_iter,
                          thrust::make_discard_iterator(),
                          result->mutable_view().begin<T>(),
                          thrust::equal_to<thrust::pair<int32_t, int32_t>>(),
                          thrust::minimum<T>());

    return result;
  }
};

}  // namespace
}  // namespace detail

std::unique_ptr<cudf::column> minimum_euclidean_distance(cudf::column_view const& xs,
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
    xs.type(), detail::polygon_separation_functor(), xs, ys, points_per_space, mr, stream);
}

}  // namespace cuspatial
