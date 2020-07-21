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

#include <cuspatial/detail/cartesian_product_group_index_iterator.cuh>
#include <cuspatial/error.hpp>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/iterator/discard_iterator.h>

#include <limits>
#include <memory>

using size_type = cudf::size_type;

namespace cuspatial {
namespace detail {
namespace {

/**
 * @brief Calculates segment-point distance
 *
 * Given a `cartesian_product_group_index` and two columns representing `x` and `y` coordinates,
 * calculates the segment-point distance of a line segment in group `a` and a point in group `b`.
 *
 * If group `a` contains two or more points, calculate segment-point distance.
 * If group `a` contains only a single point, calculate point-point distance.
 */
template <typename T>
struct segment_point_distance_calculator {
  cudf::column_device_view xs;
  cudf::column_device_view ys;

  T __device__ operator()(cartesian_product_group_index idx)
  {
    auto const a_idx_0 = idx.group_a.offset + (idx.element_a_idx);
    auto const a_idx_1 = idx.group_a.offset + (idx.element_a_idx + 1) % idx.group_a.size;
    auto const b_idx_0 = idx.group_b.offset + (idx.element_b_idx);

    auto const origin_x = xs.element<T>(a_idx_0);
    auto const origin_y = ys.element<T>(a_idx_0);
    auto const edge_x   = xs.element<T>(a_idx_1) - origin_x;
    auto const edge_y   = ys.element<T>(a_idx_1) - origin_y;
    auto const point_x  = xs.element<T>(b_idx_0) - origin_x;
    auto const point_y  = ys.element<T>(b_idx_0) - origin_y;

    auto const magnitude = edge_x * edge_x + edge_y * edge_y;
    auto const travel    = point_x * edge_x + point_y * edge_y;

    // if point is projected within line segment bounds, use segment-point distance.
    // if point is projected outside line segment bounds, use point-point distance.
    if (0 < travel && travel < magnitude) {  // 0 < travel < edge_length
      return abs(point_y * edge_x - point_x * edge_y) * rhypot(edge_x, edge_y);
    } else {
      return hypot(point_x, point_y);
    }
  }
};

struct directed_polygon_distance_functor {
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

    auto cartesian_iter = make_cartesian_product_group_index_iterator(
      num_points, num_spaces, space_offsets.begin<cudf::size_type>());

    auto cartesian_key_iter = thrust::make_transform_iterator(
      cartesian_iter, [] __device__(cartesian_product_group_index idx) {
        return thrust::make_pair(idx.group_a.idx, idx.group_b.idx);
      });

    auto d_xs = cudf::column_device_view::create(xs);
    auto d_ys = cudf::column_device_view::create(ys);

    auto separation_iter = thrust::make_transform_iterator(
      cartesian_iter, segment_point_distance_calculator<T>{*d_xs, *d_ys});

    // ===== Materialize ===========================================================================

    auto result = cudf::make_fixed_width_column(cudf::data_type{cudf::type_to_id<T>()},
                                                num_results,
                                                cudf::mask_state::UNALLOCATED,
                                                stream,
                                                mr);

    auto num_cartesian = num_points * num_points;

    thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream),
                          cartesian_key_iter,
                          cartesian_key_iter + num_cartesian,
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

std::unique_ptr<cudf::column> directed_polygon_distance(cudf::column_view const& xs,
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
    xs.type(), detail::directed_polygon_distance_functor(), xs, ys, points_per_space, mr, stream);
}

}  // namespace cuspatial
