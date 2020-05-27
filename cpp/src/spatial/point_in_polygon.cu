/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <memory>

#include <cuspatial/error.hpp>
#include <type_traits>

#include "cudf/utilities/type_dispatcher.hpp"
#include "rmm/mr/device/device_memory_resource.hpp"

namespace {

template <typename T>
__global__ void point_in_polygon_kernel(cudf::size_type num_test_points,
                                        const T* const __restrict__ test_points_x,
                                        const T* const __restrict__ test_points_y,
                                        cudf::size_type num_polys,
                                        const cudf::size_type* const __restrict__ poly_offsets,
                                        cudf::size_type num_rings,
                                        const cudf::size_type* const __restrict__ poly_ring_offsets,
                                        cudf::size_type num_points,
                                        const T* const __restrict__ poly_points_x,
                                        const T* const __restrict__ poly_points_y,
                                        int32_t* const __restrict__ result)
{
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx > num_test_points) { return; }

  int32_t hit_mask = 0;

  T x = test_points_x[idx];
  T y = test_points_y[idx];

  // for each polygon
  for (auto poly_idx = 0; poly_idx < num_polys; poly_idx++) {
    auto poly_idx_next = poly_idx + 1;
    auto poly_begin    = poly_offsets[poly_idx];
    auto poly_end      = (poly_idx_next < num_polys) ? poly_offsets[poly_idx_next] : num_rings;

    bool point_is_within = false;

    // for each ring
    for (auto ring_idx = poly_begin; ring_idx < poly_end; ring_idx++) {
      auto ring_idx_next = ring_idx + 1;
      auto ring_begin    = poly_ring_offsets[ring_idx];
      auto ring_end = (ring_idx_next < num_rings) ? poly_ring_offsets[ring_idx_next] : num_points;
      auto ring_len = ring_end - ring_begin;

      // for each line segment
      for (auto point_idx = 0; point_idx < ring_len; point_idx++) {
        T ax = poly_points_x[ring_begin + ((point_idx + 0) % ring_len)];
        T ay = poly_points_y[ring_begin + ((point_idx + 0) % ring_len)];
        T bx = poly_points_x[ring_begin + ((point_idx + 1) % ring_len)];
        T by = poly_points_y[ring_begin + ((point_idx + 1) % ring_len)];

        bool y_between_ay_by = ay <= y && y < by;  // is y in range [ay, by) when ay < by?
        bool y_between_by_ay = by <= y && y < ay;  // is y in range [by, ay) when by < ay?
        bool y_in_bounds     = y_between_ay_by || y_between_by_ay;  // is y in range [by, ay]?
        T run                = bx - ax;
        T rise               = by - ay;
        T rise_to_point      = y - ay;

        if (y_in_bounds && x < (run / rise) * rise_to_point + ax) {
          point_is_within = not point_is_within;
        }
      }
    }

    hit_mask |= point_is_within << poly_idx;
  }

  result[idx] = hit_mask;
}

struct point_in_polygon_functor {
  template <typename T>
  static constexpr bool is_supported()
  {
    return std::is_floating_point<T>::value;
  }

  template <typename T, std::enable_if_t<!is_supported<T>()>* = nullptr, typename... Args>
  std::unique_ptr<cudf::column> operator()(Args&&...)
  {
    CUSPATIAL_FAIL("Non-floating point operation is not supported");
  }

  template <typename T, std::enable_if_t<is_supported<T>()>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& test_points_x,
                                           cudf::column_view const& test_points_y,
                                           cudf::column_view const& poly_offsets,
                                           cudf::column_view const& poly_ring_offsets,
                                           cudf::column_view const& poly_points_x,
                                           cudf::column_view const& poly_points_y,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream)
  {
    auto size = test_points_x.size();
    auto tid  = cudf::type_to_id<int32_t>();
    auto type = cudf::data_type{tid};
    auto results =
      cudf::make_fixed_width_column(type, size, cudf::mask_state::UNALLOCATED, stream, mr);

    if (results->size() == 0) { return results; }

    constexpr cudf::size_type block_size = 256;

    cudf::detail::grid_1d grid{results->size(), block_size, 1};

    auto kernel = point_in_polygon_kernel<T>;

    kernel<<<grid.num_blocks, block_size, 0, stream>>>(test_points_x.size(),
                                                       test_points_x.begin<T>(),
                                                       test_points_y.begin<T>(),
                                                       poly_offsets.size(),
                                                       poly_offsets.begin<cudf::size_type>(),
                                                       poly_ring_offsets.size(),
                                                       poly_ring_offsets.begin<cudf::size_type>(),
                                                       poly_points_x.size(),
                                                       poly_points_x.begin<T>(),
                                                       poly_points_y.begin<T>(),
                                                       results->mutable_view().begin<int32_t>());

    return results;
  }
};

}  // anonymous namespace

namespace cuspatial {

namespace detail {

std::unique_ptr<cudf::column> point_in_polygon(cudf::column_view const& test_points_x,
                                               cudf::column_view const& test_points_y,
                                               cudf::column_view const& poly_offsets,
                                               cudf::column_view const& poly_ring_offsets,
                                               cudf::column_view const& poly_points_x,
                                               cudf::column_view const& poly_points_y,
                                               rmm::mr::device_memory_resource* mr,
                                               cudaStream_t stream)
{
  return cudf::type_dispatcher(test_points_x.type(),
                               point_in_polygon_functor(),
                               test_points_x,
                               test_points_y,
                               poly_offsets,
                               poly_ring_offsets,
                               poly_points_x,
                               poly_points_y,
                               mr,
                               stream);
}

}  // namespace detail

std::unique_ptr<cudf::column> point_in_polygon(cudf::column_view const& test_points_x,
                                               cudf::column_view const& test_points_y,
                                               cudf::column_view const& poly_offsets,
                                               cudf::column_view const& poly_ring_offsets,
                                               cudf::column_view const& poly_points_x,
                                               cudf::column_view const& poly_points_y,
                                               rmm::mr::device_memory_resource* mr)
{
  CUSPATIAL_EXPECTS(
    test_points_x.size() == test_points_y.size() and poly_points_x.size() == poly_points_y.size(),
    "All points must have both x and y values");

  CUSPATIAL_EXPECTS(test_points_x.type() == test_points_y.type() and
                      test_points_x.type() == poly_points_x.type() and
                      test_points_x.type() == poly_points_y.type(),
                    "All points much have the same type for both x and y");

  CUSPATIAL_EXPECTS(not test_points_x.has_nulls() && not test_points_y.has_nulls(),
                    "Test points must not contain nulls");

  CUSPATIAL_EXPECTS(not poly_points_x.has_nulls() && not poly_points_y.has_nulls(),
                    "Polygon points must not contain nulls");

  CUSPATIAL_EXPECTS(poly_offsets.size() <= std::numeric_limits<int32_t>::digits,
                    "Number of polygons cannot exceed 31");

  CUSPATIAL_EXPECTS(poly_ring_offsets.size() >= poly_offsets.size(),
                    "Each polygon must have at least one ring");

  CUSPATIAL_EXPECTS(poly_points_x.size() >= poly_offsets.size() * 4,
                    "Each ring must have at least four vertices");

  return cuspatial::detail::point_in_polygon(test_points_x,
                                             test_points_y,
                                             poly_offsets,
                                             poly_ring_offsets,
                                             poly_points_x,
                                             poly_points_y,
                                             mr,
                                             0);
}

}  // namespace cuspatial
