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

#include <memory>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/column/column_view.hpp>

#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <utilities/legacy/cuda_utils.hpp>
#include <type_traits>
#include <utility/utility.hpp>
#include <cuspatial/legacy/point_in_polygon.hpp>
#include <cuspatial/error.hpp>

#include <cudf/legacy/column.hpp>
#include "cudf/utilities/type_dispatcher.hpp"
#include "rmm/mr/device/device_memory_resource.hpp"

namespace {

template <typename T>
__global__ void point_in_polygon_kernel(cudf::size_type num_test_points,
                                        const T* const __restrict__ test_point_x,
                                        const T* const __restrict__ test_point_y,
                                        cudf::size_type num_polys,
                                        const cudf::size_type* const __restrict__ some_offsets,
                                        const cudf::size_type* const __restrict__ some_other_offsets,
                                        const T* const __restrict__ poly_point_x,
                                        const T* const __restrict__ poly_point_y,
                                        uint32_t* const __restrict__ result)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_test_points) {
        return;
    }

    uint32_t hit_mask = 0;

    T x = test_point_x[idx];
    T y = test_point_y[idx];

    // for each polygon
    for (cudf::size_type poly_idx = 0;
         poly_idx < num_polys;
         poly_idx++)
    {
        cudf::size_type poly_begin = (0 == poly_idx) ? 0 : some_offsets[poly_idx - 1];
        cudf::size_type poly_end = some_offsets[poly_idx];
        bool point_is_within = false;

        // for each ring
        for (cudf::size_type poly_ring_idx = poly_begin;
             poly_ring_idx < poly_end;
             poly_ring_idx++)
        {
            cudf::size_type poly_points_begin = (poly_ring_idx == 0) ? 0 : some_other_offsets[poly_ring_idx - 1];
            cudf::size_type poly_points_end = some_other_offsets[poly_ring_idx] - 1;

            // for each line segment
            for (cudf::size_type poly_point_idx = poly_points_begin;
                 poly_point_idx < poly_points_end;
                 poly_point_idx++)
            {
                T ax = poly_point_x[poly_point_idx];
                T ay = poly_point_y[poly_point_idx];
                T bx = poly_point_x[poly_point_idx + 1];
                T by = poly_point_y[poly_point_idx + 1];

                auto y_within_ay_by = ay <= y && y < by;          // is y in range [ay, by) when ay < by?
                auto y_within_by_ay = by <= y && y < ay;          // is y in range [by, ay) when by < ay?
                auto y_within = y_within_ay_by || y_within_by_ay; // is y in range [by, ay]?
                auto run = bx - ax;
                auto rise = by - ay;
                auto rise_to_y =  y - ay;

                // Checks if a ray cast has crossed over the line segment.
                // Signs cancel out such that the test is always performed from the same side.
                // If x is on the same side of an odd number of line segments, the point must be within the polygon.
                auto did_cross_segment = x < (run / rise) * rise_to_y + ax;

                if (y_within && did_cross_segment) {
                    point_is_within = not point_is_within;
                }
        }

        hit_mask |= point_is_within << poly_idx;
    }

    result[idx] = hit_mask;
}

struct point_in_polygon_functor
{
    template <typename T>
    static constexpr bool is_supported()
    {
        return std::is_floating_point<T>::value;
    }

    template <typename T, std::enable_if_t< !is_supported<T>() >* = nullptr, typename... Args>
    std::unique_ptr<cudf::column> operator()(Args&& ...)

    {
        CUSPATIAL_FAIL("Non-floating point operation is not supported");
    }

    template <typename T, std::enable_if_t< is_supported<T>() >* = nullptr>
    std::unique_ptr<cudf::column>
    operator()(cudf::column_view const& test_point_x,
               cudf::column_view const& test_point_y,
               cudf::column_view const& some_offsets,
               cudf::column_view const& some_other_offsets,
               cudf::column_view const& poly_point_x,
               cudf::column_view const& poly_point_y,
               rmm::mr::device_memory_resource* mr,
               cudaStream_t stream)
    {
        auto size = test_point_y.size();
        auto tid = cudf::experimental::type_to_id<uint32_t>();
        auto type = cudf::data_type{ tid };
        auto result = cudf::make_fixed_width_column(type, size, cudf::mask_state::UNALLOCATED, stream, mr);

        constexpr cudf::size_type block_size = 256;
        cudf::experimental::detail::grid_1d grid{result->size(), block_size, 1};
        cudf::size_type min_grid_size = 0;

        auto kernel = point_in_polygon_kernel<T>;

        kernel<<<grid.num_blocks, block_size, 0, stream>>>(
            test_point_x.begin<T>(),
            test_point_y.begin<T>(),
            some_offsets.begin<int32_t>(),
            some_other_offsets.begin<int32_t>(),
            poly_point_x.begin<T>(),
            poly_point_y.begin<T>(),
            result->mutable_view().begin<uint32_t>()
        );

        return result;
    }
};

} // anonymous namespace

namespace cuspatial {

namespace detail {

std::unique_ptr<cudf::column>
point_in_polygon_bitmap(cudf::column_view const& points_x,
                        cudf::column_view const& points_y,
                        cudf::column_view const& some_offsets,
                        cudf::column_view const& some_other_offsets,
                        cudf::column_view const& poly_points_y,
                        cudf::column_view const& poly_points_x,
                        rmm::mr::device_memory_resource* mr,
                        cudaStream_t stream)
{
    return cudf::experimental::type_dispatcher(points_x.type(), point_in_polygon_functor(),
                                               points_x,  points_y,
                                               some_offsets, some_other_offsets,
                                               poly_points_y,    poly_points_x,
                                               mr,
                                               stream);
}

}

std::unique_ptr<cudf::column>
point_in_polygon_bitmap(cudf::column_view const& points_x,
                        cudf::column_view const& points_y,
                        cudf::column_view const& some_offsets,
                        cudf::column_view const& some_other_offsets,
                        cudf::column_view const& poly_points_y,
                        cudf::column_view const& poly_points_x,
                        rmm::mr::device_memory_resource* mr)
{

    CUSPATIAL_EXPECTS(points_y.type() == points_x.type(),
                      "polygon vertex and point temp_bitmap type mismatch for x array ");

    CUSPATIAL_EXPECTS(not points_x.has_nulls() &&
                      not points_y.has_nulls(),
                      "this version does not support points_x/points_y contains nulls");

    CUSPATIAL_EXPECTS(some_offsets.size() > 0 && (cudf::size_type) some_offsets.size() <= sizeof(uint32_t) * 8,
                      "Number of polygons cannot exceed bitmap capacity (32 for unsigned int)");

    CUSPATIAL_EXPECTS(some_offsets.size() <= some_other_offsets.size(),
                      "Number of polygons must be equal or less than # of rings (one polygon has at least one ring");

    CUSPATIAL_EXPECTS(poly_points_x.size() == poly_points_y.size(),
                      "polygon vertices sizes mismatch between x/y arrays");

    CUSPATIAL_EXPECTS(points_y.size() == points_x.size(),
                      "query points size mismatch from between x/y arrays");

    CUSPATIAL_EXPECTS(poly_points_x.type() == poly_points_y.type(),
                      "polygon vertex temp_bitmap type mismatch between x/y arrays");

    CUSPATIAL_EXPECTS(poly_points_x.type() == points_y.type(),
                      "polygon vertex and point temp_bitmap type mismatch for y array");

    CUSPATIAL_EXPECTS(not poly_points_y.has_nulls() &&
                      not poly_points_x.has_nulls(),
                      "polygon should not contain nulls");
}

} // namespace cuspatial
