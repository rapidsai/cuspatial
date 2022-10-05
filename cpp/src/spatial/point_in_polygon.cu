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

#include <cuspatial/error.hpp>
#include <cuspatial/experimental/iterator_factory.cuh>
#include <cuspatial/experimental/point_in_polygon.cuh>
#include <cuspatial/vec_2d.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <memory>
#include <type_traits>

namespace {

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
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
  {
    auto size = test_points_x.size();
    auto tid  = cudf::type_to_id<int32_t>();
    auto type = cudf::data_type{tid};
    auto results =
      cudf::make_fixed_width_column(type, size, cudf::mask_state::UNALLOCATED, stream, mr);

    if (results->size() == 0) { return results; }

    auto points_begin =
      cuspatial::make_vec_2d_iterator(test_points_x.begin<T>(), test_points_y.begin<T>());
    auto polygon_offsets_begin = poly_offsets.begin<cudf::size_type>();
    auto ring_offsets_begin    = poly_ring_offsets.begin<cudf::size_type>();
    auto polygon_points_begin =
      cuspatial::make_vec_2d_iterator(poly_points_x.begin<T>(), poly_points_y.begin<T>());
    auto results_begin = results->mutable_view().begin<int32_t>();

    cuspatial::point_in_polygon(points_begin,
                                points_begin + test_points_x.size(),
                                polygon_offsets_begin,
                                polygon_offsets_begin + poly_offsets.size(),
                                ring_offsets_begin,
                                ring_offsets_begin + poly_ring_offsets.size(),
                                polygon_points_begin,
                                polygon_points_begin + poly_points_x.size(),
                                results_begin,
                                stream);

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
                                               rmm::cuda_stream_view stream,
                                               rmm::mr::device_memory_resource* mr)
{
  return cudf::type_dispatcher(test_points_x.type(),
                               point_in_polygon_functor(),
                               test_points_x,
                               test_points_y,
                               poly_offsets,
                               poly_ring_offsets,
                               poly_points_x,
                               poly_points_y,
                               stream,
                               mr);
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
                                             rmm::cuda_stream_default,
                                             mr);
}

}  // namespace cuspatial
