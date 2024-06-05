/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
#include <cuspatial/iterator_factory.cuh>
#include <cuspatial/points_in_range.cuh>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <type_traits>

namespace {

// Type-dispatch functor that creates the spatial range filter of the correct type.
// Only floating point types are supported.
struct points_in_range_dispatch {
  template <typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
  std::unique_ptr<cudf::table> operator()(double range_min_x,
                                          double range_max_x,
                                          double range_min_y,
                                          double range_max_y,
                                          cudf::column_view const& x,
                                          cudf::column_view const& y,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
  {
    auto points_begin = cuspatial::make_vec_2d_iterator(x.begin<T>(), y.begin<T>());

    auto range_min = cuspatial::vec_2d<T>{static_cast<T>(range_min_x), static_cast<T>(range_min_y)};
    auto range_max = cuspatial::vec_2d<T>{static_cast<T>(range_max_x), static_cast<T>(range_max_y)};

    auto output_size = cuspatial::count_points_in_range(
      range_min, range_max, points_begin, points_begin + x.size(), stream);

    std::vector<std::unique_ptr<cudf::column>> cols{};
    cols.reserve(2);
    auto mask_policy = cudf::mask_allocation_policy::NEVER;
    cols.push_back(cudf::detail::allocate_like(x, output_size, mask_policy, stream, mr));
    cols.push_back(cudf::detail::allocate_like(y, output_size, mask_policy, stream, mr));

    auto& output_x = cols[0];
    auto& output_y = cols[1];

    auto output_zip = cuspatial::make_vec_2d_output_iterator(output_x->mutable_view().begin<T>(),
                                                             output_y->mutable_view().begin<T>());

    cuspatial::copy_points_in_range(
      range_min, range_max, points_begin, points_begin + x.size(), output_zip, stream);

    return std::make_unique<cudf::table>(std::move(cols));
  }

  template <typename T,
            std::enable_if_t<not std::is_floating_point<T>::value>* = nullptr,
            typename... Args>
  std::unique_ptr<cudf::table> operator()(Args&&...)
  {
    CUSPATIAL_FAIL("Only floating-point types supported");
  }
};

}  // namespace

namespace cuspatial {

namespace detail {

/*
 * Return all points (x,y) that fall within a query range (x1,y1,x2,y2)
 * see query.hpp
 *
 * Detail version that takes a stream.
 */
std::unique_ptr<cudf::table> points_in_range(double range_min_x,
                                             double range_max_x,
                                             double range_min_y,
                                             double range_max_y,
                                             cudf::column_view const& x,
                                             cudf::column_view const& y,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  CUSPATIAL_EXPECTS(x.type() == y.type(), "Type mismatch between x and y arrays");
  CUSPATIAL_EXPECTS(x.size() == y.size(), "Size mismatch between x and y arrays");

  CUSPATIAL_EXPECTS(not(x.has_nulls() || y.has_nulls()), "NULL point data not supported");

  return cudf::type_dispatcher(x.type(),
                               points_in_range_dispatch(),
                               range_min_x,
                               range_max_x,
                               range_min_y,
                               range_max_y,
                               x,
                               y,
                               stream,
                               mr);
}

}  // namespace detail

/*
 * Return all points (x,y) that fall within a query range (x1,y1,x2,y2)
 * see query.hpp
 */
std::unique_ptr<cudf::table> points_in_range(double range_min_x,
                                             double range_max_x,
                                             double range_min_y,
                                             double range_max_y,
                                             cudf::column_view const& x,
                                             cudf::column_view const& y,
                                             rmm::device_async_resource_ref mr)
{
  return detail::points_in_range(
    range_min_x, range_max_x, range_min_y, range_max_y, x, y, rmm::cuda_stream_default, mr);
}

}  // namespace cuspatial
