/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/copy_if.cuh>

#include <cuspatial/error.hpp>

#include <memory>
#include <type_traits>

namespace {

// Functor to filter out points that are not inside the query window
// This is passed to cudf::detail::copy_if
template <typename T>
struct spatial_window_filter {
  spatial_window_filter(T left,
                        T bottom,
                        T right,
                        T top,
                        cudf::column_device_view const& x,
                        cudf::column_device_view const& y)
    : left{left}, bottom{bottom}, right{right}, top{top}, points_x{x}, points_y{y}
  {
  }

  __device__ inline bool operator()(cudf::size_type i)
  {
    auto x = points_x.element<T>(i);
    auto y = points_y.element<T>(i);
    return x > left && x < right && y > bottom && y < top;
  }

 protected:
  T left;
  T bottom;
  T right;
  T top;
  cudf::column_device_view points_x;
  cudf::column_device_view points_y;
};

// Type-dispatch functor that creates the spatial window filter of the correct type.
// Only floating point types are supported.
struct spatial_window_dispatch {
  template <typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
  std::unique_ptr<cudf::experimental::table> operator()(double left,
                                                        double bottom,
                                                        double right,
                                                        double top,
                                                        cudf::column_view const& x,
                                                        cudf::column_view const& y,
                                                        cudaStream_t stream,
                                                        rmm::mr::device_memory_resource* mr)
  {
    auto device_x = cudf::column_device_view::create(x, stream);
    auto device_y = cudf::column_device_view::create(y, stream);
    return cudf::experimental::detail::copy_if(cudf::table_view{{x, y}},
                                               spatial_window_filter<T>{static_cast<T>(left),
                                                                        static_cast<T>(bottom),
                                                                        static_cast<T>(right),
                                                                        static_cast<T>(top),
                                                                        *device_x,
                                                                        *device_y},
                                               mr,
                                               stream);
  }

  template <typename T, std::enable_if_t<not std::is_floating_point<T>::value>* = nullptr>
  std::unique_ptr<cudf::experimental::table> operator()(double left,
                                                        double bottom,
                                                        double right,
                                                        double top,
                                                        cudf::column_view const& x,
                                                        cudf::column_view const& y,
                                                        cudaStream_t stream,
                                                        rmm::mr::device_memory_resource* mr)
  {
    CUSPATIAL_FAIL("Only floating-point types supported");
  }
};

}  // namespace

namespace cuspatial {

namespace detail {

/*
 * Return all points (x,y) that fall within a query window (x1,y1,x2,y2)
 * see query.hpp
 *
 * Detail version that takes a stream.
 */
std::unique_ptr<cudf::experimental::table> points_in_spatial_window(
  double left,
  double bottom,
  double right,
  double top,
  cudf::column_view const& x,
  cudf::column_view const& y,
  cudaStream_t stream,
  rmm::mr::device_memory_resource* mr)
{
  CUSPATIAL_EXPECTS(x.type() == y.type(), "Type mismatch between x and y arrays");
  CUSPATIAL_EXPECTS(x.size() == y.size(), "Size mismatch between x and y arrays");

  CUSPATIAL_EXPECTS(not(x.has_nulls() || y.has_nulls()), "NULL point data not supported");

  return cudf::experimental::type_dispatcher(
    x.type(), spatial_window_dispatch(), left, bottom, right, top, x, y, stream, mr);
}

}  // namespace detail

/*
 * Return all points (x,y) that fall within a query window (x1,y1,x2,y2)
 * see query.hpp
 */
std::unique_ptr<cudf::experimental::table> points_in_spatial_window(
  double left,
  double bottom,
  double right,
  double top,
  cudf::column_view const& x,
  cudf::column_view const& y,
  rmm::mr::device_memory_resource* mr)
{
  return detail::points_in_spatial_window(left, bottom, right, top, x, y, 0, mr);
}

}  // namespace cuspatial
