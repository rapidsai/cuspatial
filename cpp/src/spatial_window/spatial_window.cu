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
  spatial_window_filter(T window_min_x,
                        T window_max_x,
                        T window_min_y,
                        T window_max_y,
                        cudf::column_device_view const& x,
                        cudf::column_device_view const& y)
    : min_x{std::min(window_min_x, window_max_x)},  // support mirrored rectangles
      max_x{std::max(window_min_x, window_max_x)},  // where specified min > max
      min_y{std::min(window_min_y, window_max_y)},
      max_y{std::max(window_min_y, window_max_y)},
      points_x{x},
      points_y{y}
  {
  }

  __device__ inline bool operator()(cudf::size_type i)
  {
    auto x = points_x.element<T>(i);
    auto y = points_y.element<T>(i);
    return x > min_x && x < max_x && y > min_y && y < max_y;
  }

 protected:
  T min_x;
  T max_x;
  T min_y;
  T max_y;
  cudf::column_device_view points_x;
  cudf::column_device_view points_y;
};

// Type-dispatch functor that creates the spatial window filter of the correct type.
// Only floating point types are supported.
struct spatial_window_dispatch {
  template <typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
  std::unique_ptr<cudf::table> operator()(double window_min_x,
                                          double window_max_x,
                                          double window_min_y,
                                          double window_max_y,
                                          cudf::column_view const& x,
                                          cudf::column_view const& y,
                                          cudaStream_t stream,
                                          rmm::mr::device_memory_resource* mr)
  {
    auto device_x = cudf::column_device_view::create(x, stream);
    auto device_y = cudf::column_device_view::create(y, stream);
    return cudf::experimental::detail::copy_if(
      cudf::table_view{{x, y}},
      spatial_window_filter<T>{static_cast<T>(window_min_x),
                               static_cast<T>(window_max_x),
                               static_cast<T>(window_min_y),
                               static_cast<T>(window_max_y),
                               *device_x,
                               *device_y},
      mr,
      stream);
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
 * Return all points (x,y) that fall within a query window (x1,y1,x2,y2)
 * see query.hpp
 *
 * Detail version that takes a stream.
 */
std::unique_ptr<cudf::table> points_in_spatial_window(double window_min_x,
                                                      double window_max_x,
                                                      double window_min_y,
                                                      double window_max_y,
                                                      cudf::column_view const& x,
                                                      cudf::column_view const& y,
                                                      cudaStream_t stream,
                                                      rmm::mr::device_memory_resource* mr)
{
  CUSPATIAL_EXPECTS(x.type() == y.type(), "Type mismatch between x and y arrays");
  CUSPATIAL_EXPECTS(x.size() == y.size(), "Size mismatch between x and y arrays");

  CUSPATIAL_EXPECTS(not(x.has_nulls() || y.has_nulls()), "NULL point data not supported");

  return cudf::experimental::type_dispatcher(x.type(),
                                             spatial_window_dispatch(),
                                             window_min_x,
                                             window_max_x,
                                             window_min_y,
                                             window_max_y,
                                             x,
                                             y,
                                             stream,
                                             mr);
}

}  // namespace detail

/*
 * Return all points (x,y) that fall within a query window (x1,y1,x2,y2)
 * see query.hpp
 */
std::unique_ptr<cudf::table> points_in_spatial_window(double window_min_x,
                                                      double window_max_x,
                                                      double window_min_y,
                                                      double window_max_y,
                                                      cudf::column_view const& x,
                                                      cudf::column_view const& y,
                                                      rmm::mr::device_memory_resource* mr)
{
  return detail::points_in_spatial_window(
    window_min_x, window_max_x, window_min_y, window_max_y, x, y, 0, mr);
}

}  // namespace cuspatial
