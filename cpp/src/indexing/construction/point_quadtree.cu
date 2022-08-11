/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <cuspatial/experimental/point_quadtree.cuh>
#include <cuspatial/experimental/type_utils.hpp>
#include <cuspatial/point_quadtree.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <memory>

/*
 * quadtree indexing on points using the bottom-up algorithm described at ref.
 * http://www.adms-conf.org/2019-camera-ready/zhang_adms19.pdf
 * extra care on minmizing peak device memory usage by deallocating memory as
 * early as possible
 */

namespace cuspatial {

namespace detail {

namespace {

/*
 * Construct a quad tree from the input (unsorted) x/y points. The bounding box
 * defined by the x_min, y_min, x_max, and y_max parameters is used to compute
 * keys in a one-dimensional Z-order curve (i.e. Morton codes) for each point.
 *
 * The keys are sorted and used to construct a quadtree from the "bottom" level,
 * ascending to the root.
 */
struct dispatch_construct_quadtree {
  template <typename T,
            std::enable_if_t<!std::is_floating_point<T>::value>* = nullptr,
            typename... Args>
  inline std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::table>> operator()(
    Args&&...)
  {
    CUSPATIAL_FAIL("Only floating-point types are supported");
  }

  template <typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
  inline std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::table>> operator()(
    cudf::column_view const& x,
    cudf::column_view const& y,
    double x_min,
    double x_max,
    double y_min,
    double y_max,
    double scale,
    int8_t max_depth,
    cudf::size_type min_size,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    auto pair = construct_quadtree(cuspatial::make_vec_2d_iterator(x.begin<T>(), y.begin<T>()),
                                   cuspatial::make_vec_2d_iterator(x.end<T>(), y.end<T>()),
                                   x_min,
                                   x_max,
                                   y_min,
                                   y_max,
                                   scale,
                                   max_depth,
                                   min_size,
                                   mr,
                                   stream);

    auto& idxs = pair.first;
    auto& tree = pair.second;
    auto size  = static_cast<cudf::size_type>(tree.key.size());

    auto keys_map = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::UINT32}, idxs.size(), idxs.release());

    std::vector<std::unique_ptr<cudf::column>> cols{};
    cols.push_back(std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::UINT32}, size, tree.key.release()));
    cols.push_back(std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::UINT8}, size, tree.level.release()));
    cols.push_back(std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::BOOL8}, size, tree.is_quad.release()));
    cols.push_back(std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::UINT32}, size, tree.length.release()));
    cols.push_back(std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::UINT32}, size, tree.offset.release()));

    return std::make_pair(std::move(keys_map), std::make_unique<cudf::table>(std::move(cols)));
  }
};

}  // namespace

std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::table>> quadtree_on_points(
  cudf::column_view const& x,
  cudf::column_view const& y,
  double x_min,
  double x_max,
  double y_min,
  double y_max,
  double scale,
  int8_t max_depth,
  cudf::size_type min_size,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  return cudf::type_dispatcher(x.type(),
                               dispatch_construct_quadtree{},
                               x,
                               y,
                               x_min,
                               x_max,
                               y_min,
                               y_max,
                               scale,
                               max_depth,
                               min_size,
                               stream,
                               mr);
}

}  // namespace detail

std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::table>> quadtree_on_points(
  cudf::column_view const& x,
  cudf::column_view const& y,
  double x_min,
  double x_max,
  double y_min,
  double y_max,
  double scale,
  int8_t max_depth,
  cudf::size_type min_size,
  rmm::mr::device_memory_resource* mr)
{
  CUSPATIAL_EXPECTS(x.size() == y.size(), "x and y columns must have the same length");
  CUSPATIAL_EXPECTS(x_min < x_max && y_min < y_max,
                    "invalid bounding box (x_min, x_max, y_min, y_max)");
  CUSPATIAL_EXPECTS(scale > 0, "scale must be positive");
  CUSPATIAL_EXPECTS(max_depth >= 0 && max_depth < 16,
                    "maximum depth must be positive and less than 16");
  CUSPATIAL_EXPECTS(min_size > 0, "minimum number of points for a non-leaf node must be positive");
  if (x.is_empty() || y.is_empty()) {
    std::vector<std::unique_ptr<cudf::column>> cols{};
    cols.push_back(cudf::make_empty_column(cudf::type_id::UINT32));
    cols.push_back(cudf::make_empty_column(cudf::type_id::UINT8));
    cols.push_back(cudf::make_empty_column(cudf::type_id::BOOL8));
    cols.push_back(cudf::make_empty_column(cudf::type_id::UINT32));
    cols.push_back(cudf::make_empty_column(cudf::type_id::UINT32));

    return std::make_pair(cudf::make_empty_column(cudf::type_id::UINT32),
                          std::make_unique<cudf::table>(std::move(cols)));
  }
  return detail::quadtree_on_points(
    x, y, x_min, x_max, y_min, y_max, scale, max_depth, min_size, rmm::cuda_stream_default, mr);
}

}  // namespace cuspatial
