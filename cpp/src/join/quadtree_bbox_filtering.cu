/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cuspatial/experimental/iterator_factory.cuh>
#include <cuspatial/experimental/spatial_join.cuh>

#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/spatial_join.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <tuple>

namespace cuspatial {

namespace {

struct dispatch_quadtree_bounding_box_join {
  template <typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
  inline std::unique_ptr<cudf::table> operator()(cudf::table_view const& quadtree,
                                                 cudf::table_view const& bbox,
                                                 double x_min,
                                                 double y_min,
                                                 double scale,
                                                 int8_t max_depth,
                                                 rmm::mr::device_memory_resource* mr,
                                                 rmm::cuda_stream_view stream)
  {
    auto const keys        = quadtree.column(0);  // uint32_t
    auto const levels      = quadtree.column(1);  // uint8_t
    auto const is_internal = quadtree.column(2);  // uint8_t
    auto const lengths     = quadtree.column(3);  // uint32_t
    auto const offsets     = quadtree.column(4);  // uint32_t

    auto bbox_min = cuspatial::make_vec_2d_iterator(bbox.column(0).template begin<T>(),
                                                    bbox.column(1).template begin<T>());
    auto bbox_max = cuspatial::make_vec_2d_iterator(bbox.column(2).template begin<T>(),
                                                    bbox.column(3).template begin<T>());

    auto bbox_itr = cuspatial::make_box_iterator(bbox_min, bbox_max);

    auto [bbox_offset, quad_offset] = join_quadtree_and_bounding_boxes(keys.begin<uint32_t>(),
                                                                       keys.end<uint32_t>(),
                                                                       levels.begin<uint8_t>(),
                                                                       is_internal.begin<uint8_t>(),
                                                                       lengths.begin<uint32_t>(),
                                                                       offsets.begin<uint32_t>(),
                                                                       bbox_itr,
                                                                       bbox_itr + bbox.num_rows(),
                                                                       static_cast<T>(x_min),
                                                                       static_cast<T>(y_min),
                                                                       static_cast<T>(scale),
                                                                       max_depth,
                                                                       mr,
                                                                       stream);

    std::vector<std::unique_ptr<cudf::column>> cols{};
    cols.push_back(std::make_unique<cudf::column>(std::move(bbox_offset)));
    cols.push_back(std::make_unique<cudf::column>(std::move(quad_offset)));

    return std::make_unique<cudf::table>(std::move(cols));
  }
  template <typename T,
            std::enable_if_t<!std::is_floating_point<T>::value>* = nullptr,
            typename... Args>
  inline std::unique_ptr<cudf::table> operator()(Args&&...)
  {
    CUSPATIAL_FAIL("Only floating-point types are supported");
  }
};

}  // namespace

std::unique_ptr<cudf::table> join_quadtree_and_bounding_boxes(cudf::table_view const& quadtree,
                                                              cudf::table_view const& bbox,
                                                              double x_min,
                                                              double x_max,
                                                              double y_min,
                                                              double y_max,
                                                              double scale,
                                                              int8_t max_depth,
                                                              rmm::mr::device_memory_resource* mr)
{
  CUSPATIAL_EXPECTS(quadtree.num_columns() == 5, "quadtree table must have 5 columns");
  CUSPATIAL_EXPECTS(bbox.num_columns() == 4, "bbox table must have 4 columns");
  CUSPATIAL_EXPECTS(scale > 0, "scale must be positive");
  CUSPATIAL_EXPECTS(x_min < x_max && y_min < y_max,
                    "invalid bounding box (x_min, x_max, y_min, y_max)");
  CUSPATIAL_EXPECTS(max_depth > 0 && max_depth < 16,
                    "maximum depth must be positive and less than 16");

  if (quadtree.num_rows() == 0 || bbox.num_rows() == 0) {
    std::vector<std::unique_ptr<cudf::column>> cols{};
    cols.reserve(2);
    cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::UINT32}));
    cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::UINT32}));
    return std::make_unique<cudf::table>(std::move(cols));
  }

  return cudf::type_dispatcher(bbox.column(0).type(),
                               dispatch_quadtree_bounding_box_join{},
                               quadtree,
                               bbox,
                               x_min,
                               y_min,
                               scale,
                               max_depth,
                               mr,
                               rmm::cuda_stream_default);
}

}  // namespace cuspatial
