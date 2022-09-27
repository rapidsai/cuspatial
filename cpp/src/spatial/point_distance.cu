/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <cuspatial/experimental/point_distance.cuh>
#include <cuspatial/experimental/type_utils.hpp>
#include <cuspatial/vec_2d.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <memory>
#include <type_traits>

namespace cuspatial {
namespace detail {
struct pairwise_point_distance_functor {
  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    cudf::column_view const& points1_x,
    cudf::column_view const& points1_y,
    cudf::column_view const& points2_x,
    cudf::column_view const& points2_y,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    auto distances = cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<T>()},
                                               points1_x.size(),
                                               cudf::mask_state::UNALLOCATED,
                                               stream,
                                               mr);

    auto points1_it = make_vec_2d_iterator(points1_x.begin<T>(), points1_y.begin<T>());
    auto points2_it = make_vec_2d_iterator(points2_x.begin<T>(), points2_y.begin<T>());
    cuspatial::pairwise_point_distance(points1_it,
                                       points1_it + points1_x.size(),
                                       points2_it,
                                       distances->mutable_view().begin<T>(),
                                       stream);
                            stream);

    return distances;
  }

  template <typename T, typename... Args>
  std::enable_if_t<not std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    Args&&...)
  {
    CUSPATIAL_FAIL("Point distances only supports floating point coordinates.");
  }
};

std::unique_ptr<cudf::column> pairwise_point_distance(cudf::column_view const& points1_x,
                                                      cudf::column_view const& points1_y,
                                                      cudf::column_view const& points2_x,
                                                      cudf::column_view const& points2_y,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::mr::device_memory_resource* mr)
{
  CUSPATIAL_EXPECTS(points1_x.size() == points1_y.size() and
                      points2_x.size() == points2_y.size() and points1_x.size() == points2_x.size(),
                    "Mismatch number of coordinate or number of points.");

  CUSPATIAL_EXPECTS(points1_x.type() == points1_y.type() and
                      points2_x.type() == points2_y.type() and points1_x.type() == points2_x.type(),
                    "The types of point coordinates arrays mismatch.");
  CUSPATIAL_EXPECTS(not points1_x.has_nulls() and not points1_y.has_nulls() and
                      not points2_x.has_nulls() and not points2_y.has_nulls(),
                    "The coordinate columns cannot have nulls.");

  if (points1_x.size() == 0) { return cudf::empty_like(points1_x); }

  return cudf::type_dispatcher(points1_x.type(),
                               pairwise_point_distance_functor{},
                               points1_x,
                               points1_y,
                               points2_x,
                               points2_y,
                               stream,
                               mr);
}

}  // namespace detail

std::unique_ptr<cudf::column> pairwise_point_distance(cudf::column_view const& points1_x,
                                                      cudf::column_view const& points1_y,
                                                      cudf::column_view const& points2_x,
                                                      cudf::column_view const& points2_y,
                                                      rmm::mr::device_memory_resource* mr)
{
  return detail::pairwise_point_distance(
    points1_x, points1_y, points2_x, points2_y, rmm::cuda_stream_default, mr);
}

}  // namespace cuspatial
