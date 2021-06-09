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

#include "detail/phase_1.cuh"
#include "detail/phase_2.cuh"
#include "detail/utilities.cuh"

#include <cuspatial/error.hpp>
#include <cuspatial/point_quadtree.hpp>

#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

/*
 * quadtree indexing on points using the bottom-up algorithm described at ref.
 * http://www.adms-conf.org/2019-camera-ready/zhang_adms19.pdf
 * extra care on minmizing peak device memory usage by deallocating memory as
 * early as possible
 */

namespace cuspatial {

namespace detail {

namespace {

/**
 * @brief Constructs a complete quad tree
 */
inline std::unique_ptr<cudf::table> make_quad_tree(rmm::device_uvector<uint32_t> &quad_keys,
                                                   rmm::device_uvector<uint32_t> &quad_point_count,
                                                   rmm::device_uvector<uint32_t> &quad_child_count,
                                                   rmm::device_uvector<uint8_t> &quad_levels,
                                                   cudf::size_type num_parent_nodes,
                                                   int8_t max_depth,
                                                   cudf::size_type min_size,
                                                   cudf::size_type level_1_size,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::mr::device_memory_resource *mr)
{
  // count the number of child nodes
  auto num_child_nodes = thrust::reduce(rmm::exec_policy(stream),
                                        quad_child_count.begin(),
                                        quad_child_count.begin() + num_parent_nodes);

  cudf::size_type num_valid_nodes{0};
  cudf::size_type num_invalid_parent_nodes{0};

  // prune quadrants with fewer points than required
  // lines 1, 2, 3, 4, and 5 of algorithm in Fig. 5 in ref.
  std::tie(num_invalid_parent_nodes, num_valid_nodes) = remove_unqualified_quads(quad_keys,
                                                                                 quad_point_count,
                                                                                 quad_child_count,
                                                                                 quad_levels,
                                                                                 num_parent_nodes,
                                                                                 num_child_nodes,
                                                                                 min_size,
                                                                                 level_1_size,
                                                                                 stream);

  num_parent_nodes -= num_invalid_parent_nodes;

  // Construct the indicator output column.
  // line 6 and 7 of algorithm in Fig. 5 in ref.
  auto is_quad = construct_non_leaf_indicator(
    quad_point_count, num_parent_nodes, num_valid_nodes, min_size, mr, stream);

  // Construct the offsets output column
  // lines 8, 9, and 10 of algorithm in Fig. 5 in ref.
  auto offsets = [&]() {
    // line 8 of algorithm in Fig. 5 in ref.
    // revision to line 8: adjust quad_point_pos based on last-level z-order
    // code
    auto quad_point_pos = compute_flattened_first_point_positions(
      quad_keys, quad_levels, quad_point_count, *is_quad, num_valid_nodes, max_depth, stream);

    auto quad_child_pos = make_fixed_width_column<uint32_t>(num_valid_nodes, stream, mr);
    // line 9 of algorithm in Fig. 5 in ref.
    thrust::replace_if(rmm::exec_policy(stream),
                       quad_child_count.begin(),
                       quad_child_count.begin() + num_valid_nodes,
                       is_quad->view().begin<uint8_t>(),
                       !thrust::placeholders::_1,
                       0);

    // line 10 of algorithm in Fig. 5 in ref.
    thrust::exclusive_scan(rmm::exec_policy(stream),
                           quad_child_count.begin(),
                           quad_child_count.end(),
                           quad_child_pos->mutable_view().begin<uint32_t>(),
                           level_1_size);

    auto &offsets     = quad_child_pos;
    auto offsets_iter = thrust::make_zip_iterator(is_quad->view().begin<bool>(),
                                                  quad_child_pos->view().template begin<uint32_t>(),
                                                  quad_point_pos.begin());

    // for each value in `is_quad` copy from `quad_child_pos` if true, else
    // `quad_point_pos`
    thrust::transform(rmm::exec_policy(stream),
                      offsets_iter,
                      offsets_iter + num_valid_nodes,
                      offsets->mutable_view().template begin<uint32_t>(),
                      // return bool ? lhs : rhs
                      [] __device__(auto const &t) {
                        return thrust::get<0>(t) ? thrust::get<1>(t) : thrust::get<2>(t);
                      });

    return std::move(offsets);
  }();

  // Construct the lengths output column
  auto lengths = make_fixed_width_column<uint32_t>(num_valid_nodes, stream, mr);
  // for each value in `is_quad` copy from `quad_child_count` if true, else
  // `quad_point_count`
  auto lengths_iter = thrust::make_zip_iterator(is_quad->view().begin<bool>(),  //
                                                quad_child_count.begin(),
                                                quad_point_count.begin());
  thrust::transform(rmm::exec_policy(stream),
                    lengths_iter,
                    lengths_iter + num_valid_nodes,
                    lengths->mutable_view().template begin<uint32_t>(),
                    // return bool ? lhs : rhs
                    [] __device__(auto const &t) {
                      return thrust::get<0>(t) ? thrust::get<1>(t) : thrust::get<2>(t);
                    });

  // Construct the keys output column
  auto keys = make_fixed_width_column<uint32_t>(num_valid_nodes, stream, mr);

  // Copy quad keys to keys output column
  thrust::copy(rmm::exec_policy(stream),
               quad_keys.begin(),
               quad_keys.end(),
               keys->mutable_view().begin<uint32_t>());

  // Construct the levels output column
  auto levels = make_fixed_width_column<uint8_t>(num_valid_nodes, stream, mr);

  // Copy quad levels to levels output column
  thrust::copy(rmm::exec_policy(stream),
               quad_levels.begin(),
               quad_levels.end(),
               levels->mutable_view().begin<uint8_t>());

  std::vector<std::unique_ptr<cudf::column>> cols{};
  cols.reserve(5);
  cols.push_back(std::move(keys));
  cols.push_back(std::move(levels));
  cols.push_back(std::move(is_quad));
  cols.push_back(std::move(lengths));
  cols.push_back(std::move(offsets));
  return std::make_unique<cudf::table>(std::move(cols));
}

/**
 * @brief Constructs a leaf-only quadtree
 */
inline std::unique_ptr<cudf::table> make_leaf_tree(
  rmm::device_uvector<uint32_t> const &quad_keys,
  rmm::device_uvector<uint32_t> const &quad_point_count,
  cudf::size_type num_top_quads,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource *mr)
{
  auto keys    = make_fixed_width_column<uint32_t>(num_top_quads, stream, mr);
  auto levels  = make_fixed_width_column<uint8_t>(num_top_quads, stream, mr);
  auto is_quad = make_fixed_width_column<bool>(num_top_quads, stream, mr);
  auto lengths = make_fixed_width_column<uint32_t>(num_top_quads, stream, mr);
  auto offsets = make_fixed_width_column<uint32_t>(num_top_quads, stream, mr);

  // copy quad keys from the front of the quad_keys list
  thrust::copy(rmm::exec_policy(stream),
               quad_keys.begin(),
               quad_keys.begin() + num_top_quads,
               keys->mutable_view().begin<uint32_t>());

  // copy point counts from the front of the quad_point_count list
  thrust::copy(rmm::exec_policy(stream),
               quad_point_count.begin(),
               quad_point_count.begin() + num_top_quads,
               lengths->mutable_view().begin<uint32_t>());

  // All leaves are children of the root node (level 0)
  thrust::fill(rmm::exec_policy(stream),
               levels->mutable_view().begin<uint8_t>(),
               levels->mutable_view().end<uint8_t>(),
               0);

  // Quad node indicators are false for leaf nodes
  thrust::fill(rmm::exec_policy(stream),
               is_quad->mutable_view().begin<bool>(),
               is_quad->mutable_view().end<bool>(),
               false);

  // compute offsets from lengths
  thrust::exclusive_scan(rmm::exec_policy(stream),
                         lengths->view().begin<uint32_t>(),
                         lengths->view().end<uint32_t>(),
                         offsets->mutable_view().begin<uint32_t>());

  std::vector<std::unique_ptr<cudf::column>> cols{};
  cols.reserve(5);
  cols.push_back(std::move(keys));
  cols.push_back(std::move(levels));
  cols.push_back(std::move(is_quad));
  cols.push_back(std::move(lengths));
  cols.push_back(std::move(offsets));
  return std::make_unique<cudf::table>(std::move(cols));
}

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
            std::enable_if_t<!std::is_floating_point<T>::value> * = nullptr,
            typename... Args>
  inline std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::table>> operator()(
    Args &&...)
  {
    CUSPATIAL_FAIL("Only floating-point types are supported");
  }

  template <typename T, std::enable_if_t<std::is_floating_point<T>::value> * = nullptr>
  inline std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::table>> operator()(
    cudf::column_view const &x,
    cudf::column_view const &y,
    double x_min,
    double x_max,
    double y_min,
    double y_max,
    double scale,
    int8_t max_depth,
    cudf::size_type min_size,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource *mr)
  {
    // Construct the full set of non-empty subquadrants starting from the lowest level.
    // Corresponds to "Phase 1" of quadtree construction in ref.
    auto quads = make_full_levels<T>(x,
                                     y,
                                     static_cast<T>(x_min),
                                     static_cast<T>(x_max),
                                     static_cast<T>(y_min),
                                     static_cast<T>(y_max),
                                     scale,
                                     max_depth,
                                     min_size,
                                     stream,
                                     mr);

    auto &point_indices    = std::get<0>(quads);
    auto &quad_keys        = std::get<1>(quads);
    auto &quad_point_count = std::get<2>(quads);
    auto &quad_child_count = std::get<3>(quads);
    auto &quad_levels      = std::get<4>(quads);
    auto &num_top_quads    = std::get<5>(quads);
    auto &num_parent_nodes = std::get<6>(quads);
    auto &level_1_size     = std::get<7>(quads);

    // Optimization: return early if the top level nodes are all leaves
    if (num_parent_nodes <= 0) {
      return std::make_pair(std::move(point_indices),
                            make_leaf_tree(quad_keys, quad_point_count, num_top_quads, stream, mr));
    }

    // Corresponds to "Phase 2" of quadtree construction in ref.
    return std::make_pair(std::move(point_indices),
                          make_quad_tree(quad_keys,
                                         quad_point_count,
                                         quad_child_count,
                                         quad_levels,
                                         num_parent_nodes,
                                         max_depth,
                                         min_size,
                                         level_1_size,
                                         stream,
                                         mr));
  }
};

}  // namespace

std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::table>> quadtree_on_points(
  cudf::column_view const &x,
  cudf::column_view const &y,
  double x_min,
  double x_max,
  double y_min,
  double y_max,
  double scale,
  int8_t max_depth,
  cudf::size_type min_size,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource *mr)
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
  cudf::column_view const &x,
  cudf::column_view const &y,
  double x_min,
  double x_max,
  double y_min,
  double y_max,
  double scale,
  int8_t max_depth,
  cudf::size_type min_size,
  rmm::mr::device_memory_resource *mr)
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
    cols.reserve(5);
    cols.push_back(detail::make_fixed_width_column<uint32_t>(0, rmm::cuda_stream_default, mr));
    cols.push_back(detail::make_fixed_width_column<uint8_t>(0, rmm::cuda_stream_default, mr));
    cols.push_back(detail::make_fixed_width_column<bool>(0, rmm::cuda_stream_default, mr));
    cols.push_back(detail::make_fixed_width_column<uint32_t>(0, rmm::cuda_stream_default, mr));
    cols.push_back(detail::make_fixed_width_column<uint32_t>(0, rmm::cuda_stream_default, mr));
    return std::make_pair(
      detail::make_fixed_width_column<uint32_t>(0, rmm::cuda_stream_default, mr),
      std::make_unique<cudf::table>(std::move(cols)));
  }
  return detail::quadtree_on_points(
    x, y, x_min, x_max, y_min, y_max, scale, max_depth, min_size, rmm::cuda_stream_default, mr);
}

}  // namespace cuspatial
