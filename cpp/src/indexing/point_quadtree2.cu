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

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cuspatial/detail/point_quadtree.hpp>
#include <cuspatial/error.hpp>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include "utility/z_order.cuh"

/*
 * quadtree indexing on points using the bottom-up algorithm described at ref.
 * http://www.adms-conf.org/2019-camera-ready/zhang_adms19.pdf
 * extra care on minmizing peak device memory usage by deallocating memory as
 * early as possible
 */

namespace {

template <typename Vector>
inline auto shrink_vector(Vector &v, cudf::size_type size) {
  v.resize(size, 0);
  v.shrink_to_fit();
  return v;
}

template <typename... Ts>
inline auto make_zip_iterator(Ts... its) {
  return thrust::make_zip_iterator(
      thrust::make_tuple(std::forward<Ts>(its)...));
}

template <typename T>
inline std::unique_ptr<cudf::column> make_fixed_width_column(
    cudf::size_type size, cudaStream_t stream = 0,
    rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource()) {
  return cudf::make_fixed_width_column(
      cudf::data_type{cudf::experimental::type_to_id<T>()}, size,
      cudf::mask_state::UNALLOCATED, stream, mr);
}

template <typename T, typename PointToKeyFunc>
inline std::unique_ptr<cudf::column> compute_z_keys(
    cudf::mutable_column_view &x, cudf::mutable_column_view &y,
    cudaStream_t stream, PointToKeyFunc func) {
  auto policy = rmm::exec_policy(stream);
  auto points = make_zip_iterator(x.begin<T>(), y.begin<T>());
  auto keys = make_fixed_width_column<int32_t>(x.size(), stream);

  // Compute Morton codes (z-order) for each point
  thrust::transform(policy->on(stream), points, points + x.size(),
                    keys->mutable_view().begin<uint32_t>(), func);
  // Sort the points and codes
  thrust::sort_by_key(policy->on(stream),
                      keys->mutable_view().begin<uint32_t>(),
                      keys->mutable_view().end<uint32_t>(), points);

  return keys;
}

template <typename InputIterator1, typename InputIterator2,
          typename OutputIterator1, typename OutputIterator2,
          typename BinaryPred>
inline cudf::size_type compute_full_quads(
    InputIterator1 keys_begin, InputIterator1 keys_end, InputIterator2 vals_in,
    OutputIterator1 keys_out, OutputIterator2 vals_out, BinaryPred binary_op,
    cudaStream_t stream) {
  auto policy = rmm::exec_policy(stream);
  auto result = thrust::reduce_by_key(policy->on(stream), keys_begin, keys_end,
                                      vals_in, keys_out, vals_out,
                                      thrust::equal_to<uint32_t>(), binary_op);
  return thrust::distance(keys_out, result.first);
}

template <typename T>
struct tuple_sum {
  inline __device__ thrust::tuple<T, T> operator()(
      thrust::tuple<T, T> const &a, thrust::tuple<T, T> const &b) {
    return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b),
                              thrust::get<1>(a) + thrust::get<1>(b));
  }
};

template <typename KeysIterator, typename ValsIterator>
inline std::tuple<cudf::size_type, cudf::size_type, std::vector<uint32_t>,
                  std::vector<uint32_t>>
compute_full_levels(cudf::size_type const num_levels,
                    cudf::size_type const num_top_quads,
                    KeysIterator keys_begin,
                    ValsIterator quad_point_count_begin,
                    ValsIterator quad_child_count_begin, cudaStream_t stream) {
  // begin/end offsets
  cudf::size_type begin{0};
  cudf::size_type end{num_top_quads};
  std::vector<uint32_t> b_pos(num_levels);
  std::vector<uint32_t> e_pos(num_levels);

  // iterator for the parent level's quad node keys
  auto parent_keys = thrust::make_transform_iterator(
      keys_begin, [] __device__(uint32_t const child) { return (child >> 2); });

  // iterator for the current level's quad node point and child counts
  auto child_nodes =
      make_zip_iterator(quad_point_count_begin, quad_child_count_begin);

  // iterator for the current level's initial values
  auto child_values = make_zip_iterator(
      quad_point_count_begin, thrust::make_constant_iterator<uint32_t>(1));

  for (cudf::size_type level = num_levels - 1; level >= 0; --level) {
    auto range = compute_full_quads(
        parent_keys + begin, parent_keys + end, child_values + begin,
        keys_begin + end, child_nodes + end, tuple_sum<uint32_t>{}, stream);
    e_pos[level] = end;
    b_pos[level] = begin;
    begin = end;
    end += range;
  }

  return std::make_tuple(
      // count the number of parent nodes (excluding leaf nodes)
      end - num_top_quads - 1,  //
      end, b_pos, e_pos);
}

inline std::tuple<rmm::device_vector<uint32_t>, rmm::device_vector<uint32_t>,
                  rmm::device_vector<uint32_t>, rmm::device_vector<int8_t>>
reverse_tree_levels(rmm::device_vector<uint32_t> const &quad_keys_in,
                    rmm::device_vector<uint32_t> const &quad_point_count_in,
                    rmm::device_vector<uint32_t> const &quad_child_count_in,
                    std::vector<uint32_t> b_pos, std::vector<uint32_t> e_pos,
                    cudf::size_type const num_levels, cudaStream_t stream) {
  auto policy = rmm::exec_policy(stream);
  rmm::device_vector<uint32_t> quad_keys(quad_keys_in.size());
  rmm::device_vector<int8_t> quad_level(quad_keys_in.size());
  rmm::device_vector<uint32_t> quad_point_count(quad_point_count_in.size());
  rmm::device_vector<uint32_t> quad_child_count(quad_child_count_in.size());
  cudf::size_type offset{0};

  for (cudf::size_type level{0}; level < num_levels; ++level) {
    cudf::size_type end = e_pos[level];
    cudf::size_type begin = b_pos[level];
    cudf::size_type range = e_pos[level] - b_pos[level];
    thrust::fill(policy->on(stream), quad_level.begin() + offset,
                 quad_level.begin() + offset + range, level);
    thrust::copy(policy->on(stream), quad_keys_in.begin() + begin,
                 quad_keys_in.begin() + end, quad_keys.begin() + offset);
    thrust::copy(policy->on(stream), quad_point_count_in.begin() + begin,
                 quad_point_count_in.begin() + end,
                 quad_point_count.begin() + offset);
    thrust::copy(policy->on(stream), quad_child_count_in.begin() + begin,
                 quad_child_count_in.begin() + end,
                 quad_child_count.begin() + offset);
    // thrust::reduce(policy->on(stream), quad_point_count_in.begin() + begin,
    //                quad_point_count_in.begin() + end);
    offset += range;
  }

  // Shrink vectors' underlying device allocations to reduce peak memory usage
  quad_keys.shrink_to_fit();
  quad_point_count.shrink_to_fit();
  quad_child_count.shrink_to_fit();
  quad_level.shrink_to_fit();

  return std::make_tuple(quad_keys, quad_point_count, quad_child_count,
                         quad_level);
}

inline rmm::device_vector<uint32_t> compute_parent_positions(
    rmm::device_vector<uint32_t> const &quad_child_count,
    cudf::size_type const num_parent_nodes,
    cudf::size_type const num_child_nodes, cudaStream_t stream) {
  // compute parent node start positions
  auto policy = rmm::exec_policy(stream);
  // wrap in an IEFE so `position_map` is freed on return
  auto parent_pos = [&]() {
    rmm::device_vector<uint32_t> position_map(num_parent_nodes);
    // line 1 of algorithm in Fig. 5 in ref.
    thrust::exclusive_scan(policy->on(stream), quad_child_count.begin(),
                           quad_child_count.begin() + num_parent_nodes,
                           position_map.begin());
    // line 2 of algorithm in Fig. 5 in ref.
    rmm::device_vector<uint32_t> parent_pos(num_child_nodes);
    thrust::scatter(policy->on(stream), thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(0) + num_parent_nodes,
                    position_map.begin(), parent_pos.begin());
    return parent_pos;
  }();

  // line 3 of algorithm in Fig. 5 in ref.
  thrust::inclusive_scan(policy->on(stream), parent_pos.begin(),
                         parent_pos.begin() + num_child_nodes,
                         parent_pos.begin(), thrust::maximum<uint32_t>());

  return parent_pos;
}

inline std::pair<uint32_t, uint32_t> remove_unqualified_quads(
    rmm::device_vector<uint32_t> &quad_keys,
    rmm::device_vector<uint32_t> &quad_point_count,
    rmm::device_vector<uint32_t> &quad_child_count,
    rmm::device_vector<int8_t> &quad_level,
    cudf::size_type const num_parent_nodes,
    cudf::size_type const num_child_nodes, cudf::size_type const min_size,
    cudf::size_type const level_1_size, cudaStream_t stream) {
  // remove invalid nodes, return number of valid nodes left
  auto policy = rmm::exec_policy(stream);
  // compute parent node start positions
  auto parent_positions = compute_parent_positions(
      quad_child_count, num_parent_nodes, num_child_nodes, stream);
  auto parent_point_counts = thrust::make_permutation_iterator(
      quad_point_count.begin(), parent_positions.begin());

  // Count the number of nodes whose children have fewer points than `min_size`.
  // Start counting nodes at level 2, since children of the root node should not
  // be discarded.
  auto num_invalid_parent_nodes = thrust::count_if(
      policy->on(stream), parent_point_counts,
      parent_point_counts + (num_parent_nodes - level_1_size),
      // i.e. quad_point_count[parent_pos] <= min_size
      [min_size] __device__(auto const n) { return n <= min_size; });

  // line 4 of algorithm in Fig. 5 in ref.
  // revision to line 4: copy unnecessary if using permutation_iterator stencil

  // Remove quad nodes fewer points than min_size.
  // Start counting nodes at level 2, since children of the root node should not
  // be discarded.
  // line 5 of algorithm in Fig. 5 in ref.
  auto tree = make_zip_iterator(quad_keys.begin() + level_1_size,
                                quad_point_count.begin() + level_1_size,
                                quad_child_count.begin() + level_1_size,
                                quad_level.begin() + level_1_size);

  auto last_valid = thrust::remove_if(
      policy->on(stream), tree, tree + num_child_nodes, parent_point_counts,
      // i.e. quad_point_count[parent_pos] <= min_size
      [min_size] __device__(auto const n) { return n <= min_size; });

  // add the number of level 1 nodes back in to num_valid_nodes
  auto num_valid_nodes = thrust::distance(tree, last_valid) + level_1_size;

  return std::make_pair(num_invalid_parent_nodes, num_valid_nodes);
}

inline std::unique_ptr<cudf::column> construct_non_leaf_indicator(
    rmm::device_vector<uint32_t> &quad_point_count,
    cudf::size_type const num_parent_nodes,
    cudf::size_type const num_valid_nodes, cudf::size_type const min_size,
    rmm::mr::device_memory_resource *mr, cudaStream_t stream) {
  //
  auto policy = rmm::exec_policy(stream);
  // Construct the indicator output column
  auto indicator = make_fixed_width_column<bool>(num_valid_nodes, stream, mr);

  // line 6 of algorithm in Fig. 5 in ref.
  thrust::transform(policy->on(stream), quad_point_count.begin(),
                    quad_point_count.begin() + num_parent_nodes,
                    indicator->mutable_view().begin<bool>(),
                    thrust::placeholders::_1 > min_size);

  // line 7 of algorithm in Fig. 5 in ref.
  thrust::replace_if(policy->on(stream), quad_point_count.begin(),
                     quad_point_count.begin() + num_parent_nodes,
                     indicator->view().begin<bool>(), thrust::placeholders::_1,
                     0);

  if (num_valid_nodes > num_parent_nodes) {
    // zero-fill the rest of the indicator column because
    // device_memory_resources aren't required to initialize allocations
    thrust::fill(policy->on(stream),
                 indicator->mutable_view().begin<bool>() + num_parent_nodes,
                 indicator->mutable_view().end<bool>(), 0);
  }

  return indicator;
}

inline rmm::device_vector<uint32_t> compute_leaf_positions(
    cudf::column_view const &indicator, cudf::size_type const num_valid_nodes,
    cudaStream_t stream) {
  auto policy = rmm::exec_policy(stream);
  rmm::device_vector<uint32_t> leaf_pos(num_valid_nodes);
  auto result = thrust::copy_if(
      policy->on(stream), thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(0) + num_valid_nodes,
      indicator.begin<bool>(), leaf_pos.begin(), !thrust::placeholders::_1);
  // Shrink leaf_pos's underlying device allocation
  return shrink_vector(leaf_pos, thrust::distance(leaf_pos.begin(), result));
}

inline rmm::device_vector<uint32_t> flatten_point_keys(
    rmm::device_vector<uint32_t> const &quad_keys,
    rmm::device_vector<int8_t> const &quad_level,
    cudf::column_view const &indicator, cudf::size_type const num_valid_nodes,
    cudf::size_type const num_levels, cudaStream_t stream) {
  rmm::device_vector<uint32_t> flattened_keys(num_valid_nodes);
  auto policy = rmm::exec_policy(stream);
  auto keys_and_levels = make_zip_iterator(
      quad_keys.begin(), quad_level.begin(), indicator.begin<bool>());
  thrust::transform(
      policy->on(stream), keys_and_levels, keys_and_levels + num_valid_nodes,
      flattened_keys.begin(), [M = num_levels] __device__(auto const &val) {
        bool is_node{false};
        uint32_t key{}, level{};
        thrust::tie(key, level, is_node) = val;
        return is_node ? 0xFFFFFFFF : (key << (2 * (M - 1 - level)));
      });
  flattened_keys.shrink_to_fit();
  return flattened_keys;
}

inline rmm::device_vector<uint32_t> compute_flattened_first_point_positions(
    rmm::device_vector<uint32_t> const &quad_keys,
    rmm::device_vector<int8_t> const &quad_level,
    rmm::device_vector<uint32_t> &quad_point_count,
    cudf::column_view const &indicator, cudf::size_type const num_valid_nodes,
    cudf::size_type const num_levels, cudaStream_t stream) {
  //
  // Adjust quad_point_count and quad_point_pos based on the last level's
  // z-order keys
  //
  auto policy = rmm::exec_policy(stream);

  rmm::device_vector<uint32_t> initial_sort_indices{};
  rmm::device_vector<uint32_t> quad_point_count_tmp{};
  // Sort initial indices and temporary point counts by the flattened keys
  std::tie(initial_sort_indices, quad_point_count_tmp) = [&]() {
    auto flattened_keys = flatten_point_keys(
        quad_keys, quad_level, indicator, num_valid_nodes, num_levels, stream);

    rmm::device_vector<uint32_t> initial_sort_indices(num_valid_nodes);
    thrust::sequence(policy->on(stream), initial_sort_indices.begin(),
                     initial_sort_indices.end());

    rmm::device_vector<uint32_t> quad_point_count_tmp(num_valid_nodes);
    thrust::copy(policy->on(stream), quad_point_count.begin(),
                 quad_point_count.end(), quad_point_count_tmp.begin());

    // sort indices and temporary point counts
    thrust::stable_sort_by_key(policy->on(stream), flattened_keys.begin(),
                               flattened_keys.end(),
                               make_zip_iterator(initial_sort_indices.begin(),
                                                 quad_point_count_tmp.begin()));

    thrust::remove_if(policy->on(stream), quad_point_count_tmp.begin(),
                      quad_point_count_tmp.begin() + num_valid_nodes,
                      quad_point_count_tmp.begin(),
                      thrust::placeholders::_1 == 0);

    initial_sort_indices.shrink_to_fit();
    quad_point_count_tmp.shrink_to_fit();

    return std::make_pair(initial_sort_indices, quad_point_count_tmp);
  }();

  auto leaf_pos = compute_leaf_positions(indicator, num_valid_nodes, stream);

  // Shrink the vector's underlying device allocations.
  // Only the first `num_leaf_nodes` are needed after removal, since
  // copy_if and remove_if should remove the same number of elements.
  shrink_vector(quad_point_count_tmp, leaf_pos.size());
  shrink_vector(initial_sort_indices, leaf_pos.size());

  rmm::device_vector<uint32_t> quad_point_f_pos_tmp(leaf_pos.size());

  thrust::exclusive_scan(policy->on(stream), quad_point_count_tmp.begin(),
                         quad_point_count_tmp.end(),
                         quad_point_f_pos_tmp.begin());

  auto count_and_f_pos = make_zip_iterator(quad_point_count_tmp.begin(),
                                           quad_point_f_pos_tmp.begin());

  thrust::stable_sort_by_key(policy->on(stream), initial_sort_indices.begin(),
                             initial_sort_indices.end(), count_and_f_pos);

  rmm::device_vector<uint32_t> quad_point_f_pos(num_valid_nodes);

  thrust::scatter(
      policy->on(stream), count_and_f_pos, count_and_f_pos + leaf_pos.size(),
      leaf_pos.begin(),
      make_zip_iterator(quad_point_count.begin(), quad_point_f_pos.begin()));

  quad_point_f_pos.shrink_to_fit();

  return quad_point_f_pos;
}

template <typename TypeOut, typename TypeIn>
inline std::unique_ptr<cudf::column> copy_if_else(
    rmm::device_vector<TypeIn> const &lhs,
    rmm::device_vector<TypeIn> const &rhs, cudf::column_view const &mask,
    cudf::size_type const size, rmm::mr::device_memory_resource *mr,
    cudaStream_t stream) {
  // for each value in `mask` copy from `lhs` if true, else `rhs`
  auto policy = rmm::exec_policy(stream);
  auto output = make_fixed_width_column<TypeOut>(size, stream, mr);
  auto iter = make_zip_iterator(mask.begin<bool>(), lhs.begin(), rhs.begin());

  thrust::transform(policy->on(stream), iter, iter + size,
                    output->mutable_view().template begin<TypeIn>(),
                    // return bool ? lhs : rhs
                    [] __device__(auto const &t) {
                      return thrust::get<0>(t) ? thrust::get<1>(t)
                                               : thrust::get<2>(t);
                    });

  return output;
}

inline std::unique_ptr<cudf::experimental::table> make_full_quadtree(
    rmm::device_vector<uint32_t> &quad_keys,
    rmm::device_vector<uint32_t> &quad_point_count,
    rmm::device_vector<uint32_t> &quad_child_count,
    rmm::device_vector<int8_t> &quad_level, cudf::size_type num_parent_nodes,
    cudf::size_type const quad_tree_size, cudf::size_type const num_levels,
    cudf::size_type const min_size, cudf::size_type const level_1_size,
    rmm::mr::device_memory_resource *mr, cudaStream_t stream) {
  auto policy = rmm::exec_policy(stream);
  // count the number of child nodes
  auto num_child_nodes =
      thrust::reduce(policy->on(stream), quad_child_count.begin(),
                     quad_child_count.begin() + num_parent_nodes);

  cudf::size_type num_valid_nodes{0};
  cudf::size_type num_invalid_parent_nodes{0};

  // prune quadrants with fewer points than required
  std::tie(num_invalid_parent_nodes, num_valid_nodes) =
      remove_unqualified_quads(quad_keys, quad_point_count, quad_child_count,
                               quad_level, num_parent_nodes, num_child_nodes,
                               min_size, level_1_size, stream);

  num_parent_nodes -= num_invalid_parent_nodes;

  // construct indicator output column
  // line 6 and 7 of algorithm in Fig. 5 in ref.
  auto indicator =
      construct_non_leaf_indicator(quad_point_count, num_parent_nodes,
                                   num_valid_nodes, min_size, mr, stream);

  // Construct the f_pos output column
  auto f_pos = [&]() {
    // line 8 of algorithm in Fig. 5 in ref.
    // revision to line 8: adjust quad_point_pos based on last-level z-order
    // code
    auto quad_point_pos = compute_flattened_first_point_positions(
        quad_keys, quad_level, quad_point_count, *indicator, num_valid_nodes,
        num_levels, stream);

    // line 9 and 10 of algorithm in Fig. 5 in ref.
    rmm::device_vector<uint32_t> quad_child_pos(num_valid_nodes);
    // line 9 of algorithm in Fig. 5 in ref.
    thrust::replace_if(policy->on(stream), quad_child_count.begin(),
                       quad_child_count.begin() + num_valid_nodes,
                       indicator->view().begin<int8_t>(),
                       !thrust::placeholders::_1, 0);

    // line 10 of algorithm in Fig. 5 in ref.
    thrust::exclusive_scan(policy->on(stream), quad_child_count.begin(),
                           quad_child_count.end(), quad_child_pos.begin(),
                           level_1_size);

    // shrink intermediate device allocation
    shrink_vector(quad_child_pos, num_valid_nodes);

    return copy_if_else<int32_t>(quad_child_pos, quad_point_pos, *indicator,
                                 num_valid_nodes, mr, stream);
  }();

  // Construct the lengths output column
  auto lengths = copy_if_else<int32_t>(quad_child_count, quad_point_count,
                                       *indicator, num_valid_nodes, mr, stream);

  // Construct the keys output column
  auto keys = make_fixed_width_column<int32_t>(num_valid_nodes, stream, mr);

  // Copy quad keys to keys output column
  thrust::copy(policy->on(stream), quad_keys.begin(), quad_keys.end(),
               keys->mutable_view().begin<uint32_t>());

  // Construct the levels output column
  auto levels = make_fixed_width_column<int8_t>(num_valid_nodes, stream, mr);

  // Copy quad levels to levels output column
  thrust::copy(policy->on(stream), quad_level.begin(), quad_level.end(),
               levels->mutable_view().begin<int8_t>());

  std::vector<std::unique_ptr<cudf::column>> cols{};
  cols.reserve(5);
  cols.push_back(std::move(keys));
  cols.push_back(std::move(levels));
  cols.push_back(std::move(indicator));
  cols.push_back(std::move(lengths));
  cols.push_back(std::move(f_pos));
  return std::make_unique<cudf::experimental::table>(std::move(cols));
}

inline std::unique_ptr<cudf::experimental::table> make_empty_quadtree(
    rmm::device_vector<uint32_t> const &quad_keys,
    rmm::device_vector<uint32_t> const &quad_point_count,
    int32_t const num_top_quads, rmm::mr::device_memory_resource *mr,
    cudaStream_t stream) {
  auto keys = make_fixed_width_column<int32_t>(num_top_quads, stream, mr);
  auto levels = make_fixed_width_column<int8_t>(num_top_quads, stream, mr);
  auto indicator = make_fixed_width_column<bool>(num_top_quads, stream, mr);
  auto lengths = make_fixed_width_column<int32_t>(num_top_quads, stream, mr);
  auto f_pos = make_fixed_width_column<int32_t>(num_top_quads, stream, mr);

  auto policy = rmm::exec_policy(stream);
  // copy quad keys from the front of the quad_keys list
  thrust::copy(policy->on(stream), quad_keys.begin(),
               quad_keys.begin() + num_top_quads,
               keys->mutable_view().begin<uint32_t>());

  // copy point counts from the front of the quad_point_count list
  thrust::copy(policy->on(stream), quad_point_count.begin(),
               quad_point_count.begin() + num_top_quads,
               lengths->mutable_view().begin<uint32_t>());

  // All leaves are children of the root node (level 0)
  thrust::fill(policy->on(stream), levels->mutable_view().begin<int8_t>(),
               levels->mutable_view().end<int8_t>(), 0);

  // Quad node indicators are false for leaf nodes
  thrust::fill(policy->on(stream), indicator->mutable_view().begin<bool>(),
               indicator->mutable_view().end<bool>(), false);

  // compute f_pos offsets from sizes
  thrust::exclusive_scan(policy->on(stream), lengths->view().begin<uint32_t>(),
                         lengths->view().end<uint32_t>(),
                         f_pos->mutable_view().begin<uint32_t>());

  std::vector<std::unique_ptr<cudf::column>> cols{};
  cols.reserve(5);
  cols.push_back(std::move(keys));
  cols.push_back(std::move(levels));
  cols.push_back(std::move(indicator));
  cols.push_back(std::move(lengths));
  cols.push_back(std::move(f_pos));
  return std::make_unique<cudf::experimental::table>(std::move(cols));
}

template <typename T>
inline std::unique_ptr<cudf::experimental::table> construct_quad_tree(
    cudf::mutable_column_view &x, cudf::mutable_column_view &y, double const x1,
    double const y1, double const x2, double const y2, double const scale,
    cudf::size_type const num_levels, cudf::size_type const min_size,
    rmm::mr::device_memory_resource *mr, cudaStream_t stream) {
  // Compute z-order for each point
  auto point_keys = compute_z_keys<T>(
      x, y, stream,
      [x1, y1, x2, y2, num_levels, scale] __device__(auto const &point) {
        T x, y;
        thrust::tie(x, y) = point;
        if (x < x1 || x > x2 || y < y1 || y > y2) {
          // If the point is outside the bbox, return a max_level key
          return static_cast<uint32_t>((1 << (2 * num_levels)) - 1);
        }
        return z_order((x - x1) / scale, (y - y1) / scale);
      });

  rmm::device_vector<uint32_t> quad_keys(x.size());
  rmm::device_vector<uint32_t> quad_point_count(x.size());
  rmm::device_vector<uint32_t> quad_child_count(x.size());

  auto const num_top_quads = compute_full_quads(
      point_keys->view().template begin<uint32_t>(),
      point_keys->view().template end<uint32_t>(),
      thrust::make_constant_iterator<uint32_t>(1), quad_keys.begin(),
      quad_point_count.begin(), thrust::plus<uint32_t>(), stream);

  std::vector<uint32_t> b_pos{};
  std::vector<uint32_t> e_pos{};
  cudf::size_type quad_tree_size{};
  cudf::size_type num_parent_nodes{};

  // compute "full" quads for the tree at each level
  std::tie(num_parent_nodes, quad_tree_size, b_pos, e_pos) =
      compute_full_levels(num_levels, num_top_quads, quad_keys.begin(),
                          quad_point_count.begin(), quad_child_count.begin(),
                          stream);

  // Shrink vectors' underlying device allocations to reduce peak memory usage
  shrink_vector(quad_keys, quad_tree_size);
  shrink_vector(quad_point_count, quad_tree_size);
  shrink_vector(quad_child_count, quad_tree_size);

  // Optimization: can return early if the top level nodes are all leaves
  if (num_parent_nodes <= 0) {
    return make_empty_quadtree(quad_keys, quad_point_count, num_top_quads, mr,
                               stream);
  }

  rmm::device_vector<uint32_t> quad_keys_f{};
  rmm::device_vector<uint32_t> quad_point_count_f{};
  rmm::device_vector<uint32_t> quad_child_count_f{};
  rmm::device_vector<int8_t> quad_level_f{};

  // Reverse the quadtree nodes for easier manipulation (skips the root node)
  std::tie(quad_keys_f, quad_point_count_f, quad_child_count_f, quad_level_f) =
      reverse_tree_levels(quad_keys, quad_point_count, quad_child_count, b_pos,
                          e_pos, num_levels, stream);

  return make_full_quadtree(quad_keys_f, quad_point_count_f, quad_child_count_f,
                            quad_level_f, num_parent_nodes, quad_tree_size,
                            num_levels, min_size, e_pos[0] - b_pos[0], mr,
                            stream);
}

struct dispatch_construct_quadtree {
  template <typename T,
            std::enable_if_t<std::is_floating_point<T>::value> * = nullptr>
  inline std::unique_ptr<cudf::experimental::table> operator()(
      cudf::mutable_column_view &x, cudf::mutable_column_view &y,
      double const x1, double const y1, double const x2, double const y2,
      double const scale, int32_t const num_level, int32_t const min_size,
      rmm::mr::device_memory_resource *mr, cudaStream_t stream) {
    return construct_quad_tree<T>(x, y, x1, y1, x2, y2, scale, num_level,
                                  min_size, mr, stream);
  }

  template <typename T,
            std::enable_if_t<!std::is_floating_point<T>::value> * = nullptr>
  inline std::unique_ptr<cudf::experimental::table> operator()(
      cudf::mutable_column_view &x, cudf::mutable_column_view &y,
      double const x1, double const y1, double const x2, double const y2,
      double const scale, int32_t const num_level, int32_t const min_size,
      rmm::mr::device_memory_resource *mr, cudaStream_t stream) {
    CUDF_FAIL("Non-floating point operation is not supported");
  }
};

}  // namespace

namespace cuspatial {

std::unique_ptr<cudf::experimental::table> quadtree_on_points(
    cudf::mutable_column_view x, cudf::mutable_column_view y, double const x1,
    double const y1, double const x2, double const y2, double const scale,
    int32_t const num_levels, int32_t const min_size,
    rmm::mr::device_memory_resource *mr) {
  CUSPATIAL_EXPECTS(x.size() == y.size(),
                    "x and y columns might have the same length");
  CUSPATIAL_EXPECTS(x.size() > 0, "point dataset can not be empty");
  CUSPATIAL_EXPECTS(x1 < x2 && y1 < y2, "invalid bounding box (x1,y1,x2,y2)");
  CUSPATIAL_EXPECTS(scale > 0, "scale must be positive");
  CUSPATIAL_EXPECTS(num_levels >= 0 && num_levels < 16,
                    "maximum of levels might be in [0,16)");
  CUSPATIAL_EXPECTS(
      min_size > 0,
      "minimum number of points for a non-leaf node must be larger than zero");

  // detail::quadtree_on_points(x, y, x1, y1, x2, y2, scale, num_levels,
  // min_size, mr, 0);

  return cudf::experimental::type_dispatcher(
      x.type(), dispatch_construct_quadtree{}, x, y, x1, y1, x2, y2, scale,
      num_levels, min_size, mr, cudaStream_t{0});
}

}  // namespace cuspatial
