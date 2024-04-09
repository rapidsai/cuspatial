/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required point_b_y applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuspatial/distance.cuh>
#include <cuspatial/error.hpp>
#include <cuspatial/iterator_factory.cuh>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

#include <memory>
#include <type_traits>

namespace {

/**
 * @brief Split `col` into equal size chunks, each has `size`.
 *
 * @note only applicable to fixed width type.
 * @note only applicable to columns of `size*size`.
 */
template <typename T>
std::vector<cudf::column_view> split_by_size(cudf::column_view const& col, cudf::size_type size)
{
  std::vector<cudf::column_view> res;
  cudf::size_type num_splits = col.size() / size;
  std::transform(thrust::counting_iterator(0),
                 thrust::counting_iterator(num_splits),
                 std::back_inserter(res),
                 [size, num_splits, &col](int i) {
                   return cudf::column_view(
                     col.type(), size, col.data<T>(), nullptr, 0, size * i, {});
                 });
  return res;
}

struct hausdorff_functor {
  template <typename T, typename... Args>
  std::enable_if_t<not std::is_floating_point<T>::value,
                   std::pair<std::unique_ptr<cudf::column>, cudf::table_view>>
  operator()(Args&&...)
  {
    CUSPATIAL_FAIL("Non-floating point operation is not supported");
  }

  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value,
                   std::pair<std::unique_ptr<cudf::column>, cudf::table_view>>
  operator()(cudf::column_view const& xs,
             cudf::column_view const& ys,
             cudf::column_view const& space_offsets,
             rmm::cuda_stream_view stream,
             rmm::device_async_resource_ref mr)
  {
    auto const num_points = static_cast<uint32_t>(xs.size());
    auto const num_spaces = static_cast<uint32_t>(space_offsets.size());

    CUSPATIAL_EXPECTS(num_spaces < (1 << 15), "Total number of spaces must be less than 2^16");

    auto const num_results = num_spaces * num_spaces;

    auto tid    = cudf::type_to_id<T>();
    auto result = cudf::make_fixed_width_column(
      cudf::data_type{tid}, num_results, cudf::mask_state::UNALLOCATED, stream, mr);

    if (result->size() == 0) { return {std::move(result), cudf::table_view{}}; }

    auto const result_view = result->mutable_view();

    auto points_iter        = cuspatial::make_vec_2d_iterator(xs.begin<T>(), ys.begin<T>());
    auto space_offsets_iter = space_offsets.begin<cudf::size_type>();

    cuspatial::directed_hausdorff_distance(points_iter,
                                           points_iter + num_points,
                                           space_offsets_iter,
                                           space_offsets_iter + num_spaces,
                                           result_view.begin<T>(),
                                           stream);

    return {std::move(result), cudf::table_view(split_by_size<T>(result->view(), num_spaces))};
  }
};

}  // namespace

namespace cuspatial {

std::pair<std::unique_ptr<cudf::column>, cudf::table_view> directed_hausdorff_distance(
  cudf::column_view const& xs,
  cudf::column_view const& ys,
  cudf::column_view const& space_offsets,
  rmm::device_async_resource_ref mr)
{
  CUSPATIAL_EXPECTS(xs.type() == ys.type(), "Inputs `xs` and `ys` must have same type.");
  CUSPATIAL_EXPECTS(xs.size() == ys.size(), "Inputs `xs` and `ys` must have same length.");

  CUSPATIAL_EXPECTS(not xs.has_nulls() and not ys.has_nulls() and not space_offsets.has_nulls(),
                    "Inputs must not have nulls.");

  return cudf::type_dispatcher(
    xs.type(), hausdorff_functor(), xs, ys, space_offsets, rmm::cuda_stream_default, mr);
}

}  // namespace cuspatial
