/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include "detail/cartesian_product_group_index_iterator.cuh"
#include "detail/hausdorff.cuh"

#include <utility/scatter_output_iterator.cuh>
#include <utility/size_from_offsets.cuh>

#include <cuspatial/error.hpp>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/iterator/transform_iterator.h>

#include <limits>
#include <memory>

namespace cuspatial {
namespace detail {
namespace {

template <typename T>
struct hausdorff_accumulator_factory {
  cudf::column_device_view const xs;
  cudf::column_device_view const ys;

  hausdorff_acc<T> inline __device__ operator()(cartesian_product_group_index const& idx)
  {
    auto const a_idx = idx.group_a.offset + idx.element_a_idx;
    auto const b_idx = idx.group_b.offset + idx.element_b_idx;

    auto const distance = hypot(xs.element<T>(b_idx) - xs.element<T>(a_idx),
                                ys.element<T>(b_idx) - ys.element<T>(a_idx));

    return hausdorff_acc<T>{b_idx, b_idx, distance, distance, 0};
  }
};

struct hausdorff_functor {
  template <typename T, typename... Args>
  std::enable_if_t<not std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    Args&&...)
  {
    CUSPATIAL_FAIL("Non-floating point operation is not supported");
  }

  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    cudf::column_view const& xs,
    cudf::column_view const& ys,
    cudf::column_view const& space_offsets,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    auto const num_points  = static_cast<uint32_t>(xs.size());
    auto const num_spaces  = static_cast<uint32_t>(space_offsets.size());
    auto const num_results = static_cast<uint64_t>(num_spaces) * static_cast<uint64_t>(num_spaces);

    CUSPATIAL_EXPECTS(
      num_results < static_cast<uint64_t>(std::numeric_limits<cudf::size_type>::max()),
      "Matrix of spaces must be less than 2^31");

    if (num_results == 0) {
      return cudf::make_empty_column(cudf::data_type{cudf::type_to_id<T>()});
    }

    // ===== Make Hausdorff Accumulator ============================================================

    auto gcp_iter = make_cartesian_product_group_index_iterator(
      num_points, num_spaces, space_offsets.begin<uint32_t>());

    auto d_xs = cudf::column_device_view::create(xs);
    auto d_ys = cudf::column_device_view::create(ys);

    auto hausdorff_acc_iter =
      thrust::make_transform_iterator(gcp_iter, hausdorff_accumulator_factory<T>{*d_xs, *d_ys});

    // ===== Materialize ===========================================================================

    auto result = cudf::make_fixed_width_column(cudf::data_type{cudf::type_to_id<T>()},
                                                static_cast<cudf::size_type>(num_results),
                                                cudf::mask_state::UNALLOCATED,
                                                stream,
                                                mr);

    auto result_temp = rmm::device_uvector<hausdorff_acc<T>>(num_results, stream);

    auto scatter_map = thrust::make_transform_iterator(
      gcp_iter, [num_spaces] __device__(cartesian_product_group_index const& idx) {
        // the given output is only a "result" if it is the last output for a given pair-of-spaces
        bool const is_result = idx.element_a_idx + 1 == idx.group_a.size &&  //
                               idx.element_b_idx + 1 == idx.group_b.size;

        if (not is_result) { return static_cast<uint32_t>(-1); }

        // the destination for the result is determined per- pair-of-spaces
        return idx.group_b.idx * num_spaces + idx.group_a.idx;
      });

    auto scatter_out = make_scatter_output_iterator(result_temp.begin(), scatter_map);

    auto gpc_key_iter = thrust::make_transform_iterator(
      gcp_iter, [] __device__(cartesian_product_group_index const& idx) {
        return thrust::make_pair(idx.group_a.idx, idx.group_b.idx);
      });

    // the following output iterator and `inclusive_scan_by_key` could be replaced by a
    // reduce_by_key, if it supported non-commutative operators.

    auto const num_cartesian =
      static_cast<uint64_t>(num_points) * static_cast<uint64_t>(num_points);

    //
    // `thrust::inclusive_scan_by_key` causes an out-of-memory
    // error on input sizes close to (but not exactly) INT_MAX.
    //
    // Doing the reduction in chunks is a temporary workaround until this is fixed.
    //
    // The magic number between OOM vs. no OOM is somewhere between:
    //                                             (1uL << 31uL) - 2048uL;
    auto const magic_inclusive_scan_by_key_limit = (1uL << 31uL) - 4096uL;

    for (auto itr = gpc_key_iter, end = gpc_key_iter + num_cartesian; itr < end;) {
      auto const len = static_cast<uint32_t>(std::min(
        static_cast<uint64_t>(thrust::distance(itr, end)), magic_inclusive_scan_by_key_limit));

      thrust::inclusive_scan_by_key(rmm::exec_policy(stream),
                                    itr,
                                    itr + len,
                                    hausdorff_acc_iter,
                                    scatter_out,
                                    thrust::equal_to<thrust::pair<uint32_t, uint32_t>>());
      thrust::advance(itr, len);
    }

    thrust::transform(rmm::exec_policy(stream),
                      result_temp.begin(),
                      result_temp.end(),
                      result->mutable_view().begin<T>(),
                      [] __device__(hausdorff_acc<T> const& a) { return static_cast<T>(a); });

    return result;
  }
};

}  // namespace
}  // namespace detail

std::unique_ptr<cudf::column> directed_hausdorff_distance(cudf::column_view const& xs,
                                                          cudf::column_view const& ys,
                                                          cudf::column_view const& space_offsets,
                                                          rmm::mr::device_memory_resource* mr)
{
  CUSPATIAL_EXPECTS(xs.type() == ys.type(), "Inputs `xs` and `ys` must have same type.");
  CUSPATIAL_EXPECTS(xs.size() == ys.size(), "Inputs `xs` and `ys` must have same length.");

  CUSPATIAL_EXPECTS(not xs.has_nulls() and not ys.has_nulls() and not space_offsets.has_nulls(),
                    "Inputs must not have nulls.");

  CUSPATIAL_EXPECTS(xs.size() >= space_offsets.size(),
                    "At least one point is required for each space");

  return cudf::type_dispatcher(
    xs.type(), detail::hausdorff_functor(), xs, ys, space_offsets, rmm::cuda_stream_default, mr);
}

}  // namespace cuspatial
