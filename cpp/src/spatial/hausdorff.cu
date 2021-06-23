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

    auto gcp_key_iter = thrust::make_transform_iterator(
      gcp_iter, [] __device__(cartesian_product_group_index const& idx) {
        return thrust::make_pair(idx.group_a.idx, idx.group_b.idx);
      });

    // the following output iterator and `inclusive_scan_by_key` could be replaced by a
    // reduce_by_key, if it supported non-commutative operators and more inputs (int 64).

    // copy offsets to host to be used in batch-size deduction.

    thrust::host_vector<uint32_t> h_space_offsets(space_offsets.size());

    CUDA_TRY(cudaMemcpyAsync(h_space_offsets.data(),
                             space_offsets.data<uint32_t>(),
                             space_offsets.size() * sizeof(uint32_t),
                             cudaMemcpyDeviceToHost,
                             stream.value()));

    stream.synchronize();

    // This algorithm processes n^2 comparisons, which cannot be handled in a single pass.
    // The following loop enables multiple passes over the data, and the inner loop
    // combines batches of elements to fit as many as possible in a single pass.

    for (uint32_t i = 1; i <= h_space_offsets.size(); i++) {
      uint32_t elements_in_pass = 0;

      // deduce appropriate pass size based on input.
      for (;i <= h_space_offsets.size(); i++) {
        uint32_t space_size =
          (i < h_space_offsets.size() ? h_space_offsets[i] : xs.size()) - h_space_offsets[i - 1];
        uint32_t elements_in_batch = xs.size() * space_size;
        if (elements_in_pass > std::numeric_limits<int32_t>::max() - elements_in_batch) { break; }
        elements_in_pass += elements_in_batch;
      }

      thrust::inclusive_scan_by_key(rmm::exec_policy(stream),
                                    gcp_key_iter,
                                    gcp_key_iter + elements_in_batch,
                                    hausdorff_acc_iter,
                                    scatter_out,
                                    thrust::equal_to<>());

      hausdorff_acc_iter += elements_in_batch;
      gcp_key_iter += elements_in_batch;
      scatter_out += elements_in_batch;
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
