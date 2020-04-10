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
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/table/table.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include "trajectory.hpp"

namespace cuspatial {
namespace experimental {

namespace {

template <typename Element>
__global__ void compute_bounding_boxes_kernel(
    // Point X
    cudf::column_device_view const x,
    // Point Y
    cudf::column_device_view const y,
    // Offset for each trip in the group of trajectories
    cudf::column_device_view const offset,
    cudf::mutable_column_device_view bbox_x1,
    cudf::mutable_column_device_view bbox_y1,
    cudf::mutable_column_device_view bbox_x2,
    cudf::mutable_column_device_view bbox_y2) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < offset.size()) {
    auto const end = offset.element<int32_t>(tid) - 1;
    auto const idx = tid == 0 ? 0 : offset.element<int32_t>(tid - 1);

    auto x1 = x.element<Element>(idx);
    auto y1 = y.element<Element>(idx);
    auto x2 = x.element<Element>(idx);
    auto y2 = y.element<Element>(idx);

    for (int32_t i = idx; ++i < end;) {
      x1 = min(x1, x.element<Element>(i));
      y1 = min(y1, y.element<Element>(i));
      x2 = max(x2, x.element<Element>(i));
      y2 = max(y2, y.element<Element>(i));
    }

    bbox_x1.element<Element>(tid) = x1;
    bbox_y1.element<Element>(tid) = y1;
    bbox_x2.element<Element>(tid) = x2;
    bbox_y2.element<Element>(tid) = y2;
  }
}

struct dispatch_element {
  template <typename Element>
  std::enable_if_t<std::is_floating_point<Element>::value,
                   std::unique_ptr<cudf::experimental::table>>
  operator()(cudf::column_view const& x, cudf::column_view const& y,
             cudf::column_view const& offset,
             rmm::mr::device_memory_resource* mr, cudaStream_t stream) {
    // Construct output columns
    auto size = offset.size();
    auto type = cudf::data_type{cudf::experimental::type_to_id<Element>()};
    std::vector<std::unique_ptr<cudf::column>> cols{};
    cols.reserve(4);
    // allocate bbox_x1 output column
    cols.push_back(cudf::make_numeric_column(
        type, size, cudf::mask_state::UNALLOCATED, stream, mr));
    // allocate bbox_y1 output column
    cols.push_back(cudf::make_numeric_column(
        type, size, cudf::mask_state::UNALLOCATED, stream, mr));
    // allocate bbox_x2 output column
    cols.push_back(cudf::make_numeric_column(
        type, size, cudf::mask_state::UNALLOCATED, stream, mr));
    // allocate bbox_y2 output column
    cols.push_back(cudf::make_numeric_column(
        type, size, cudf::mask_state::UNALLOCATED, stream, mr));

    // Query for launch bounds
    cudf::size_type block_size{0};
    cudf::size_type min_grid_size{0};
    CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size, &block_size, compute_bounding_boxes_kernel<Element>));

    cudf::experimental::detail::grid_1d grid{size, block_size};

    // Call compute kernel
    compute_bounding_boxes_kernel<Element>
        <<<grid.num_blocks, block_size, 0, stream>>>(
            *cudf::column_device_view::create(x, stream),
            *cudf::column_device_view::create(y, stream),
            *cudf::column_device_view::create(offset, stream),
            *cudf::mutable_column_device_view::create(*cols.at(0), stream),
            *cudf::mutable_column_device_view::create(*cols.at(1), stream),
            *cudf::mutable_column_device_view::create(*cols.at(2), stream),
            *cudf::mutable_column_device_view::create(*cols.at(3), stream));

    // check for errors
    CHECK_CUDA(stream);

    return std::make_unique<cudf::experimental::table>(std::move(cols));
  }

  template <typename Element>
  std::enable_if_t<not std::is_floating_point<Element>::value,
                   std::unique_ptr<cudf::experimental::table>>
  operator()(cudf::column_view const& x, cudf::column_view const& y,
             cudf::column_view const& offset,
             rmm::mr::device_memory_resource* mr, cudaStream_t stream) {
    CUDF_FAIL("X and Y must be floating point types");
  }
};

}  // namespace

namespace detail {
std::unique_ptr<cudf::experimental::table> compute_bounding_boxes(
    cudf::column_view const& x, cudf::column_view const& y,
    cudf::column_view const& offset, rmm::mr::device_memory_resource* mr,
    cudaStream_t stream) {
  return cudf::experimental::type_dispatcher(x.type(), dispatch_element{}, x, y,
                                             offset, mr, stream);
}
}  // namespace detail

std::unique_ptr<cudf::experimental::table> compute_bounding_boxes(
    cudf::column_view const& x, cudf::column_view const& y,
    cudf::column_view const& offset, rmm::mr::device_memory_resource* mr) {
  CUSPATIAL_EXPECTS(!(x.is_empty() || y.is_empty() || offset.is_empty()),
                    "Insufficient trajectory data");
  CUSPATIAL_EXPECTS(x.size() == y.size(), "Data size mismatch");
  CUSPATIAL_EXPECTS(x.type().id() == y.type().id(), "Data type mismatch");
  CUSPATIAL_EXPECTS(offset.type().id() == cudf::INT32,
                    "Invalid trajectory offset type");
  CUSPATIAL_EXPECTS(!(x.has_nulls() || y.has_nulls() || offset.has_nulls()),
                    "NULL support unimplemented");
  CUSPATIAL_EXPECTS(offset.size() > 0 && x.size() >= offset.size(),
                    "Insufficient trajectory data");

  return detail::compute_bounding_boxes(x, y, offset, mr, 0);
}

}  // namespace experimental
}  // namespace cuspatial
