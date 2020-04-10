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

template <typename Element, typename Timestamp>
__global__ void compute_distance_and_speed_kernel(
    // Point X
    cudf::column_device_view const x,
    // Point Y
    cudf::column_device_view const y,
    // Timestamps for each x/y point
    cudf::column_device_view const timestamp,
    // Offset for each trip in the group of trajectories
    cudf::column_device_view const offset,
    // Output distance column (in meters)
    cudf::mutable_column_device_view distance,
    // Output speed column (in meters per second)
    cudf::mutable_column_device_view speed) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < offset.size()) {
    auto const end = offset.element<int32_t>(tid) - 1;
    auto const idx = tid == 0 ? 0 : offset.element<int32_t>(tid - 1);
    auto const time_ms = simt::std::chrono::floor<cudf::timestamp_ms::duration>(
        timestamp.element<Timestamp>(end) - timestamp.element<Timestamp>(idx));

    if ((end - idx) < 2) {
      speed.element<Element>(tid) = -2.0;
      distance.element<Element>(tid) = -2.0;
    } else if (time_ms.count() == 0) {
      speed.element<Element>(tid) = -3.0;
      distance.element<Element>(tid) = -3.0;
    } else {
      // Reduce one trajectory group per thread
      Element dist_km{0.0};
      for (int32_t i = idx; i < end; ++i) {
        auto const x0 = x.element<Element>(i + 0);
        auto const x1 = x.element<Element>(i + 1);
        auto const y0 = y.element<Element>(i + 0);
        auto const y1 = y.element<Element>(i + 1);
        dist_km += sqrt(pow(x1 - x0, 2) + pow(y1 - y0, 2));
      }
      distance.element<Element>(tid) = dist_km * 1000;  // km to m
      speed.element<Element>(tid) = dist_km * 1000000 / time_ms.count();  // m/s
    }
  }
}

template <typename Element>
struct dispatch_timestamp {
  template <typename Timestamp>
  std::enable_if_t<cudf::is_timestamp<Timestamp>(),
                   std::unique_ptr<cudf::experimental::table>>
  operator()(cudf::column_view const& x, cudf::column_view const& y,
             cudf::column_view const& timestamp,
             cudf::column_view const& offset,
             rmm::mr::device_memory_resource* mr, cudaStream_t stream) {
    // Construct output columns
    auto size = offset.size();
    auto type = cudf::data_type{cudf::experimental::type_to_id<Element>()};
    std::vector<std::unique_ptr<cudf::column>> cols{};
    cols.reserve(2);
    // allocate distance output column
    cols.push_back(cudf::make_numeric_column(
        type, size, cudf::mask_state::UNALLOCATED, stream, mr));
    // allocate speed output column
    cols.push_back(cudf::make_numeric_column(
        type, size, cudf::mask_state::UNALLOCATED, stream, mr));

    // Query for launch bounds
    cudf::size_type block_size{0};
    cudf::size_type min_grid_size{0};
    CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size, &block_size,
        compute_distance_and_speed_kernel<Element, Timestamp>));

    cudf::experimental::detail::grid_1d grid{size, block_size};

    // Call compute kernel
    compute_distance_and_speed_kernel<Element, Timestamp>
        <<<grid.num_blocks, block_size, 0, stream>>>(
            *cudf::column_device_view::create(x, stream),
            *cudf::column_device_view::create(y, stream),
            *cudf::column_device_view::create(timestamp, stream),
            *cudf::column_device_view::create(offset, stream),
            *cudf::mutable_column_device_view::create(*cols.at(0), stream),
            *cudf::mutable_column_device_view::create(*cols.at(1), stream));

    // check for errors
    CHECK_CUDA(stream);

    return std::make_unique<cudf::experimental::table>(std::move(cols));
  }

  template <typename Timestamp>
  std::enable_if_t<not cudf::is_timestamp<Timestamp>(),
                   std::unique_ptr<cudf::experimental::table>>
  operator()(cudf::column_view const& x, cudf::column_view const& y,
             cudf::column_view const& timestamp,
             cudf::column_view const& offset,
             rmm::mr::device_memory_resource* mr, cudaStream_t stream) {
    CUDF_FAIL("Timestamp must be a timestamp type");
  }
};

struct dispatch_element {
  template <typename Element>
  std::enable_if_t<std::is_floating_point<Element>::value,
                   std::unique_ptr<cudf::experimental::table>>
  operator()(cudf::column_view const& x, cudf::column_view const& y,
             cudf::column_view const& timestamp,
             cudf::column_view const& offset,
             rmm::mr::device_memory_resource* mr, cudaStream_t stream) {
    return cudf::experimental::type_dispatcher(
        timestamp.type(), dispatch_timestamp<Element>{}, x, y, timestamp,
        offset, mr, stream);
  }

  template <typename Element>
  std::enable_if_t<not std::is_floating_point<Element>::value,
                   std::unique_ptr<cudf::experimental::table>>
  operator()(cudf::column_view const& x, cudf::column_view const& y,
             cudf::column_view const& timestamp,
             cudf::column_view const& offset,
             rmm::mr::device_memory_resource* mr, cudaStream_t stream) {
    CUDF_FAIL("X and Y must be floating point types");
  }
};

}  // namespace

namespace detail {
std::unique_ptr<cudf::experimental::table> compute_distance_and_speed(
    cudf::column_view const& x, cudf::column_view const& y,
    cudf::column_view const& timestamp, cudf::column_view const& offset,
    rmm::mr::device_memory_resource* mr, cudaStream_t stream) {
  return cudf::experimental::type_dispatcher(x.type(), dispatch_element{}, x, y,
                                             timestamp, offset, mr, stream);
}
}  // namespace detail

std::unique_ptr<cudf::experimental::table> compute_distance_and_speed(
    cudf::column_view const& x, cudf::column_view const& y,
    cudf::column_view const& timestamp, cudf::column_view const& offset,
    rmm::mr::device_memory_resource* mr) {
  CUSPATIAL_EXPECTS(!(x.is_empty() || y.is_empty() || offset.is_empty() ||
                      timestamp.is_empty()),
                    "Insufficient trajectory data");
  CUSPATIAL_EXPECTS(x.size() == y.size() && x.size() == timestamp.size(),
                    "Data size mismatch");
  CUSPATIAL_EXPECTS(x.type().id() == y.type().id(), "Data type mismatch");
  CUSPATIAL_EXPECTS(cudf::is_timestamp(timestamp.type()),
                    "Invalid timestamp datatype");
  CUSPATIAL_EXPECTS(offset.type().id() == cudf::INT32,
                    "Invalid trajectory offset type");
  CUSPATIAL_EXPECTS(!(x.has_nulls() || y.has_nulls() || timestamp.has_nulls() ||
                      offset.has_nulls()),
                    "NULL support unimplemented");
  CUSPATIAL_EXPECTS(offset.size() > 0 && x.size() >= offset.size(),
                    "Insufficient trajectory data");

  return detail::compute_distance_and_speed(x, y, timestamp, offset, mr, 0);
}

}  // namespace experimental
}  // namespace cuspatial
