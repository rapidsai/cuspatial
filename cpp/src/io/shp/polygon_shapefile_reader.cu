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

#include <cuspatial/error.hpp>

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/thrust_rmm_allocator.h>

#include <thrust/scan.h>

#include <memory>
#include <string>

namespace cuspatial {
namespace detail {

template<typename T>
std::unique_ptr<cudf::column>
make_column(std::vector<T> source, cudaStream_t stream, rmm::mr::device_memory_resource* mr)
{
    auto tid = cudf::experimental::type_to_id<T>();
    auto type = cudf::data_type{tid};
    auto buffer = rmm::device_buffer(source.data(), sizeof(T) * source.size(), stream, mr);
    return std::make_unique<cudf::column>(type, source.size(), buffer);
}

std::tuple<std::vector<cudf::size_type>,
           std::vector<cudf::size_type>,
           std::vector<double>,
           std::vector<double>>
read_polygon_shapefile(std::string const& filename);

std::tuple<std::unique_ptr<cudf::column>,
           std::unique_ptr<cudf::column>,
           std::unique_ptr<cudf::column>,
           std::unique_ptr<cudf::column>>
read_polygon_shapefile(std::string const& filename,
                       rmm::mr::device_memory_resource* mr,
                       cudaStream_t stream)
{
    CUSPATIAL_EXPECTS(not filename.empty(), "Filename cannot be empty.");

    auto poly_vectors = detail::read_polygon_shapefile(filename);

    auto index_tid = cudf::experimental::type_to_id<cudf::size_type>();
    auto point_tid = cudf::experimental::type_to_id<double>();

    auto index_type = cudf::data_type{index_tid};
    auto point_type = cudf::data_type{point_tid};

    auto polygon_offsets = make_column<cudf::size_type>(std::get<0>(poly_vectors), stream, mr);
    auto ring_offsets    = make_column<cudf::size_type>(std::get<1>(poly_vectors), stream, mr);
    auto xs              = make_column<double>(std::get<2>(poly_vectors), stream, mr);
    auto ys              = make_column<double>(std::get<3>(poly_vectors), stream, mr);

    // transform polygon lengths to polygon offsets
    thrust::exclusive_scan(rmm::exec_policy(stream)->on(stream),
                           polygon_offsets->view().begin<cudf::size_type>(),
                           polygon_offsets->view().end<cudf::size_type>(),
                           polygon_offsets->mutable_view().begin<cudf::size_type>());

    // transform ring lengths to ring offsets
    thrust::exclusive_scan(rmm::exec_policy(stream)->on(stream),
                           ring_offsets->view().begin<cudf::size_type>(),
                           ring_offsets->view().end<cudf::size_type>(),
                           ring_offsets->mutable_view().begin<cudf::size_type>());

    return std::make_tuple(std::move(polygon_offsets),
                           std::move(ring_offsets),
                           std::move(xs),
                           std::move(ys));
}

} // namespace detail

std::tuple<std::unique_ptr<cudf::column>,
           std::unique_ptr<cudf::column>,
           std::unique_ptr<cudf::column>,
           std::unique_ptr<cudf::column>>
read_polygon_shapefile(std::string const& filename, rmm::mr::device_memory_resource* mr)
{
    return detail::read_polygon_shapefile(filename, mr, 0);
}

} // namespace cuspatial
