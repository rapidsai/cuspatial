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

#include <utility/utility.hpp>
#include <cuspatial/error.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <thrust/copy.h>
#include <thrust/scan.h>

#include <memory>
#include <string>

namespace cuspatial {
namespace detail {

polygon_vectors read_polygon_shapefile(std::string const& filename);

std::unique_ptr<cudf::experimental::table>
read_polygon_shapefile(std::string const& filename,
                       rmm::mr::device_memory_resource* mr,
                       cudaStream_t stream)
{
    CUSPATIAL_EXPECTS(not filename.empty(), "Filename cannot be empty.");

    auto poly = detail::read_polygon_shapefile(filename);

    auto index_tid = cudf::experimental::type_to_id<cudf::size_type>();
    auto point_tid = cudf::experimental::type_to_id<double>();

    auto index_type = cudf::data_type{index_tid};
    auto point_type = cudf::data_type{point_tid};

    auto polygon_offsets = cudf::make_fixed_width_column(index_type, poly.feature_lengths.size());
    auto ring_offsets    = cudf::make_fixed_width_column(index_type, poly.ring_lengths.size());
    auto xs              = cudf::make_fixed_width_column(point_type, poly.xs.size());
    auto ys              = cudf::make_fixed_width_column(point_type, poly.ys.size());

    thrust::copy(poly.feature_lengths.cbegin(),
                 poly.feature_lengths.cend(),
                 polygon_offsets->mutable_view().begin<cudf::size_type>());

    thrust::copy(poly.ring_lengths.cbegin(),
                 poly.ring_lengths.cend(),
                 ring_offsets->mutable_view().begin<cudf::size_type>());

    thrust::copy(poly.xs.cbegin(),
                 poly.xs.cend(),
                 xs->mutable_view().begin<double>());

    thrust::copy(poly.ys.cbegin(),
                 poly.ys.cend(),
                 ys->mutable_view().begin<double>());

    // transform polygon lengths to polygon offsets
    thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream),
                           polygon_offsets->view().begin<cudf::size_type>(),
                           polygon_offsets->view().end<cudf::size_type>(),
                           polygon_offsets->mutable_view().begin<cudf::size_type>());

    // transform ring lengths to ring offsets
    thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream),
                           ring_offsets->view().begin<cudf::size_type>(),
                           ring_offsets->view().end<cudf::size_type>(),
                           ring_offsets->mutable_view().begin<cudf::size_type>());

    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.reserve(4);
    columns.emplace_back(std::move(polygon_offsets));
    columns.emplace_back(std::move(ring_offsets));
    columns.emplace_back(std::move(xs));
    columns.emplace_back(std::move(ys));

    return std::make_unique<cudf::experimental::table>(std::move(columns));
}

} // namespace detail

std::unique_ptr<cudf::experimental::table>
read_polygon_shapefile(std::string const& filename, rmm::mr::device_memory_resource* mr)
{
    return detail::read_polygon_shapefile(filename, mr, 0);
}

} // namespace cuspatial
