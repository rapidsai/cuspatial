/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <rmm/thrust_rmm_allocator.h>
#include <memory>
#include <string>
#include <utility/utility.hpp>

namespace cuspatial {

namespace detail {

polygons read_polygon_shapefile(const char *filename);

} // namespace detail

std::unique_ptr<cudf::experimental::table>
read_polygon_shapefile(std::string const& filename,
                       rmm::mr::device_memory_resource* mr,
                       cudaStream_t stream)
{
    auto poly = detail::read_polygon_shapefile(filename.c_str());

    auto tid  = cudf::experimental::type_to_id<double>();
    auto type = cudf::data_type{tid};

    auto polygon_offsets = cudf::make_fixed_width_column(type, poly.feature_lengths.size());
    auto ring_offsets    = cudf::make_fixed_width_column(type, poly.ring_lengths.size());
    auto xs              = cudf::make_fixed_width_column(type, poly.xs.size());
    auto ys              = cudf::make_fixed_width_column(type, poly.ys.size());

    thrust::copy(poly.feature_lengths.begin(),
                 poly.feature_lengths.end(),
                 polygon_offsets->mutable_view().begin<cudf::size_type>());

    thrust::copy(poly.ring_lengths.begin(),
                 poly.ring_lengths.end(),
                 ring_offsets->mutable_view().begin<cudf::size_type>());

    thrust::copy(poly.xs.begin(), poly.xs.end(), xs->mutable_view().begin<double>());
    thrust::copy(poly.ys.begin(), poly.ys.end(), ys->mutable_view().begin<double>());

    // transform polygon lengths to polygon offsets
    thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream),
                           polygon_offsets->view().begin<double>(),
                           polygon_offsets->view().end<double>(),
                           polygon_offsets->mutable_view().begin<double>());

    // transform ring lengths to ring offsets
    thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream),
                           ring_offsets->view().begin<double>(),
                           ring_offsets->view().end<double>(),
                           ring_offsets->mutable_view().begin<double>());

    auto output_columns = std::vector<std::unique_ptr<cudf::column>>();

    return std::make_unique<cudf::experimental::table>(std::move(output_columns));
}

} // namespace cuspatial
