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

// #include <stdio.h>
// #include <string.h>
// #include <math.h>
// #include <algorithm>
// #include <cuda_runtime.h>
// #include <thrust/device_vector.h>
// #include <rmm/thrust_rmm_allocator.h>
// #include <cudf/types.h>
// #include <cudf/legacy/column.hpp>
// #include <cudf/utilities/error.hpp>
// #include <cuspatial/legacy/shapefile_readers.hpp>
// #include <cuspatial/error.hpp>
// #include <utility/utility.hpp>

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <rmm/thrust_rmm_allocator.h>
#include <memory>
#include <string>
#include "rmm/device_buffer.hpp"
#include "utility/utility.hpp"

namespace cuspatial
{
    // probably just import this here.
// namespace detail
// {
//     /*
//     * Read a polygon shapefile and fill in a polygons structure
//     * ToDo: read associated relational data into a CUDF Table
//     *
//     * filename: ESRI shapefile name (wtih .shp extension
//     * pm: structure polygons (fixed to double type) to hold polygon data

//     * Note: only the first layer is read - shapefiles have only one layer in GDALDataset model
//     */
//     void polygon_from_shapefile(std::string const& filename, polygons<double>& pm);
// } // namespace detail

/*
* read polygon data from file in ESRI Shapefile format; data type of vertices is fixed to double (GDF_FLOAT64)
* see shp_readers.hpp
*/

std::unique_ptr<cudf::experimental::table>
read_polygon_shapefile(std::string const& filename,
                       rmm::mr::device_memory_resource* mr,
                       cudaStream_t stream)
{
    auto pm = polygons<double>{};

    auto tid  = cudf::experimental::type_to_id<double>();
    auto type = cudf::data_type{tid};

    auto polygon_offsets = cudf::make_fixed_width_column(type, pm.num_feature);
    auto ring_offsets    = cudf::make_fixed_width_column(type, pm.num_ring);
    auto point_x         = cudf::make_fixed_width_column(type, pm.num_vertex);
    auto point_y         = cudf::make_fixed_width_column(type, pm.num_vertex);

    thrust::copy(pm.feature_length,
                 pm.feature_length + pm.num_feature,
                 polygon_offsets->mutable_view().end<cudf::size_type>());

    thrust::copy(pm.ring_length,
                 pm.ring_length + pm.num_ring,
                 ring_offsets->mutable_view().end<cudf::size_type>());

    thrust::copy(pm.x, pm.x + pm.num_vertex, ring_offsets->mutable_view().end<double>());
    thrust::copy(pm.y, pm.y + pm.num_vertex, ring_offsets->mutable_view().end<double>());

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
