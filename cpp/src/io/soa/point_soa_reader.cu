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

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cudf/types.h>
#include <cudf/column/column.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <rmm/rmm.h>
#include <cuspatial/soa_readers.hpp>
#include <utility/legacy/utility.hpp>

namespace cuspatial
{
namespace detail
{

std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>>
read_lonlat_points_soa(std::string const& filename,
    cudaStream_t stream, rmm::mr::device_memory_resource* mr)
{
    // Read the lon and lat points from the soa file into host memory
    double* p_lon=nullptr, *p_lat=nullptr;
    int num_p = read_point_lonlat<double>(filename.c_str(), p_lon, p_lat);

    auto tid = cudf::experimental::type_to_id<double>();
    auto type = cudf::data_type{ tid };

    // Allocate a cudf::column with the lon host memory
    auto lon = std::make_unique<cudf::column>(cudf::column(type,
        num_p, rmm::device_buffer(p_lon, num_p)));
    auto lat = std::make_unique<cudf::column>(cudf::column(type,
        num_p, rmm::device_buffer(p_lat, num_p)));

    return std::make_pair(std::move(lon), std::move(lat));
}

std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>>
read_xy_points_soa(std::string const& filename,
    cudaStream_t stream, rmm::mr::device_memory_resource* mr)
{
    double * p_x=nullptr, *p_y=nullptr;
    int num_p = read_point_xy<double>(filename.c_str(), p_x, p_y);

    auto tid = cudf::experimental::type_to_id<double>();
    auto type = cudf::data_type{ tid };

    auto x = std::make_unique<cudf::column>(cudf::column(type,
        num_p, rmm::device_buffer(p_x, num_p)));
    auto y = std::make_unique<cudf::column>(cudf::column(type,
        num_p, rmm::device_buffer(p_y, num_p)));
   
    return std::make_pair(std::move(x), std::move(y));
}

} // detail namespace

namespace experimental {

/**
* @brief read lon/lat from file into two columns; data type is fixed to double (GDF_FLOAT64)
*
* see soa_readers.hpp
*/
std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>> read_lonlat_points_soa(std::string const& filename)
{
    auto result = detail::read_lonlat_points_soa(filename.c_str(), cudaStream_t{0}, rmm::mr::get_default_resource());
    return result;
}

/**
* @brief read x/y from file into two columns; data type is fixed to double (GDF_FLOAT64)
*
* see soa_readers.hpp
*/
std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>> read_xy_points_soa(std::string const& filename)
{
    auto result = detail::read_xy_points_soa(filename.c_str(), cudaStream_t{0}, rmm::mr::get_default_resource());
    return result;
}

} // experimental namespace

} // cuspatial namespace
