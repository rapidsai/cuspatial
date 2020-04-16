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
#include <cudf/legacy/column.hpp>
#include <cudf/utilities/error.hpp>
#include <rmm/rmm.h>
#include <cuspatial/soa_readers.hpp>
#include <utility/utility.hpp>

namespace cuspatial
{
namespace detail
{
    std::pair<cudf::column, cudf::column> read_lonlat_points_soa(const char *filename,
        cudaStream_t stream, rmm::device_memory_resource* mr)
    {
        // Read the lon and lat points from the soa file into host memory
        double* p_lon=nullptr, *p_lat=nullptr;
        int num_p = read_point_lonlat<double>(filename, p_lon, p_lat);

        // Allocate a cudf::column with the lon host memory
        auto lon = cudf::test::fixed_width_column_wrapper<double>(p_lon, stream, mr);
        auto lat = cudf::test::fixed_width_column_wrapper<double>(p_lat, stream, mr);

        return std::pair<lon, lat>
    }
   
    std::pair<cudf::column, cudf::column> read_xy_points_soa(const char* filename,
        cudaStream_t stream, rmm::device_memory_resource* mr) 
    {
        double * p_x=nullptr, *p_y=nullptr;
        int num_p = read_point_xy<double>(filename,p_x,p_y);

        auto x = cudf::test::fixed_width_column_wrapper<double>(p_x, stream, mr);
        auto y = cudf::test::fixed_width_column_wrapper<double>(p_y, stream, mr);
       
        return std::make_pair(x, y);
    }	
} // detail namespace

    /**
    * @brief read lon/lat from file into two columns; data type is fixed to double (GDF_FLOAT64)
    *
    * see soa_readers.hpp
    */
    std::pair<cudf::column, cudf::column> read_lonlat_points_soa(const char *filename)
    {
      return detail::read_lonlat_points_soa(filename, cudaStream_t{0}, rmm::mr::get_default_resource());
    }
	
    /**
    * @brief read x/y from file into two columns; data type is fixed to double (GDF_FLOAT64)
    *
    * see soa_readers.hpp
    */
    std::pair<cudf::column, cudf::column> read_xy_points_soa(const char *filename)                              
    {
      return detail::read_xy_points_soa(filename, cudaStream_t{0}, rmm::mr::get_default_resource());
    }	
}
