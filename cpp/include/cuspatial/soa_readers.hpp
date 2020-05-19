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

#pragma once

namespace cuspatial {
namespace experimental {

/**
 * @brief read int32_t data from file as column
 *
 * @param[in] filepath path to file.
 * @param[in] mr Optional resource to use for allocation
 *
 * @return cudf::column of integer data.
 **/
std::unique_ptr<cudf::column> read_int32_soa(const char *filename,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Read a column of timestamp data from file.
 *
 * @param[in] filepath path to file.
 * @param[in] mr Optional resource to use for allocating output device memory.
 *
 * @return cudf::column of timestamp data.
**/
std::unique_ptr<cudf::column> read_timestamp_soa(const char *filename,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief read lon/lat from file as two columns; data type is fixed to FLOAT64
 *
 * 
 * The file referred to by `filepath` contains data in location_3d layout (longitude/latitude/altitude, but altitude is not returned).
 *
 * @param[in] filepath path to file.
 * @param[in] mr Optional resource to use for allocating output device memory.
 *
 * @return A `std::pair` of two `FLOAT64` `cudf::column`s containing longitude and latitude data.
**/
std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>>
read_lonlat_points(std::string filepath, 
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Read x and y coordinate columns from file.
 * 
* 
 * The file referred to by `filepath` contains data in coordinate_2d layout (x/y).
 *
 * @param[in] filepath path to file.
 * @param[in] mr Optional resource to use for allocating output device memory.
 * 
 * @return A `std::pair` of two `FLOAT64` `cudf::column`s containing x and y data.
**/
std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>>
read_xy_points_soa(const char *filename,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Read polygon data from file.
 * 
 *
 * @param[in] filepath path to file.
 * @param[in] mr Optional resource to use for allocating output device memory.
 *
 * @note x/y can also be longitude and latitude.
 *
 * @return `std::vector` of `cudf::column`s:
 *          column(0): index polygons: INT64 offsets to the start of each polygon. The size of this column equals the number of polygons.
 *          column(1): index rings: INT64 offset to the start of each ring in the vertex data.
 *          column(2): FLOAT64 x-coordinates of concatenated polygons.
 *          column(3): FLOAT64 y-coordinates of concatenated polygons.
**/
std::vector<cudf::column>
read_polygons(std::string filepath, 
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

} // namespace experimental

}// namespace cuspatial
