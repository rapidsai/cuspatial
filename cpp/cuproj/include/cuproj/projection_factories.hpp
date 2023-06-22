/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cuproj/projection.cuh>
#include <cuproj/projection_parameters.hpp>

template <typename Coordinate, typename T = typename Coordinate::value_type>
cuproj::projection<Coordinate> make_utm_projection(int zone, cuproj::hemisphere hemisphere)
{
  cuproj::projection_parameters<T> tmerc_proj_params{
    cuproj::make_ellipsoid_wgs84<T>(), zone, hemisphere, T{0}, T{0}};

  std::vector<cuproj::operation_type> h_utm_pipeline{
    cuproj::operation_type::AXIS_SWAP,
    cuproj::operation_type::DEGREES_TO_RADIANS,
    cuproj::operation_type::CLAMP_ANGULAR_COORDINATES,
    cuproj::operation_type::TRANSVERSE_MERCATOR,
    cuproj::operation_type::OFFSET_SCALE_CARTESIAN_COORDINATES};

  return cuproj::projection<Coordinate>{h_utm_pipeline, tmerc_proj_params};
}
