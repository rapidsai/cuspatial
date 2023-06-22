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

#include <cuproj/error.hpp>
#include <cuproj/projection.cuh>
#include <cuproj/projection_parameters.hpp>

namespace cuproj {

template <typename Coordinate, typename T = typename Coordinate::value_type>
projection<Coordinate> make_utm_projection(int zone, hemisphere hemisphere)
{
  projection_parameters<T> tmerc_proj_params{
    make_ellipsoid_wgs84<T>(), zone, hemisphere, T{0}, T{0}};

  std::vector<cuproj::operation_type> h_utm_pipeline{
    operation_type::AXIS_SWAP,
    operation_type::DEGREES_TO_RADIANS,
    operation_type::CLAMP_ANGULAR_COORDINATES,
    operation_type::TRANSVERSE_MERCATOR,
    operation_type::OFFSET_SCALE_CARTESIAN_COORDINATES};

  return projection<Coordinate>{h_utm_pipeline, tmerc_proj_params};
}

inline auto epsg_to_utm_zone(std::string const& epsg_str)
{
  int epsg = [&]() {
    try {
      CUPROJ_EXPECTS(epsg_str.find("EPSG:") == 0, "EPSG code must start with 'EPSG:'");
      return std::stoi(epsg_str.substr(epsg_str.find_first_not_of("EPSG:")));
    } catch (std::invalid_argument const&) {
      CUPROJ_FAIL("Invalid EPSG code");
    }
  }();

  if (epsg >= 32601 && epsg <= 32660) {
    return std::make_pair(epsg - 32600, hemisphere::NORTH);
  } else if (epsg >= 32701 && epsg <= 32760) {
    return std::make_pair(epsg - 32700, hemisphere::SOUTH);
  } else {
    CUPROJ_FAIL("Unsupported UTM EPSG code. Must be in range [32601, 32760] or [32701, 32760]]");
  }
}

template <typename Coordinate>
cuproj::projection<Coordinate> make_projection(std::string const& src_epsg,
                                               std::string const& dst_epsg)
{
  // TODO make this work forward or inverse
  CUPROJ_EXPECTS(src_epsg == "EPSG:4326", "Source EPSG must be WGS84 (EPSG:4326)");
  auto [dst_zone, dst_hemisphere] = epsg_to_utm_zone(dst_epsg);
  return make_utm_projection<Coordinate>(dst_zone, dst_hemisphere);
}

}  // namespace cuproj
