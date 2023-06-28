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

namespace detail {

/**
 * @internal
 * @brief Check if the given EPSG code string is valid
 *
 * @param epsg_str the EPSG code string
 * @return true if the EPSG code is valid, false otherwise
 */
inline bool is_epsg(std::string const& epsg_str) { return epsg_str.find("EPSG:") == 0; }

/**
 * @internal
 * @brief Convert an EPSG code string to its integer value (the part after 'EPSG:')
 *
 * @param epsg_str the EPSG code string
 * @return the integer value of the EPSG code
 */
inline int epsg_stoi(std::string const& epsg_str)
{
  try {
    CUPROJ_EXPECTS(is_epsg(epsg_str), "EPSG code must start with 'EPSG:'");
    return std::stoi(epsg_str.substr(epsg_str.find_first_not_of("EPSG:")));
  } catch (std::invalid_argument const&) {
    CUPROJ_FAIL("Invalid EPSG code");
  }
}

/**
 * @internal
 * @brief Check if the given EPSG code string is for WGS84
 *
 * @param epsg_str the EPSG code string
 * @return true if the EPSG code is for WGS84, false otherwise
 */
inline bool is_wgs_84(std::string const& epsg_str) { return epsg_str == "EPSG:4326"; }

/**
 * @internal
 * @brief Convert an EPSG code string to a UTM zone and hemisphere
 *
 * @param epsg_str the EPSG code string
 * @return a pair of UTM zone and hemisphere
 */
inline auto epsg_to_utm_zone(std::string const& epsg_str)
{
  int epsg = epsg_stoi(epsg_str);

  if (epsg >= 32601 && epsg <= 32660) {
    return std::make_pair(epsg - 32600, hemisphere::NORTH);
  } else if (epsg >= 32701 && epsg <= 32760) {
    return std::make_pair(epsg - 32700, hemisphere::SOUTH);
  } else {
    CUPROJ_FAIL("Unsupported UTM EPSG code. Must be in range [32601, 32760] or [32701, 32760]]");
  }
}

}  // namespace detail

/**
 * @brief Create a WGS84<-->UTM projection for the given UTM zone and hemisphere
 *
 * @tparam Coordinate the coordinate type
 * @tparam Coordinate::value_type the coordinate value type
 * @param zone the UTM zone
 * @param hemisphere the UTM hemisphere
 * @param inverse if true, create a projection from UTM to WGS84 (default is false, meaning WGS84 to
 * UTM)
 * @return a projection object implementing the requested transformation
 */
template <typename Coordinate, typename T = typename Coordinate::value_type>
projection<Coordinate> make_utm_projection(int zone, hemisphere hemisphere, bool inverse = false)
{
  projection_parameters<T> tmerc_proj_params{
    make_ellipsoid_wgs84<T>(), zone, hemisphere, T{0}, T{0}};

  std::vector<cuproj::operation_type> h_utm_pipeline{
    operation_type::AXIS_SWAP,
    operation_type::DEGREES_TO_RADIANS,
    operation_type::CLAMP_ANGULAR_COORDINATES,
    operation_type::TRANSVERSE_MERCATOR,
    operation_type::OFFSET_SCALE_CARTESIAN_COORDINATES};

  if (inverse) { std::reverse(h_utm_pipeline.begin(), h_utm_pipeline.end()); }

  return projection<Coordinate>{h_utm_pipeline, tmerc_proj_params};
}

/**
 * @brief Create a projection object from EPSG codes
 *
 * @throw cuproj::logic_error if the EPSG codes describe a transformation that is not supported
 *
 * @note Currently only WGS84 to UTM and UTM to WGS84 are supported, so one of the EPSG codes must
 * be "EPSG:4326" (WGS84) and the other must be a UTM EPSG code.
 *
 * @tparam Coordinate the coordinate type
 * @param src_epsg the source EPSG code
 * @param dst_epsg the destination EPSG code
 * @return a projection object implementing the transformation between the two EPSG codes
 */
template <typename Coordinate>
cuproj::projection<Coordinate> make_projection(std::string const& src_epsg,
                                               std::string const& dst_epsg)
{
  if (detail::is_wgs_84(src_epsg)) {
    auto [dst_zone, dst_hemisphere] = detail::epsg_to_utm_zone(dst_epsg);
    return make_utm_projection<Coordinate>(dst_zone, dst_hemisphere);
  } else {
    CUPROJ_EXPECTS(detail::is_wgs_84(dst_epsg),
                   "Source or Destination EPSG must be WGS84 (EPSG:4326)");
    auto [src_zone, src_hemisphere] = detail::epsg_to_utm_zone(src_epsg);
    return make_utm_projection<Coordinate>(src_zone, src_hemisphere, true);
  }
}

}  // namespace cuproj
