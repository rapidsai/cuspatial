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

#include <memory>

namespace cuproj {

/**
 * @addtogroup projection_factories
 * @{
 */

namespace detail {

/**
 * @internal
 * @brief A class to represent EPSG codes
 */
class epsg_code {
 public:
  /// Construct an epsg_code from a string
  explicit epsg_code(std::string const& str) : str_(str)
  {
    std::transform(str_.begin(), str_.end(), str_.begin(), ::toupper);
    CUPROJ_EXPECTS(valid_prefix(), "EPSG code must start with 'EPSG:'");

    try {
      epsg_ = std::stoi(str_.substr(str_.find_first_not_of("EPSG:")));
    } catch (std::invalid_argument const&) {
      CUPROJ_FAIL("Invalid EPSG code");
    }
  }

  /// Construct an epsg_code from an integer
  explicit epsg_code(int code) : str_("EPSG:" + std::to_string(code)), epsg_(code) {}

  explicit operator std::string() const { return str_; }  //< Return the EPSG code as a string
  explicit operator int() const { return epsg_; }         //< Return the EPSG code as an integer

  // Return true if the EPSG code is for WGS84 (4326), false otherwise
  inline bool is_wgs_84() const { return epsg_ == 4326; }

  /// Return a [zone, hemisphere] pair for the UTM zone corresponding to the EPSG code
  inline auto to_utm_zone()
  {
    if (epsg_ >= 32601 && epsg_ <= 32660) {
      return std::make_pair(epsg_ - 32600, hemisphere::NORTH);
    } else if (epsg_ >= 32701 && epsg_ <= 32760) {
      return std::make_pair(epsg_ - 32700, hemisphere::SOUTH);
    } else {
      CUPROJ_FAIL("Unsupported UTM EPSG code. Must be in range [32601, 32760] or [32701, 32760]]");
    }
  }

 private:
  std::string str_;
  int epsg_;

  /// Return true if the EPSG code is valid, false otherwise
  inline bool valid_prefix() const { return str_.find("EPSG:") == 0; }
};

}  // namespace detail

/**
 * @brief Create a WGS84<-->UTM projection for the given UTM zone and hemisphere
 *
 * @tparam Coordinate the coordinate type
 * @tparam Coordinate::value_type the coordinate value type
 * @param zone the UTM zone
 * @param hemisphere the UTM hemisphere
 * @param dir if FORWARD, create a projection from UTM to WGS84, otherwise create a projection
 * from WGS84 to UTM
 * @return a unique_ptr to a projection object implementing the requested transformation
 */
template <typename Coordinate, typename T = typename Coordinate::value_type>
projection<Coordinate>* make_utm_projection(int zone,
                                            hemisphere hemisphere,
                                            direction dir = direction::FORWARD)
{
  projection_parameters<T> tmerc_proj_params{
    make_ellipsoid_wgs84<T>(), zone, hemisphere, T{0}, T{0}};

  std::vector<cuproj::operation_type> h_utm_pipeline{
    operation_type::AXIS_SWAP,
    operation_type::DEGREES_TO_RADIANS,
    operation_type::CLAMP_ANGULAR_COORDINATES,
    operation_type::TRANSVERSE_MERCATOR,
    operation_type::OFFSET_SCALE_CARTESIAN_COORDINATES};

  return new projection<Coordinate>(h_utm_pipeline, tmerc_proj_params, dir);
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
 * @return a unique_ptr to a projection object implementing the transformation between the two EPSG
 * codes
 */
template <typename Coordinate>
cuproj::projection<Coordinate>* make_projection(detail::epsg_code const& src_epsg,
                                                detail::epsg_code const& dst_epsg)
{
  detail::epsg_code src_code{src_epsg};
  detail::epsg_code dst_code{dst_epsg};

  auto dir = [&]() {
    if (src_code.is_wgs_84()) {
      return direction::FORWARD;
    } else {
      std::swap(src_code, dst_code);
      CUPROJ_EXPECTS(src_code.is_wgs_84(), "Unsupported CRS combination.");
      return direction::INVERSE;
    }
  }();

  auto [dst_zone, dst_hemisphere] = dst_code.to_utm_zone();
  return make_utm_projection<Coordinate>(dst_zone, dst_hemisphere, dir);
}

/**
 * @brief Create a projection object from EPSG codes as "EPSG:XXXX" strings
 *
 * @throw cuproj::logic_error if the EPSG codes describe a transformation that is not supported
 *
 * @note Currently only WGS84 to UTM and UTM to WGS84 are supported, so one of the EPSG codes must
 * be "EPSG:4326" (WGS84) and the other must be a UTM EPSG code.
 *
 * @note Auth strings are case insensitive
 *
 * @tparam Coordinate the coordinate type
 * @param src_epsg the source EPSG code
 * @param dst_epsg the destination EPSG code
 * @return a pointer to a projection object implementing the transformation between the two EPSG
 * codes
 */
template <typename Coordinate>
cuproj::projection<Coordinate>* make_projection(std::string const& src_epsg,
                                                std::string const& dst_epsg)
{
  detail::epsg_code src_code{src_epsg};
  detail::epsg_code dst_code{dst_epsg};

  return make_projection<Coordinate>(src_code, dst_code);
}

/**
 * @brief Create a projection object from integer EPSG codes
 *
 * @throw cuproj::logic_error if the EPSG codes describe a transformation that is not supported
 *
 * @note Currently only WGS84 to UTM and UTM to WGS84 are supported, so one of the EPSG codes must
 * be 4326 (WGS84) and the other must be a UTM EPSG code.
 *
 * @tparam Coordinate the coordinate type
 * @param src_epsg the source EPSG code
 * @param dst_epsg the destination EPSG code
 * @return a pointer to a projection object implementing the transformation between the two EPSG
 * codes
 */
template <typename Coordinate>
cuproj::projection<Coordinate>* make_projection(int src_epsg, int const& dst_epsg)
{
  detail::epsg_code src_code{src_epsg};
  detail::epsg_code dst_code{dst_epsg};

  return make_projection<Coordinate>(detail::epsg_code(src_epsg), detail::epsg_code(dst_epsg));
}

/**
 * @} // end of doxygen group
 */

}  // namespace cuproj
