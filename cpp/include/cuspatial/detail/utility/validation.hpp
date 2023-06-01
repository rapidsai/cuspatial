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

#include <cuspatial/error.hpp>

/**
 * @brief Macro for validating the data array sizes for linestrings.
 *
 * Raises an exception if any of the following are false:
 *  - The number of linestring offsets is greater than zero.
 *
 * Linestrings follow [GeoArrow data layout][1]. Offsets arrays have one more element than the
 * number of items in the array. The last offset is always the sum of the previous offset and the
 * size of that element. For example the last value in the linestring offsets array is the
 * last linestring offset plus one. See [Arrow Variable-Size Binary layout](2). Note that an
 * empty list still has one offset: {0}.
 *
 * The following is not explicitly checked in this macro, but is always assumed in cuspatial:
 * 1. LineString can contain zero or two or more vertices.
 *
 * [1]: https://github.com/geoarrow/geoarrow/blob/main/format.md
 * [2]: https://arrow.apache.org/docs/format/Columnar.html#variable-size-binary-layout
 */
#define CUSPATIAL_EXPECTS_VALID_LINESTRING_SIZES(num_linestring_points, num_linestring_offsets) \
  CUSPATIAL_HOST_DEVICE_EXPECTS(num_linestring_offsets > 0,                                     \
                                "Polygon offsets must contain at least one (1) value");

/**
 * @brief Macro for validating the data array sizes for multilinestrings.
 *
 * Raises an exception if any of the following are false:
 *  - The number of multilinestring offsets is greater than zero.
 *  - The number of linestring offsets is greater than zero.
 *
 * Multilinestrings follow [GeoArrow data layout][1]. Offsets arrays have one more element than the
 * number of items in the array. The last offset is always the sum of the previous offset and the
 * size of that element. For example the last value in the linestring offsets array is the
 * last linestring offset plus one. See [Arrow Variable-Size Binary layout](2). Note that an
 * empty list still has one offset: {0}.
 *
 * [1]: https://github.com/geoarrow/geoarrow/blob/main/format.md
 * [2]: https://arrow.apache.org/docs/format/Columnar.html#variable-size-binary-layout
 */
#define CUSPATIAL_EXPECTS_VALID_MULTILINESTRING_SIZES(                                          \
  num_linestring_points, num_multilinestring_offsets, num_linestring_offsets)                   \
  CUSPATIAL_HOST_DEVICE_EXPECTS(num_multilinestring_offsets > 0,                                \
                                "Multilinestring offsets must contain at least one (1) value"); \
  CUSPATIAL_EXPECTS_VALID_LINESTRING_SIZES(num_linestring_points, num_linestring_offsets);

/**
 * @brief Macro for validating the data array sizes for a polygon.
 *
 * Raises an exception if any of the following are false:
 *  - The number of polygon offsets is greater than zero.
 *  - The number of ring offsets is greater than zero.
 *
 * Polygons follow [GeoArrow data layout][1]. Offsets arrays (polygons and rings) have one more
 * element than the number of items in the array. The last offset is always the sum of the previous
 * offset and the size of that element. For example the last value in the ring offsets array is the
 * last ring offset plus the number of rings in the last polygon. See
 * [Arrow Variable-Size Binary layout](2). Note that an empty list still has one offset: {0}.
 *
 * The following are not explicitly checked in this macro, but is always assumed in cuspatial:
 *
 * 1. Polygon can contain zero or more rings. A polygon with zero rings is an empty polygon.
 * 2. Rings are assumed to be closed (closed means the first and last vertices of each ring are
 * equal). Rings can also be empty. Therefore each ring must contain zero or four or more vertices.
 *
 * [1]: https://github.com/geoarrow/geoarrow/blob/main/format.md
 * [2]: https://arrow.apache.org/docs/format/Columnar.html#variable-size-binary-layout
 */
#define CUSPATIAL_EXPECTS_VALID_POLYGON_SIZES(                                          \
  num_poly_points, num_poly_offsets, num_poly_ring_offsets)                             \
  CUSPATIAL_HOST_DEVICE_EXPECTS(num_poly_offsets > 0,                                   \
                                "Polygon offsets must contain at least one (1) value"); \
  CUSPATIAL_HOST_DEVICE_EXPECTS(num_poly_ring_offsets > 0,                              \
                                "Polygon ring offsets must contain at least one (1) value");

/**
 * @brief Macro for validating the data array sizes for a multipolygon.
 *
 * Raises an exception if any of the following are false:
 *  - The number of multipolygon offsets is greater than zero.
 *  - The number of polygon offsets is greater than zero.
 *  - The number of ring offsets is greater than zero.
 *
 * MultiPolygons follow [GeoArrow data layout][1]. Offsets arrays (polygons and rings) have one more
 * element than the number of items in the array. The last offset is always the sum of the previous
 * offset and the size of that element. For example the last value in the ring offsets array is the
 * last ring offset plus the number of rings in the last polygon. See
 * [Arrow Variable-Size Binary layout](2). Note that an empty list still has one offset: {0}.
 *
 * The following are not explicitly checked in this macro, but is always assumed in cuspatial:
 *
 * 1. Polygon can contain zero or more rings. A polygon with zero rings is an empty polygon.
 * 2. Rings are assumed to be closed (closed means the first and last vertices of each ring are
 * equal). Rings can also be empty. Therefore each ring must contain zero or four or more
 * coordinates.
 *
 * [1]: https://github.com/geoarrow/geoarrow/blob/main/format.md
 * [2]: https://arrow.apache.org/docs/format/Columnar.html#variable-size-binary-layout
 */
#define CUSPATIAL_EXPECTS_VALID_MULTIPOLYGON_SIZES(                                          \
  num_poly_points, num_multipoly_offsets, num_poly_offsets, num_poly_ring_offsets)           \
  CUSPATIAL_HOST_DEVICE_EXPECTS(num_multipoly_offsets > 0,                                   \
                                "Multipolygon offsets must contain at least one (1) value"); \
  CUSPATIAL_EXPECTS_VALID_POLYGON_SIZES(num_poly_points, num_poly_offsets, num_poly_ring_offsets);
