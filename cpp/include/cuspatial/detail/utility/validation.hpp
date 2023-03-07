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
 * @brief Macro for validating the data array sizes for a polygon.
 *
 * Raises an exception if any of the following are false:
 *  - There are either zero or at least 2 polygon offsets (number of polygons + 1).
 *  - There are either zero or at least 2 polygon ring offsets (number of rings + 1).
 *  - There is at least one ring offset per polygon offset.
 *  - There are at least 3 vertices per ring offset.
 *
 * Polygons follow [GeoArrow data layout][1]. Offsets arrays (polygons and rings) have one more
 * element than the number of items in the array. The last offset is always the sum of the previous
 * offset and the size of that element. For example the last value in the ring offsets array is the
 * last ring offset plus the number of rings in the last polygon. See
 * [Arrow Variable-Size Binary layout](2).
 *
 * Rings are not assumed to be open or closed (closed means the first and last vertices of
 * each ring are equal). Rings must have at least 3 vertices.
 *
 * [1]: https://github.com/geoarrow/geoarrow/blob/main/format.md
 * [2]: https://arrow.apache.org/docs/format/Columnar.html#variable-size-binary-layout
 */
#define CUSPATIAL_EXPECTS_VALID_POLYGON_SIZES(                                    \
  num_poly_points, num_poly_offsets, num_poly_ring_offsets)                       \
  CUSPATIAL_EXPECTS(num_poly_offsets == 0 || num_poly_offsets > 1,                \
                    "Polygon offsets must contain at least two (2) values");      \
  CUSPATIAL_EXPECTS(num_poly_ring_offsets == 0 || num_poly_ring_offsets > 1,      \
                    "Polygon ring offsets must contain at least two (2) values"); \
  CUSPATIAL_EXPECTS(num_poly_ring_offsets >= num_poly_offsets,                    \
                    "Each polygon must have at least one ring");                  \
  CUSPATIAL_EXPECTS(num_poly_points >= 3 * (num_poly_ring_offsets - 1),           \
                    "Each ring must have at least three (3) vertices");
