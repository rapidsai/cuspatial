/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cuspatial/constants.hpp>
#include <cuspatial/traits.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <iterator>

namespace cuspatial {

/**
 * @addtogroup distance
 * @{
 */

/**
 * @brief Compute haversine distances between points in set A to the corresponding points in set B.
 *
 * Computes N haversine distances, where N is `std::distance(a_lonlat_first, a_lonlat_last)`.
 * The distance for each `a_lonlat[i]` and `b_lonlat[i]` point pair is assigned to
 * `distance_first[i]`. `distance_first` must be an iterator to output storage allocated for N
 * distances.
 *
 * Computed distances will have the same units as `radius`.
 *
 * https://en.wikipedia.org/wiki/Haversine_formula
 *
 * @param[in]  a_lonlat_first: beginning of range of (longitude, latitude) locations in set A
 * @param[in]  a_lonlat_last: end of range of (longitude, latitude) locations in set A
 * @param[in]  b_lonlat_first: beginning of range of (longitude, latitude) locations in set B
 * @param[out] distance_first: beginning of output range of haversine distances
 * @param[in]  radius: radius of the sphere on which the points reside. default: 6371.0
 *            (approximate radius of Earth in km)
 * @param[in]  stream: The CUDA stream on which to perform computations and allocate memory.
 *
 * @tparam LonLatItA Iterator to input location set A. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam LonLatItB Iterator to input location set B. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam OutputIt Output iterator. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible and mutable.
 * @tparam T The underlying coordinate type. Must be a floating-point type.
 *
 * @pre All iterators must have the same `Location` type, with  the same underlying floating-point
 * coordinate type (e.g. `cuspatial::vec_2d<float>`).
 *
 * @return Output iterator to the element past the last distance computed.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <class LonLatItA,
          class LonLatItB,
          class OutputIt,
          class T = typename cuspatial::iterator_vec_base_type<LonLatItA>>
OutputIt haversine_distance(LonLatItA a_lonlat_first,
                            LonLatItA a_lonlat_last,
                            LonLatItB b_lonlat_first,
                            OutputIt distance_first,
                            T const radius               = EARTH_RADIUS_KM,
                            rmm::cuda_stream_view stream = rmm::cuda_stream_default);

/**
 * @brief Computes Hausdorff distances for all pairs in a collection of spaces
 *
 * https://en.wikipedia.org/wiki/Hausdorff_distance
 *
 * Example in 1D (this function operates in 2D):
 * ```
 * spaces
 * [0 2 5] [9] [3 7]
 *
 * spaces represented as points per space and concatenation of all points
 * [0 2 5 9 3 7] [3 1 2]
 *
 * note: the following matrices are visually separated to highlight the relationship of a pair of
 * points with the pair of spaces from which it is produced
 *
 * cartesian product of all
 * points by pair of spaces     distance between points
 * +----------+----+-------+    +---------+---+------+
 * : 00 02 05 : 09 : 03 07 :    : 0  2  5 : 9 : 3  7 :
 * : 20 22 25 : 29 : 23 27 :    : 2  0  3 : 7 : 1  5 :
 * : 50 52 55 : 59 : 53 57 :    : 5  3  0 : 4 : 2  2 :
 * +----------+----+-------+    +---------+---+------+
 * : 90 92 95 : 99 : 93 97 :    : 9  7  4 : 0 : 6  2 :
 * +----------+----+-------+    +---------+---+------+
 * : 30 32 35 : 39 : 33 37 :    : 3  1  2 : 6 : 0  4 :
 * : 70 72 75 : 79 : 73 77 :    : 7  5  2 : 2 : 4  0 :
 * +----------+----+-------+    +---------+---+------+

 * minimum distance from
 * every point in one           Hausdorff distance is
 * space to any point in        the maximum of the
 * the other space              minimum distances
 * +----------+----+-------+    +---------+---+------+
 * :  0       :  9 :  3    :    : 0       : 9 : 3    :
 * :     0    :  7 :  1    :    :         :   :      :
 * :        0 :  4 :  2    :    :         :   :      :
 * +----------+----+-------+    +---------+---+------+
 * :        4 :  0 :     2 :    :       4 : 0 :    2 :
 * +----------+----+-------+    +---------+---+------+
 * :     1    :  6 :  0    :    :         : 6 : 0    :
 * :        2 :  2 :     0 :    :       2 :   :      :
 * +----------+----+-------+    +---------+---+------+
 *
 * returned as concatenation of columns
 * [0 2 4 3 0 2 9 6 0]
 * ```
 *
 * @param[in] points_first: xs: beginning of range of (x,y) points
 * @param[in] points_lasts: xs: end of range of (x,y) points
 * @param[in] space_offsets_first: beginning of range of indices to each space.
 * @param[in] space_offsets_first: end of range of indices to each space. Last index is the last
 * @param[in] distance_first: beginning of range of output Hausdorff distance for each pair of
 * spaces
 *
 * @tparam PointIt Iterator to input points. Points must be of a type that is convertible to
 * `cuspatial::vec_2d<T>`. Must meet the requirements of [LegacyRandomAccessIterator][LinkLRAI] and
 * be device-accessible.
 * @tparam OffsetIt Iterator to space offsets. Value type must be integral. Must meet the
 * requirements of [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam OutputIt Output iterator. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible and mutable.
 *
 * @pre All iterators must have the same underlying floating-point value type.
 *
 * @return Output iterator to the element past the last distance computed.
 *
 * @note Hausdorff distances are asymmetrical
 */
template <class PointIt, class OffsetIt, class OutputIt>
OutputIt directed_hausdorff_distance(PointIt points_first,
                                     PointIt points_last,
                                     OffsetIt space_offsets_first,
                                     OffsetIt space_offsets_last,
                                     OutputIt distance_first,
                                     rmm::cuda_stream_view stream = rmm::cuda_stream_default);

/**
 * @brief Compute pairwise (multi)point-to-(multi)point Cartesian distance
 *
 * Computes the cartesian distance between each pair of multipoints.
 *
 * @tparam MultiPointArrayViewA An instance of template type `array_view::multipoint_array`
 * @tparam MultiPointArrayViewB An instance of template type `array_view::multipoint_array`
 *
 * @param multipoints1 Range of first multipoint in each distance pair.
 * @param multipoints2 Range of second multipoint in each distance pair.
 * @return Iterator past the last distance computed
 */
template <class MultiPointArrayViewA, class MultiPointArrayViewB, class OutputIt>
OutputIt pairwise_point_distance(MultiPointArrayViewA multipoints1,
                                 MultiPointArrayViewB multipoints2,
                                 OutputIt distances_first,
                                 rmm::cuda_stream_view stream = rmm::cuda_stream_default);

/**
 * @brief Compute pairwise multipoint to multilinestring distance
 *
 * @tparam MultiPointRange an instance of template type `multipoint_range`
 * @tparam MultiLinestringRange an instance of template type `multilinestring_range`
 * @tparam OutputIt iterator type for output array. Must meet the requirements of [LRAI](LinkLRAI).
 *
 * @param multipoints The range of multipoints, one per computed distance pair
 * @param multilinestrings The range of multilinestrings, one per computed distance pair
 * @param stream The CUDA stream to use for device memory operations and kernel launches.
 * @return Output iterator to the element past the last distance computed.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <class MultiPointRange, class MultiLinestringRange, class OutputIt>
OutputIt pairwise_point_linestring_distance(
  MultiPointRange multipoints,
  MultiLinestringRange multilinestrings,
  OutputIt distances_first,
  rmm::cuda_stream_view stream = rmm::cuda_stream_default);

/**
 * @brief Computes pairwise multipoint to multipolygon distance
 *
 * @tparam MultiPointRange An instance of template type `multipoint_range`
 * @tparam MultiPolygonRange An instance of template type `multipolygon_range`
 * @tparam OutputIt iterator type for output array. Must meet the requirements of [LRAI](LinkLRAI).
 * Must be an iterator to type convertible from floating points.
 *
 * @param multipoints Range of multipoints, one per computed distance pair.
 * @param multipolygons Range of multilinestrings, one per computed distance pair.
 * @param stream The CUDA stream on which to perform computations
 * @return Output Iterator past the last distance computed
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <class MultiPointRange, class MultiPolygonRange, class OutputIt>
OutputIt pairwise_point_polygon_distance(MultiPointRange multipoints,
                                         MultiPolygonRange multipoiygons,
                                         OutputIt distances_first,
                                         rmm::cuda_stream_view stream = rmm::cuda_stream_default);

/**
 * @copybrief cuspatial::pairwise_linestring_distance
 *
 * The shortest distance between two linestrings is defined as the shortest distance
 * between all pairs of segments of the two linestrings. If any of the segments intersect,
 * the distance is 0.
 *
 * @tparam MultiLinestringRange an instance of template type `multilinestring_range`
 * @tparam OutputIt iterator type for output array. Must meet the requirements of [LRAI](LinkLRAI)
 * and be device writable.
 *
 * @param multilinestrings1 Range object of the lhs multilinestring array
 * @param multilinestrings2 Range object of the rhs multilinestring array
 * @param stream The CUDA stream to use for device memory operations and kernel launches.
 * @return Output iterator to the element past the last distance computed.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <class MultiLinestringRange1, class MultiLinestringRange2, class OutputIt>
OutputIt pairwise_linestring_distance(MultiLinestringRange1 multilinestrings1,
                                      MultiLinestringRange2 multilinestrings2,
                                      OutputIt distances_first,
                                      rmm::cuda_stream_view stream = rmm::cuda_stream_default);

/**
 * @brief Computes pairwise multilinestring to multipolygon distance
 *
 * @tparam MultiLinestringRange An instance of template type `multipoint_range`
 * @tparam MultiPolygonRange An instance of template type `multipolygon_range`
 * @tparam OutputIt iterator type for output array. Must meet the requirements of [LRAI](LinkLRAI).
 * Must be an iterator to type convertible from floating points.
 *
 * @param multilinestrings Range of multilinestrings, one per computed distance pair.
 * @param multipolygons Range of multipolygons, one per computed distance pair.
 * @param stream The CUDA stream on which to perform computations
 * @return Output Iterator past the last distance computed
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <class MultiLinestringRange, class MultiPolygonRange, class OutputIt>
OutputIt pairwise_linestring_polygon_distance(
  MultiLinestringRange multilinestrings,
  MultiPolygonRange multipoiygons,
  OutputIt distances_first,
  rmm::cuda_stream_view stream = rmm::cuda_stream_default);

/**
 * @brief Computes pairwise multipolygon to multipolygon distance
 *
 * @tparam MultiPolygonRangeA An instance of template type `multipolygon_range`
 * @tparam MultiPolygonRangeB An instance of template type `multipolygon_range`
 * @tparam OutputIt iterator type for output array. Must meet the requirements of [LRAI](LinkLRAI).
 * Must be an iterator to type convertible from floating points.
 *
 * @param lhs The first multipolygon range to compute distance from
 * @param rhs The second multipolygon range to compute distance to
 * @param stream The CUDA stream on which to perform computations
 * @return Output Iterator past the last distance computed
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <class MultipolygonRangeA, class MultipolygonRangeB, class OutputIt>
OutputIt pairwise_polygon_distance(MultipolygonRangeA lhs,
                                   MultipolygonRangeB rhs,
                                   OutputIt distances_first,
                                   rmm::cuda_stream_view stream = rmm::cuda_stream_default);

/**
 * @} // end of doxygen group
 */

}  // namespace cuspatial

#include <cuspatial/detail/distance/hausdorff.cuh>
#include <cuspatial/detail/distance/haversine.cuh>
#include <cuspatial/detail/distance/linestring_distance.cuh>
#include <cuspatial/detail/distance/linestring_polygon_distance.cuh>
#include <cuspatial/detail/distance/point_distance.cuh>
#include <cuspatial/detail/distance/point_linestring_distance.cuh>
#include <cuspatial/detail/distance/point_polygon_distance.cuh>
#include <cuspatial/detail/distance/polygon_distance.cuh>
