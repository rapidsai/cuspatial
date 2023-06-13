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

#include <cuspatial_test/random.cuh>
#include <cuspatial_test/vector_factories.cuh>

#include <cuspatial/cuda_utils.hpp>
#include <cuspatial/detail/utility/zero_data.cuh>
#include <cuspatial/error.hpp>
#include <cuspatial/geometry/vec_2d.hpp>
#include <cuspatial/range/multipolygon_range.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/random/normal_distribution.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/tabulate.h>

#include <optional>

namespace cuspatial {
namespace test {

namespace detail {
template <typename Generator>
rmm::device_uvector<std::size_t> make_offsets(Generator gen,
                                              std::size_t size,
                                              rmm::cuda_stream_view stream)
{
  rmm::device_uvector<std::size_t> offsets(size, stream);

  if (gen.is_random()) {
    zero_data_async(offsets.begin(), offsets.end(), stream);
    thrust::tabulate(rmm::exec_policy(stream), thrust::next(offsets.begin()), offsets.end(), gen);
    thrust::inclusive_scan(rmm::exec_policy(stream),
                           thrust::next(offsets.begin()),
                           offsets.end(),
                           thrust::next(offsets.begin()));
  } else {
    thrust::sequence(rmm::exec_policy(stream),
                     offsets.begin(),
                     offsets.end(),
                     std::size_t{0},
                     static_cast<std::size_t>(gen.mean()));
  }

  return offsets;
}

}  // namespace detail

/**
 * @brief Struct to store the parameters of the multipolygon array generator
 *
 * @tparam T Type of the coordinates
 */
template <typename T>
struct multipolygon_generator_parameter {
  using element_t = T;

  std::size_t num_multipolygons;
  std::size_t num_polygons_per_multipolygon;
  std::size_t num_holes_per_polygon;
  std::size_t num_edges_per_ring;
  vec_2d<T> centroid;
  T radius;

  CUSPATIAL_HOST_DEVICE std::size_t num_polygons()
  {
    return num_multipolygons * num_polygons_per_multipolygon;
  }
  CUSPATIAL_HOST_DEVICE std::size_t num_rings() { return num_polygons() * num_rings_per_polygon(); }
  CUSPATIAL_HOST_DEVICE std::size_t num_coords() { return num_rings() * num_vertices_per_ring(); }
  CUSPATIAL_HOST_DEVICE std::size_t num_vertices_per_ring() { return num_edges_per_ring + 1; }
  CUSPATIAL_HOST_DEVICE std::size_t num_rings_per_polygon() { return num_holes_per_polygon + 1; }
  CUSPATIAL_HOST_DEVICE T hole_radius() { return radius / (num_holes_per_polygon + 1); }
};

/**
 * @brief Generate coordinates for the ring based on the local index of the point.
 *
 * The ring is generated by walking a point around a centroid with a fixed radius.
 * Each step has equal angles.
 *
 * @tparam T Type of coordinate
 * @param point_local_idx Local index of the point
 * @param num_edges Number of sides of the polygon
 * @param centroid Centroid of the ring
 * @param radius Radius of the ring
 * @return Coordinate of the point
 */
template <typename T>
vec_2d<T> __device__ generate_ring_coordinate(std::size_t point_local_idx,
                                              std::size_t num_edges,
                                              vec_2d<T> centroid,
                                              T radius)
{
  // Overrides last coordinate to make sure ring is closed.
  if (point_local_idx == num_edges) return vec_2d<T>{centroid.x + radius, centroid.y};

  T angle = (2.0 * M_PI * point_local_idx) / num_edges;

  return vec_2d<T>{centroid.x + radius * cos(angle), centroid.y + radius * sin(angle)};
}

/**
 * @brief Apply displacement to the centroid of a polygon.
 *
 * The `i`th polygon's centroid is displaced by (3*radius*i, 0). This makes sure
 * polygons within a multipolygon does not overlap.
 *
 * @tparam T Type of the coordinates
 * @param centroid The first centroid of the polygons
 * @param part_local_idx Local index of the polygon
 * @param radius Radius of each polygon
 * @return Displaced centroid
 */
template <typename T>
vec_2d<T> __device__ polygon_centroid_displacement(vec_2d<T> centroid,
                                                   std::size_t part_local_idx,
                                                   T radius)
{
  return centroid + vec_2d<T>{part_local_idx * radius * T{3.0}, T{0.0}};
}

/**
 * @brief Given a ring centroid, displace it based on its ring index.
 *
 * A Polygon contains at least 1 shell. It may contain 0 or more holes.
 * The shell is the leading ring of the polygon (index 0). All holes' centroid
 * has the same y value as the shell's centroid. Holes are aligned from left
 * to right on the center axis, with no overlapping areas. It may look like:
 *
 *            ******
 *        **          **
 *      *                *
 *    *                    *
 *   *                      *
 *  @@@ @@@ @@@ @@@          *
 * @   @   @   @   @          *
 * @   @   @   @   @          *
 * @   @   @   @   @          *
 * *@@@ @@@ @@@ @@@           *
 *  *                        *
 *   *                      *
 *    *                    *
 *      *                *
 *        **          **
 *            ******
 *
 *
 * @tparam T Type of the coordinates
 * @param centroid The center of the polygon
 * @param ring_local_idx Local index of the ring
 * @param radius Radius of the polygon
 * @param hole_radius Radius of each hole
 * @return Centroid of the ring
 */
template <typename T>
vec_2d<T> __device__
ring_centroid_displacement(vec_2d<T> centroid, std::size_t ring_local_idx, T radius, T hole_radius)
{
  // This is a shell
  if (ring_local_idx == 0) { return centroid; }

  // This is a hole
  ring_local_idx -= 1;  // offset hole indices to be 0-based
  T max_hole_displacement = radius - hole_radius;
  T displacement_x        = -max_hole_displacement + ring_local_idx * hole_radius * 2;
  T displacement_y        = 0.0;
  return centroid + vec_2d<T>{displacement_x, displacement_y};
}

/**
 * @brief Kernel to generate coordinates for multipolygon arrays.
 *
 * @pre This kernel requires that the three offset arrays (geometry, part, ring) has been prefilled
 * with the correct offsets.
 *
 * @tparam T Type of the coordinate
 * @tparam MultipolygonRange A specialization of `multipolygon_range`
 * @param multipolygons The range of multipolygons
 * @param params Parameters to generate the mulitpolygons
 */
template <typename T, typename MultipolygonRange>
void __global__ generate_multipolygon_array_coordinates(MultipolygonRange multipolygons,
                                                        multipolygon_generator_parameter<T> params)
{
  for (auto idx = threadIdx.x + blockIdx.x * blockDim.x; idx < multipolygons.num_points();
       idx += gridDim.x * blockDim.x) {
    auto ring_idx     = multipolygons.ring_idx_from_point_idx(idx);
    auto part_idx     = multipolygons.part_idx_from_ring_idx(ring_idx);
    auto geometry_idx = multipolygons.geometry_idx_from_part_idx(part_idx);

    auto point_local_idx = idx - params.num_vertices_per_ring() * ring_idx;
    auto ring_local_idx  = ring_idx - params.num_rings_per_polygon() * part_idx;
    auto part_local_idx  = part_idx - params.num_polygons_per_multipolygon * geometry_idx;

    auto centroid = ring_centroid_displacement(
      polygon_centroid_displacement(params.centroid, part_local_idx, params.radius),
      ring_local_idx,
      params.radius,
      params.hole_radius());

    if (ring_local_idx == 0)  // Generate coordinate for shell
      multipolygons.point_begin()[idx] = generate_ring_coordinate(
        point_local_idx, params.num_edges_per_ring, centroid, params.radius);
    else  // Generate coordinate for holes
      multipolygons.point_begin()[idx] = generate_ring_coordinate(
        point_local_idx, params.num_edges_per_ring, centroid, params.hole_radius());
  }
}

/**
 * @brief Helper to generate multipolygon arrays used for tests and benchmarks.
 *
 * @tparam T The floating point type for the coordinates
 * @param params The parameters to set for the multipolygon array
 * @param stream The CUDA stream to use for device memory operations and kernel launches
 * @return A cuspatial::test::multipolygon_array object.
 */
template <typename T>
auto generate_multipolygon_array(multipolygon_generator_parameter<T> params,
                                 rmm::cuda_stream_view stream)
{
  rmm::device_uvector<std::size_t> geometry_offsets(params.num_multipolygons + 1, stream);
  rmm::device_uvector<std::size_t> part_offsets(params.num_polygons() + 1, stream);
  rmm::device_uvector<std::size_t> ring_offsets(params.num_rings() + 1, stream);
  rmm::device_uvector<vec_2d<T>> coordinates(params.num_coords(), stream);

  thrust::sequence(rmm::exec_policy(stream),
                   ring_offsets.begin(),
                   ring_offsets.end(),
                   std::size_t{0},
                   params.num_vertices_per_ring());

  thrust::sequence(rmm::exec_policy(stream),
                   part_offsets.begin(),
                   part_offsets.end(),
                   std::size_t{0},
                   params.num_rings_per_polygon());

  thrust::sequence(rmm::exec_policy(stream),
                   geometry_offsets.begin(),
                   geometry_offsets.end(),
                   std::size_t{0},
                   params.num_polygons_per_multipolygon);

  auto multipolygons = multipolygon_range(geometry_offsets.begin(),
                                          geometry_offsets.end(),
                                          part_offsets.begin(),
                                          part_offsets.end(),
                                          ring_offsets.begin(),
                                          ring_offsets.end(),
                                          coordinates.begin(),
                                          coordinates.end());

  auto [tpb, nblocks] = grid_1d(multipolygons.num_points());

  generate_multipolygon_array_coordinates<T><<<nblocks, tpb, 0, stream>>>(multipolygons, params);

  CUSPATIAL_CHECK_CUDA(stream.value());

  return make_multipolygon_array<std::size_t, vec_2d<T>>(std::move(geometry_offsets),
                                                         std::move(part_offsets),
                                                         std::move(ring_offsets),
                                                         std::move(coordinates));
}

/**
 * @brief Struct to store the parameters of the multilinestring generator
 *
 * @tparam T Underlying type of the coordinates
 */
template <typename CoordType>
class multilinestring_normal_distribution_generator_parameter {
 private:
  static int constexpr NUM_LINESTRING_GEN_SEED = 0;
  static int constexpr NUM_SEGMENT_GEN_SEED    = 1;

  std::size_t _num_multilinestrings;
  cuspatial::test::normal_random_variable<double> _num_linestrings_per_multilinestring;
  cuspatial::test::normal_random_variable<double> _num_segments_per_linestring;
  CoordType _segment_length;
  vec_2d<CoordType> _origin;

 public:
  template <typename index_t>
  struct _direction_functor {
    vec_2d<CoordType> __device__ operator()(index_t i)
    {
      return vec_2d<CoordType>{cos(static_cast<CoordType>(i)), sin(static_cast<CoordType>(i))};
    }
  };

  struct _random_walk_functor {
    CoordType segment_length;

    vec_2d<CoordType> __device__ operator()(vec_2d<CoordType> prev, vec_2d<CoordType> rad)
    {
      return prev + segment_length * rad;
    }
  };

  multilinestring_normal_distribution_generator_parameter(
    std::size_t num_multilinestrings,
    cuspatial::test::normal_random_variable<double> num_linestrings_per_multilinestring,
    cuspatial::test::normal_random_variable<double> num_segments_per_linestring,
    CoordType segment_length,
    vec_2d<CoordType> origin)
    : _num_multilinestrings(num_multilinestrings),
      _num_linestrings_per_multilinestring(num_linestrings_per_multilinestring),
      _num_segments_per_linestring(num_segments_per_linestring),
      _segment_length(segment_length),
      _origin(origin)
  {
  }

  std::size_t num_multilinestrings() { return _num_multilinestrings; }
  auto num_linestrings_per_multilinestring() { return _num_linestrings_per_multilinestring; }
  auto num_segments_per_linestring() { return _num_segments_per_linestring; }
  CoordType segment_length() { return _segment_length; }
  vec_2d<CoordType> origin() { return _origin; }

  auto num_linestrings_generator()
  {
    auto lower = std::max(
      std::size_t{1}, static_cast<std::size_t>(_num_linestrings_per_multilinestring.neg_6stddev()));
    auto upper = static_cast<std::size_t>(_num_linestrings_per_multilinestring.plus_6stddev());
    return make_clipped_normal_distribution_value_generator(lower, upper, NUM_LINESTRING_GEN_SEED);
  }

  auto num_points_generator()
  {
    auto lower = std::max(std::size_t{1},
                          static_cast<std::size_t>(_num_segments_per_linestring.neg_6stddev()));
    auto upper = static_cast<std::size_t>(_num_segments_per_linestring.plus_6stddev());
    return make_clipped_normal_distribution_value_generator(lower, upper, NUM_SEGMENT_GEN_SEED);
  }

  auto direction_functor() { return _direction_functor<std::size_t>{}; }
  auto random_walk_functor() { return _random_walk_functor{}; }
};

/**
 * @brief
 *
 * @tparam
 */
template <typename CoordType>
struct multilinestring_fixed_generator_parameter
  : public multilinestring_normal_distribution_generator_parameter<CoordType> {
  multilinestring_fixed_generator_parameter(std::size_t num_multilinestrings,
                                            std::size_t num_linestrings_per_multilinestring,
                                            std::size_t num_segments_per_linestring,
                                            CoordType segment_length,
                                            vec_2d<CoordType> origin)
    : multilinestring_normal_distribution_generator_parameter<CoordType>(
        num_multilinestrings,
        {static_cast<double>(num_linestrings_per_multilinestring), 0.0},
        {static_cast<double>(num_segments_per_linestring), 0.0},
        segment_length,
        origin)
  {
  }
};

/**
 * @brief Helper to generate linestrings used for benchmarks.
 *
 * The generator adopts a walking algorithm. The ith point is computed by
 * walking (cos(i) * segment_length, sin(i) * segment_length) from the `i-1`
 * point. The initial point of the linestring is at `(init_xy, init_xy)`.
 *
 * The number of line segments per linestring is constrolled by
 * `num_segment_per_string`.
 *
 * Since the outreach upper bound of the linestring group is
 * `(init_xy + total_num_segments * segment_length)`, user may control the
 * locality of the linestring group via these five arguments.
 *
 * The locality of the multilinestrings is important to the computation and
 * and carefully designing the parameters can make the multilinestrings intersect/disjoint.
 * which could affect whether the benchmark is testing against best or worst case.
 *
 * @tparam T The floating point type for the coordinates
 * @param params The parameters used to specify the generator
 * @param stream The CUDA stream to use for device memory operations and kernel launches
 * @return The generated multilinestring array
 */
template <typename T>
auto generate_multilinestring_array(
  multilinestring_normal_distribution_generator_parameter<T> params, rmm::cuda_stream_view stream)
{
  auto geometry_offset = detail::make_offsets(
    params.num_linestrings_generator(), params.num_multilinestrings() + 1, stream);
  auto num_linestrings = geometry_offset.element(geometry_offset.size() - 1, stream);
  auto part_offset = detail::make_offsets(params.num_points_generator(), num_linestrings, stream);
  auto num_points  = part_offset.element(part_offset.size() - 1, stream);

  rmm::device_uvector<vec_2d<T>> points(num_points, stream);
  thrust::tabulate(
    rmm::exec_policy(stream), points.begin(), points.end(), params.direction_functor());

  thrust::exclusive_scan(rmm::exec_policy(stream),
                         points.begin(),
                         points.end(),
                         points.begin(),
                         params.origin(),
                         params.random_walk_functor());

  return make_multilinestring_array(
    std::move(geometry_offset), std::move(part_offset), std::move(points));
}

/**
 * @brief Creates a parameter set that configures the multipoint generator
 *
 * The number of point in each multipoint is sampled from a normal distribution.
 *
 * @tparam CoordType The type of coordinate
 */
template <typename CoordType>
class multipoint_normal_distribution_generator_parameter {
 protected:
  std::size_t _num_multipoints;
  cuspatial::test::normal_random_variable<double> _num_points_per_multipoints;
  vec_2d<CoordType> _lower_left;
  vec_2d<CoordType> _upper_right;

 public:
  multipoint_normal_distribution_generator_parameter(
    std::size_t num_multipoints,
    normal_random_variable<double> num_points_per_multipoints,
    vec_2d<CoordType> lower_left,
    vec_2d<CoordType> upper_right)
    : _num_multipoints(num_multipoints),
      _num_points_per_multipoints(num_points_per_multipoints),
      _lower_left(lower_left),
      _upper_right(upper_right)
  {
  }

  bool count_has_variance() { return _num_points_per_multipoints.stddev != 0.0; }

  auto multipoint_count_generator()
  {
    auto lower = std::max(1, static_cast<int>(_num_points_per_multipoints.neg_6stddev()));
    auto upper = static_cast<int>(_num_points_per_multipoints.plus_6stddev());
    return make_clipped_normal_distribution_value_generator(lower, upper);
  }

  auto points_generator()
  {
    auto engine_x = deterministic_engine(0);
    auto engine_y = deterministic_engine(1);

    auto x_dist = make_uniform_dist(_lower_left.x, _upper_right.x);
    auto y_dist = make_uniform_dist(_lower_left.y, _upper_right.y);

    return point_generator(_lower_left, _upper_right, engine_x, engine_y, x_dist, y_dist);
  }

  std::size_t num_multipoints() { return _num_multipoints; }
  auto num_points_per_multipoints() { return _num_points_per_multipoints; }
  vec_2d<CoordType> lower_left() { return _lower_left; }
  vec_2d<CoordType> upper_right() { return _upper_right; }
};

/**
 * @brief Parameters to configure a multipoint generator to generate identical multipoint for each
 * element
 *
 * Idendity function is a special case of normal distribution where deviation is 0.
 *
 * @tparam CoordType The type of underlying coordinates
 */
template <typename CoordType>
class multipoint_fixed_generator_parameter
  : public multipoint_normal_distribution_generator_parameter<CoordType> {
 public:
  multipoint_fixed_generator_parameter(std::size_t num_multipoints,
                                       std::size_t num_points_per_multipoints,
                                       vec_2d<CoordType> lower_left,
                                       vec_2d<CoordType> upper_right)
    : multipoint_normal_distribution_generator_parameter<CoordType>(
        num_multipoints,
        {static_cast<double>(num_points_per_multipoints), 0.0},
        lower_left,
        upper_right)
  {
  }
};

/**
 * @brief Generate a multipoint array, the number of point in each multipoint follows a normal
 * distribution
 *
 * @tparam T The floating point type for the coordinates
 * @param params Parameters to specify for the multipoints
 * @param stream The CUDA stream to use for device memory operations and kernel launches
 * @return a cuspatial::test::multipoint_array object
 */
template <typename CoordType>
auto generate_multipoint_array(multipoint_normal_distribution_generator_parameter<CoordType> params,
                               rmm::cuda_stream_view stream)
{
  auto offsets =
    detail::make_offsets(params.multipoint_count_generator(), params.num_multipoints() + 1, stream);
  auto num_points = offsets.element(offsets.size() - 1, stream);

  rmm::device_uvector<vec_2d<CoordType>> coordinates(num_points, stream);
  thrust::tabulate(
    rmm::exec_policy(stream), coordinates.begin(), coordinates.end(), params.points_generator());

  return make_multipoint_array(std::move(offsets), std::move(coordinates));
}

}  // namespace test
}  // namespace cuspatial
