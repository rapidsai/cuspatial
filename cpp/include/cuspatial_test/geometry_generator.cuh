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

#include <cuspatial_test/vector_factories.cuh>

#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/sequence.h>
#include <thrust/tabulate.h>

namespace cuspatial {
namespace test {

constexpr double PI = 3.14159265358979323846;

template <typename T>
struct multipolygon_generator_parameter {
  using element_t = T;

  std::size_t num_multipolygons;
  std::size_t num_polygons_per_multipolygon;
  std::size_t num_holes_per_polygon;
  std::size_t num_sides_per_ring;
  vec_2d<T> centroid;
  T radius;

  std::size_t num_polygons() { return num_multipolygons * num_polygons_per_multipolygon; }
  std::size_t num_rings() { return num_polygons() * num_rings_per_polygon(); }
  std::size_t num_coords() { return num_rings() * num_vertices_per_ring(); }
  std::size_t num_vertices_per_ring() { return num_sides_per_ring + 1; }
  std::size_t num_rings_per_polygon() { return num_holes_per_polygon + 1; }
};

template <typename T, typename VecIterator>
VecIterator generate_ring(std::size_t num_sides,
                          vec_2d<T> centroid,
                          T radius,
                          VecIterator points_it,
                          rmm::cuda_stream_view stream)
{
  std::size_t num_vertices = num_sides + 1;
  std::cout << "generate_ring: \n num_vertices: " << num_vertices << std::endl;
  thrust::tabulate(
    rmm::exec_policy(stream),
    points_it,
    points_it + num_vertices,
    [num_sides, centroid, radius] __device__(int32_t i) {
      // Overrides last coordinate to make sure ring is closed.
      if (i == num_sides) return vec_2d<T>{centroid.x + radius, centroid.y};

      T angle = 2.0 * PI * i / num_sides;
      return vec_2d<T>{centroid.x + radius * cos(angle), centroid.y + radius * sin(angle)};
    });

  std::cout << "Done generate_ring: \n" << std::endl;

  return points_it + num_vertices;
}

template <typename T, typename VecIterator>
VecIterator generate_polygon(std::size_t num_sides_per_ring,
                             std::size_t num_holes,
                             vec_2d<T> centroid,
                             T radius,
                             VecIterator points_it,
                             rmm::cuda_stream_view stream)
{
  std::cout << "generate_polygon: \n" << std::endl;
  // make shell
  points_it = generate_ring(num_sides_per_ring, centroid, radius, points_it, stream);

  // Align hole centroid to the horizontal axis of the polygon centroid
  T hole_radius           = radius / num_holes;
  T max_hole_displacement = radius - hole_radius;

  // make hole
  for (std::size_t i = 0; i < num_holes; ++i) {
    T displacement_x        = -max_hole_displacement + i * hole_radius * 2;
    T displacement_y        = 0.0;
    vec_2d<T> hole_centroid = centroid + vec_2d<T>{displacement_x, displacement_y};
    points_it = generate_ring(num_sides_per_ring, hole_centroid, hole_radius, points_it, stream);
  }

  return points_it;
}

template <typename T, typename VecIterator>
VecIterator generate_multipolygon(std::size_t num_polygons,
                                  std::size_t num_sides_per_ring,
                                  std::size_t num_holes,
                                  vec_2d<T> origin,
                                  T radius,
                                  VecIterator points_it,
                                  rmm::cuda_stream_view stream)
{
  std::cout << "generate_multipolygon: \n" << std::endl;
  for (std::size_t i = 0; i < num_polygons; ++i) {
    vec_2d<T> centroid = origin + vec_2d<T>{i * radius * 3, 0};
    points_it =
      generate_polygon(num_sides_per_ring, num_holes, centroid, radius, points_it, stream);
  }

  return points_it;
}

/**
 * @brief Helper to generate multipolygon arrays used for benchmarks.
 *
 * @tparam T The floating point type for the coordinates
 * @param stream The CUDA stream to use for device memory operations and kernel launches
 * @return A tuple of x and y coordinates of points and offsets to which the first point
 * of each linestring starts.
 */
template <typename T>
auto generate_multipolygon_array(multipolygon_generator_parameter<T> params,
                                 rmm::cuda_stream_view stream)
{
  std::cout << "Num_coords: " << params.num_coords() << std::endl;
  rmm::device_uvector<vec_2d<T>> coordinates(params.num_coords(), stream);
  auto points_it = coordinates.begin();

  for (std::size_t i = 0; i < params.num_multipolygons; ++i) {
    points_it = generate_multipolygon(params.num_polygons_per_multipolygon,
                                      params.num_sides_per_ring,
                                      params.num_holes_per_polygon,
                                      params.centroid,
                                      params.radius,
                                      points_it,
                                      stream);
  }

  rmm::device_uvector<std::size_t> ring_offsets(params.num_rings() + 1, stream);
  rmm::device_uvector<std::size_t> part_offsets(params.num_polygons() + 1, stream);
  rmm::device_uvector<std::size_t> geometry_offsets(params.num_multipolygons + 1, stream);

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

  std::cout << "num_polygons_per_multipolygon:" << params.num_polygons_per_multipolygon
            << std::endl;
  thrust::sequence(rmm::exec_policy(stream),
                   geometry_offsets.begin(),
                   geometry_offsets.end(),
                   std::size_t{0},
                   params.num_polygons_per_multipolygon);

  return make_multipolygon_array_from_uvector<std::size_t, vec_2d<T>>(std::move(geometry_offsets),
                                                                      std::move(part_offsets),
                                                                      std::move(ring_offsets),
                                                                      std::move(coordinates));
}

}  // namespace test
}  // namespace cuspatial
