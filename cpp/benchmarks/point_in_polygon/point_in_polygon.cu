/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <benchmarks/fixture/rmm_pool_raii.hpp>

#include <cuspatial_test/geometry_generator.cuh>

#include <cuspatial/geometry/vec_2d.hpp>
#include <cuspatial/point_in_polygon.cuh>

#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <nvbench/nvbench.cuh>

#include <memory>
#include <numeric>

using namespace cuspatial;

auto constexpr radius                = 10.0;
auto constexpr num_polygons          = 31ul;
auto constexpr num_rings_per_polygon = 1ul;  // only 1 ring for now

template <typename T>
void point_in_polygon_benchmark(nvbench::state& state, nvbench::type_list<T>)
{
  // TODO: to be replaced by nvbench fixture once it's ready
  cuspatial::rmm_pool_raii rmm_pool;
  rmm::cuda_stream_view stream(rmm::cuda_stream_default);

  auto const minXY = vec_2d<T>{-radius * 2, -radius * 2};
  auto const maxXY = vec_2d<T>{radius * 2, radius * 2};

  auto const num_test_points{state.get_int64("NumTestPoints")},
    num_sides_per_ring{state.get_int64("NumSidesPerRing")};

  auto const num_rings = num_polygons * num_rings_per_polygon;
  auto const num_polygon_points =
    num_rings * (num_sides_per_ring + 1);  // +1 for the overlapping start and end point of the ring

  auto point_gen_param = test::multipoint_generator_parameter<T>{
    static_cast<std::size_t>(num_test_points), 1, minXY, maxXY};
  auto poly_gen_param =
    test::multipolygon_generator_parameter<T>{static_cast<std::size_t>(num_polygons),
                                              1,
                                              0,
                                              static_cast<std::size_t>(num_sides_per_ring),
                                              vec_2d<T>{0, 0},
                                              radius};
  auto test_points   = test::generate_multipoint_array<T>(point_gen_param, stream);
  auto test_polygons = test::generate_multipolygon_array<T>(poly_gen_param, stream);

  auto [_, points]                                             = test_points.release();
  auto [__, part_offset_array, ring_offset_array, poly_coords] = test_polygons.release();

  auto points_range = make_multipoint_range(
    num_test_points, thrust::make_counting_iterator(0), points.size(), points.begin());
  auto polys_range = make_multipolygon_range(num_polygons,
                                             thrust::make_counting_iterator(0),
                                             part_offset_array.size() - 1,
                                             part_offset_array.begin(),
                                             ring_offset_array.size() - 1,
                                             ring_offset_array.begin(),
                                             poly_coords.size(),
                                             poly_coords.begin());

  rmm::device_vector<int32_t> result(num_test_points);

  state.add_element_count(num_polygon_points, "NumPolygonPoints");
  state.add_global_memory_reads<T>(num_test_points * 2, "TotalMemoryReads");
  state.add_global_memory_reads<T>(num_polygon_points);
  state.add_global_memory_reads<int32_t>(num_rings);
  state.add_global_memory_reads<int32_t>(num_polygons);
  state.add_global_memory_writes<int32_t>(num_test_points, "TotalMemoryWrites");

  state.exec(nvbench::exec_tag::sync,
             [points_range, polys_range, &result, stream](nvbench::launch& launch) {
               point_in_polygon(points_range, polys_range, result.begin(), stream);
             });
}

using floating_point_types = nvbench::type_list<float, double>;
NVBENCH_BENCH_TYPES(point_in_polygon_benchmark, NVBENCH_TYPE_AXES(floating_point_types))
  .set_type_axes_names({"CoordsType"})
  .add_int64_axis("NumTestPoints", {1'000, 100'000, 10'000'000})
  .add_int64_axis("NumSidesPerRing", {4, 10, 100});
