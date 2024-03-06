/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cuspatial/distance.cuh>
#include <cuspatial/geometry/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <nvbench/nvbench.cuh>

using namespace cuspatial;
using namespace cuspatial::test;

template <typename T>
void pairwise_point_polygon_distance_benchmark(nvbench::state& state, nvbench::type_list<T>)
{
  // TODO: to be replaced by nvbench fixture once it's ready
  cuspatial::rmm_pool_raii rmm_pool;
  rmm::cuda_stream_view stream{rmm::cuda_stream_default};

  auto const num_pairs{static_cast<std::size_t>(state.get_int64("num_pairs"))};

  auto const num_polygons_per_multipolygon{
    static_cast<std::size_t>(state.get_int64("num_polygons_per_multipolygon"))};
  auto const num_holes_per_polygon{
    static_cast<std::size_t>(state.get_int64("num_holes_per_polygon"))};
  auto const num_edges_per_ring{static_cast<std::size_t>(state.get_int64("num_edges_per_ring"))};

  auto const num_points_per_multipoint{
    static_cast<std::size_t>(state.get_int64("num_points_per_multipoint"))};

  auto mpoly_generator_param = multipolygon_generator_parameter<T>{
    num_pairs, num_polygons_per_multipolygon, num_holes_per_polygon, num_edges_per_ring};

  auto mpoint_generator_param = multipoint_generator_parameter<T>{
    num_pairs, num_points_per_multipoint, vec_2d<T>{-1, -1}, vec_2d<T>{0, 0}};

  auto multipolygons = generate_multipolygon_array<T>(mpoly_generator_param, stream);
  auto multipoints   = generate_multipoint_array<T>(mpoint_generator_param, stream);

  auto distances = rmm::device_vector<T>(num_pairs);
  auto out_it    = distances.begin();

  auto mpoly_view  = multipolygons.range();
  auto mpoint_view = multipoints.range();

  state.add_element_count(num_pairs, "NumPairs");
  state.add_element_count(mpoly_generator_param.num_polygons(), "NumPolygons");
  state.add_element_count(mpoly_generator_param.num_rings(), "NumRings");
  state.add_element_count(mpoly_generator_param.num_coords(), "NumPoints (in mpoly)");
  state.add_element_count(static_cast<std::size_t>(mpoly_generator_param.num_coords() *
                                                   mpoly_generator_param.num_rings() *
                                                   mpoly_generator_param.num_polygons()),
                          "Multipolygon Complexity");
  state.add_element_count(mpoint_generator_param.num_points(), "NumPoints (in multipoints)");

  state.add_global_memory_reads<T>(
    mpoly_generator_param.num_coords() + mpoint_generator_param.num_points(),
    "CoordinatesReadSize");
  state.add_global_memory_reads<std::size_t>(
    (mpoly_generator_param.num_rings() + 1) + (mpoly_generator_param.num_polygons() + 1) +
      (mpoly_generator_param.num_multipolygons + 1) + (mpoint_generator_param.num_multipoints + 1),
    "OffsetsDataSize");

  state.add_global_memory_writes<T>(num_pairs);

  state.exec(nvbench::exec_tag::sync,
             [&mpoly_view, &mpoint_view, &out_it, &stream](nvbench::launch& launch) {
               pairwise_point_polygon_distance(mpoint_view, mpoly_view, out_it, stream);
             });
}

using floating_point_types = nvbench::type_list<float, double>;

// Benchmark scalability with simple multipolygon (3 sides, 0 hole, 1 poly)
NVBENCH_BENCH_TYPES(pairwise_point_polygon_distance_benchmark,
                    NVBENCH_TYPE_AXES(floating_point_types))
  .set_type_axes_names({"CoordsType"})
  .add_int64_axis("num_pairs", {1, 1'00, 10'000, 1'000'000, 100'000'000})
  .add_int64_axis("num_polygons_per_multipolygon", {1})
  .add_int64_axis("num_holes_per_polygon", {0})
  .add_int64_axis("num_edges_per_ring", {3})
  .add_int64_axis("num_points_per_multipoint", {1})
  .set_name("point_polygon_distance_benchmark_simple_polygon");

// Benchmark scalability with complex multipolygon (100 sides, 10 holes, 3 polys)
NVBENCH_BENCH_TYPES(pairwise_point_polygon_distance_benchmark,
                    NVBENCH_TYPE_AXES(floating_point_types))
  .set_type_axes_names({"CoordsType"})
  .add_int64_axis("num_pairs", {1'000, 10'000, 100'000, 1'000'000})
  .add_int64_axis("num_polygons_per_multipolygon", {2})
  .add_int64_axis("num_holes_per_polygon", {3})
  .add_int64_axis("num_edges_per_ring", {50})
  .add_int64_axis("num_points_per_multipoint", {1})
  .set_name("point_polygon_distance_benchmark_complex_polygon");

// // Benchmark impact of rings (100K pairs, 1 polygon, 3 sides)
NVBENCH_BENCH_TYPES(pairwise_point_polygon_distance_benchmark,
                    NVBENCH_TYPE_AXES(floating_point_types))
  .set_type_axes_names({"CoordsType"})
  .add_int64_axis("num_pairs", {10'000})
  .add_int64_axis("num_polygons_per_multipolygon", {1})
  .add_int64_axis("num_holes_per_polygon", {0, 10, 100, 1000})
  .add_int64_axis("num_edges_per_ring", {3})
  .add_int64_axis("num_points_per_multipoint", {1})
  .set_name("point_polygon_distance_benchmark_ring_numbers");

// Benchmark impact of rings (1M pairs, 1 polygon, 0 holes, 3 sides)
NVBENCH_BENCH_TYPES(pairwise_point_polygon_distance_benchmark,
                    NVBENCH_TYPE_AXES(floating_point_types))
  .set_type_axes_names({"CoordsType"})
  .add_int64_axis("num_pairs", {100})
  .add_int64_axis("num_polygons_per_multipolygon", {1})
  .add_int64_axis("num_holes_per_polygon", {0})
  .add_int64_axis("num_edges_per_ring", {3})
  .add_int64_axis("num_points_per_multipoint", {50, 5'00, 5'000, 50'000, 500'000})
  .set_name("point_polygon_distance_benchmark_points_in_multipoint");
