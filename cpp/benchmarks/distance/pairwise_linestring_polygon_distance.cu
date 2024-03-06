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

#include <rmm/cuda_stream_view.hpp>

#include <nvbench/nvbench.cuh>

using namespace cuspatial;

template <typename T>
void pairwise_linestring_polygon_distance_benchmark(nvbench::state& state, nvbench::type_list<T>)
{
  // TODO: to be replaced by nvbench fixture once it's ready
  cuspatial::rmm_pool_raii rmm_pool;
  rmm::cuda_stream_view stream = rmm::cuda_stream_default;

  auto const num_pairs{static_cast<std::size_t>(state.get_int64("NumPairs"))};
  auto const num_linestrings_per_multilinestring{
    static_cast<std::size_t>(state.get_int64("NumLineStringPerMultiLineString"))};
  auto const num_segments_per_linestring{
    static_cast<std::size_t>(state.get_int64("NumSegmentsPerLineString"))};

  auto const num_polygon_per_multipolygon{
    static_cast<std::size_t>(state.get_int64("NumPolygonPerMultiPolygon"))};
  auto const num_ring_per_polygon{static_cast<std::size_t>(state.get_int64("NumRingsPerPolygon"))};
  auto const num_points_per_ring{static_cast<std::size_t>(state.get_int64("NumPointsPerRing"))};

  auto params1 = test::multilinestring_generator_parameter<T>{
    num_pairs, num_linestrings_per_multilinestring, num_segments_per_linestring, 1.0, {0., 0.}};
  auto params2 = test::multipolygon_generator_parameter<T>{num_pairs,
                                                           num_polygon_per_multipolygon,
                                                           num_ring_per_polygon - 1,
                                                           num_points_per_ring - 1,
                                                           {10000, 10000},
                                                           1};

  auto lines = generate_multilinestring_array(params1, stream);
  auto polys = generate_multipolygon_array(params2, stream);

  auto lines_range = lines.range();
  auto poly_range  = polys.range();

  auto output = rmm::device_uvector<T>(num_pairs, stream);
  auto out_it = output.begin();

  auto const total_points = lines_range.num_points() + poly_range.num_points();

  state.add_element_count(num_pairs, "NumPairs");
  state.add_element_count(total_points, "NumPoints");
  state.add_global_memory_reads<T>(total_points * 2, "CoordinatesDataSize");
  state.add_global_memory_reads<int32_t>(params1.num_multilinestrings + params1.num_linestrings() +
                                           params2.num_multipolygons + params2.num_polygons() +
                                           params2.num_rings() + 5,
                                         "OffsetsDataSize");
  state.add_global_memory_writes<T>(num_pairs);

  state.exec(nvbench::exec_tag::sync,
             [&lines_range, &poly_range, &out_it](nvbench::launch& launch) {
               pairwise_linestring_polygon_distance(lines_range, poly_range, out_it);
             });
}

using floating_point_types = nvbench::type_list<float, double>;
NVBENCH_BENCH_TYPES(pairwise_linestring_polygon_distance_benchmark,
                    NVBENCH_TYPE_AXES(floating_point_types))
  .set_type_axes_names({"CoordsType"})
  .add_int64_axis("NumPairs", {100'00})
  .add_int64_axis("NumLineStringPerMultiLineString", {10, 100, 1'000})
  .add_int64_axis("NumSegmentsPerLineString", {100})
  .add_int64_axis("NumPolygonPerMultiPolygon", {100})
  .add_int64_axis("NumRingsPerPolygon", {10})
  .add_int64_axis("NumPointsPerRing", {100});
