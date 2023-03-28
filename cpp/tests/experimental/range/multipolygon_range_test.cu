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

#include <cuspatial_test/base_fixture.hpp>
#include <cuspatial_test/vector_equality.hpp>
#include <cuspatial_test/vector_factories.cuh>

#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <initializer_list>

using namespace cuspatial;
using namespace cuspatial::test;

struct polygon_ref_exact_equal_comparator {
  template <typename PolygonRef>
  bool operator()(PolygonRef first, PolygonRef second)
  {
    if (first.num_rings() != second.num_rings()) return false;

    bool equals = true;
    for (std::size_t i = 0; equals && i < first.num_rings(); ++i) {
      auto ring_first  = first[i];
      auto ring_second = second[i];
      if (ring_first.size() != ring_second.size()) equals = false;
      for (std::size_t j = 0; j < ring_first.size(); ++j)
        equals = equals && ring_first[j] == ring_second[j];
    }
    return equals;
  }
};

template <typename T>
struct MultipolygonRangeTest : public BaseFixture {
  void run_polygon_iterator_test_single(std::initializer_list<std::size_t> geometry_offset,
                                        std::initializer_list<std::size_t> part_offset,
                                        std::initializer_list<std::size_t> ring_offset,
                                        std::initializer_list<vec_2d<T>> coordinates,
                                        std::initializer_list<std::size_t> expected_geometry_offset,
                                        std::initializer_list<std::size_t> expected_part_offset,
                                        std::initializer_list<std::size_t> expected_ring_offset,
                                        std::initializer_list<vec_2d<T>> expected_coordinates)
  {
    auto multipolygon_array =
      make_multipolygon_array(geometry_offset, part_offset, ring_offset, coordinates);

    auto expected_multipolygon_array = make_multipolygon_array(
      expected_geometry_offset, expected_part_offset, expected_ring_offset, expected_coordinates);

    auto rng          = multipolygon_array.range();
    auto expected_rng = expected_multipolygon_array.range();

    auto equal = rmm::device_uvector(expected_multipolygon_array.size(), stream());

    thrust::transform(multipolygon_array.polygon_begin(),
                      multipolygon_array.polygon_end(),
                      expected_multipolygon_array.begin(),
                      equal.begin(),
                      polygon_ref_exact_equal_comparator{});
  }
};
