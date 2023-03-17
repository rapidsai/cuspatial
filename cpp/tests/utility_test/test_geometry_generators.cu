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
#include <cuspatial_test/geometry_generator.cuh>
#include <cuspatial_test/vector_equality.hpp>

#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>

using namespace cuspatial;
using namespace cuspatial::test;

template <typename T>
struct GeometryFactoryTest : public BaseFixture {
  rmm::cuda_stream_view stream() { return rmm::cuda_stream_default; }

  template <typename MultipolygonArray>
  void run(multipolygon_generator_parameter<T> params, MultipolygonArray expected)
  {
    auto got = generate_multipolygon_array(params, stream());

    std::cout << "done generate multipolygon array" << std::endl;

    stream().synchronize();

    std::cout << "done sync" << std::endl;

    auto [got_geometry_offsets, got_part_offsets, got_ring_offsets, got_coordinates] =
      got.to_host();

    std::cout << "done sync to host" << std::endl;

    auto [expected_geometry_offsets,
          expected_part_offsets,
          expected_ring_offsets,
          expected_coordinates] = expected.to_host();

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_geometry_offsets, got_geometry_offsets);
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_part_offsets, got_part_offsets);
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_ring_offsets, got_ring_offsets);
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_coordinates, got_coordinates);
  }
};

using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_CASE(GeometryFactoryTest, TestTypes);

TYPED_TEST(GeometryFactoryTest, multipolygonarray_basic)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  auto params   = multipolygon_generator_parameter<T>{1, 1, 0, 3, {0.0, 0.0}, 1.0};
  auto expected = make_multipolygon_array(
    {0, 1},
    {0, 1},
    {0, 4},
    {P{1.0, 0.0}, P{-0.5, 0.8660254037844386}, P{-0.5, -0.8660254037844386}, P{1.0, 0.0}});

  CUSPATIAL_RUN_TEST(this->run, params, std::move(expected));
}
