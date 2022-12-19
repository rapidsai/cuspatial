/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

using namespace cuspatial;
using namespace cuspatial::detail;
using namespace cuspatial::test;

template <typename T>
struct MultiPointFactoryTest : public BaseFixture {
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(MultiPointFactoryTest, TestTypes);

TYPED_TEST(MultiPointFactoryTest, simple)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  auto multipoints =
    make_multipoints_array({{P{0.0, 0.0}, P{1.0, 0.0}}, {P{2.0, 0.0}, P{2.0, 2.0}}});

  auto [offsets, coords] = multipoints.release();

  auto expected_offsets = make_device_vector<std::size_t>({0, 2, 4});
  auto expected_coords =
    make_device_vector<vec_2d<T>>({P{0.0, 0.0}, P{1.0, 0.0}, P{2.0, 0.0}, P{2.0, 2.0}});

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(offsets, expected_offsets);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(coords, expected_coords);
}

TYPED_TEST(MultiPointFactoryTest, empty)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  auto multipoints = make_multipoints_array<T>({});

  auto [offsets, coords] = multipoints.release();

  auto expected_offsets = make_device_vector<std::size_t>({0});
  auto expected_coords  = make_device_vector<vec_2d<T>>({});

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(offsets, expected_offsets);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(coords, expected_coords);
}

TYPED_TEST(MultiPointFactoryTest, mixed_empty_multipoint)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  auto multipoints = make_multipoints_array({{P{1.0, 0.0}}, {}, {P{2.0, 3.0}, P{4.0, 5.0}}});

  auto [offsets, coords] = multipoints.release();

  auto expected_offsets = make_device_vector<std::size_t>({0, 1, 1, 3});
  auto expected_coords  = make_device_vector<vec_2d<T>>({P{1.0, 0.0}, P{2.0, 3.0}, P{4.0, 5.0}});

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(offsets, expected_offsets);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(coords, expected_coords);
}

TYPED_TEST(MultiPointFactoryTest, mixed_empty_multipoint2)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  auto multipoints = make_multipoints_array({{}, {P{1.0, 0.0}}, {P{2.0, 3.0}, P{4.0, 5.0}}});

  auto [offsets, coords] = multipoints.release();

  auto expected_offsets = make_device_vector<std::size_t>({0, 0, 1, 3});
  auto expected_coords  = make_device_vector<vec_2d<T>>({P{1.0, 0.0}, P{2.0, 3.0}, P{4.0, 5.0}});

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(offsets, expected_offsets);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(coords, expected_coords);
}

TYPED_TEST(MultiPointFactoryTest, mixed_empty_multipoint3)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  auto multipoints = make_multipoints_array({{P{1.0, 0.0}}, {P{2.0, 3.0}, P{4.0, 5.0}}, {}});

  auto [offsets, coords] = multipoints.release();

  auto expected_offsets = make_device_vector<std::size_t>({0, 1, 3, 3});
  auto expected_coords  = make_device_vector<vec_2d<T>>({P{1.0, 0.0}, P{2.0, 3.0}, P{4.0, 5.0}});

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(offsets, expected_offsets);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(coords, expected_coords);
}
