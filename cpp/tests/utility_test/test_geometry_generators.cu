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
#include <cuspatial_test/base_fixture.hpp>
#include <cuspatial_test/geometry_generator.cuh>
#include <cuspatial_test/vector_equality.hpp>

#include <cuspatial/geometry/vec_2d.hpp>

#include <gtest/gtest-param-test.h>

#include <type_traits>

using namespace cuspatial;
using namespace cuspatial::test;

template <typename T>
auto constexpr abs_error(T radius)
{
  return radius * (std::is_same_v<T, float> ? 1e-7 : 1e-15);
};

template <typename T>
struct GeometryFactoryTest : public BaseFixture {
  template <typename MultipolygonArray>
  void run(multipolygon_generator_parameter<T> params,
           MultipolygonArray expected,
           T abs_error = 0.0)
  {
    auto got = generate_multipolygon_array(params, stream());

    auto [got_geometry_offsets, got_part_offsets, got_ring_offsets, got_coordinates] =
      got.to_host();

    auto [expected_geometry_offsets,
          expected_part_offsets,
          expected_ring_offsets,
          expected_coordinates] = expected.to_host();

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_geometry_offsets, got_geometry_offsets);
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_part_offsets, got_part_offsets);
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_ring_offsets, got_ring_offsets);
    if (abs_error == 0.0)
      CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_coordinates, got_coordinates);
    else
      CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_coordinates, got_coordinates, abs_error);
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

TYPED_TEST(GeometryFactoryTest, multipolygonarray_basic2)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  auto params   = multipolygon_generator_parameter<T>{1, 1, 0, 4, {0.0, 0.0}, 1.0};
  auto expected = make_multipolygon_array(
    {0, 1}, {0, 1}, {0, 5}, {P{1.0, 0.0}, P{0.0, 1.0}, P{-1.0, 0.0}, P{0.0, -1.0}, P{1.0, 0.0}});

  CUSPATIAL_RUN_TEST(this->run, params, std::move(expected), abs_error<T>(params.radius));
}

TYPED_TEST(GeometryFactoryTest, multipolygonarray_1poly1hole)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  auto params   = multipolygon_generator_parameter<T>{1, 1, 1, 4, {0.0, 0.0}, 1.0};
  auto expected = make_multipolygon_array({0, 1},
                                          {0, 2},
                                          {0, 5, 10},
                                          {P{1.0, 0.0},
                                           P{0.0, 1.0},
                                           P{-1.0, 0.0},
                                           P{0.0, -1.0},
                                           P{1.0, 0.0},
                                           P{0.0, 0.0},
                                           P{-0.5, 0.5},
                                           P{-1.0, 0.0},
                                           P{-0.5, -0.5},
                                           P{0.0, 0.0}});

  CUSPATIAL_RUN_TEST(this->run, params, std::move(expected), abs_error<T>(params.radius));
}

TYPED_TEST(GeometryFactoryTest, multipolygonarray_1poly2holes)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  auto params   = multipolygon_generator_parameter<T>{1, 1, 2, 4, {0.0, 0.0}, 1.0};
  auto expected = make_multipolygon_array({0, 1},
                                          {0, 3},
                                          {0, 5, 10, 15},
                                          {
                                            P{1.0, 0.0},
                                            P{0.0, 1.0},
                                            P{-1.0, 0.0},
                                            P{0.0, -1.0},
                                            P{1.0, 0.0},
                                            P{-0.3333333333333333, 0.0},
                                            P{-0.6666666666666666, 0.3333333333333333},
                                            P{-1.0, 0.0},
                                            P{-0.6666666666666666, -0.3333333333333333},
                                            P{-0.3333333333333333, 0.0},
                                            P{0.3333333333333333, 0.0},
                                            P{0.0, 0.3333333333333333},
                                            P{-0.3333333333333333, 0.0},
                                            P{0.0, -0.3333333333333333},
                                            P{0.3333333333333333, 0.0},
                                          });

  CUSPATIAL_RUN_TEST(this->run, params, std::move(expected), abs_error<T>(params.radius));
}

TYPED_TEST(GeometryFactoryTest, multipolygonarray_2poly0hole)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  auto params   = multipolygon_generator_parameter<T>{1, 2, 0, 4, {0.0, 0.0}, 1.0};
  auto expected = make_multipolygon_array({0, 2},
                                          {0, 1, 2},
                                          {0, 5, 10},
                                          {
                                            P{1.0, 0.0},
                                            P{0.0, 1.0},
                                            P{-1.0, 0.0},
                                            P{0.0, -1.0},
                                            P{1.0, 0.0},
                                            P{4.0, 0.0},
                                            P{3.0, 1.0},
                                            P{2.0, 0.0},
                                            P{3.0, -1.0},
                                            P{4.0, 0.0},
                                          });

  CUSPATIAL_RUN_TEST(this->run, params, std::move(expected), abs_error<T>(params.radius));
}

TYPED_TEST(GeometryFactoryTest, multipolygonarray_2poly1hole)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  auto params = multipolygon_generator_parameter<T>{1, 2, 1, 4, {0.0, 0.0}, 1.0};
  auto expected =
    make_multipolygon_array({0, 2},
                            {0, 2, 4},
                            {0, 5, 10, 15, 20},
                            {
                              P{1.0, 0.0}, P{0.0, 1.0},  P{-1.0, 0.0}, P{0.0, -1.0},  P{1.0, 0.0},
                              P{0.0, 0.0}, P{-0.5, 0.5}, P{-1.0, 0.0}, P{-0.5, -0.5}, P{0.0, 0.0},
                              P{4.0, 0.0}, P{3.0, 1.0},  P{2.0, 0.0},  P{3.0, -1.0},  P{4.0, 0.0},
                              P{3.0, 0.0}, P{2.5, 0.5},  P{2.0, 0.0},  P{2.5, -0.5},  P{3.0, 0.0},
                            });

  CUSPATIAL_RUN_TEST(this->run, params, std::move(expected), abs_error<T>(params.radius));
}

TYPED_TEST(GeometryFactoryTest, multipolygonarray_2multipolygon1poly0hole)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  auto params   = multipolygon_generator_parameter<T>{2, 1, 0, 4, {0.0, 0.0}, 1.0};
  auto expected = make_multipolygon_array({0, 1, 2},
                                          {0, 1, 2},
                                          {0, 5, 10},
                                          {
                                            P{1.0, 0.0},
                                            P{0.0, 1.0},
                                            P{-1.0, 0.0},
                                            P{0.0, -1.0},
                                            P{1.0, 0.0},
                                            P{1.0, 0.0},
                                            P{0.0, 1.0},
                                            P{-1.0, 0.0},
                                            P{0.0, -1.0},
                                            P{1.0, 0.0},
                                          });

  CUSPATIAL_RUN_TEST(this->run, params, std::move(expected), abs_error<T>(params.radius));
}

TYPED_TEST(GeometryFactoryTest, multipolygonarray_2multipolygon1poly1hole)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  auto params = multipolygon_generator_parameter<T>{2, 1, 1, 4, {0.0, 0.0}, 1.0};
  auto expected =
    make_multipolygon_array({0, 1, 2},
                            {0, 2, 4},
                            {0, 5, 10, 15, 20},
                            {
                              P{1.0, 0.0}, P{0.0, 1.0},  P{-1.0, 0.0}, P{0.0, -1.0},  P{1.0, 0.0},
                              P{0.0, 0.0}, P{-0.5, 0.5}, P{-1.0, 0.0}, P{-0.5, -0.5}, P{0.0, 0.0},
                              P{1.0, 0.0}, P{0.0, 1.0},  P{-1.0, 0.0}, P{0.0, -1.0},  P{1.0, 0.0},
                              P{0.0, 0.0}, P{-0.5, 0.5}, P{-1.0, 0.0}, P{-0.5, -0.5}, P{0.0, 0.0},
                            });

  CUSPATIAL_RUN_TEST(this->run, params, std::move(expected), abs_error<T>(params.radius));
}

TYPED_TEST(GeometryFactoryTest, multipolygonarray_2multipolygon2poly1hole)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  auto params   = multipolygon_generator_parameter<T>{2, 2, 1, 4, {0.0, 0.0}, 1.0};
  auto expected = make_multipolygon_array(
    {0, 2, 4},
    {0, 2, 4, 6, 8},
    {0, 5, 10, 15, 20, 25, 30, 35, 40},
    {
      P{1.0, 0.0},  P{0.0, 1.0},  P{-1.0, 0.0},  P{0.0, -1.0}, P{1.0, 0.0},   P{0.0, 0.0},
      P{-0.5, 0.5}, P{-1.0, 0.0}, P{-0.5, -0.5}, P{0.0, 0.0},  P{4.0, 0.0},   P{3.0, 1.0},
      P{2.0, 0.0},  P{3.0, -1.0}, P{4.0, 0.0},   P{3.0, 0.0},  P{2.5, 0.5},   P{2.0, 0.0},
      P{2.5, -0.5}, P{3.0, 0.0},  P{1.0, 0.0},   P{0.0, 1.0},  P{-1.0, 0.0},  P{0.0, -1.0},
      P{1.0, 0.0},  P{0.0, 0.0},  P{-0.5, 0.5},  P{-1.0, 0.0}, P{-0.5, -0.5}, P{0.0, 0.0},
      P{4.0, 0.0},  P{3.0, 1.0},  P{2.0, 0.0},   P{3.0, -1.0}, P{4.0, 0.0},   P{3.0, 0.0},
      P{2.5, 0.5},  P{2.0, 0.0},  P{2.5, -0.5},  P{3.0, 0.0},
    });

  CUSPATIAL_RUN_TEST(this->run, params, std::move(expected), abs_error<T>(params.radius));
}

TYPED_TEST(GeometryFactoryTest, multipolygonarray_basic_centroid)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  auto params   = multipolygon_generator_parameter<T>{1, 1, 0, 4, {2.0, 3.0}, 1.0};
  auto expected = make_multipolygon_array(
    {0, 1}, {0, 1}, {0, 5}, {P{3.0, 3.0}, P{2.0, 4.0}, P{1.0, 3.0}, P{2.0, 2.0}, P{3.0, 3.0}});

  CUSPATIAL_RUN_TEST(this->run, params, std::move(expected), abs_error<T>(params.radius));
}

TYPED_TEST(GeometryFactoryTest, multipolygonarray_basic_radius)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  auto params   = multipolygon_generator_parameter<T>{1, 1, 0, 4, {0.0, 0.0}, 6.0};
  auto expected = make_multipolygon_array(
    {0, 1}, {0, 1}, {0, 5}, {P{6.0, 0.0}, P{0.0, 6.0}, P{-6.0, 0.0}, P{0.0, -6.0}, P{6.0, 0.0}});

  CUSPATIAL_RUN_TEST(this->run, params, std::move(expected), abs_error<T>(params.radius));
}

struct GeometryFactoryCountVerificationTest
  : public BaseFixtureWithParam<std::size_t, std::size_t, std::size_t, std::size_t> {
  void run(multipolygon_generator_parameter<float> params)
  {
    auto got = generate_multipolygon_array(params, stream());

    auto [got_geometry_offsets, got_part_offsets, got_ring_offsets, got_coordinates] =
      got.to_host();

    EXPECT_EQ(got_geometry_offsets.size(), params.num_multipolygons + 1);
    EXPECT_EQ(got_part_offsets.size(), params.num_polygons() + 1);
    EXPECT_EQ(got_ring_offsets.size(), params.num_rings() + 1);
    EXPECT_EQ(got_coordinates.size(), params.num_coords());
  }
};

TEST_P(GeometryFactoryCountVerificationTest, CountsVerification)
{
  // Structured binding unsupported by Gtest
  std::size_t num_multipolygons             = std::get<0>(GetParam());
  std::size_t num_polygons_per_multipolygon = std::get<1>(GetParam());
  std::size_t num_holes_per_polygon         = std::get<2>(GetParam());
  std::size_t num_sides_per_ring            = std::get<3>(GetParam());

  auto params = multipolygon_generator_parameter<float>{num_multipolygons,
                                                        num_polygons_per_multipolygon,
                                                        num_holes_per_polygon,
                                                        num_sides_per_ring,
                                                        vec_2d<float>{0.0, 0.0},
                                                        1.0};
  CUSPATIAL_RUN_TEST(this->run, params);
}

INSTANTIATE_TEST_SUITE_P(
  GeometryFactoryCountVerificationTests,
  GeometryFactoryCountVerificationTest,
  ::testing::Combine(::testing::Values<std::size_t>(1, 100),  // num_multipolygons
                     ::testing::Values<std::size_t>(1, 30),   // num_polygons_per_multipolygon
                     ::testing::Values<std::size_t>(0, 100),  // num_holes_per_polygon
                     ::testing::Values<std::size_t>(3, 100)   // num_sides_per_ring
                     ));
