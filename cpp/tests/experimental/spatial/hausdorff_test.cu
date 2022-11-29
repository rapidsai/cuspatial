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

#include <cuspatial_test/vector_equality.hpp>

#include <cuspatial/error.hpp>
#include <cuspatial/experimental/hausdorff.cuh>
#include <cuspatial/vec_2d.hpp>

#include <rmm/device_vector.hpp>

#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <iterator>

template <typename T>
struct HausdorffTest : public ::testing::Test {
  template <typename Point, typename Index>
  void test(std::vector<Point> const& points,
            std::vector<Index> const& space_offsets,
            std::vector<T> const& expected)
  {
    auto const d_points        = rmm::device_vector<Point>{points};
    auto const d_space_offsets = rmm::device_vector<Index>{space_offsets};

    auto const num_distances = space_offsets.size() * space_offsets.size();
    auto distances           = rmm::device_vector<T>{num_distances};

    auto const distances_end = cuspatial::directed_hausdorff_distance(d_points.begin(),
                                                                      d_points.end(),
                                                                      d_space_offsets.begin(),
                                                                      d_space_offsets.end(),
                                                                      distances.begin());

    thrust::host_vector<T> h_distances(distances);

    cuspatial::test::expect_vector_equivalent(distances, expected);
    EXPECT_EQ(num_distances, std::distance(distances.begin(), distances_end));
  }
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(HausdorffTest, TestTypes);

TYPED_TEST(HausdorffTest, Empty)
{
  this->template test<cuspatial::vec_2d<TypeParam>, uint32_t>({}, {}, {});
}

TYPED_TEST(HausdorffTest, Simple)
{
  this->template test<cuspatial::vec_2d<TypeParam>, uint32_t>(
    {{0, 0}, {1, 0}, {0, 1}, {0, 2}},
    {{0, 2}},
    {{0.0, static_cast<TypeParam>(std::sqrt(2.0)), 2.0, 0.0}});
}

TYPED_TEST(HausdorffTest, SingleTrajectorySinglePoint)
{
  this->template test<cuspatial::vec_2d<TypeParam>, uint32_t>({{152.2, 2351.0}}, {{0}}, {{0.0}});
}

TYPED_TEST(HausdorffTest, TwoShortSpaces)
{
  this->template test<cuspatial::vec_2d<TypeParam>, uint32_t>(
    {{0, 0}, {5, 12}, {4, 3}}, {{0, 1}}, {{0.0, 5.0, 13.0, 0.0}});
}

TYPED_TEST(HausdorffTest, TwoShortSpaces2)
{
  this->template test<cuspatial::vec_2d<TypeParam>, uint32_t>(
    {{1, 1}, {5, 12}, {4, 3}, {2, 8}, {3, 4}, {7, 7}},
    {{0, 3, 4}},
    {{0.0,
      7.0710678118654755,
      5.3851648071345037,
      5.0000000000000000,
      0.0,
      4.1231056256176606,
      5.0,
      5.0990195135927854,
      0.0}});
}

TYPED_TEST(HausdorffTest, ThreeSpacesLengths543)
{
  this->template test<cuspatial::vec_2d<TypeParam>, uint32_t>({{0.0, 1.0},
                                                               {1.0, 2.0},
                                                               {2.0, 3.0},
                                                               {3.0, 5.0},
                                                               {1.0, 7.0},
                                                               {3.0, 0.0},
                                                               {5.0, 2.0},
                                                               {6.0, 3.0},
                                                               {5.0, 6.0},
                                                               {4.0, 1.0},
                                                               {7.0, 3.0},
                                                               {4.0, 6.0}},
                                                              {{0, 5, 9}},
                                                              {{0.0000000000000000,
                                                                4.1231056256176606,
                                                                4.0000000000000000,
                                                                3.6055512754639896,
                                                                0.0000000000000000,
                                                                1.4142135623730951,
                                                                4.4721359549995796,
                                                                1.4142135623730951,
                                                                0.0000000000000000}});
}

TYPED_TEST(HausdorffTest, MoreSpacesThanPoints)
{
  EXPECT_THROW(
    (this->template test<cuspatial::vec_2d<TypeParam>, uint32_t>({{0, 0}}, {{0, 1}}, {{0.0}})),
    cuspatial::logic_error);
}

template <typename T, uint32_t num_spaces, uint32_t elements_per_space>
void generic_hausdorff_test()
{
  constexpr uint64_t num_points =
    static_cast<uint64_t>(elements_per_space) * static_cast<uint64_t>(num_spaces);
  constexpr auto num_distances = num_spaces * num_spaces;

  using vec_2d = cuspatial::vec_2d<T>;

  auto zero_iter         = thrust::make_constant_iterator<vec_2d>({0, 0});
  auto counting_iter     = thrust::make_counting_iterator<uint32_t>(0);
  auto space_offset_iter = thrust::make_transform_iterator(
    counting_iter, [] __device__(auto idx) { return idx * elements_per_space; });

  auto distances = rmm::device_vector<T>{num_distances};
  auto expected  = rmm::device_vector<T>{num_distances, 0};

  auto distances_end = cuspatial::directed_hausdorff_distance(zero_iter,
                                                              zero_iter + num_points,
                                                              space_offset_iter,
                                                              space_offset_iter + num_spaces,
                                                              distances.begin());

  cuspatial::test::expect_vector_equivalent(distances, expected);
  EXPECT_EQ(num_distances, std::distance(distances.begin(), distances_end));
}

TYPED_TEST(HausdorffTest, 500Spaces100Points) { generic_hausdorff_test<TypeParam, 500, 100>(); }

TYPED_TEST(HausdorffTest, 10000Spaces10Points) { generic_hausdorff_test<TypeParam, 10000, 10>(); }

TYPED_TEST(HausdorffTest, 10Spaces10000Points) { generic_hausdorff_test<TypeParam, 10, 10000>(); }
