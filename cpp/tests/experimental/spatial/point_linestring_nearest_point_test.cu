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

#include "../../utility/vector_equality.hpp"
#include "cuspatial/experimental/point_distance.cuh"

#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/experimental/point_linestring_nearest_point.cuh>
#include <cuspatial/experimental/type_utils.hpp>
#include <cuspatial/vec_2d.hpp>

#include <rmm/device_vector.hpp>

#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <type_traits>

namespace cuspatial {
namespace test {

template <typename T>
struct PairwisePointLinestringNearestPointTest : public ::testing::Test {
};

// float and double are logically the same but would require seperate tests due to precision.
using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(PairwisePointLinestringNearestPointTest, TestTypes);

TYPED_TEST(PairwisePointLinestringNearestPointTest, Empty)
{
  using T       = TypeParam;
  using CartVec = std::vector<vec_2d<T>>;

  auto num_pairs          = 0;
  auto points_geometry_it = thrust::make_counting_iterator(0);
  auto points_it          = rmm::device_vector<vec_2d<T>>(CartVec{});

  auto linestring_geometry_it  = thrust::make_counting_iterator(0);
  auto linestring_part_offsets = rmm::device_vector<int32_t>(std::vector<int32_t>{0});
  auto linestring_points_it    = rmm::device_vector<vec_2d<T>>(CartVec{});

  auto nearest_linestring_parts_id   = rmm::device_vector<int32_t>(num_pairs);
  auto nearest_linestring_segment_id = rmm::device_vector<int32_t>(num_pairs);
  auto neartest_point_coordinate     = rmm::device_vector<vec_2d<T>>(num_pairs);

  auto output_it =
    thrust::make_zip_iterator(thrust::make_tuple(nearest_linestring_parts_id.begin(),
                                                 nearest_linestring_segment_id.begin(),
                                                 neartest_point_coordinate.begin()));

  auto ret = pairwise_point_linestring_nearest_point(points_geometry_it,
                                                     points_geometry_it + num_pairs + 1,
                                                     points_it.begin(),
                                                     points_it.end(),
                                                     linestring_geometry_it,
                                                     linestring_part_offsets.begin(),
                                                     linestring_part_offsets.end(),
                                                     linestring_points_it.begin(),
                                                     linestring_points_it.end(),
                                                     output_it);

  EXPECT_EQ(nearest_linestring_parts_id, std::vector<int32_t>{});
  EXPECT_EQ(nearest_linestring_segment_id, std::vector<int32_t>{});
  expect_vec2d_vector_equivalent(thrust::host_vector<vec_2d<T>>(neartest_point_coordinate),
                                 std::vector<vec_2d<T>>{});
  EXPECT_EQ(std::distance(output_it, ret), num_pairs);
}

TYPED_TEST(PairwisePointLinestringNearestPointTest, OnePairSingleComponent)
{
  using T       = TypeParam;
  using CartVec = std::vector<vec_2d<T>>;

  auto num_pairs          = 1;
  auto points_geometry_it = thrust::make_counting_iterator(0);
  auto points_it          = rmm::device_vector<vec_2d<T>>(CartVec{{0, 0}});

  auto linestring_geometry_it  = thrust::make_counting_iterator(0);
  auto linestring_part_offsets = rmm::device_vector<int32_t>(std::vector<int32_t>{0, 3});
  auto linestring_points_it    = rmm::device_vector<vec_2d<T>>(CartVec{{1, -1}, {1, 0}, {0, 1}});

  auto nearest_linestring_segment_id = rmm::device_vector<int32_t>(num_pairs);
  auto neartest_point_coordinate     = rmm::device_vector<vec_2d<T>>(num_pairs);

  auto output_it =
    thrust::make_zip_iterator(thrust::make_tuple(thrust::make_discard_iterator(),
                                                 nearest_linestring_segment_id.begin(),
                                                 neartest_point_coordinate.begin()));

  auto ret = pairwise_point_linestring_nearest_point(points_geometry_it,
                                                     points_geometry_it + num_pairs + 1,
                                                     points_it.begin(),
                                                     points_it.end(),
                                                     linestring_geometry_it,
                                                     linestring_part_offsets.begin(),
                                                     linestring_part_offsets.end(),
                                                     linestring_points_it.begin(),
                                                     linestring_points_it.end(),
                                                     output_it);

  EXPECT_EQ(nearest_linestring_segment_id, std::vector<int32_t>{1});
  auto expected_coordinate = CartVec{{0.5, 0.5}};
  expect_vec2d_vector_equivalent(thrust::host_vector<vec_2d<T>>(neartest_point_coordinate),
                                 expected_coordinate);
  EXPECT_EQ(std::distance(output_it, ret), num_pairs);
}

TYPED_TEST(PairwisePointLinestringNearestPointTest, TwoPairsSingleComponent)
{
  using T       = TypeParam;
  using CartVec = std::vector<vec_2d<T>>;

  auto num_pairs          = 2;
  auto points_geometry_it = thrust::make_counting_iterator(0);
  auto points_it          = rmm::device_vector<vec_2d<T>>(CartVec{{0, 0}, {1, 2}});

  auto linestring_geometry_it  = thrust::make_counting_iterator(0);
  auto linestring_part_offsets = rmm::device_vector<int32_t>(std::vector<int32_t>{0, 3, 7});
  auto linestring_points_it    = rmm::device_vector<vec_2d<T>>(
    CartVec{{1, -1}, {1, 0}, {0, 1}, {0, 0}, {3, 1}, {3.9, 4}, {5.5, 1.2}});

  auto nearest_linestring_segment_id = rmm::device_vector<int32_t>(num_pairs);
  auto neartest_point_coordinate     = rmm::device_vector<vec_2d<T>>(num_pairs);

  auto output_it =
    thrust::make_zip_iterator(thrust::make_tuple(thrust::make_discard_iterator(),
                                                 nearest_linestring_segment_id.begin(),
                                                 neartest_point_coordinate.begin()));

  auto ret = pairwise_point_linestring_nearest_point(points_geometry_it,
                                                     points_geometry_it + num_pairs + 1,
                                                     points_it.begin(),
                                                     points_it.end(),
                                                     linestring_geometry_it,
                                                     linestring_part_offsets.begin(),
                                                     linestring_part_offsets.end(),
                                                     linestring_points_it.begin(),
                                                     linestring_points_it.end(),
                                                     output_it);

  EXPECT_EQ(nearest_linestring_segment_id, std::vector<int32_t>({1, 0}));
  auto expected_coordinate = CartVec{{0.5, 0.5}, {1.5, 0.5}};
  expect_vec2d_vector_equivalent(thrust::host_vector<vec_2d<T>>(neartest_point_coordinate),
                                 expected_coordinate);
  EXPECT_EQ(std::distance(output_it, ret), num_pairs);
}

TYPED_TEST(PairwisePointLinestringNearestPointTest, OnePairMultiComponent)
{
  using T       = TypeParam;
  using CartVec = std::vector<vec_2d<T>>;

  auto num_pairs          = 1;
  auto points_geometry_it = rmm::device_vector<int32_t>(std::vector<int>{0, 4, 6});
  auto points_it =
    rmm::device_vector<vec_2d<T>>(CartVec{{0, 0}, {1, 2}, {3, 4}, {5, 6}, {-1, -2}, {-3, -4}});

  auto linestring_geometry_it  = rmm::device_vector<int32_t>(std::vector<int>{0, 2, 5});
  auto linestring_part_offsets = rmm::device_vector<int32_t>(std::vector<int32_t>{0, 3, 5, 8, 12});
  auto linestring_points_it    = rmm::device_vector<vec_2d<T>>(CartVec{/*TODO*/});

  auto nearest_linestring_segment_id = rmm::device_vector<int32_t>(num_pairs);
  auto neartest_point_coordinate     = rmm::device_vector<vec_2d<T>>(num_pairs);

  auto output_it =
    thrust::make_zip_iterator(thrust::make_tuple(thrust::make_discard_iterator(),
                                                 nearest_linestring_segment_id.begin(),
                                                 neartest_point_coordinate.begin()));

  auto ret = pairwise_point_linestring_nearest_point(points_geometry_it,
                                                     points_geometry_it + num_pairs + 1,
                                                     points_it.begin(),
                                                     points_it.end(),
                                                     linestring_geometry_it,
                                                     linestring_part_offsets.begin(),
                                                     linestring_part_offsets.end(),
                                                     linestring_points_it.begin(),
                                                     linestring_points_it.end(),
                                                     output_it);

  EXPECT_EQ(nearest_linestring_segment_id, std::vector<int32_t>({1, 0}));
  auto expected_coordinate = CartVec{{0.5, 0.5}, {1.5, 0.5}};
  expect_vec2d_vector_equivalent(thrust::host_vector<vec_2d<T>>(neartest_point_coordinate),
                                 expected_coordinate);
  EXPECT_EQ(std::distance(output_it, ret), num_pairs);
}

}  // namespace test
}  // namespace cuspatial
