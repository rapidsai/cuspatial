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

#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/experimental/iterator_factory.cuh>
#include <cuspatial/experimental/point_linestring_nearest_points.cuh>
#include <cuspatial/vec_2d.hpp>

#include <rmm/device_vector.hpp>

#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <type_traits>

using namespace cuspatial;
using namespace cuspatial::test;

template <typename T>
struct PairwisePointLinestringNearestPointsTest : public ::testing::Test {
};

// float and double are logically the same but would require seperate tests due to precision.
using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(PairwisePointLinestringNearestPointsTest, TestTypes);

TYPED_TEST(PairwisePointLinestringNearestPointsTest, Empty)
{
  using T       = TypeParam;
  using CartVec = std::vector<vec_2d<T>>;

  auto num_pairs          = 0;
  auto points_geometry_it = thrust::make_counting_iterator(0);
  auto points_it          = rmm::device_vector<vec_2d<T>>(CartVec{});

  auto linestring_geometry_it  = thrust::make_counting_iterator(0);
  auto linestring_part_offsets = rmm::device_vector<int32_t>(std::vector<int32_t>{0});
  auto linestring_points_it    = rmm::device_vector<vec_2d<T>>(CartVec{});

  auto nearest_point_id              = rmm::device_vector<int32_t>(num_pairs);
  auto nearest_linestring_parts_id   = rmm::device_vector<int32_t>(num_pairs);
  auto nearest_linestring_segment_id = rmm::device_vector<int32_t>(0);
  auto neartest_point_coordinate     = rmm::device_vector<vec_2d<T>>(0);

  auto output_it =
    thrust::make_zip_iterator(thrust::make_tuple(nearest_point_id.begin(),
                                                 nearest_linestring_parts_id.begin(),
                                                 nearest_linestring_segment_id.begin(),
                                                 neartest_point_coordinate.begin()));

  auto ret = pairwise_point_linestring_nearest_points(points_geometry_it,
                                                      points_geometry_it + num_pairs + 1,
                                                      points_it.begin(),
                                                      points_it.end(),
                                                      linestring_geometry_it,
                                                      linestring_part_offsets.begin(),
                                                      linestring_part_offsets.end(),
                                                      linestring_points_it.begin(),
                                                      linestring_points_it.end(),
                                                      output_it);

  EXPECT_EQ(nearest_point_id, std::vector<int32_t>{});
  EXPECT_EQ(nearest_linestring_parts_id, std::vector<int32_t>{});
  EXPECT_EQ(nearest_linestring_segment_id, std::vector<int32_t>{});
  expect_vector_equivalent(neartest_point_coordinate, std::vector<vec_2d<T>>{});
  EXPECT_EQ(std::distance(output_it, ret), num_pairs);
}

TYPED_TEST(PairwisePointLinestringNearestPointsTest, OnePairSingleComponent)
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
                                                 thrust::make_discard_iterator(),
                                                 nearest_linestring_segment_id.begin(),
                                                 neartest_point_coordinate.begin()));

  auto ret = pairwise_point_linestring_nearest_points(points_geometry_it,
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
  expect_vector_equivalent(neartest_point_coordinate, expected_coordinate);
  EXPECT_EQ(std::distance(output_it, ret), num_pairs);
}

TYPED_TEST(PairwisePointLinestringNearestPointsTest, NearestAtLeftEndPoint)
{
  using T       = TypeParam;
  using CartVec = std::vector<vec_2d<T>>;

  auto num_pairs          = 1;
  auto points_geometry_it = thrust::make_counting_iterator(0);
  auto points_it          = rmm::device_vector<vec_2d<T>>(CartVec{{0, 0}});

  auto linestring_geometry_it  = thrust::make_counting_iterator(0);
  auto linestring_part_offsets = rmm::device_vector<int32_t>(std::vector<int32_t>{0, 2});
  auto linestring_points_it    = rmm::device_vector<vec_2d<T>>(CartVec{{1, 1}, {2, 2}});

  auto nearest_linestring_segment_id = rmm::device_vector<int32_t>(num_pairs);
  auto neartest_point_coordinate     = rmm::device_vector<vec_2d<T>>(num_pairs);

  auto output_it =
    thrust::make_zip_iterator(thrust::make_tuple(thrust::make_discard_iterator(),
                                                 thrust::make_discard_iterator(),
                                                 nearest_linestring_segment_id.begin(),
                                                 neartest_point_coordinate.begin()));

  auto ret = pairwise_point_linestring_nearest_points(points_geometry_it,
                                                      points_geometry_it + num_pairs + 1,
                                                      points_it.begin(),
                                                      points_it.end(),
                                                      linestring_geometry_it,
                                                      linestring_part_offsets.begin(),
                                                      linestring_part_offsets.end(),
                                                      linestring_points_it.begin(),
                                                      linestring_points_it.end(),
                                                      output_it);

  EXPECT_EQ(nearest_linestring_segment_id, std::vector<int32_t>{0});
  auto expected_coordinate = CartVec{{1, 1}};
  expect_vector_equivalent(neartest_point_coordinate, expected_coordinate);
  EXPECT_EQ(std::distance(output_it, ret), num_pairs);
}

TYPED_TEST(PairwisePointLinestringNearestPointsTest, NearestAtRightEndPoint)
{
  using T       = TypeParam;
  using CartVec = std::vector<vec_2d<T>>;

  auto num_pairs          = 1;
  auto points_geometry_it = thrust::make_counting_iterator(0);
  auto points_it          = rmm::device_vector<vec_2d<T>>(CartVec{{3, 3}});

  auto linestring_geometry_it  = thrust::make_counting_iterator(0);
  auto linestring_part_offsets = rmm::device_vector<int32_t>(std::vector<int32_t>{0, 2});
  auto linestring_points_it    = rmm::device_vector<vec_2d<T>>(CartVec{{1, 1}, {2, 2}});

  auto nearest_linestring_segment_id = rmm::device_vector<int32_t>(num_pairs);
  auto neartest_point_coordinate     = rmm::device_vector<vec_2d<T>>(num_pairs);

  auto output_it =
    thrust::make_zip_iterator(thrust::make_tuple(thrust::make_discard_iterator(),
                                                 thrust::make_discard_iterator(),
                                                 nearest_linestring_segment_id.begin(),
                                                 neartest_point_coordinate.begin()));

  auto ret = pairwise_point_linestring_nearest_points(points_geometry_it,
                                                      points_geometry_it + num_pairs + 1,
                                                      points_it.begin(),
                                                      points_it.end(),
                                                      linestring_geometry_it,
                                                      linestring_part_offsets.begin(),
                                                      linestring_part_offsets.end(),
                                                      linestring_points_it.begin(),
                                                      linestring_points_it.end(),
                                                      output_it);

  EXPECT_EQ(nearest_linestring_segment_id, std::vector<int32_t>{0});
  auto expected_coordinate = CartVec{{2, 2}};
  expect_vector_equivalent(neartest_point_coordinate, expected_coordinate);
  EXPECT_EQ(std::distance(output_it, ret), num_pairs);
}

TYPED_TEST(PairwisePointLinestringNearestPointsTest, PointAtEndPoints)
{
  using T       = TypeParam;
  using CartVec = std::vector<vec_2d<T>>;

  auto num_pairs          = 3;
  auto points_geometry_it = thrust::make_counting_iterator(0);
  auto points_it          = rmm::device_vector<vec_2d<T>>(CartVec{{0, 0}, {1, 1}, {2, 2}});

  auto linestring_geometry_it  = thrust::make_counting_iterator(0);
  auto linestring_part_offsets = rmm::device_vector<int32_t>(std::vector<int32_t>{0, 3, 6, 9});
  auto linestring_points_it    = rmm::device_vector<vec_2d<T>>(
    CartVec{{0, 0}, {1, 1}, {2, 2}, {0, 0}, {1, 1}, {2, 2}, {0, 0}, {1, 1}, {2, 2}});

  auto nearest_linestring_segment_id = rmm::device_vector<int32_t>(num_pairs);
  auto neartest_point_coordinate     = rmm::device_vector<vec_2d<T>>(num_pairs);

  auto output_it =
    thrust::make_zip_iterator(thrust::make_tuple(thrust::make_discard_iterator(),
                                                 thrust::make_discard_iterator(),
                                                 nearest_linestring_segment_id.begin(),
                                                 neartest_point_coordinate.begin()));

  auto ret = pairwise_point_linestring_nearest_points(points_geometry_it,
                                                      points_geometry_it + num_pairs + 1,
                                                      points_it.begin(),
                                                      points_it.end(),
                                                      linestring_geometry_it,
                                                      linestring_part_offsets.begin(),
                                                      linestring_part_offsets.end(),
                                                      linestring_points_it.begin(),
                                                      linestring_points_it.end(),
                                                      output_it);

  EXPECT_EQ(nearest_linestring_segment_id, std::vector<int32_t>({0, 0, 1}));
  auto expected_coordinate = CartVec{{0, 0}, {1, 1}, {2, 2}};
  expect_vector_equivalent(neartest_point_coordinate, expected_coordinate);
  EXPECT_EQ(std::distance(output_it, ret), num_pairs);
}

TYPED_TEST(PairwisePointLinestringNearestPointsTest, PointOnLineString)
{
  using T       = TypeParam;
  using CartVec = std::vector<vec_2d<T>>;

  auto num_pairs          = 2;
  auto points_geometry_it = thrust::make_counting_iterator(0);
  auto points_it          = rmm::device_vector<vec_2d<T>>(CartVec{{0.5, 0.5}, {1.5, 1.5}});

  auto linestring_geometry_it  = thrust::make_counting_iterator(0);
  auto linestring_part_offsets = rmm::device_vector<int32_t>(std::vector<int32_t>{0, 3, 6});
  auto linestring_points_it =
    rmm::device_vector<vec_2d<T>>(CartVec{{0, 0}, {1, 1}, {2, 2}, {0, 0}, {1, 1}, {2, 2}});

  auto nearest_linestring_segment_id = rmm::device_vector<int32_t>(num_pairs);
  auto neartest_point_coordinate     = rmm::device_vector<vec_2d<T>>(num_pairs);

  auto output_it =
    thrust::make_zip_iterator(thrust::make_tuple(thrust::make_discard_iterator(),
                                                 thrust::make_discard_iterator(),
                                                 nearest_linestring_segment_id.begin(),
                                                 neartest_point_coordinate.begin()));

  auto ret = pairwise_point_linestring_nearest_points(points_geometry_it,
                                                      points_geometry_it + num_pairs + 1,
                                                      points_it.begin(),
                                                      points_it.end(),
                                                      linestring_geometry_it,
                                                      linestring_part_offsets.begin(),
                                                      linestring_part_offsets.end(),
                                                      linestring_points_it.begin(),
                                                      linestring_points_it.end(),
                                                      output_it);

  EXPECT_EQ(nearest_linestring_segment_id, std::vector<int32_t>({0, 1}));
  auto expected_coordinate = CartVec{{0.5, 0.5}, {1.5, 1.5}};
  expect_vector_equivalent(neartest_point_coordinate, expected_coordinate);
  EXPECT_EQ(std::distance(output_it, ret), num_pairs);
}

TYPED_TEST(PairwisePointLinestringNearestPointsTest, TwoPairsSingleComponent)
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
                                                 thrust::make_discard_iterator(),
                                                 nearest_linestring_segment_id.begin(),
                                                 neartest_point_coordinate.begin()));

  auto ret = pairwise_point_linestring_nearest_points(points_geometry_it,
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
  expect_vector_equivalent(neartest_point_coordinate, expected_coordinate);
  EXPECT_EQ(std::distance(output_it, ret), num_pairs);
}

TYPED_TEST(PairwisePointLinestringNearestPointsTest, OnePairMultiComponent)
{
  using T       = TypeParam;
  using CartVec = std::vector<vec_2d<T>>;

  auto num_pairs               = 1;
  auto points_geometry_offsets = rmm::device_vector<int32_t>(std::vector<int>{0, 4});
  auto points_it = rmm::device_vector<vec_2d<T>>(CartVec{{0, 0}, {1, 2}, {3, 4}, {5, 6}});

  auto linestring_geometry_offsets = rmm::device_vector<int32_t>(std::vector<int>{0, 2});
  auto linestring_part_offsets     = rmm::device_vector<int32_t>(std::vector<int32_t>{0, 3, 5});
  auto linestring_points           = rmm::device_vector<vec_2d<T>>(
    CartVec{{1.0, 1.5}, {2.3, 3.7}, {-5, 4.0}, {0.0, 1.0}, {-2.0, 0.5}});

  auto nearest_point_id                 = rmm::device_vector<int32_t>(num_pairs);
  auto nearest_linestring_linestring_id = rmm::device_vector<int32_t>(num_pairs);
  auto nearest_linestring_segment_id    = rmm::device_vector<int32_t>(num_pairs);
  auto neartest_point_coordinate        = rmm::device_vector<vec_2d<T>>(num_pairs);

  auto output_it =
    thrust::make_zip_iterator(thrust::make_tuple(nearest_point_id.begin(),
                                                 nearest_linestring_linestring_id.begin(),
                                                 nearest_linestring_segment_id.begin(),
                                                 neartest_point_coordinate.begin()));

  auto ret = pairwise_point_linestring_nearest_points(points_geometry_offsets.begin(),
                                                      points_geometry_offsets.end(),
                                                      points_it.begin(),
                                                      points_it.end(),
                                                      linestring_geometry_offsets.begin(),
                                                      linestring_part_offsets.begin(),
                                                      linestring_part_offsets.end(),
                                                      linestring_points.begin(),
                                                      linestring_points.end(),
                                                      output_it);

  EXPECT_EQ(nearest_point_id, std::vector<int32_t>({1}));
  EXPECT_EQ(nearest_linestring_linestring_id, std::vector<int32_t>({0}));
  EXPECT_EQ(nearest_linestring_segment_id, std::vector<int32_t>({0}));
  auto expected_coordinate = CartVec{{1.2189892802450228, 1.8705972434915774}};
  expect_vector_equivalent(neartest_point_coordinate, expected_coordinate);
  EXPECT_EQ(std::distance(output_it, ret), num_pairs);
}

TYPED_TEST(PairwisePointLinestringNearestPointsTest, ThreePairMultiComponent)
{
  using T       = TypeParam;
  using CartVec = std::vector<vec_2d<T>>;

  auto num_pairs               = 3;
  auto points_geometry_offsets = rmm::device_vector<int32_t>(std::vector<int>{0, 2, 3, 6});
  auto points                  = rmm::device_vector<vec_2d<T>>(
    CartVec{{1.1, 3.0}, {3.6, 2.4}, {10.0, 15.0}, {-5.0, -8.7}, {-6.28, -7.0}, {-10.0, -10.0}});

  auto linestring_geometry_offsets = rmm::device_vector<int32_t>(std::vector<int>{0, 2, 4, 5});
  auto linestring_part_offsets =
    rmm::device_vector<int32_t>(std::vector<int32_t>{0, 3, 5, 7, 9, 13});
  auto linestring_points_it = rmm::device_vector<vec_2d<T>>(CartVec{{2.1, 3.14},
                                                                    {8.4, -0.5},
                                                                    {6.0, 1.4},
                                                                    {-1.0, 0.0},
                                                                    {-1.7, 0.83},
                                                                    {20.14, 13.5},
                                                                    {18.3, 14.3},
                                                                    {8.34, 9.1},
                                                                    {9.9, 9.4},
                                                                    {-20.0, 0.0},
                                                                    {-15.0, -15.0},
                                                                    {0.0, -18.0},
                                                                    {0.0, 0.0}});

  auto nearest_point_id          = rmm::device_vector<int32_t>(num_pairs);
  auto nearest_linestring_id     = rmm::device_vector<int32_t>(num_pairs);
  auto nearest_segment_id        = rmm::device_vector<int32_t>(num_pairs);
  auto neartest_point_coordinate = rmm::device_vector<vec_2d<T>>(num_pairs);

  auto output_it = thrust::make_zip_iterator(thrust::make_tuple(nearest_point_id.begin(),
                                                                nearest_linestring_id.begin(),
                                                                nearest_segment_id.begin(),
                                                                neartest_point_coordinate.begin()));

  auto ret = pairwise_point_linestring_nearest_points(points_geometry_offsets.begin(),
                                                      points_geometry_offsets.end(),
                                                      points.begin(),
                                                      points.end(),
                                                      linestring_geometry_offsets.begin(),
                                                      linestring_part_offsets.begin(),
                                                      linestring_part_offsets.end(),
                                                      linestring_points_it.begin(),
                                                      linestring_points_it.end(),
                                                      output_it);

  EXPECT_EQ(thrust::host_vector<int32_t>(nearest_point_id), std::vector<int32_t>({1, 0, 0}));
  EXPECT_EQ(thrust::host_vector<int32_t>(nearest_linestring_id), std::vector<int32_t>({0, 1, 0}));
  EXPECT_EQ(thrust::host_vector<int32_t>(nearest_segment_id), std::vector<int32_t>({0, 0, 2}));
  expect_vector_equivalent(neartest_point_coordinate,
                           CartVec{{3.545131432802666, 2.30503517215846}, {9.9, 9.4}, {0.0, -8.7}});
  EXPECT_EQ(std::distance(output_it, ret), num_pairs);
}
