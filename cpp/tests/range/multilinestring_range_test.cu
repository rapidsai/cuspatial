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
#include <cuspatial_test/test_util.cuh>
#include <cuspatial_test/vector_equality.hpp>
#include <cuspatial_test/vector_factories.cuh>

#include <cuspatial/detail/utility/zero_data.cuh>
#include <cuspatial/geometry/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <thrust/sequence.h>
#include <thrust/tabulate.h>

#include <initializer_list>
#include <limits>

using namespace cuspatial;
using namespace cuspatial::test;

template <typename MultiLineStringRange, typename OutputIt>
CUSPATIAL_KERNEL void array_access_tester(MultiLineStringRange mls,
                                          std::size_t i,
                                          OutputIt output_points)
{
  thrust::copy(thrust::seq, mls[i].point_begin(), mls[i].point_end(), output_points);
}

template <typename T>
struct MultilinestringRangeTest : public BaseFixture {
  void run_segment_test_single(std::initializer_list<std::size_t> geometry_offset,
                               std::initializer_list<std::size_t> part_offset,
                               std::initializer_list<vec_2d<T>> coordinates,
                               std::initializer_list<segment<T>> expected)
  {
    auto multilinestring_array =
      make_multilinestring_array(geometry_offset, part_offset, coordinates);
    auto rng            = multilinestring_array.range();
    auto segments       = rng._segments(stream());
    auto segments_range = segments.segment_range();

    rmm::device_uvector<segment<T>> got(segments_range.num_segments(), stream());
    thrust::copy(
      rmm::exec_policy(stream()), segments_range.begin(), segments_range.end(), got.begin());

    auto d_expected = thrust::device_vector<segment<T>>(expected.begin(), expected.end());
    CUSPATIAL_EXPECT_VEC2D_PAIRS_EQUIVALENT(d_expected, got);
  }

  void run_multilinestring_point_count_test(std::initializer_list<std::size_t> geometry_offset,
                                            std::initializer_list<std::size_t> part_offset,
                                            std::initializer_list<vec_2d<T>> coordinates,
                                            std::initializer_list<std::size_t> expected)
  {
    auto multilinestring_array =
      make_multilinestring_array(geometry_offset, part_offset, coordinates);

    auto d_expected = thrust::device_vector<std::size_t>(expected.begin(), expected.end());

    auto rng = multilinestring_array.range();

    rmm::device_uvector<std::size_t> got(rng.num_multilinestrings(), stream());
    thrust::copy(rmm::exec_policy(stream()),
                 rng.multilinestring_point_count_begin(),
                 rng.multilinestring_point_count_end(),
                 got.begin());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_expected, got);
  }

  void run_multilinestring_segment_count_test(std::initializer_list<std::size_t> geometry_offset,
                                              std::initializer_list<std::size_t> part_offset,
                                              std::initializer_list<vec_2d<T>> coordinates,
                                              std::initializer_list<std::size_t> expected)
  {
    auto multilinestring_array =
      make_multilinestring_array(geometry_offset, part_offset, coordinates);
    auto rng            = multilinestring_array.range();
    auto segments       = rng._segments(stream());
    auto segments_range = segments.segment_range();

    auto d_expected = thrust::device_vector<std::size_t>(expected.begin(), expected.end());

    rmm::device_uvector<std::size_t> got(rng.num_multilinestrings(), stream());
    thrust::copy(rmm::exec_policy(stream()),
                 segments_range.multigeometry_count_begin(),
                 segments_range.multigeometry_count_end(),
                 got.begin());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_expected, got);
  }

  void run_multilinestring_segment_offset_test(std::initializer_list<std::size_t> geometry_offset,
                                               std::initializer_list<std::size_t> part_offset,
                                               std::initializer_list<vec_2d<T>> coordinates,
                                               std::initializer_list<std::size_t> expected)
  {
    auto multilinestring_array =
      make_multilinestring_array(geometry_offset, part_offset, coordinates);
    auto rng            = multilinestring_array.range();
    auto segments       = rng._segments(stream());
    auto segments_range = segments.segment_range();

    auto d_expected = thrust::device_vector<std::size_t>(expected.begin(), expected.end());

    rmm::device_uvector<std::size_t> got(rng.num_multilinestrings() + 1, stream());

    thrust::copy(rmm::exec_policy(stream()),
                 segments_range.multigeometry_offset_begin(),
                 segments_range.multigeometry_offset_end(),
                 got.begin());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_expected, got);
  }

  void run_multilinestring_linestring_count_test(std::initializer_list<std::size_t> geometry_offset,
                                                 std::initializer_list<std::size_t> part_offset,
                                                 std::initializer_list<vec_2d<T>> coordinates,
                                                 std::initializer_list<std::size_t> expected)
  {
    auto multilinestring_array =
      make_multilinestring_array(geometry_offset, part_offset, coordinates);

    auto d_expected = thrust::device_vector<std::size_t>(expected.begin(), expected.end());

    auto rng = multilinestring_array.range();

    rmm::device_uvector<std::size_t> got(rng.num_multilinestrings(), stream());
    thrust::copy(rmm::exec_policy(stream()),
                 rng.multilinestring_linestring_count_begin(),
                 rng.multilinestring_linestring_count_end(),
                 got.begin());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_expected, got);
  }

  void run_multilinestring_as_multipoint_test(
    std::initializer_list<std::size_t> geometry_offset,
    std::initializer_list<std::size_t> part_offset,
    std::initializer_list<vec_2d<T>> coordinates,
    std::initializer_list<std::initializer_list<vec_2d<T>>> expected)
  {
    auto multilinestring_array =
      make_multilinestring_array(geometry_offset, part_offset, coordinates);
    auto multilinestring_range = multilinestring_array.range();

    auto multipoint_range = multilinestring_range.as_multipoint_range();

    thrust::device_vector<std::size_t> got_geometry_offset(multipoint_range.offsets_begin(),
                                                           multipoint_range.offsets_end());
    thrust::device_vector<vec_2d<T>> got_coordinates(multipoint_range.point_begin(),
                                                     multipoint_range.point_end());

    auto expected_multipoint = make_multipoint_array(expected);
    auto expected_range      = expected_multipoint.range();

    thrust::device_vector<std::size_t> expected_geometry_offset(expected_range.offsets_begin(),
                                                                expected_range.offsets_end());
    thrust::device_vector<vec_2d<T>> expected_coordinates(expected_range.point_begin(),
                                                          expected_range.point_end());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_geometry_offset, got_geometry_offset);
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_coordinates, got_coordinates);
  }
};

using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_CASE(MultilinestringRangeTest, TestTypes);

TYPED_TEST(MultilinestringRangeTest, SegmentIteratorOneSegmentTest)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(
    this->run_segment_test_single, {0, 1}, {0, 2}, {P{0, 0}, P{1, 1}}, {S{P{0, 0}, P{1, 1}}});
}

TYPED_TEST(MultilinestringRangeTest, SegmentIteratorTwoSegmentTest)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_segment_test_single,
                     {0, 2},
                     {0, 2, 4},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{3, 3}},
                     {S{P{0, 0}, P{1, 1}}, S{P{2, 2}, P{3, 3}}});
}

TYPED_TEST(MultilinestringRangeTest, SegmentIteratorTwoSegmentTest2)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_segment_test_single,
                     {0, 1, 2},
                     {0, 2, 4},
                     {P{0, 0}, P{1, 1}, P{0, 0}, P{1, 1}},
                     {S{P{0, 0}, P{1, 1}}, S{P{0, 0}, P{1, 1}}});
}

TYPED_TEST(MultilinestringRangeTest, SegmentIteratorManyPairTest)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_segment_test_single,
                     {0, 1, 2, 3},
                     {0, 6, 11, 14},
                     {

                       P{0, 0},
                       P{1, 1},
                       P{2, 2},
                       P{3, 3},
                       P{4, 4},
                       P{5, 5},

                       P{10, 10},
                       P{11, 11},
                       P{12, 12},
                       P{13, 13},
                       P{14, 14},

                       P{20, 20},
                       P{21, 21},
                       P{22, 22}

                     },

                     {S{P{0, 0}, P{1, 1}},
                      S{P{1, 1}, P{2, 2}},
                      S{P{2, 2}, P{3, 3}},
                      S{P{3, 3}, P{4, 4}},
                      S{P{4, 4}, P{5, 5}},
                      S{P{10, 10}, P{11, 11}},
                      S{P{11, 11}, P{12, 12}},
                      S{P{12, 12}, P{13, 13}},
                      S{P{13, 13}, P{14, 14}},
                      S{P{20, 20}, P{21, 21}},
                      S{P{21, 21}, P{22, 22}}});
}

TYPED_TEST(MultilinestringRangeTest, SegmentIteratorWithEmptyLineTest)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(
    this->run_segment_test_single,
    {0, 1, 2, 3},
    {0, 3, 3, 6},
    {P{0, 0}, P{1, 1}, P{2, 2}, P{10, 10}, P{11, 11}, P{12, 12}},
    {S{P{0, 0}, P{1, 1}}, S{P{1, 1}, P{2, 2}}, S{P{10, 10}, P{11, 11}}, S{P{11, 11}, P{12, 12}}});
}

TYPED_TEST(MultilinestringRangeTest, SegmentIteratorWithEmptyMultiLineStringTest)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(
    this->run_segment_test_single,
    {0, 1, 1, 2},
    {0, 3, 6},
    {P{0, 0}, P{1, 1}, P{2, 2}, P{10, 10}, P{11, 11}, P{12, 12}},
    {S{P{0, 0}, P{1, 1}}, S{P{1, 1}, P{2, 2}}, S{P{10, 10}, P{11, 11}}, S{P{11, 11}, P{12, 12}}});
}

TYPED_TEST(MultilinestringRangeTest, SegmentIteratorWithEmptyMultiLineStringTest2)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_segment_test_single,

                     {0, 1, 1, 2},
                     {0, 4, 7},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{3, 3}, P{10, 10}, P{11, 11}, P{12, 12}},
                     {S{P{0, 0}, P{1, 1}},
                      S{P{1, 1}, P{2, 2}},
                      S{P{2, 2}, P{3, 3}},
                      S{P{10, 10}, P{11, 11}},
                      S{P{11, 11}, P{12, 12}}});
}

TYPED_TEST(MultilinestringRangeTest, PerMultilinestringCountTest)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_multilinestring_point_count_test,
                     {0, 1, 2, 3},
                     {0, 3, 3, 6},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{10, 10}, P{11, 11}, P{12, 12}},
                     {3, 0, 3});
}

TYPED_TEST(MultilinestringRangeTest, PerMultilinestringCountTest2)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_multilinestring_point_count_test,
                     {0, 1, 3},
                     {0, 3, 3, 6},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{10, 10}, P{11, 11}, P{12, 12}},
                     {3, 3});
}

TYPED_TEST(MultilinestringRangeTest, MultilinestringSegmentCountTest)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(
    this->run_multilinestring_segment_count_test, {0, 1}, {0, 3}, {P{0, 0}, P{1, 1}, P{2, 2}}, {2});
}

TYPED_TEST(MultilinestringRangeTest, MultilinestringSegmentCountTest2)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_multilinestring_segment_count_test,
                     {0, 1, 3},
                     {0, 3, 3, 6},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{10, 10}, P{11, 11}, P{12, 12}},
                     {2, 2});
}

TYPED_TEST(MultilinestringRangeTest, MultilinestringSegmentCountTest3)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_multilinestring_segment_count_test,
                     {0, 1, 2},
                     {0, 3, 6},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{10, 10}, P{11, 11}, P{12, 12}},
                     {2, 2});
}

TYPED_TEST(MultilinestringRangeTest, MultilinestringSegmentCountTest4)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_multilinestring_segment_count_test,
                     {0, 1, 3},
                     {0, 3, 5, 7},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{10, 10}, P{11, 11}, P{20, 20}, P{21, 21}},
                     {2, 2});
}

// contains empty linestring
TYPED_TEST(MultilinestringRangeTest, MultilinestringSegmentCountTest5)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_multilinestring_segment_count_test,
                     {0, 1, 2, 3},
                     {0, 3, 3, 6},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{10, 10}, P{11, 11}, P{12, 12}},
                     {2, 0, 2});
}

// contains empty multilinestring
TYPED_TEST(MultilinestringRangeTest, MultilinestringSegmentCountTest6)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_multilinestring_segment_count_test,
                     {0, 1, 1, 2},
                     {0, 3, 6},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{10, 10}, P{11, 11}, P{12, 12}},
                     {2, 0, 2});
}

// contains empty multilinestring
TYPED_TEST(MultilinestringRangeTest, MultilinestringSegmentCountTest7)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_multilinestring_segment_count_test,
                     {0, 1, 1, 2},
                     {0, 4, 7},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{3, 3}, P{10, 10}, P{11, 11}, P{12, 12}},
                     {3, 0, 2});
}

// contains empty multilinestring
TYPED_TEST(MultilinestringRangeTest, MultilinestringSegmentOffsetTest)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_multilinestring_segment_offset_test,
                     {0, 1, 1, 2},
                     {0, 4, 7},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{3, 3}, P{10, 10}, P{11, 11}, P{12, 12}},
                     {0, 3, 3, 5});
}

TYPED_TEST(MultilinestringRangeTest, MultilinestringLinestringCountTest)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(
    this->run_multilinestring_linestring_count_test,
    {0, 1, 2, 3},
    {0, 3, 6, 9},
    {P{0, 0}, P{1, 1}, P{2, 2}, P{10, 10}, P{11, 11}, P{12, 12}, P{20, 20}, P{21, 21}, P{22, 22}},
    {1, 1, 1});
}

TYPED_TEST(MultilinestringRangeTest, MultilinestringLinestringCountTest2)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(
    this->run_multilinestring_linestring_count_test,
    {0, 1, 3},
    {0, 3, 6, 9},
    {P{0, 0}, P{1, 1}, P{2, 2}, P{10, 10}, P{11, 11}, P{12, 12}, P{20, 20}, P{21, 21}, P{22, 22}},
    {1, 2});
}

TYPED_TEST(MultilinestringRangeTest, MultilinestringAsMultipointTest)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_multilinestring_as_multipoint_test,
                     {0, 1},
                     {0, 3},
                     {P{0, 0}, P{1, 1}, P{2, 2}},
                     {{P{0, 0}, P{1, 1}, P{2, 2}}});
}

TYPED_TEST(MultilinestringRangeTest, MultilinestringAsMultipointTest2)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_multilinestring_as_multipoint_test,
                     {0, 2},
                     {0, 3, 5},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{3, 3}, P{4, 4}},
                     {{P{0, 0}, P{1, 1}, P{2, 2}, P{3, 3}, P{4, 4}}});
}

TYPED_TEST(MultilinestringRangeTest, MultilinestringAsMultipointTest3)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_multilinestring_as_multipoint_test,
                     {0, 1, 2},
                     {0, 3, 5},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{10, 10}, P{11, 11}},
                     {{P{0, 0}, P{1, 1}, P{2, 2}}, {P{10, 10}, P{11, 11}}});
}

TYPED_TEST(MultilinestringRangeTest, MultilinestringAsMultipointTest4)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_multilinestring_as_multipoint_test,
                     {0, 1, 3},
                     {0, 3, 5, 7},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{10, 10}, P{11, 11}, P{12, 12}, P{13, 13}},
                     {{P{0, 0}, P{1, 1}, P{2, 2}}, {P{10, 10}, P{11, 11}, P{12, 12}, P{13, 13}}});
}

TYPED_TEST(MultilinestringRangeTest, MultilinestringAsMultipointTest5)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_multilinestring_as_multipoint_test,
                     {0, 1},
                     {0, 4},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{2, 3}},
                     {{P{0, 0}, P{1, 1}, P{2, 2}, P{2, 3}}});
}

TYPED_TEST(MultilinestringRangeTest, MultilinestringAsMultipointTest6)
{
  using T = TypeParam;
  using P = vec_2d<T>;
  using S = segment<T>;

  CUSPATIAL_RUN_TEST(this->run_multilinestring_as_multipoint_test,
                     {0, 2},
                     {0, 2, 4},
                     {P{1, 1}, P{0, 0}, P{6, 6}, P{6, 7}},
                     {{P{1, 1}, P{0, 0}, P{6, 6}, P{6, 7}}});
}

template <typename T>
class MultilinestringRangeTestBase : public BaseFixture {
 public:
  struct copy_leading_point_per_multilinestring {
    template <typename MultiLineStringRef>
    vec_2d<T> __device__ operator()(MultiLineStringRef m)
    {
      return m.size() > 0 ? m[0].point_begin()[0] : vec_2d<T>{-1, -1};
    }
  };

  template <typename MultiLineStringRange>
  struct part_idx_from_point_idx_functor {
    MultiLineStringRange _rng;
    std::size_t __device__ operator()(std::size_t point_idx)
    {
      return _rng.part_idx_from_point_idx(point_idx);
    }
  };

  template <typename MultiLineStringRange>
  struct part_idx_from_segment_idx_functor {
    MultiLineStringRange _rng;
    std::size_t __device__ operator()(std::size_t segment_idx)
    {
      auto opt = _rng.part_idx_from_segment_idx(segment_idx);
      if (opt.has_value()) {
        return opt.value();
      } else {
        return std::numeric_limits<std::size_t>::max();
      }
    }
  };

  template <typename MultiLineStringRange>
  struct geometry_idx_from_point_idx_functor {
    MultiLineStringRange _rng;
    std::size_t __device__ operator()(std::size_t point_idx)
    {
      return _rng.geometry_idx_from_point_idx(point_idx);
    }
  };

  template <typename MultiLineStringRange>
  struct intra_part_idx_functor {
    MultiLineStringRange _rng;
    std::size_t __device__ operator()(std::size_t i) { return _rng.intra_part_idx(i); }
  };

  template <typename MultiLineStringRange>
  struct intra_point_idx_functor {
    MultiLineStringRange _rng;
    std::size_t __device__ operator()(std::size_t i) { return _rng.intra_point_idx(i); }
  };

  template <typename MultiLineStringRange>
  struct is_valid_segment_id_functor {
    MultiLineStringRange _rng;
    bool __device__ operator()(std::size_t i)
    {
      auto part_idx = _rng.part_idx_from_point_idx(i);
      return _rng.is_valid_segment_id(i, part_idx);
    }
  };

  template <typename MultiLineStringRange>
  struct segment_functor {
    MultiLineStringRange _rng;
    segment<T> __device__ operator()(std::size_t i)
    {
      auto part_idx = _rng.part_idx_from_point_idx(i);
      return _rng.is_valid_segment_id(i, part_idx)
               ? _rng.segment(i)
               : segment<T>{vec_2d<T>{-1, -1}, vec_2d<T>{-1, -1}};
    }
  };

  void SetUp() { make_test_multilinestring(); }

  virtual void make_test_multilinestring() = 0;

  auto range() { return test_multilinestring->range(); }

  void run_test()
  {
    test_size();

    test_num_multilinestrings();

    test_num_linestrings();

    test_num_points();

    test_multilinestring_it();

    test_begin();

    test_end();

    test_point_it();

    test_part_offset_it();

    test_part_idx_from_point_idx();

    test_part_idx_from_segment_idx();

    test_geometry_idx_from_point_idx();

    test_intra_part_idx();

    test_intra_point_idx();

    test_is_valid_segment_id();

    test_segment();

    test_multilinestring_point_count_it();

    test_multilinestring_linestring_count_it();

    test_array_access_operator();

    test_geometry_offset_it();
  }

  void test_size() { EXPECT_EQ(this->range().size(), this->range().num_multilinestrings()); }

  virtual void test_num_multilinestrings() = 0;

  virtual void test_num_linestrings() = 0;

  virtual void test_num_points() = 0;

  virtual void test_multilinestring_it() = 0;

  void test_begin() { EXPECT_EQ(this->range().begin(), this->range().multilinestring_begin()); }

  void test_end() { EXPECT_EQ(this->range().end(), this->range().multilinestring_end()); }

  virtual void test_point_it() = 0;

  virtual void test_geometry_offset_it() = 0;

  virtual void test_part_offset_it() = 0;

  virtual void test_part_idx_from_point_idx() = 0;

  virtual void test_part_idx_from_segment_idx() = 0;

  virtual void test_geometry_idx_from_point_idx() = 0;

  virtual void test_intra_part_idx() = 0;

  virtual void test_intra_point_idx() = 0;

  virtual void test_is_valid_segment_id() = 0;

  virtual void test_segment() = 0;

  virtual void test_multilinestring_point_count_it() = 0;

  virtual void test_multilinestring_linestring_count_it() = 0;

  virtual void test_array_access_operator() = 0;

  // Helper functions to be used by all subclass (test cases).
  rmm::device_uvector<vec_2d<T>> copy_leading_points()
  {
    auto rng             = this->range();
    auto d_leading_point = rmm::device_uvector<vec_2d<T>>(rng.num_multilinestrings(), stream());
    thrust::transform(rmm::exec_policy(stream()),
                      rng.begin(),
                      rng.end(),
                      d_leading_point.begin(),
                      copy_leading_point_per_multilinestring());
    return d_leading_point;
  }

  rmm::device_uvector<vec_2d<T>> copy_all_points()
  {
    auto rng          = this->range();
    auto d_all_points = rmm::device_uvector<vec_2d<T>>(rng.num_points(), stream());
    thrust::copy(
      rmm::exec_policy(stream()), rng.point_begin(), rng.point_end(), d_all_points.begin());
    return d_all_points;
  }

  rmm::device_uvector<std::size_t> copy_geometry_offsets()
  {
    auto rng = this->range();
    auto d_geometry_offsets =
      rmm::device_uvector<std::size_t>(rng.num_multilinestrings() + 1, stream());
    thrust::copy(rmm::exec_policy(stream()),
                 rng.geometry_offset_begin(),
                 rng.geometry_offset_end(),
                 d_geometry_offsets.begin());
    return d_geometry_offsets;
  }

  rmm::device_uvector<std::size_t> copy_part_offset()
  {
    auto rng           = this->range();
    auto d_part_offset = rmm::device_uvector<std::size_t>(rng.num_linestrings() + 1, stream());
    thrust::copy(rmm::exec_policy(stream()),
                 rng.part_offset_begin(),
                 rng.part_offset_end(),
                 d_part_offset.begin());
    return d_part_offset;
  }

  rmm::device_uvector<std::size_t> copy_part_idx_from_point_idx()
  {
    auto rng        = this->range();
    auto d_part_idx = rmm::device_uvector<std::size_t>(rng.num_points(), stream());
    auto f          = part_idx_from_point_idx_functor<decltype(rng)>{rng};
    thrust::tabulate(rmm::exec_policy(stream()), d_part_idx.begin(), d_part_idx.end(), f);
    return d_part_idx;
  }

  rmm::device_uvector<std::size_t> copy_part_idx_from_segment_idx()
  {
    auto rng        = this->range();
    auto d_part_idx = rmm::device_uvector<std::size_t>(rng.num_points(), stream());
    auto f          = part_idx_from_segment_idx_functor<decltype(rng)>{rng};

    thrust::tabulate(rmm::exec_policy(stream()), d_part_idx.begin(), d_part_idx.end(), f);
    return d_part_idx;
  }

  rmm::device_uvector<std::size_t> copy_geometry_idx_from_point_idx()
  {
    auto rng            = this->range();
    auto d_geometry_idx = rmm::device_uvector<std::size_t>(rng.num_points(), stream());
    thrust::tabulate(rmm::exec_policy(stream()),
                     d_geometry_idx.begin(),
                     d_geometry_idx.end(),
                     geometry_idx_from_point_idx_functor<decltype(rng)>{rng});
    return d_geometry_idx;
  }

  rmm::device_uvector<std::size_t> copy_intra_part_idx()
  {
    auto rng              = this->range();
    auto d_intra_part_idx = rmm::device_uvector<std::size_t>(rng.num_linestrings(), stream());
    thrust::tabulate(rmm::exec_policy(stream()),
                     d_intra_part_idx.begin(),
                     d_intra_part_idx.end(),
                     intra_part_idx_functor<decltype(rng)>{rng});
    return d_intra_part_idx;
  }

  rmm::device_uvector<std::size_t> copy_intra_point_idx()
  {
    auto rng               = this->range();
    auto d_intra_point_idx = rmm::device_uvector<std::size_t>(rng.num_points(), stream());
    thrust::tabulate(rmm::exec_policy(stream()),
                     d_intra_point_idx.begin(),
                     d_intra_point_idx.end(),
                     intra_point_idx_functor<decltype(rng)>{rng});
    return d_intra_point_idx;
  }

  rmm::device_uvector<uint8_t> copy_is_valid_segment_id()
  {
    auto rng                   = this->range();
    auto d_is_valid_segment_id = rmm::device_uvector<uint8_t>(rng.num_points(), stream());
    thrust::tabulate(rmm::exec_policy(stream()),
                     d_is_valid_segment_id.begin(),
                     d_is_valid_segment_id.end(),
                     is_valid_segment_id_functor<decltype(rng)>{rng});
    return d_is_valid_segment_id;
  }

  rmm::device_uvector<segment<T>> copy_segments()
  {
    auto rng        = this->range();
    auto d_segments = rmm::device_uvector<segment<T>>(rng.num_points(), stream());
    thrust::tabulate(rmm::exec_policy(stream()),
                     d_segments.begin(),
                     d_segments.end(),
                     segment_functor<decltype(rng)>{rng});
    return d_segments;
  }

  rmm::device_uvector<std::size_t> copy_multilinestring_point_count()
  {
    auto rng = this->range();
    auto d_multilinestring_point_count =
      rmm::device_uvector<std::size_t>(rng.num_multilinestrings(), stream());
    thrust::copy(rmm::exec_policy(stream()),
                 rng.multilinestring_point_count_begin(),
                 rng.multilinestring_point_count_end(),
                 d_multilinestring_point_count.begin());
    return d_multilinestring_point_count;
  }

  rmm::device_uvector<std::size_t> copy_multilinestring_linestring_count()
  {
    auto rng = this->range();
    auto d_multilinestring_linestring_count =
      rmm::device_uvector<std::size_t>(rng.num_multilinestrings(), stream());
    thrust::copy(rmm::exec_policy(stream()),
                 rng.multilinestring_linestring_count_begin(),
                 rng.multilinestring_linestring_count_end(),
                 d_multilinestring_linestring_count.begin());
    return d_multilinestring_linestring_count;
  }

  rmm::device_uvector<vec_2d<T>> copy_all_points_of_ith_multilinestring(std::size_t i)
  {
    auto rng = this->range();
    rmm::device_scalar<std::size_t> num_points(stream());

    thrust::copy_n(rmm::exec_policy(stream()),
                   rng.multilinestring_point_count_begin() + i,
                   1,
                   num_points.data());

    auto d_all_points = rmm::device_uvector<vec_2d<T>>(num_points.value(stream()), stream());

    array_access_tester<<<1, 1, 0, stream()>>>(rng, i, d_all_points.data());
    return d_all_points;
  }

 protected:
  std::unique_ptr<multilinestring_array<rmm::device_vector<std::size_t>,
                                        rmm::device_vector<std::size_t>,
                                        rmm::device_vector<vec_2d<T>>>>
    test_multilinestring;
};

template <typename T>
class MultilinestringRangeEmptyTest : public MultilinestringRangeTestBase<T> {
  void make_test_multilinestring()
  {
    auto array                 = make_multilinestring_array<T>({0}, {0}, {});
    this->test_multilinestring = std::make_unique<decltype(array)>(std::move(array));
  }

  void test_num_multilinestrings() { EXPECT_EQ(this->range().num_multilinestrings(), 0); }

  void test_num_linestrings() { EXPECT_EQ(this->range().num_linestrings(), 0); }

  void test_num_points() { EXPECT_EQ(this->range().num_points(), 0); }

  void test_multilinestring_it()
  {
    auto leading_points = this->copy_leading_points();
    auto expected       = rmm::device_uvector<vec_2d<T>>(0, this->stream());
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(leading_points, expected);
  }

  void test_point_it()
  {
    auto all_points = this->copy_all_points();
    auto expected   = rmm::device_uvector<vec_2d<T>>(0, this->stream());
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(all_points, expected);
  }

  void test_part_offset_it()
  {
    auto part_offset = this->copy_part_offset();
    auto expected    = make_device_vector<std::size_t>({0});
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(part_offset, expected);
  }

  void test_part_idx_from_point_idx()
  {
    auto part_idx = this->copy_part_idx_from_point_idx();
    auto expected = rmm::device_vector<std::size_t>(0);
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(part_idx, expected);
  }

  void test_part_idx_from_segment_idx()
  {
    auto part_idx = this->copy_part_idx_from_segment_idx();
    auto expected = rmm::device_vector<std::size_t>(0);
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(part_idx, expected);
  }

  void test_geometry_idx_from_point_idx()
  {
    auto geometry_idx = this->copy_geometry_idx_from_point_idx();
    auto expected     = rmm::device_vector<std::size_t>(0);
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(geometry_idx, expected);
  }

  void test_intra_part_idx()
  {
    auto intra_part_idx = this->copy_intra_part_idx();
    auto expected       = rmm::device_vector<std::size_t>(0);
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(intra_part_idx, expected);
  }

  void test_intra_point_idx()
  {
    auto intra_point_idx = this->copy_intra_point_idx();
    auto expected        = rmm::device_vector<std::size_t>(0);
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(intra_point_idx, expected);
  }

  void test_is_valid_segment_id()
  {
    auto is_valid_segment_id = this->copy_is_valid_segment_id();
    auto expected            = rmm::device_vector<uint8_t>(0);
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(is_valid_segment_id, expected);
  }

  void test_segment()
  {
    auto segments = this->copy_segments();
    auto expected = rmm::device_vector<segment<T>>(0);
    CUSPATIAL_EXPECT_VEC2D_PAIRS_EQUIVALENT(segments, expected);
  }

  void test_multilinestring_point_count_it()
  {
    auto multilinestring_point_count = this->copy_multilinestring_point_count();
    auto expected                    = rmm::device_vector<std::size_t>(0);
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(multilinestring_point_count, expected);
  }

  void test_multilinestring_linestring_count_it()
  {
    auto multilinestring_linestring_count = this->copy_multilinestring_linestring_count();
    auto expected                         = rmm::device_vector<std::size_t>(0);
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(multilinestring_linestring_count, expected);
  }

  void test_array_access_operator()
  {
    // Nothing to access
    SUCCEED();
  }

  void test_geometry_offset_it()
  {
    auto geometry_offsets = this->copy_geometry_offsets();
    auto expected         = make_device_vector<std::size_t>({0});
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(geometry_offsets, expected);
  }

  void test_part_offsets_it()
  {
    auto part_offsets = this->copy_part_offsets();
    auto expected     = make_device_vector<std::size_t>({0});
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(part_offsets, expected);
  }
};

TYPED_TEST_CASE(MultilinestringRangeEmptyTest, FloatingPointTypes);
TYPED_TEST(MultilinestringRangeEmptyTest, EmptyTest) { this->run_test(); }

template <typename T>
class MultilinestringRangeOneTest : public MultilinestringRangeTestBase<T> {
  void make_test_multilinestring()
  {
    auto array = make_multilinestring_array<T>(
      {0, 2}, {0, 2, 5}, {{10, 10}, {20, 20}, {100, 100}, {200, 200}, {300, 300}});
    this->test_multilinestring = std::make_unique<decltype(array)>(std::move(array));
  }

  void test_num_multilinestrings() { EXPECT_EQ(this->range().num_multilinestrings(), 1); }

  void test_num_linestrings() { EXPECT_EQ(this->range().num_linestrings(), 2); }

  void test_num_points() { EXPECT_EQ(this->range().num_points(), 5); }

  void test_multilinestring_it()
  {
    auto leading_points = this->copy_leading_points();
    auto expected       = make_device_vector<vec_2d<T>>({{10, 10}});
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(leading_points, expected);
  }

  void test_point_it()
  {
    auto all_points = this->copy_all_points();
    auto expected =
      make_device_vector<vec_2d<T>>({{10, 10}, {20, 20}, {100, 100}, {200, 200}, {300, 300}});
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(all_points, expected);
  }

  void test_part_offset_it()
  {
    auto part_offset = this->copy_part_offset();
    auto expected    = make_device_vector<std::size_t>({0, 2, 5});
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(part_offset, expected);
  }

  void test_part_idx_from_point_idx()
  {
    auto part_idx = this->copy_part_idx_from_point_idx();
    auto expected = make_device_vector<std::size_t>({0, 0, 1, 1, 1});
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(part_idx, expected);
  }

  void test_part_idx_from_segment_idx()
  {
    auto part_idx = this->copy_part_idx_from_segment_idx();
    auto expected = make_device_vector<std::size_t>(
      {0, std::numeric_limits<std::size_t>::max(), 1, 1, std::numeric_limits<std::size_t>::max()});
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(part_idx, expected);
  }

  void test_geometry_idx_from_point_idx()
  {
    auto geometry_idx = this->copy_geometry_idx_from_point_idx();
    auto expected     = make_device_vector<std::size_t>({0, 0, 0, 0, 0});
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(geometry_idx, expected);
  }

  void test_intra_part_idx()
  {
    auto intra_part_idx = this->copy_intra_part_idx();
    auto expected       = make_device_vector<std::size_t>({0, 1});
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(intra_part_idx, expected);
  }

  void test_intra_point_idx()
  {
    auto intra_point_idx = this->copy_intra_point_idx();
    auto expected        = make_device_vector<std::size_t>({0, 1, 0, 1, 2});
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(intra_point_idx, expected);
  }

  void test_is_valid_segment_id()
  {
    auto is_valid_segment_id = this->copy_is_valid_segment_id();
    auto expected            = make_device_vector<uint8_t>({1, 0, 1, 1, 0});
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(is_valid_segment_id, expected);
  }

  void test_segment()
  {
    auto segments = this->copy_segments();
    auto expected = make_device_vector<segment<T>>({{{10, 10}, {20, 20}},
                                                    {{-1, -1}, {-1, -1}},
                                                    {{100, 100}, {200, 200}},
                                                    {{200, 200}, {300, 300}},
                                                    {{-1, -1}, {-1, -1}}});
    CUSPATIAL_EXPECT_VEC2D_PAIRS_EQUIVALENT(segments, expected);
  }

  void test_multilinestring_point_count_it()
  {
    auto multilinestring_point_count = this->copy_multilinestring_point_count();
    auto expected                    = make_device_vector<std::size_t>({5});
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(multilinestring_point_count, expected);
  }

  void test_multilinestring_linestring_count_it()
  {
    auto multilinestring_linestring_count = this->copy_multilinestring_linestring_count();
    auto expected                         = make_device_vector<std::size_t>({2});
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(multilinestring_linestring_count, expected);
  }

  void test_array_access_operator()
  {
    auto all_points = this->copy_all_points_of_ith_multilinestring(0);
    auto expected =
      make_device_vector<vec_2d<T>>({{10, 10}, {20, 20}, {100, 100}, {200, 200}, {300, 300}});
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(all_points, expected);
  }

  void test_geometry_offset_it()
  {
    auto geometry_offsets = this->copy_geometry_offsets();
    auto expected         = make_device_vector<std::size_t>({0, 2});
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(geometry_offsets, expected);
  }

  void test_part_offsets_it()
  {
    auto part_offsets = this->copy_part_offsets();
    auto expected     = make_device_vector<std::size_t>({0, 2, 5});
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(part_offsets, expected);
  }
};

TYPED_TEST_CASE(MultilinestringRangeOneTest, FloatingPointTypes);
TYPED_TEST(MultilinestringRangeOneTest, OneTest) { this->run_test(); }

template <typename T>
class MultilinestringRangeOneThousandTest : public MultilinestringRangeTestBase<T> {
 public:
  struct make_points_functor {
    vec_2d<T> __device__ operator()(std::size_t i)
    {
      auto part_idx        = i / 2;
      auto intra_point_idx = i % 2;
      return vec_2d<T>{static_cast<T>(part_idx * 10 + intra_point_idx),
                       static_cast<T>(part_idx * 10 + intra_point_idx)};
    }
  };

  void make_test_multilinestring()
  {
    rmm::device_vector<std::size_t> geometry_offset(1001);
    rmm::device_vector<std::size_t> part_offset(1001);
    rmm::device_vector<vec_2d<T>> points(2000);

    thrust::sequence(
      rmm::exec_policy(this->stream()), geometry_offset.begin(), geometry_offset.end());

    thrust::sequence(
      rmm::exec_policy(this->stream()), part_offset.begin(), part_offset.end(), 0, 2);

    thrust::tabulate(
      rmm::exec_policy(this->stream()), points.begin(), points.end(), make_points_functor{});

    auto array = make_multilinestring_array(
      std::move(geometry_offset), std::move(part_offset), std::move(points));

    this->test_multilinestring = std::make_unique<decltype(array)>(std::move(array));
  }

  void test_num_multilinestrings() { EXPECT_EQ(this->range().num_multilinestrings(), 1000); }

  void test_num_linestrings() { EXPECT_EQ(this->range().num_linestrings(), 1000); }

  void test_num_points() { EXPECT_EQ(this->range().num_points(), 2000); }

  void test_multilinestring_it()
  {
    auto leading_points = this->copy_leading_points();
    auto expected       = rmm::device_uvector<vec_2d<T>>(1000, this->stream());

    thrust::tabulate(rmm::exec_policy(this->stream()),
                     expected.begin(),
                     expected.end(),
                     [] __device__(std::size_t i) {
                       return vec_2d<T>{i * T{10.}, i * T{10.}};
                     });
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(leading_points, expected);
  }

  void test_point_it()
  {
    auto all_points = this->copy_all_points();
    auto expected   = rmm::device_uvector<vec_2d<T>>(2000, this->stream());

    thrust::tabulate(
      rmm::exec_policy(this->stream()), expected.begin(), expected.end(), make_points_functor{});
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(all_points, expected);
  }

  void test_part_offset_it()
  {
    auto part_offset = this->copy_part_offset();
    auto expected    = rmm::device_uvector<std::size_t>(1001, this->stream());

    thrust::sequence(rmm::exec_policy(this->stream()), expected.begin(), expected.end(), 0, 2);

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(part_offset, expected);
  }

  void test_part_idx_from_point_idx()
  {
    auto part_idx = this->copy_part_idx_from_point_idx();
    auto expected = rmm::device_uvector<std::size_t>(2000, this->stream());

    thrust::tabulate(rmm::exec_policy(this->stream()),
                     expected.begin(),
                     expected.end(),
                     [] __device__(std::size_t i) { return i / 2; });

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(part_idx, expected);
  }

  void test_part_idx_from_segment_idx()
  {
    auto part_idx = this->copy_part_idx_from_segment_idx();
    auto expected = rmm::device_uvector<std::size_t>(2000, this->stream());

    thrust::tabulate(rmm::exec_policy(this->stream()),
                     expected.begin(),
                     expected.end(),
                     [] __device__(std::size_t i) {
                       return i % 2 == 0 ? i / 2 : std::numeric_limits<std::size_t>::max();
                     });

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(part_idx, expected);
  }

  void test_geometry_idx_from_point_idx()
  {
    auto geometry_idx = this->copy_geometry_idx_from_point_idx();
    auto expected     = rmm::device_uvector<std::size_t>(2000, this->stream());

    thrust::tabulate(rmm::exec_policy(this->stream()),
                     expected.begin(),
                     expected.end(),
                     [] __device__(std::size_t i) { return i / 2; });

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(geometry_idx, expected);
  }

  void test_intra_part_idx()
  {
    auto intra_part_idx = this->copy_intra_part_idx();
    auto expected       = rmm::device_uvector<std::size_t>(1000, this->stream());

    detail::zero_data_async(expected.begin(), expected.end(), this->stream());
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(intra_part_idx, expected);
  }

  void test_intra_point_idx()
  {
    auto intra_point_idx = this->copy_intra_point_idx();
    auto expected        = rmm::device_uvector<std::size_t>(2000, this->stream());

    thrust::tabulate(rmm::exec_policy(this->stream()),
                     expected.begin(),
                     expected.end(),
                     [] __device__(std::size_t i) { return i % 2; });

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(intra_point_idx, expected);
  }

  void test_is_valid_segment_id()
  {
    auto is_valid_segment_id = this->copy_is_valid_segment_id();
    auto expected            = rmm::device_uvector<uint8_t>(2000, this->stream());

    thrust::tabulate(rmm::exec_policy(this->stream()),
                     expected.begin(),
                     expected.end(),
                     [] __device__(std::size_t i) { return (i + 1) % 2; });
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(is_valid_segment_id, expected);
  }

  void test_segment()
  {
    auto segments = this->copy_segments();
    auto expected = rmm::device_uvector<segment<T>>(2000, this->stream());

    thrust::tabulate(
      rmm::exec_policy(this->stream()),
      expected.begin(),
      expected.end(),
      [] __device__(std::size_t i) {
        auto part_idx        = i / 2;
        auto intra_point_idx = i % 2;
        return i % 2 == 0
                 ? segment<T>{vec_2d<T>{static_cast<T>(part_idx * 10 + intra_point_idx),
                                        static_cast<T>(part_idx * 10 + intra_point_idx)},
                              vec_2d<T>{static_cast<T>(part_idx * 10 + intra_point_idx + 1),
                                        static_cast<T>(part_idx * 10 + intra_point_idx + 1)}}
                 : segment<T>{vec_2d<T>{-1, -1}, vec_2d<T>{-1, -1}};
      });
    CUSPATIAL_EXPECT_VEC2D_PAIRS_EQUIVALENT(segments, expected);
  }

  void test_multilinestring_point_count_it()
  {
    auto multilinestring_point_count = this->copy_multilinestring_point_count();
    auto expected                    = rmm::device_uvector<std::size_t>(1000, this->stream());

    thrust::fill(rmm::exec_policy(this->stream()), expected.begin(), expected.end(), 2);
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(multilinestring_point_count, expected);
  }

  void test_multilinestring_linestring_count_it()
  {
    auto multilinestring_linestring_count = this->copy_multilinestring_linestring_count();
    auto expected                         = rmm::device_uvector<std::size_t>(1000, this->stream());

    thrust::fill(rmm::exec_policy(this->stream()), expected.begin(), expected.end(), 1);
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(multilinestring_linestring_count, expected);
  }

  void test_array_access_operator()
  {
    auto all_points = this->copy_all_points_of_ith_multilinestring(513);
    auto expected   = make_device_vector<vec_2d<T>>({{5130, 5130}, {5131, 5131}});

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(all_points, expected);
  }

  void test_geometry_offset_it()
  {
    auto geometry_offsets = this->copy_geometry_offsets();
    auto expected         = rmm::device_uvector<std::size_t>(1001, this->stream());

    thrust::sequence(rmm::exec_policy(this->stream()), expected.begin(), expected.end());
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(geometry_offsets, expected);
  }

  void test_part_offsets_it()
  {
    auto part_offsets = this->copy_part_offsets();
    auto expected     = rmm::device_uvector<std::size_t>(1001, this->stream());

    thrust::sequence(rmm::exec_policy(this->stream()), expected.begin(), expected.end(), 0, 2);
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(part_offsets, expected);
  }
};

TYPED_TEST_CASE(MultilinestringRangeOneThousandTest, FloatingPointTypes);
TYPED_TEST(MultilinestringRangeOneThousandTest, OneThousandTest) { this->run_test(); }
