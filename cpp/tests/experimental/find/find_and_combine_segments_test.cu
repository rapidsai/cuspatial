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
#include <cuspatial_test/vector_factories.cuh>
#include <tests/base_fixture.hpp>

#include <cuspatial/experimental/detail/find/find_and_combine_segment.cuh>
#include <cuspatial/experimental/geometry/segment.cuh>
#include <cuspatial/experimental/ranges/range.cuh>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>

using namespace cuspatial;
using namespace cuspatial::detail;
using namespace cuspatial::test;

template <typename SegmentVector, typename T = typename SegmentVector::value_type::value_type>
std::pair<rmm::device_vector<vec_2d<T>>, rmm::device_vector<vec_2d<T>>> unpack_segment_vector(
  SegmentVector const& segments)
{
  rmm::device_vector<vec_2d<T>> first(segments.size()), second(segments.size());
  auto zipped_output = thrust::make_zip_iterator(first.begin(), second.begin());
  thrust::transform(
    segments.begin(), segments.end(), zipped_output, [] __device__(segment<T> const& segment) {
      return thrust::make_tuple(segment.first, segment.second);
    });
  return {std::move(first), std::move(second)};
}

template <typename SegmentVector1, typename SegmentVector2>
void expect_segment_equivalent(SegmentVector1 expected, SegmentVector2 got)
{
  auto [expected_first, expected_second] = unpack_segment_vector(expected);
  auto [got_first, got_second]           = unpack_segment_vector(got);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_first, got_first);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_second, got_second);
}

template <typename T>
struct FindAndCombineSegmentsTest : public BaseFixture {
  rmm::cuda_stream_view stream() { return rmm::cuda_stream_default; }
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(FindAndCombineSegmentsTest, TestTypes);

TYPED_TEST(FindAndCombineSegmentsTest, Simple1)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto segment_offsets = make_device_vector<index_t>({0, 3});
  auto segments        = make_device_vector<segment<T>>(
    {S{P{0.0, 0.0}, P{0.0, 0.5}}, S{P{0.0, 0.25}, P{0.0, 0.75}}, S{P{0.0, 0.5}, P{0.0, 1.0}}});
  auto merged_flag = rmm::device_vector<uint8_t>(segments.size());

  auto expected_segments = make_device_vector<segment<T>>(
    {S{P{0.0, 0.0}, P{0.0, 1.0}}, S{P{0.0, 0.25}, P{0.0, 0.75}}, S{P{0.0, 0.5}, P{0.0, 1.0}}});
  auto expected_merged_flag = make_device_vector<uint8_t>({0, 1, 1});

  find_and_combine_segment(range(segment_offsets.begin(), segment_offsets.end()),
                           range(segments.begin(), segments.end()),
                           merged_flag.begin(),
                           this->stream());

  expect_segment_equivalent(segments, expected_segments);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(merged_flag, expected_merged_flag);
}

TYPED_TEST(FindAndCombineSegmentsTest, Simple2)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto segment_offsets = make_device_vector<index_t>({0, 3});
  auto segments        = make_device_vector<segment<T>>(
    {S{P{0.0, 0.0}, P{0.5, 0.0}}, S{P{0.25, 0.0}, P{0.75, 0.0}}, S{P{0.5, 0.0}, P{1.0, 0.0}}});
  auto merged_flag = rmm::device_vector<uint8_t>(segments.size());

  auto expected_segments = make_device_vector<segment<T>>(
    {S{P{0.0, 0.0}, P{1.0, 0.0}}, S{P{0.25, 0.0}, P{0.75, 0.0}}, S{P{0.5, 0.0}, P{1.0, 0.0}}});
  auto expected_merged_flag = make_device_vector<uint8_t>({0, 1, 1});

  find_and_combine_segment(range(segment_offsets.begin(), segment_offsets.end()),
                           range(segments.begin(), segments.end()),
                           merged_flag.begin(),
                           this->stream());

  expect_segment_equivalent(segments, expected_segments);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(merged_flag, expected_merged_flag);
}

TYPED_TEST(FindAndCombineSegmentsTest, Simple3)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto segment_offsets = make_device_vector<index_t>({0, 3});
  auto segments        = make_device_vector<segment<T>>(
    {S{P{0.0, 0.0}, P{0.5, 0.5}}, S{P{0.25, 0.25}, P{0.75, 0.75}}, S{P{0.5, 0.5}, P{1.0, 1.0}}});
  auto merged_flag = rmm::device_vector<uint8_t>(segments.size());

  auto expected_segments = make_device_vector<segment<T>>(
    {S{P{0.0, 0.0}, P{1.0, 1.0}}, S{P{0.25, 0.25}, P{0.75, 0.75}}, S{P{0.5, 0.5}, P{1.0, 1.0}}});
  auto expected_merged_flag = make_device_vector<uint8_t>({0, 1, 1});

  find_and_combine_segment(range(segment_offsets.begin(), segment_offsets.end()),
                           range(segments.begin(), segments.end()),
                           merged_flag.begin(),
                           this->stream());

  expect_segment_equivalent(segments, expected_segments);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(merged_flag, expected_merged_flag);
}

TYPED_TEST(FindAndCombineSegmentsTest, Touching1)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto segment_offsets = make_device_vector<index_t>({0, 2});
  auto segments =
    make_device_vector<segment<T>>({S{P{0.0, 0.0}, P{0.0, 0.5}}, S{P{0.0, 0.5}, P{0.0, 1.0}}});
  auto merged_flag = rmm::device_vector<uint8_t>(segments.size());

  auto expected_segments =
    make_device_vector<segment<T>>({S{P{0.0, 0.0}, P{0.0, 1.0}}, S{P{0.0, 0.5}, P{0.0, 1.0}}});
  auto expected_merged_flag = make_device_vector<uint8_t>({0, 1});

  find_and_combine_segment(range(segment_offsets.begin(), segment_offsets.end()),
                           range(segments.begin(), segments.end()),
                           merged_flag.begin(),
                           this->stream());

  expect_segment_equivalent(segments, expected_segments);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(merged_flag, expected_merged_flag);
}

TYPED_TEST(FindAndCombineSegmentsTest, Touching2)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto segment_offsets = make_device_vector<index_t>({0, 2});
  auto segments =
    make_device_vector<segment<T>>({S{P{0.0, 0.0}, P{0.5, 0.0}}, S{P{0.5, 0.0}, P{1.0, 0.0}}});
  auto merged_flag = rmm::device_vector<uint8_t>(segments.size());

  auto expected_segments =
    make_device_vector<segment<T>>({S{P{0.0, 0.0}, P{1.0, 0.0}}, S{P{0.5, 0.0}, P{1.0, 0.0}}});
  auto expected_merged_flag = make_device_vector<uint8_t>({0, 1});

  find_and_combine_segment(range(segment_offsets.begin(), segment_offsets.end()),
                           range(segments.begin(), segments.end()),
                           merged_flag.begin(),
                           this->stream());

  expect_segment_equivalent(segments, expected_segments);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(merged_flag, expected_merged_flag);
}

TYPED_TEST(FindAndCombineSegmentsTest, Touching3)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto segment_offsets = make_device_vector<index_t>({0, 2});
  auto segments =
    make_device_vector<segment<T>>({S{P{0.0, 0.0}, P{0.5, 0.5}}, S{P{0.5, 0.5}, P{1.0, 1.0}}});
  auto merged_flag = rmm::device_vector<uint8_t>(segments.size());

  auto expected_segments =
    make_device_vector<segment<T>>({S{P{0.0, 0.0}, P{1.0, 1.0}}, S{P{0.5, 0.5}, P{1.0, 1.0}}});
  auto expected_merged_flag = make_device_vector<uint8_t>({0, 1});

  find_and_combine_segment(range(segment_offsets.begin(), segment_offsets.end()),
                           range(segments.begin(), segments.end()),
                           merged_flag.begin(),
                           this->stream());

  expect_segment_equivalent(segments, expected_segments);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(merged_flag, expected_merged_flag);
}

TYPED_TEST(FindAndCombineSegmentsTest, contains1)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto segment_offsets = make_device_vector<index_t>({0, 2});
  auto segments =
    make_device_vector<segment<T>>({S{P{0.0, 0.0}, P{1.0, 1.0}}, S{P{0.25, 0.25}, P{0.75, 0.75}}});
  auto merged_flag = rmm::device_vector<uint8_t>(segments.size());

  auto expected_segments =
    make_device_vector<segment<T>>({S{P{0.0, 0.0}, P{1.0, 1.0}}, S{P{0.25, 0.25}, P{0.75, 0.75}}});
  auto expected_merged_flag = make_device_vector<uint8_t>({0, 1});

  find_and_combine_segment(range(segment_offsets.begin(), segment_offsets.end()),
                           range(segments.begin(), segments.end()),
                           merged_flag.begin(),
                           this->stream());

  expect_segment_equivalent(segments, expected_segments);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(merged_flag, expected_merged_flag);
}

TYPED_TEST(FindAndCombineSegmentsTest, contains2)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto segment_offsets = make_device_vector<index_t>({0, 2});
  auto segments =
    make_device_vector<segment<T>>({S{P{0.0, 0.0}, P{0.0, 1.0}}, S{P{0.0, 0.25}, P{0.0, 0.75}}});
  auto merged_flag = rmm::device_vector<uint8_t>(segments.size());

  auto expected_segments =
    make_device_vector<segment<T>>({S{P{0.0, 0.0}, P{0.0, 1.0}}, S{P{0.0, 0.25}, P{0.0, 0.75}}});
  auto expected_merged_flag = make_device_vector<uint8_t>({0, 1});

  find_and_combine_segment(range(segment_offsets.begin(), segment_offsets.end()),
                           range(segments.begin(), segments.end()),
                           merged_flag.begin(),
                           this->stream());

  expect_segment_equivalent(segments, expected_segments);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(merged_flag, expected_merged_flag);
}

TYPED_TEST(FindAndCombineSegmentsTest, contains3)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto segment_offsets = make_device_vector<index_t>({0, 2});
  auto segments =
    make_device_vector<segment<T>>({S{P{0.0, 0.0}, P{1.0, 0.0}}, S{P{0.25, 0.0}, P{0.75, 0.0}}});
  auto merged_flag = rmm::device_vector<uint8_t>(segments.size());

  auto expected_segments =
    make_device_vector<segment<T>>({S{P{0.0, 0.0}, P{1.0, 0.0}}, S{P{0.25, 0.0}, P{0.75, 0.0}}});
  auto expected_merged_flag = make_device_vector<uint8_t>({0, 1});

  find_and_combine_segment(range(segment_offsets.begin(), segment_offsets.end()),
                           range(segments.begin(), segments.end()),
                           merged_flag.begin(),
                           this->stream());

  expect_segment_equivalent(segments, expected_segments);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(merged_flag, expected_merged_flag);
}

TYPED_TEST(FindAndCombineSegmentsTest, nooverlap1)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto segment_offsets = make_device_vector<index_t>({0, 2});
  auto segments =
    make_device_vector<segment<T>>({S{P{0.0, 0.0}, P{1.0, 0.0}}, S{P{0.0, 1.0}, P{0.0, 2.0}}});
  auto merged_flag = rmm::device_vector<uint8_t>(segments.size());

  auto expected_segments =
    make_device_vector<segment<T>>({S{P{0.0, 0.0}, P{1.0, 0.0}}, S{P{0.0, 1.0}, P{0.0, 2.0}}});
  auto expected_merged_flag = make_device_vector<uint8_t>({0, 0});

  find_and_combine_segment(range(segment_offsets.begin(), segment_offsets.end()),
                           range(segments.begin(), segments.end()),
                           merged_flag.begin(),
                           this->stream());

  expect_segment_equivalent(segments, expected_segments);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(merged_flag, expected_merged_flag);
}

TYPED_TEST(FindAndCombineSegmentsTest, nooverlap2)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto segment_offsets = make_device_vector<index_t>({0, 2});
  auto segments =
    make_device_vector<segment<T>>({S{P{0.0, 0.0}, P{1.0, 1.0}}, S{P{2.0, 2.0}, P{3.0, 3.0}}});
  auto merged_flag = rmm::device_vector<uint8_t>(segments.size());

  auto expected_segments =
    make_device_vector<segment<T>>({S{P{0.0, 0.0}, P{1.0, 1.0}}, S{P{2.0, 2.0}, P{3.0, 3.0}}});
  auto expected_merged_flag = make_device_vector<uint8_t>({0, 0});

  find_and_combine_segment(range(segment_offsets.begin(), segment_offsets.end()),
                           range(segments.begin(), segments.end()),
                           merged_flag.begin(),
                           this->stream());

  expect_segment_equivalent(segments, expected_segments);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(merged_flag, expected_merged_flag);
}

TYPED_TEST(FindAndCombineSegmentsTest, nooverlap3)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto segment_offsets = make_device_vector<index_t>({0, 2});
  auto segments =
    make_device_vector<segment<T>>({S{P{0.0, 0.0}, P{1.0, 1.0}}, S{P{0.0, 1.0}, P{1.0, 0.0}}});
  auto merged_flag = rmm::device_vector<uint8_t>(segments.size());

  auto expected_segments =
    make_device_vector<segment<T>>({S{P{0.0, 0.0}, P{1.0, 1.0}}, S{P{0.0, 1.0}, P{1.0, 0.0}}});
  auto expected_merged_flag = make_device_vector<uint8_t>({0, 0});

  find_and_combine_segment(range(segment_offsets.begin(), segment_offsets.end()),
                           range(segments.begin(), segments.end()),
                           merged_flag.begin(),
                           this->stream());

  expect_segment_equivalent(segments, expected_segments);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(merged_flag, expected_merged_flag);
}
