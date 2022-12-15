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

#include <cuspatial/experimental/detail/find/find_and_combine_segment.cuh>
#include <cuspatial/experimental/geometry/segment.cuh>
#include <cuspatial/experimental/ranges/range.cuh>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>

using namespace cuspatial;
using namespace cuspatial::detail;
using namespace cuspatial::test;

template <typename OffsetArray, typename CoordinateArray>
class multisegment_array {
 public:
  multisegment_array(OffsetArray offsets, CoordinateArray coordinates)
    : _offsets(offsets), _coordinates(coordinates)
  {
  }

  auto offsets_range() { return range(_offsets.begin(), _offsets.end()); }
  auto coordinates_range() { return range(_coordinates.begin(), _coordinates.end()); }
  auto release() { return std::pair{std::move(_offsets), std::move(_coordinates)}; }

 protected:
  OffsetArray _offsets;
  CoordinateArray _coordinates;
};

template <typename OffsetArray, typename CoordinateArray>
multisegment_array(OffsetArray, CoordinateArray)
  -> multisegment_array<OffsetArray, CoordinateArray>;

template <typename IndexType, typename T>
auto make_segment_array(std::initializer_list<IndexType> offsets,
                        std::initializer_list<segment<T>> segments)
{
  auto d_offsets = make_device_vector(offsets);
  auto d_coords  = make_device_vector(segments);
  return multisegment_array{d_offsets, d_coords};
}

template <typename T>
struct FindAndCombineSegmentsTest : public BaseFixture {
  rmm::cuda_stream_view stream() { return rmm::cuda_stream_default; }

  template <typename MultiSegmentArray>
  void run_single_test(MultiSegmentArray segments,
                       std::initializer_list<uint8_t> expected_flags,
                       std::initializer_list<segment<T>> expected_segment)
  {
    auto d_expected          = make_device_vector(expected_flags);
    auto d_expected_segments = make_device_vector(expected_segment);
    auto flags               = rmm::device_vector<uint8_t>(d_expected.size());
    find_and_combine_segment(
      segments.offsets_range(), segments.coordinates_range(), flags.begin(), this->stream());

    auto [_, merged_segments] = segments.release();

    expect_vec_2d_pair_equivalent(d_expected_segments, merged_segments);
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_expected, flags);
  }
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(FindAndCombineSegmentsTest, TestTypes);

TYPED_TEST(FindAndCombineSegmentsTest, Simple1)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto segments = make_segment_array<index_t, T>(
    {0, 3},
    {S{P{0.0, 0.0}, P{0.0, 0.5}}, S{P{0.0, 0.25}, P{0.0, 0.75}}, S{P{0.0, 0.5}, P{0.0, 1.0}}});

  CUSPATIAL_RUN_TEST(
    this->run_single_test,
    segments,
    {0, 1, 1},
    {S{P{0.0, 0.0}, P{0.0, 1.0}}, S{P{0.0, 0.25}, P{0.0, 0.75}}, S{P{0.0, 0.5}, P{0.0, 1.0}}});
}

TYPED_TEST(FindAndCombineSegmentsTest, Simple2)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto segments = make_segment_array<index_t, T>(
    {0, 3},
    {S{P{0.0, 0.0}, P{0.5, 0.0}}, S{P{0.25, 0.0}, P{0.75, 0.0}}, S{P{0.5, 0.0}, P{1.0, 0.0}}});

  CUSPATIAL_RUN_TEST(
    this->run_single_test,
    segments,
    {0, 1, 1},
    {S{P{0.0, 0.0}, P{1.0, 0.0}}, S{P{0.25, 0.0}, P{0.75, 0.0}}, S{P{0.5, 0.0}, P{1.0, 0.0}}});
}
TYPED_TEST(FindAndCombineSegmentsTest, Simple3)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto segments = make_segment_array<index_t, T>(
    {0, 3},
    {S{P{0.0, 0.0}, P{0.5, 0.5}}, S{P{0.25, 0.25}, P{0.75, 0.75}}, S{P{0.5, 0.5}, P{1.0, 1.0}}});

  CUSPATIAL_RUN_TEST(
    this->run_single_test,
    segments,
    {0, 1, 1},
    {S{P{0.0, 0.0}, P{1.0, 1.0}}, S{P{0.25, 0.25}, P{0.75, 0.75}}, S{P{0.5, 0.5}, P{1.0, 1.0}}});
}

TYPED_TEST(FindAndCombineSegmentsTest, Touching1)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto segments = make_segment_array<index_t, T>(
    {0, 2}, {S{P{0.0, 0.0}, P{0.0, 0.5}}, S{P{0.0, 0.5}, P{0.0, 1.0}}});

  CUSPATIAL_RUN_TEST(this->run_single_test,
                     segments,
                     {0, 1},
                     {S{P{0.0, 0.0}, P{0.0, 1.0}}, S{P{0.0, 0.5}, P{0.0, 1.0}}});
}

TYPED_TEST(FindAndCombineSegmentsTest, Touching2)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto segments = make_segment_array<index_t, T>(
    {0, 2}, {S{P{0.0, 0.0}, P{0.5, 0.0}}, S{P{0.5, 0.0}, P{1.0, 0.0}}});

  CUSPATIAL_RUN_TEST(this->run_single_test,
                     segments,
                     {0, 1},
                     {S{P{0.0, 0.0}, P{1.0, 0.0}}, S{P{0.5, 0.0}, P{1.0, 0.0}}});
}

TYPED_TEST(FindAndCombineSegmentsTest, Touching3)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto segments = make_segment_array<index_t, T>(
    {0, 2}, {S{P{0.0, 0.0}, P{0.5, 0.5}}, S{P{0.5, 0.5}, P{1.0, 1.0}}});

  CUSPATIAL_RUN_TEST(this->run_single_test,
                     segments,
                     {0, 1},
                     {S{P{0.0, 0.0}, P{1.0, 1.0}}, S{P{0.5, 0.5}, P{1.0, 1.0}}});
}

TYPED_TEST(FindAndCombineSegmentsTest, contains1)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto segments = make_segment_array<index_t, T>(
    {0, 2}, {S{P{0.0, 0.0}, P{1.0, 1.0}}, S{P{0.25, 0.25}, P{0.75, 0.75}}});

  CUSPATIAL_RUN_TEST(this->run_single_test,
                     segments,
                     {0, 1},
                     {S{P{0.0, 0.0}, P{1.0, 1.0}}, S{P{0.25, 0.25}, P{0.75, 0.75}}});
}

TYPED_TEST(FindAndCombineSegmentsTest, contains2)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto segments = make_segment_array<index_t, T>(
    {0, 2}, {S{P{0.0, 0.0}, P{0.0, 1.0}}, S{P{0.0, 0.25}, P{0.0, 0.75}}});

  CUSPATIAL_RUN_TEST(this->run_single_test,
                     segments,
                     {0, 1},
                     {S{P{0.0, 0.0}, P{0.0, 1.0}}, S{P{0.0, 0.25}, P{0.0, 0.75}}});
}

TYPED_TEST(FindAndCombineSegmentsTest, contains3)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto segments = make_segment_array<index_t, T>(
    {0, 2}, {S{P{0.0, 0.0}, P{1.0, 0.0}}, S{P{0.25, 0.0}, P{0.75, 0.0}}});

  CUSPATIAL_RUN_TEST(this->run_single_test,
                     segments,
                     {0, 1},
                     {S{P{0.0, 0.0}, P{1.0, 0.0}}, S{P{0.25, 0.0}, P{0.75, 0.0}}});
}

TYPED_TEST(FindAndCombineSegmentsTest, nooverlap1)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto segments = make_segment_array<index_t, T>(
    {0, 2}, {S{P{0.0, 0.0}, P{1.0, 0.0}}, S{P{0.0, 1.0}, P{0.0, 2.0}}});

  CUSPATIAL_RUN_TEST(this->run_single_test,
                     segments,
                     {0, 0},
                     {S{P{0.0, 0.0}, P{1.0, 0.0}}, S{P{0.0, 1.0}, P{0.0, 2.0}}});
}

TYPED_TEST(FindAndCombineSegmentsTest, nooverlap2)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto segments = make_segment_array<index_t, T>(
    {0, 2}, {S{P{0.0, 0.0}, P{1.0, 1.0}}, S{P{2.0, 2.0}, P{3.0, 3.0}}});

  CUSPATIAL_RUN_TEST(this->run_single_test,
                     segments,
                     {0, 0},
                     {S{P{0.0, 0.0}, P{1.0, 1.0}}, S{P{2.0, 2.0}, P{3.0, 3.0}}});
}

TYPED_TEST(FindAndCombineSegmentsTest, nooverlap3)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto segments = make_segment_array<index_t, T>(
    {0, 2}, {S{P{0.0, 0.0}, P{1.0, 1.0}}, S{P{0.0, 1.0}, P{1.0, 0.0}}});

  CUSPATIAL_RUN_TEST(this->run_single_test,
                     segments,
                     {0, 0},
                     {S{P{0.0, 0.0}, P{1.0, 1.0}}, S{P{0.0, 1.0}, P{1.0, 0.0}}});
}
