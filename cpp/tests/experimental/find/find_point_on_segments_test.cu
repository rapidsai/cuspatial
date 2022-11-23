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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>

#include <cuspatial/experimental/detail/find/find_point_on_segments.cuh>
#include <cuspatial/experimental/ranges/range.cuh>
#include <cuspatial/vec_2d.hpp>

using namespace cuspatial;
using namespace cuspatial::detail;
using namespace cuspatial::test;

template <typename T>
struct FindPointOnSegmentTest : public BaseFixture {
  rmm::cuda_stream_view stream() { return rmm::cuda_stream_default; }
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(FindPointOnSegmentTest, TestTypes);

TYPED_TEST(FindPointOnSegmentTest, Simple1)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto multipoints = make_multipoints_array({{P{0.0, 0.0}, P{1.0, 0.0}}});

  auto segment_offsets = make_device_vector<index_t>({0, 1});
  auto segments        = make_device_vector<segment<T>>({S{P{1.0, 1.0}, P{1.0, -1.0}}});

  rmm::device_vector<uint8_t> flags(multipoints.range().num_points());
  std::vector<uint8_t> expected_flags{0, 1};

  find_point_on_segments(multipoints.range(),
                         range(segment_offsets.begin(), segment_offsets.end()),
                         range(segments.begin(), segments.end()),
                         flags.begin(),
                         this->stream());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(flags, expected_flags);
}

TYPED_TEST(FindPointOnSegmentTest, Simple2)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto multipoints = make_multipoints_array({{P{0.0, 0.0}, P{1.0, 1.0}}});

  auto segment_offsets = make_device_vector<index_t>({0, 1});
  auto segments        = make_device_vector<segment<T>>({S{P{2.0, 0.0}, P{0.0, 2.0}}});

  rmm::device_vector<uint8_t> flags(multipoints.range().num_points());
  std::vector<uint8_t> expected_flags{0, 1};

  find_point_on_segments(multipoints.range(),
                         range(segment_offsets.begin(), segment_offsets.end()),
                         range(segments.begin(), segments.end()),
                         flags.begin(),
                         this->stream());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(flags, expected_flags);
}

TYPED_TEST(FindPointOnSegmentTest, Simple3)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto multipoints = make_multipoints_array({{P{0.0, 0.0}, P{1.0, 1.0}}});

  auto segment_offsets = make_device_vector<index_t>({0, 1});
  auto segments        = make_device_vector<segment<T>>({S{P{0.0, 1.0}, P{1.0, 1.0}}});

  rmm::device_vector<uint8_t> flags(multipoints.range().num_points());
  std::vector<uint8_t> expected_flags{0, 1};

  find_point_on_segments(multipoints.range(),
                         range(segment_offsets.begin(), segment_offsets.end()),
                         range(segments.begin(), segments.end()),
                         flags.begin(),
                         this->stream());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(flags, expected_flags);
}

TYPED_TEST(FindPointOnSegmentTest, Simple4)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto multipoints = make_multipoints_array({{P{0.0, 0.0}, P{1.0, 1.0}}});

  auto segment_offsets = make_device_vector<index_t>({0, 1});
  auto segments        = make_device_vector<segment<T>>({S{P{1.0, 1.0}, P{1.0, 0.0}}});

  rmm::device_vector<uint8_t> flags(multipoints.range().num_points());
  std::vector<uint8_t> expected_flags{0, 1};

  find_point_on_segments(multipoints.range(),
                         range(segment_offsets.begin(), segment_offsets.end()),
                         range(segments.begin(), segments.end()),
                         flags.begin(),
                         this->stream());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(flags, expected_flags);
}

TYPED_TEST(FindPointOnSegmentTest, NoPointOnSegment1)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto multipoints = make_multipoints_array({{P{0.0, 0.0}, P{1.0, 1.0}}});

  auto segment_offsets = make_device_vector<index_t>({0, 1});
  auto segments        = make_device_vector<segment<T>>({S{P{0.0, 0.5}, P{1.0, 0.0}}});

  rmm::device_vector<uint8_t> flags(multipoints.range().num_points());
  std::vector<uint8_t> expected_flags{0, 0};

  find_point_on_segments(multipoints.range(),
                         range(segment_offsets.begin(), segment_offsets.end()),
                         range(segments.begin(), segments.end()),
                         flags.begin(),
                         this->stream());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(flags, expected_flags);
}

TYPED_TEST(FindPointOnSegmentTest, NoPointOnSegment2)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto multipoints = make_multipoints_array({{P{0.0, 0.0}, P{1.0, 1.0}}});

  auto segment_offsets = make_device_vector<index_t>({0, 1});
  auto segments        = make_device_vector<segment<T>>({S{P{2.0, 2.0}, P{3.0, 3.0}}});

  rmm::device_vector<uint8_t> flags(multipoints.range().num_points());
  std::vector<uint8_t> expected_flags{0, 0};

  find_point_on_segments(multipoints.range(),
                         range(segment_offsets.begin(), segment_offsets.end()),
                         range(segments.begin(), segments.end()),
                         flags.begin(),
                         this->stream());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(flags, expected_flags);
}

TYPED_TEST(FindPointOnSegmentTest, TwoPairs)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto multipoints = make_multipoints_array({{P{0.0, 0.0}, P{1.0, 1.0}}, {P{2.0, 2.0}}});

  auto segment_offsets = make_device_vector<index_t>({0, 1, 2});
  auto segments =
    make_device_vector<segment<T>>({S{P{2.0, 2.0}, P{3.0, 3.0}}, S{P{1.0, 3.0}, P{3.0, 1.0}}});

  rmm::device_vector<uint8_t> flags(multipoints.range().num_points());
  std::vector<uint8_t> expected_flags{0, 0, 1};

  find_point_on_segments(multipoints.range(),
                         range(segment_offsets.begin(), segment_offsets.end()),
                         range(segments.begin(), segments.end()),
                         flags.begin(),
                         this->stream());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(flags, expected_flags);
}
