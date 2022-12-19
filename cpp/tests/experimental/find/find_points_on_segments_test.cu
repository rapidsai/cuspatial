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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>

#include <cuspatial/experimental/detail/find/find_points_on_segments.cuh>
#include <cuspatial/experimental/ranges/range.cuh>
#include <cuspatial/vec_2d.hpp>

#include <initializer_list>

using namespace cuspatial;
using namespace cuspatial::detail;
using namespace cuspatial::test;

template <typename T>
struct FindPointOnSegmentTest : public BaseFixture {
  rmm::cuda_stream_view stream() { return rmm::cuda_stream_default; }

  template <typename IndexType>
  void run_single(std::initializer_list<std::initializer_list<vec_2d<T>>> multipoints,
                  std::initializer_list<IndexType> segment_offsets,
                  std::initializer_list<segment<T>> segments,
                  std::initializer_list<uint8_t> expected_flags)
  {
    auto d_multipoints = make_multipoints_array(multipoints);

    auto d_segment_offsets = make_device_vector<IndexType>(segment_offsets);
    auto d_segments        = make_device_vector<segment<T>>(segments);

    rmm::device_vector<uint8_t> d_flags(d_multipoints.range().num_points());

    find_points_on_segments(d_multipoints.range(),
                            range(d_segment_offsets.begin(), d_segment_offsets.end()),
                            range(d_segments.begin(), d_segments.end()),
                            d_flags.begin(),
                            this->stream());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_flags, std::vector<uint8_t>(expected_flags));
  }
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(FindPointOnSegmentTest, TestTypes);

TYPED_TEST(FindPointOnSegmentTest, VerticalSegment1)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  CUSPATIAL_RUN_TEST(this->template run_single<index_t>,
                     {{P{0.0, 0.0}, P{1.0, 0.0}}},
                     {0, 1},
                     {S{P{1.0, 1.0}, P{1.0, -1.0}}},
                     {0, 1});
}

TYPED_TEST(FindPointOnSegmentTest, VerticalSegment2)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  CUSPATIAL_RUN_TEST(this->template run_single<index_t>,
                     {{P{0.0, 0.0}, P{2.0, 0.0}}},
                     {0, 1},
                     {S{P{1.0, 1.0}, P{1.0, -1.0}}},
                     {0, 0});
}

TYPED_TEST(FindPointOnSegmentTest, VerticalSegment3)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  CUSPATIAL_RUN_TEST(this->template run_single<index_t>,
                     {{P{1.0, 0.0}, P{2.0, 0.0}}},
                     {0, 1},
                     {S{P{1.0, 1.0}, P{1.0, -1.0}}},
                     {1, 0});
}

TYPED_TEST(FindPointOnSegmentTest, VerticalSegment4)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  CUSPATIAL_RUN_TEST(this->template run_single<index_t>,
                     {{P{1.0, 0.0}, P{1.0, 0.5}}},
                     {0, 1},
                     {S{P{1.0, 1.0}, P{1.0, -1.0}}},
                     {1, 1});
}

TYPED_TEST(FindPointOnSegmentTest, DiagnalSegment1)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  CUSPATIAL_RUN_TEST(this->template run_single<index_t>,
                     {{P{0.0, 0.0}, P{1.0, 1.0}}},
                     {0, 1},
                     {S{P{2.0, 0.0}, P{0.0, 2.0}}},
                     {0, 1});
}

TYPED_TEST(FindPointOnSegmentTest, DiagnalSegment2)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  CUSPATIAL_RUN_TEST(this->template run_single<index_t>,
                     {{P{0.0, 0.0}, P{1.0, 0.0}}},
                     {0, 1},
                     {S{P{2.0, 0.0}, P{0.0, 2.0}}},
                     {0, 0});
}

TYPED_TEST(FindPointOnSegmentTest, DiagnalSegment3)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  CUSPATIAL_RUN_TEST(this->template run_single<index_t>,
                     {{P{1.0, 1.0}, P{1.0, 0.0}}},
                     {0, 1},
                     {S{P{2.0, 0.0}, P{0.0, 2.0}}},
                     {1, 0});
}

TYPED_TEST(FindPointOnSegmentTest, DiagnalSegment4)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  CUSPATIAL_RUN_TEST(this->template run_single<index_t>,
                     {{P{1.0, 1.0}, P{1.5, 0.5}}},
                     {0, 1},
                     {S{P{2.0, 0.0}, P{0.0, 2.0}}},
                     {1, 1});
}

TYPED_TEST(FindPointOnSegmentTest, HorizontalSegment1)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  CUSPATIAL_RUN_TEST(this->template run_single<index_t>,
                     {{P{0.0, 0.0}, P{1.0, 1.0}}},
                     {0, 1},
                     {S{P{0.0, 1.0}, P{1.0, 1.0}}},
                     {0, 1});
}

TYPED_TEST(FindPointOnSegmentTest, HorizontalSegment2)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  CUSPATIAL_RUN_TEST(this->template run_single<index_t>,
                     {{P{0.0, 0.0}, P{2.0, 1.0}}},
                     {0, 1},
                     {S{P{0.0, 1.0}, P{1.0, 1.0}}},
                     {0, 0});
}

TYPED_TEST(FindPointOnSegmentTest, HorizontalSegment3)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  CUSPATIAL_RUN_TEST(this->template run_single<index_t>,
                     {{P{0.5, 1.0}, P{2.0, 1.0}}},
                     {0, 1},
                     {S{P{0.0, 1.0}, P{1.0, 1.0}}},
                     {1, 0});
}

TYPED_TEST(FindPointOnSegmentTest, HorizontalSegment4)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  CUSPATIAL_RUN_TEST(this->template run_single<index_t>,
                     {{P{0.5, 1.0}, P{0.75, 1.0}}},
                     {0, 1},
                     {S{P{0.0, 1.0}, P{2.0, 1.0}}},
                     {1, 1});
}

TYPED_TEST(FindPointOnSegmentTest, OnVertex)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  CUSPATIAL_RUN_TEST(this->template run_single<index_t>,
                     {{P{0.0, 0.0}, P{1.0, 1.0}}},
                     {0, 1},
                     {S{P{1.0, 1.0}, P{1.0, 0.0}}},
                     {0, 1});
}

TYPED_TEST(FindPointOnSegmentTest, NoPointOnSegment1)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  CUSPATIAL_RUN_TEST(this->template run_single<index_t>,
                     {{P{0.0, 0.0}, P{1.0, 1.0}}},
                     {0, 1},
                     {S{P{0.0, 0.5}, P{1.0, 0.0}}},
                     {0, 0});
}

TYPED_TEST(FindPointOnSegmentTest, NoPointOnSegment2)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  CUSPATIAL_RUN_TEST(this->template run_single<index_t>,
                     {{P{0.0, 0.0}, P{1.0, 1.0}}},
                     {0, 1},
                     {S{P{2.0, 2.0}, P{3.0, 3.0}}},
                     {0, 0});
}

TYPED_TEST(FindPointOnSegmentTest, TwoPairs)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  CUSPATIAL_RUN_TEST(this->template run_single<index_t>,
                     {{P{0.0, 0.0}, P{1.0, 1.0}}, {P{2.0, 2.0}}},
                     {0, 1, 2},
                     {S{P{2.0, 2.0}, P{3.0, 3.0}}, S{P{1.0, 3.0}, P{3.0, 1.0}}},
                     {0, 0, 1});
}
