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

#include <cuspatial/experimental/detail/combine/combine_segments.cuh>
#include <cuspatial/experimental/geometry/segment.cuh>
#include <cuspatial/experimental/ranges/range.cuh>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>

using namespace cuspatial;
using namespace cuspatial::detail;
using namespace cuspatial::test;

template <typename T>
struct CombineSegmentsTest : public BaseFixture {
  rmm::cuda_stream_view stream() { return rmm::cuda_stream_default; }
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(CombineSegmentsTest, TestTypes);

TYPED_TEST(CombineSegmentsTest, simple)
{
  using T       = TypeParam;
  using index_t = std::size_t;
  using P       = vec_2d<T>;
  using S       = segment<T>;

  auto segment_offsets = make_device_vector<index_t>({0, 3});
  auto segments        = make_device_vector<segment<T>>(
    {S{P{0.0, 0.0}, P{0.0, 0.5}}, S{P{0.0, 0.25}, P{0.0, 0.75}}, S{P{0.0, 0.5}, P{0.0, 1.0}}});
  auto stencils = rmm::device_vector<uint8_t>(segments.size());

  auto expected_segments = make_device_vector<segment<T>>(
    {S{P{0.0, 0.0}, P{0.0, 1.0}}, S{P{0.0, 0.25}, P{0.0, 0.75}}, S{P{0.0, 0.5}, P{0.0, 1.0}}});
  auto expected_stencil = make_device_vector<uint8_t>({0, 1, 1});

  combine_segments(range(segment_offsets.begin(), segment_offsets.end()),
                   range(segments.begin(), segments.end()),
                   stencils.begin(),
                   this->stream());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(segments, expected_segments);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(stencils, expected_stencil);
}
