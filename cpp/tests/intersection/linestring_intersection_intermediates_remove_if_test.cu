/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <cuspatial/detail/intersection/linestring_intersection_with_duplicates.cuh>
#include <cuspatial/error.hpp>
#include <cuspatial/geometry/vec_2d.hpp>
#include <cuspatial/iterator_factory.cuh>
#include <cuspatial/range/range.cuh>
#include <cuspatial/traits.hpp>

#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/iterator/zip_iterator.h>

#include <initializer_list>
#include <type_traits>

using namespace cuspatial;
using namespace cuspatial::test;

template <typename IndexType, typename Geom>
auto make_linestring_intersection_intermediates(std::initializer_list<IndexType> offsets,
                                                std::initializer_list<Geom> geoms,
                                                std::initializer_list<IndexType> lhs_linestring_ids,
                                                std::initializer_list<IndexType> lhs_segment_ids,
                                                std::initializer_list<IndexType> rhs_linestring_ids,
                                                std::initializer_list<IndexType> rhs_segment_ids,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr)
{
  auto d_offset             = make_device_uvector<IndexType>(offsets, stream, mr);
  auto d_geoms              = make_device_uvector<Geom>(geoms, stream, mr);
  auto d_lhs_linestring_ids = make_device_uvector<IndexType>(lhs_linestring_ids, stream, mr);
  auto d_lhs_segment_ids    = make_device_uvector<IndexType>(lhs_segment_ids, stream, mr);
  auto d_rhs_linestring_ids = make_device_uvector<IndexType>(rhs_linestring_ids, stream, mr);
  auto d_rhs_segment_ids    = make_device_uvector<IndexType>(rhs_segment_ids, stream, mr);

  return linestring_intersection_intermediates<Geom, IndexType>{
    std::make_unique<rmm::device_uvector<IndexType>>(d_offset, stream),
    std::make_unique<rmm::device_uvector<Geom>>(d_geoms, stream),
    std::make_unique<rmm::device_uvector<IndexType>>(d_lhs_linestring_ids, stream),
    std::make_unique<rmm::device_uvector<IndexType>>(d_lhs_segment_ids, stream),
    std::make_unique<rmm::device_uvector<IndexType>>(d_rhs_linestring_ids, stream),
    std::make_unique<rmm::device_uvector<IndexType>>(d_rhs_segment_ids, stream),
  };
}

template <typename T>
struct LinestringIntersectionIntermediatesRemoveIfTest : public ::testing::Test {
  rmm::cuda_stream_view stream() { return rmm::cuda_stream_default; }
  rmm::device_async_resource_ref mr() { return rmm::mr::get_current_device_resource(); }

  template <typename FlagType, typename IntermediateType>
  void run_single(IntermediateType& intermediates,
                  std::initializer_list<FlagType> flags,
                  IntermediateType const& expected)
  {
    using GeomType = typename IntermediateType::geometry_t;
    auto d_flags   = make_device_uvector<FlagType>(flags, this->stream(), this->mr());
    intermediates.remove_if(range(d_flags.begin(), d_flags.end()), this->stream());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(*intermediates.offsets, *expected.offsets);
    if constexpr (cuspatial::is_vec_2d<GeomType>)
      CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(*intermediates.geoms, *expected.geoms);
    else
      CUSPATIAL_EXPECT_VEC2D_PAIRS_EQUIVALENT(*intermediates.geoms, *expected.geoms);
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(*intermediates.lhs_linestring_ids,
                                        *expected.lhs_linestring_ids);
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(*intermediates.lhs_segment_ids, *expected.lhs_segment_ids);
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(*intermediates.rhs_linestring_ids,
                                        *expected.rhs_linestring_ids);
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(*intermediates.rhs_segment_ids, *expected.rhs_segment_ids);
  }
};

// float and double are logically the same but would require separate tests due to precision.
using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(LinestringIntersectionIntermediatesRemoveIfTest, TestTypes);

TYPED_TEST(LinestringIntersectionIntermediatesRemoveIfTest, EmptyInput)
{
  using T       = TypeParam;
  using P       = vec_2d<T>;
  using index_t = std::size_t;
  using flag_t  = uint8_t;

  auto intermediates = make_linestring_intersection_intermediates<index_t, P>(
    {0}, {}, {}, {}, {}, {}, this->stream(), this->mr());

  auto expected = make_linestring_intersection_intermediates<index_t, P>(
    {0}, {}, {}, {}, {}, {}, this->stream(), this->mr());

  CUSPATIAL_RUN_TEST(this->template run_single<flag_t>, intermediates, {}, expected);
}

TYPED_TEST(LinestringIntersectionIntermediatesRemoveIfTest, OnePairOnePointNotRemoved)
{
  using T       = TypeParam;
  using P       = vec_2d<T>;
  using index_t = std::size_t;
  using flag_t  = uint8_t;

  auto intermediates = make_linestring_intersection_intermediates<index_t, P>(
    {0, 1}, {{0.5, 0.5}}, {0}, {0}, {0}, {0}, this->stream(), this->mr());

  auto expected = make_linestring_intersection_intermediates<index_t, P>(
    {0, 1}, {{0.5, 0.5}}, {0}, {0}, {0}, {0}, this->stream(), this->mr());

  CUSPATIAL_RUN_TEST(this->template run_single<flag_t>, intermediates, {0}, expected);
}

TYPED_TEST(LinestringIntersectionIntermediatesRemoveIfTest, OnePairOnePointRemoved)
{
  using T       = TypeParam;
  using P       = vec_2d<T>;
  using index_t = std::size_t;
  using flag_t  = uint8_t;

  auto intermediates = make_linestring_intersection_intermediates<index_t, P>(
    {0, 1}, {{0.5, 0.5}}, {0}, {0}, {0}, {0}, this->stream(), this->mr());

  auto expected = make_linestring_intersection_intermediates<index_t, P>(
    {0, 0}, {}, {}, {}, {}, {}, this->stream(), this->mr());

  CUSPATIAL_RUN_TEST(this->template run_single<flag_t>, intermediates, {1}, expected);
}

TYPED_TEST(LinestringIntersectionIntermediatesRemoveIfTest, SimplePoint)
{
  using T       = TypeParam;
  using P       = vec_2d<T>;
  using index_t = std::size_t;
  using flag_t  = uint8_t;

  auto intermediates = make_linestring_intersection_intermediates<index_t, P>(
    {0, 2, 4},
    {{0.0, 0.0}, {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}},
    {0, 1, 2, 3},
    {0, 1, 2, 3},
    {0, 1, 2, 3},
    {0, 1, 2, 3},
    this->stream(),
    this->mr());

  auto expected = make_linestring_intersection_intermediates<index_t, P>({0, 1, 2},
                                                                         {{0.0, 0.0}, {2.0, 2.0}},
                                                                         {0, 2},
                                                                         {0, 2},
                                                                         {0, 2},
                                                                         {0, 2},
                                                                         this->stream(),
                                                                         this->mr());

  CUSPATIAL_RUN_TEST(this->template run_single<flag_t>, intermediates, {0, 1, 0, 1}, expected);
}

TYPED_TEST(LinestringIntersectionIntermediatesRemoveIfTest, SimplePoint2)
{
  using T       = TypeParam;
  using P       = vec_2d<T>;
  using index_t = std::size_t;
  using flag_t  = uint8_t;

  auto intermediates = make_linestring_intersection_intermediates<index_t, P>(
    {0, 2}, {{0.0, 0.5}, {0.5, 0.5}}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, this->stream(), this->mr());

  auto expected = make_linestring_intersection_intermediates<index_t, P>(
    {0, 1}, {{0.0, 0.5}}, {0}, {0}, {0}, {0}, this->stream(), this->mr());

  CUSPATIAL_RUN_TEST(this->template run_single<flag_t>, intermediates, {0, 1}, expected);
}

TYPED_TEST(LinestringIntersectionIntermediatesRemoveIfTest, SimpleSegment1)
{
  using T       = TypeParam;
  using S       = segment<T>;
  using index_t = std::size_t;
  using flag_t  = uint8_t;

  auto intermediates = make_linestring_intersection_intermediates<index_t, S>(
    {0, 2},
    {S{{0.0, 0.5}, {0.5, 0.5}}, S{{0.0, 0.0}, {1.0, 1.0}}},
    {0, 0},
    {0, 0},
    {0, 0},
    {0, 0},
    this->stream(),
    this->mr());

  auto expected = make_linestring_intersection_intermediates<index_t, S>(
    {0, 1}, {S{{0.0, 0.5}, {0.5, 0.5}}}, {0}, {0}, {0}, {0}, this->stream(), this->mr());

  CUSPATIAL_RUN_TEST(this->template run_single<flag_t>, intermediates, {0, 1}, expected);
}

TYPED_TEST(LinestringIntersectionIntermediatesRemoveIfTest, SimpleSegment2)
{
  using T       = TypeParam;
  using S       = segment<T>;
  using index_t = std::size_t;
  using flag_t  = uint8_t;

  auto intermediates = make_linestring_intersection_intermediates<index_t, S>(
    {0, 2},
    {S{{0.0, 0.5}, {0.5, 0.5}}, S{{0.0, 0.0}, {1.0, 1.0}}},
    {0, 0},
    {0, 0},
    {0, 0},
    {0, 0},
    this->stream(),
    this->mr());

  auto expected = make_linestring_intersection_intermediates<index_t, S>(
    {0, 0}, {}, {}, {}, {}, {}, this->stream(), this->mr());

  CUSPATIAL_RUN_TEST(this->template run_single<flag_t>, intermediates, {1, 1}, expected);
}

TYPED_TEST(LinestringIntersectionIntermediatesRemoveIfTest, FourSegmentsKeep)
{
  using T       = TypeParam;
  using S       = segment<T>;
  using index_t = std::size_t;
  using flag_t  = uint8_t;

  auto intermediates =
    make_linestring_intersection_intermediates<index_t, S>({0, 0, 0, 1, 3, 4, 4, 4},
                                                           {S{{0.5, 0.5}, {1, 1}},
                                                            S{{0, 0}, {0.25, 0.25}},
                                                            S{{0.75, 0.75}, {1, 1}},
                                                            S{{0.75, 0.75}, {1, 1}}},
                                                           {0, 0, 0, 0},
                                                           {0, 0, 0, 0},
                                                           {0, 0, 0, 0},
                                                           {0, 0, 3, 2},
                                                           this->stream(),
                                                           this->mr());

  auto expected = make_linestring_intersection_intermediates<index_t, S>({0, 0, 0, 1, 3, 4, 4, 4},
                                                                         {S{{0.5, 0.5}, {1, 1}},
                                                                          S{{0, 0}, {0.25, 0.25}},
                                                                          S{{0.75, 0.75}, {1, 1}},
                                                                          S{{0.75, 0.75}, {1, 1}}},
                                                                         {0, 0, 0, 0},
                                                                         {0, 0, 0, 0},
                                                                         {0, 0, 0, 0},
                                                                         {0, 0, 3, 2},
                                                                         this->stream(),
                                                                         this->mr());

  CUSPATIAL_RUN_TEST(this->template run_single<flag_t>, intermediates, {0, 0, 0, 0}, expected);
}

TYPED_TEST(LinestringIntersectionIntermediatesRemoveIfTest, FourSegmentsRemoveOne1)
{
  using T       = TypeParam;
  using S       = segment<T>;
  using index_t = std::size_t;
  using flag_t  = uint8_t;

  auto intermediates =
    make_linestring_intersection_intermediates<index_t, S>({0, 0, 0, 1, 3, 4, 4, 4},
                                                           {S{{0.5, 0.5}, {1, 1}},
                                                            S{{0, 0}, {0.25, 0.25}},
                                                            S{{0.75, 0.75}, {1, 1}},
                                                            S{{0.75, 0.75}, {1, 1}}},
                                                           {0, 0, 0, 0},
                                                           {0, 0, 0, 0},
                                                           {0, 0, 0, 0},
                                                           {0, 0, 3, 2},
                                                           this->stream(),
                                                           this->mr());

  auto expected = make_linestring_intersection_intermediates<index_t, S>(
    {0, 0, 0, 1, 2, 3, 3, 3},
    {S{{0.5, 0.5}, {1, 1}}, S{{0.75, 0.75}, {1, 1}}, S{{0.75, 0.75}, {1, 1}}},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 3, 2},
    this->stream(),
    this->mr());

  CUSPATIAL_RUN_TEST(this->template run_single<flag_t>, intermediates, {0, 1, 0, 0}, expected);
}

TYPED_TEST(LinestringIntersectionIntermediatesRemoveIfTest, FourSegmentsRemoveOne2)
{
  using T       = TypeParam;
  using S       = segment<T>;
  using index_t = std::size_t;
  using flag_t  = uint8_t;

  auto intermediates =
    make_linestring_intersection_intermediates<index_t, S>({0, 0, 0, 1, 3, 4, 4, 4},
                                                           {S{{0.5, 0.5}, {1, 1}},
                                                            S{{0, 0}, {0.25, 0.25}},
                                                            S{{0.75, 0.75}, {1, 1}},
                                                            S{{0.75, 0.75}, {1, 1}}},
                                                           {0, 0, 0, 0},
                                                           {0, 0, 0, 0},
                                                           {0, 0, 0, 0},
                                                           {0, 0, 3, 2},
                                                           this->stream(),
                                                           this->mr());

  auto expected = make_linestring_intersection_intermediates<index_t, S>(
    {0, 0, 0, 0, 2, 3, 3, 3},
    {S{{0, 0}, {0.25, 0.25}}, S{{0.75, 0.75}, {1, 1}}, S{{0.75, 0.75}, {1, 1}}},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 3, 2},
    this->stream(),
    this->mr());

  CUSPATIAL_RUN_TEST(this->template run_single<flag_t>, intermediates, {1, 0, 0, 0}, expected);
}

TYPED_TEST(LinestringIntersectionIntermediatesRemoveIfTest, FourSegmentsRemoveOne3)
{
  using T       = TypeParam;
  using S       = segment<T>;
  using index_t = std::size_t;
  using flag_t  = uint8_t;

  auto intermediates =
    make_linestring_intersection_intermediates<index_t, S>({0, 0, 0, 1, 3, 4, 4, 4},
                                                           {S{{0.5, 0.5}, {1, 1}},
                                                            S{{0, 0}, {0.25, 0.25}},
                                                            S{{0.75, 0.75}, {1, 1}},
                                                            S{{0.75, 0.75}, {1, 1}}},
                                                           {0, 0, 0, 0},
                                                           {0, 0, 0, 0},
                                                           {0, 0, 0, 0},
                                                           {0, 0, 3, 2},
                                                           this->stream(),
                                                           this->mr());

  auto expected = make_linestring_intersection_intermediates<index_t, S>(
    {0, 0, 0, 1, 3, 3, 3, 3},
    {S{{0.5, 0.5}, {1, 1}}, S{{0, 0}, {0.25, 0.25}}, S{{0.75, 0.75}, {1, 1}}},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 3},
    this->stream(),
    this->mr());

  CUSPATIAL_RUN_TEST(this->template run_single<flag_t>, intermediates, {0, 0, 0, 1}, expected);
}

TYPED_TEST(LinestringIntersectionIntermediatesRemoveIfTest, FourSegmentsRemoveOne4)
{
  using T       = TypeParam;
  using S       = segment<T>;
  using index_t = std::size_t;
  using flag_t  = uint8_t;

  auto intermediates =
    make_linestring_intersection_intermediates<index_t, S>({0, 0, 0, 1, 3, 4, 4, 4},
                                                           {S{{0.5, 0.5}, {1, 1}},
                                                            S{{0, 0}, {0.25, 0.25}},
                                                            S{{0.75, 0.75}, {1, 1}},
                                                            S{{0.75, 0.75}, {1, 1}}},
                                                           {0, 0, 0, 0},
                                                           {0, 0, 0, 0},
                                                           {0, 0, 0, 0},
                                                           {0, 0, 3, 2},
                                                           this->stream(),
                                                           this->mr());

  auto expected = make_linestring_intersection_intermediates<index_t, S>(
    {0, 0, 0, 1, 2, 3, 3, 3},
    {S{{0.5, 0.5}, {1, 1}}, S{{0, 0}, {0.25, 0.25}}, S{{0.75, 0.75}, {1, 1}}},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 2},
    this->stream(),
    this->mr());

  CUSPATIAL_RUN_TEST(this->template run_single<flag_t>, intermediates, {0, 0, 1, 0}, expected);
}

TYPED_TEST(LinestringIntersectionIntermediatesRemoveIfTest, FourSegmentsRemoveTwo1)
{
  using T       = TypeParam;
  using S       = segment<T>;
  using index_t = std::size_t;
  using flag_t  = uint8_t;

  auto intermediates =
    make_linestring_intersection_intermediates<index_t, S>({0, 0, 0, 1, 3, 4, 4, 4},
                                                           {S{{0.5, 0.5}, {1, 1}},
                                                            S{{0, 0}, {0.25, 0.25}},
                                                            S{{0.75, 0.75}, {1, 1}},
                                                            S{{0.75, 0.75}, {1, 1}}},
                                                           {0, 0, 0, 0},
                                                           {0, 0, 0, 0},
                                                           {0, 0, 0, 0},
                                                           {0, 0, 3, 2},
                                                           this->stream(),
                                                           this->mr());

  auto expected = make_linestring_intersection_intermediates<index_t, S>(
    {0, 0, 0, 1, 1, 2, 2, 2},
    {S{{0.5, 0.5}, {1, 1}}, S{{0.75, 0.75}, {1, 1}}},
    {0, 0},
    {0, 0},
    {0, 0},
    {0, 2},
    this->stream(),
    this->mr());

  CUSPATIAL_RUN_TEST(this->template run_single<flag_t>, intermediates, {0, 1, 1, 0}, expected);
}
