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

#include <cuspatial/column/geometry_column_view.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/geometry/vec_2d.hpp>
#include <cuspatial/intersection.hpp>
#include <cuspatial/types.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <initializer_list>
#include <memory>
#include <optional>
#include <utility>

using namespace cuspatial;
using namespace cuspatial::test;

using namespace cudf;
using namespace cudf::test;

template <typename T>
using wrapper = cudf::test::fixed_width_column_wrapper<T>;

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::ALL_ERRORS};

// helper function to make a linestring column
template <typename T>
std::pair<collection_type_id, std::unique_ptr<cudf::column>> make_linestring_column(
  std::initializer_list<cudf::size_type>&& linestring_offsets,
  std::initializer_list<T>&& linestring_coords,
  rmm::cuda_stream_view stream)
{
  auto num_points = linestring_coords.size() / 2;
  auto size       = linestring_offsets.size() - 1;

  auto zero           = make_fixed_width_scalar<size_type>(0, stream);
  auto two            = make_fixed_width_scalar<size_type>(2, stream);
  auto offsets_column = wrapper<cudf::size_type>(linestring_offsets).release();
  auto coords_offset  = cudf::sequence(num_points + 1, *zero, *two);
  auto coords_column  = wrapper<T>(linestring_coords).release();

  return {collection_type_id::SINGLE,
          cudf::make_lists_column(
            size,
            std::move(offsets_column),
            cudf::make_lists_column(
              num_points, std::move(coords_offset), std::move(coords_column), 0, {}),
            0,
            {})};
}

// helper function to make a multilinestring column
template <typename T>
std::pair<collection_type_id, std::unique_ptr<cudf::column>> make_linestring_column(
  std::initializer_list<cudf::size_type>&& multilinestring_offsets,
  std::initializer_list<cudf::size_type>&& linestring_offsets,
  std::initializer_list<T> linestring_coords,
  rmm::cuda_stream_view stream)
{
  auto geometry_size = multilinestring_offsets.size() - 1;
  auto part_size     = linestring_offsets.size() - 1;
  auto num_points    = linestring_coords.size() / 2;

  auto zero            = make_fixed_width_scalar<size_type>(0, stream);
  auto two             = make_fixed_width_scalar<size_type>(2, stream);
  auto geometry_column = wrapper<cudf::size_type>(multilinestring_offsets).release();
  auto part_column     = wrapper<cudf::size_type>(linestring_offsets).release();
  auto coords_offset   = cudf::sequence(num_points + 1, *zero, *two);
  auto coord_column    = wrapper<T>(linestring_coords).release();

  return {collection_type_id::MULTI,
          cudf::make_lists_column(
            geometry_size,
            std::move(geometry_column),
            cudf::make_lists_column(
              part_size,
              std::move(part_column),
              cudf::make_lists_column(
                num_points, std::move(coords_offset), std::move(coord_column), 0, {}),
              0,
              {}),
            0,
            {})};
}

struct LinestringIntersectionTestBase : public BaseFixture {
  rmm::cuda_stream_view stream{rmm::cuda_stream_default};
  rmm::device_async_resource_ref mr{rmm::mr::get_current_device_resource()};
};

template <typename T>
struct LinestringIntersectionTest : public LinestringIntersectionTestBase {
  void run_single(geometry_column_view lhs,
                  geometry_column_view rhs,
                  linestring_intersection_column_result const& expected)
  {
    auto result = pairwise_linestring_intersection(lhs, rhs);

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result.geometry_collection_offset->view(),
                                        expected.geometry_collection_offset->view(),
                                        verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      result.types_buffer->view(), expected.types_buffer->view(), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      result.offset_buffer->view(), expected.offset_buffer->view(), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result.points->view(), expected.points->view(), verbosity);

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      result.segments->view(), expected.segments->view(), verbosity);

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      result.lhs_linestring_id->view(), expected.lhs_linestring_id->view(), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      result.lhs_segment_id->view(), expected.lhs_segment_id->view(), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      result.rhs_linestring_id->view(), expected.rhs_linestring_id->view(), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      result.rhs_segment_id->view(), expected.rhs_segment_id->view(), verbosity);
  }

  // helper function to make a linestring_intersection_column_result
  auto make_linestring_intersection_result(
    std::initializer_list<cudf::size_type> geometry_collection_offset,
    std::initializer_list<uint8_t> types_buffer,
    std::initializer_list<cudf::size_type> offset_buffer,
    std::initializer_list<cudf::size_type> points_offsets,
    std::initializer_list<T> points_xy,
    std::initializer_list<cudf::size_type> segments_offsets,
    std::initializer_list<cudf::size_type> segments_coords_offsets,
    std::initializer_list<T> segments_xy,
    std::initializer_list<cudf::size_type> lhs_linestring_ids,
    std::initializer_list<cudf::size_type> lhs_segment_ids,
    std::initializer_list<cudf::size_type> rhs_linestring_ids,
    std::initializer_list<cudf::size_type> rhs_segment_ids)
  {
    auto d_geometry_collection_offset =
      wrapper<cudf::size_type>(geometry_collection_offset).release();
    auto d_types_buffer  = wrapper<uint8_t>(types_buffer).release();
    auto d_offset_buffer = wrapper<cudf::size_type>(offset_buffer).release();

    auto num_points       = points_offsets.size() - 1;
    auto d_points_offsets = wrapper<cudf::size_type>(points_offsets).release();
    auto d_points_xy      = wrapper<T>(points_xy).release();
    auto d_points =
      make_lists_column(num_points, std::move(d_points_offsets), std::move(d_points_xy), 0, {});

    auto d_segments_offsets        = wrapper<size_type>(segments_offsets).release();
    auto d_segments_coords_offsets = wrapper<size_type>(segments_coords_offsets).release();
    auto d_segments_xy             = wrapper<T>(segments_xy).release();

    auto num_segments       = d_segments_offsets->size() - 1;
    auto num_segment_points = d_segments_coords_offsets->size() - 1;
    auto d_segments         = make_lists_column(
      num_segments,
      std::move(d_segments_offsets),
      make_lists_column(
        num_segment_points, std::move(d_segments_coords_offsets), std::move(d_segments_xy), 0, {}),
      0,
      {});

    auto d_lhs_linestring_ids = wrapper<cudf::size_type>(lhs_linestring_ids).release();
    auto d_lhs_segment_ids    = wrapper<cudf::size_type>(lhs_segment_ids).release();
    auto d_rhs_linestring_ids = wrapper<cudf::size_type>(rhs_linestring_ids).release();
    auto d_rhs_segment_ids    = wrapper<cudf::size_type>(rhs_segment_ids).release();

    return linestring_intersection_column_result{std::move(d_geometry_collection_offset),
                                                 std::move(d_types_buffer),
                                                 std::move(d_offset_buffer),
                                                 std::move(d_points),
                                                 std::move(d_segments),
                                                 std::move(d_lhs_linestring_ids),
                                                 std::move(d_lhs_segment_ids),
                                                 std::move(d_rhs_linestring_ids),
                                                 std::move(d_rhs_segment_ids)};
  }
};

struct LinestringIntersectionTestUntyped : public LinestringIntersectionTestBase {};

// float and double are logically the same but would require separate tests due to precision.
using TestTypes = FloatingPointTypes;
TYPED_TEST_CASE(LinestringIntersectionTest, TestTypes);

TYPED_TEST(LinestringIntersectionTest, SingleToSingleEmpty)
{
  using T = TypeParam;

  auto [ltype, lhs] = make_linestring_column<T>({0}, std::initializer_list<T>{}, this->stream);
  auto [rtype, rhs] = make_linestring_column<T>({0}, std::initializer_list<T>{}, this->stream);

  auto expected =
    this->make_linestring_intersection_result({0},
                                              {},
                                              std::initializer_list<size_type>{},
                                              {0},
                                              std::initializer_list<T>{},
                                              {0},
                                              {0},
                                              std::initializer_list<T>{},
                                              std::initializer_list<cudf::size_type>{},
                                              std::initializer_list<cudf::size_type>{},
                                              std::initializer_list<cudf::size_type>{},
                                              std::initializer_list<cudf::size_type>{});

  CUSPATIAL_RUN_TEST(this->run_single,
                     geometry_column_view(lhs->view(), ltype, geometry_type_id::LINESTRING),
                     geometry_column_view(rhs->view(), rtype, geometry_type_id::LINESTRING),
                     expected);
}

TYPED_TEST(LinestringIntersectionTest, SingleToSingleOnePair)
{
  using T           = TypeParam;
  auto [ltype, lhs] = make_linestring_column<T>({0, 2}, {0, 0, 1, 1}, this->stream);
  auto [rtype, rhs] = make_linestring_column<T>({0, 2}, {0, 1, 1, 0}, this->stream);

  auto expected = this->make_linestring_intersection_result(
    {0, 1}, {0}, {0}, {0, 2}, {0.5, 0.5}, {0}, {0}, {}, {0}, {0}, {0}, {0});

  CUSPATIAL_RUN_TEST(this->run_single,
                     geometry_column_view(lhs->view(), ltype, geometry_type_id::LINESTRING),
                     geometry_column_view(rhs->view(), rtype, geometry_type_id::LINESTRING),
                     expected);
}

TYPED_TEST(LinestringIntersectionTest, MultiToSingleEmpty)
{
  using T = TypeParam;

  auto [ltype, lhs] = make_linestring_column<T>({0}, {0}, std::initializer_list<T>{}, this->stream);
  auto [rtype, rhs] = make_linestring_column<T>({0}, std::initializer_list<T>{}, this->stream);

  auto expected =
    this->make_linestring_intersection_result({0},
                                              {},
                                              std::initializer_list<cudf::size_type>{},
                                              {0},
                                              std::initializer_list<T>{},
                                              {0},
                                              {0},
                                              std::initializer_list<T>{},
                                              std::initializer_list<cudf::size_type>{},
                                              std::initializer_list<cudf::size_type>{},
                                              std::initializer_list<cudf::size_type>{},
                                              std::initializer_list<cudf::size_type>{});

  CUSPATIAL_RUN_TEST(this->run_single,
                     geometry_column_view(lhs->view(), ltype, geometry_type_id::LINESTRING),
                     geometry_column_view(rhs->view(), rtype, geometry_type_id::LINESTRING),
                     expected);
}

TYPED_TEST(LinestringIntersectionTest, MultiToSingleOnePair)
{
  using T = TypeParam;

  auto [ltype, lhs] = make_linestring_column<T>({0, 1}, {0, 2}, {0, 2, 2, 2}, this->stream);
  auto [rtype, rhs] = make_linestring_column<T>({0, 2}, {1, 3, 1, 0}, this->stream);

  auto expected = this->make_linestring_intersection_result(
    {0, 1}, {0}, {0}, {0, 2}, {1, 2}, {0}, {0}, std::initializer_list<T>{}, {0}, {0}, {0}, {0});

  CUSPATIAL_RUN_TEST(this->run_single,
                     geometry_column_view(lhs->view(), ltype, geometry_type_id::LINESTRING),
                     geometry_column_view(rhs->view(), rtype, geometry_type_id::LINESTRING),
                     expected);
}

TYPED_TEST(LinestringIntersectionTest, SingleToMultiEmpty)
{
  using T = TypeParam;

  auto [ltype, lhs] = make_linestring_column<T>({0}, std::initializer_list<T>{}, this->stream);
  auto [rtype, rhs] = make_linestring_column<T>({0}, {0}, std::initializer_list<T>{}, this->stream);

  auto expected =
    this->make_linestring_intersection_result({0},
                                              {},
                                              std::initializer_list<cudf::size_type>{},
                                              {0},
                                              std::initializer_list<T>{},
                                              {0},
                                              {0},
                                              std::initializer_list<T>{},
                                              std::initializer_list<cudf::size_type>{},
                                              std::initializer_list<cudf::size_type>{},
                                              std::initializer_list<cudf::size_type>{},
                                              std::initializer_list<cudf::size_type>{});

  CUSPATIAL_RUN_TEST(this->run_single,
                     geometry_column_view(lhs->view(), ltype, geometry_type_id::LINESTRING),
                     geometry_column_view(rhs->view(), rtype, geometry_type_id::LINESTRING),
                     expected);
}

TYPED_TEST(LinestringIntersectionTest, SingleToMultiOnePair)
{
  using T           = TypeParam;
  auto [ltype, lhs] = make_linestring_column<T>({0, 2}, {0, 2, 2, 2}, this->stream);
  auto [rtype, rhs] = make_linestring_column<T>({0, 1}, {0, 2}, {1, 3, 1, 0}, this->stream);

  auto expected = this->make_linestring_intersection_result(
    {0, 1}, {0}, {0}, {0, 2}, {1, 2}, {0}, {0}, {}, {0}, {0}, {0}, {0});

  CUSPATIAL_RUN_TEST(this->run_single,
                     geometry_column_view(lhs->view(), ltype, geometry_type_id::LINESTRING),
                     geometry_column_view(rhs->view(), rtype, geometry_type_id::LINESTRING),
                     expected);
}

TYPED_TEST(LinestringIntersectionTest, MultiToMultiEmpty)
{
  using T = TypeParam;

  auto [ltype, lhs] = make_linestring_column<T>({0}, {0}, std::initializer_list<T>{}, this->stream);
  auto [rtype, rhs] = make_linestring_column<T>({0}, {0}, std::initializer_list<T>{}, this->stream);

  auto expected =
    this->make_linestring_intersection_result({0},
                                              {},
                                              std::initializer_list<cudf::size_type>{},
                                              {0},
                                              std::initializer_list<T>{},
                                              {0},
                                              {0},
                                              std::initializer_list<T>{},
                                              std::initializer_list<cudf::size_type>{},
                                              std::initializer_list<cudf::size_type>{},
                                              std::initializer_list<cudf::size_type>{},
                                              std::initializer_list<cudf::size_type>{});

  CUSPATIAL_RUN_TEST(this->run_single,
                     geometry_column_view(lhs->view(), ltype, geometry_type_id::LINESTRING),
                     geometry_column_view(rhs->view(), rtype, geometry_type_id::LINESTRING),
                     expected);
}

TYPED_TEST(LinestringIntersectionTest, MultiToMultiOnePair)
{
  using T = TypeParam;

  auto [ltype, lhs] = make_linestring_column<T>({0, 1}, {0, 2}, {0, 2, 2, 2}, this->stream);
  auto [rtype, rhs] = make_linestring_column<T>({0, 1}, {0, 2}, {1, 3, 1, 0}, this->stream);

  auto expected = this->make_linestring_intersection_result(
    {0, 1}, {0}, {0}, {0, 2}, {1, 2}, {0}, {0}, std::initializer_list<T>{}, {0}, {0}, {0}, {0});

  CUSPATIAL_RUN_TEST(this->run_single,
                     geometry_column_view(lhs->view(), ltype, geometry_type_id::LINESTRING),
                     geometry_column_view(rhs->view(), rtype, geometry_type_id::LINESTRING),
                     expected);
}

TEST_F(LinestringIntersectionTestUntyped, MismatchCoordinateType)
{
  auto lhs_geom   = fixed_width_column_wrapper<cudf::size_type>{0, 2};
  auto lhs_coords = fixed_width_column_wrapper<float>{0, 1, 1, 0};
  auto lhs        = cudf::make_lists_column(1, lhs_geom.release(), lhs_coords.release(), 0, {});
  auto lhs_view =
    geometry_column_view(lhs->view(), collection_type_id::SINGLE, geometry_type_id::LINESTRING);

  auto rhs_geom   = fixed_width_column_wrapper<cudf::size_type>{0, 2};
  auto rhs_coords = fixed_width_column_wrapper<double>{0, 1, 1, 0};
  auto rhs        = cudf::make_lists_column(1, rhs_geom.release(), rhs_coords.release(), 0, {});
  auto rhs_view =
    geometry_column_view(rhs->view(), collection_type_id::SINGLE, geometry_type_id::LINESTRING);

  EXPECT_THROW(cuspatial::pairwise_linestring_intersection(lhs_view, rhs_view),
               cuspatial::logic_error);
}

TEST_F(LinestringIntersectionTestUntyped, MismatchSizeSingleToSingle)
{
  auto lhs_geom   = fixed_width_column_wrapper<cudf::size_type>{0, 2, 4};
  auto lhs_coords = fixed_width_column_wrapper<double>{0, 1, 1, 0, 1, 1, 2, 2};
  auto lhs        = cudf::make_lists_column(2, lhs_geom.release(), lhs_coords.release(), 0, {});
  auto lhs_view =
    geometry_column_view(lhs->view(), collection_type_id::SINGLE, geometry_type_id::LINESTRING);

  auto rhs_geom   = fixed_width_column_wrapper<cudf::size_type>{0, 2};
  auto rhs_coords = fixed_width_column_wrapper<double>{0, 1, 1, 0};
  auto rhs        = cudf::make_lists_column(1, rhs_geom.release(), rhs_coords.release(), 0, {});
  auto rhs_view =
    geometry_column_view(rhs->view(), collection_type_id::SINGLE, geometry_type_id::LINESTRING);

  EXPECT_THROW(cuspatial::pairwise_linestring_intersection(lhs_view, rhs_view),
               cuspatial::logic_error);
}

TEST_F(LinestringIntersectionTestUntyped, MismatchSizeSingleToMulti)
{
  auto lhs_geom   = fixed_width_column_wrapper<cudf::size_type>{0, 2};
  auto lhs_coords = fixed_width_column_wrapper<double>{0, 1, 1, 0};
  auto lhs        = cudf::make_lists_column(1, lhs_geom.release(), lhs_coords.release(), 0, {});
  auto lhs_view =
    geometry_column_view(lhs->view(), collection_type_id::SINGLE, geometry_type_id::LINESTRING);

  auto rhs_geom   = fixed_width_column_wrapper<cudf::size_type>{0, 1, 2};
  auto rhs_part   = fixed_width_column_wrapper<cudf::size_type>{0, 2, 4};
  auto rhs_coords = fixed_width_column_wrapper<double>{0, 1, 1, 0, 0, 2, 2, 0};
  auto rhs        = cudf::make_lists_column(2, rhs_geom.release(), rhs_coords.release(), 0, {});
  auto rhs_view =
    geometry_column_view(rhs->view(), collection_type_id::MULTI, geometry_type_id::LINESTRING);

  EXPECT_THROW(cuspatial::pairwise_linestring_intersection(lhs_view, rhs_view),
               cuspatial::logic_error);
}

TEST_F(LinestringIntersectionTestUntyped, MismatchSizeMultiToSingle)
{
  auto lhs_geom   = fixed_width_column_wrapper<cudf::size_type>{0, 2};
  auto lhs_coords = fixed_width_column_wrapper<double>{0, 1, 1, 0};
  auto lhs        = cudf::make_lists_column(1, lhs_geom.release(), lhs_coords.release(), 0, {});
  auto lhs_view =
    geometry_column_view(lhs->view(), collection_type_id::SINGLE, geometry_type_id::LINESTRING);

  auto rhs_geom   = fixed_width_column_wrapper<cudf::size_type>{0, 1, 2};
  auto rhs_part   = fixed_width_column_wrapper<cudf::size_type>{0, 2, 4};
  auto rhs_coords = fixed_width_column_wrapper<double>{0, 1, 1, 0, 0, 2, 2, 0};
  auto rhs        = cudf::make_lists_column(2, rhs_geom.release(), rhs_coords.release(), 0, {});
  auto rhs_view =
    geometry_column_view(rhs->view(), collection_type_id::MULTI, geometry_type_id::LINESTRING);

  EXPECT_THROW(cuspatial::pairwise_linestring_intersection(lhs_view, rhs_view),
               cuspatial::logic_error);
}

TEST_F(LinestringIntersectionTestUntyped, MismatchSizeMultiToMulti)
{
  auto lhs_geom   = fixed_width_column_wrapper<cudf::size_type>{0, 1, 2};
  auto lhs_part   = fixed_width_column_wrapper<cudf::size_type>{0, 2, 4};
  auto lhs_coords = fixed_width_column_wrapper<double>{0, 1, 1, 0, 0, 2, 2, 2};
  auto lhs        = cudf::make_lists_column(2, lhs_geom.release(), lhs_coords.release(), 0, {});
  auto lhs_view =
    geometry_column_view(lhs->view(), collection_type_id::MULTI, geometry_type_id::LINESTRING);

  auto rhs_geom   = fixed_width_column_wrapper<cudf::size_type>{0, 1};
  auto rhs_part   = fixed_width_column_wrapper<cudf::size_type>{0, 2};
  auto rhs_coords = fixed_width_column_wrapper<double>{0, 1, 1, 0};
  auto rhs        = cudf::make_lists_column(1, rhs_geom.release(), rhs_coords.release(), 0, {});
  auto rhs_view =
    geometry_column_view(rhs->view(), collection_type_id::MULTI, geometry_type_id::LINESTRING);

  EXPECT_THROW(cuspatial::pairwise_linestring_intersection(lhs_view, rhs_view),
               cuspatial::logic_error);
}

TEST_F(LinestringIntersectionTestUntyped, NotLineString)
{
  auto lhs_offsets = fixed_width_column_wrapper<cudf::size_type>{0, 2, 4};
  auto lhs_coords  = fixed_width_column_wrapper<double>{0, 1, 1, 2};
  auto lhs         = cudf::make_lists_column(2, lhs_offsets.release(), lhs_coords.release(), 0, {});
  auto lhs_view =
    geometry_column_view(lhs->view(), collection_type_id::SINGLE, geometry_type_id::POINT);

  auto [rhs_type, rhs] = make_linestring_column<double>({0, 2}, {0, 1, 1, 0}, this->stream);
  auto rhs_view        = geometry_column_view(rhs->view(), rhs_type, geometry_type_id::LINESTRING);

  EXPECT_THROW(cuspatial::pairwise_linestring_intersection(lhs_view, rhs_view),
               cuspatial::logic_error);
}

TEST_F(LinestringIntersectionTestUntyped, NotLinestring2)
{
  auto lhs_geom   = fixed_width_column_wrapper<cudf::size_type>{0, 2};
  auto lhs_coords = fixed_width_column_wrapper<double>{0, 1, 1, 0};
  auto lhs        = cudf::make_lists_column(1, lhs_geom.release(), lhs_coords.release(), 0, {});
  auto lhs_view =
    geometry_column_view(lhs->view(), collection_type_id::SINGLE, geometry_type_id::LINESTRING);

  auto rhs_geom  = fixed_width_column_wrapper<cudf::size_type>{0, 1};
  auto rhs_part  = fixed_width_column_wrapper<cudf::size_type>{0, 1};
  auto rhs_ring  = fixed_width_column_wrapper<cudf::size_type>{0, 5};
  auto rhs_coord = fixed_width_column_wrapper<double>{0, 1, 1, 1, 1, 0, 0, 0, 0, 1};
  auto rhs       = cudf::make_lists_column(
    1,
    rhs_geom.release(),
    cudf::make_lists_column(
      1,
      rhs_part.release(),
      cudf::make_lists_column(1, rhs_ring.release(), rhs_coord.release(), 0, {}),
      0,
      {}),
    0,
    {});
  auto rhs_view =
    geometry_column_view(rhs->view(), collection_type_id::MULTI, geometry_type_id::POLYGON);

  EXPECT_THROW(cuspatial::pairwise_linestring_intersection(lhs_view, rhs_view),
               cuspatial::logic_error);
}
