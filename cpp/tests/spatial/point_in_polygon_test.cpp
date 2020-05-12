/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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


#include <type_traits>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>
#include <cuspatial/point_in_polygon.hpp>
#include <cuspatial/error.hpp>

using namespace cudf::test;

template <typename T>
struct PointInPolygonTest : public BaseFixture {};

// float and double are logically the same but would require separate tests due to precision.
using TestTypes = Types<double>;
TYPED_TEST_CASE(PointInPolygonTest, TestTypes);

TYPED_TEST(PointInPolygonTest, Empty)
{
    using T = TypeParam;

    auto xs = fixed_width_column_wrapper<T>({ 0 });
    auto ys = fixed_width_column_wrapper<T>({ 0 });
    auto poly_offsets = fixed_width_column_wrapper<cudf::size_type>({ });
    auto poly_ring_offsets = fixed_width_column_wrapper<cudf::size_type>({ });
    auto poly_point_xs = fixed_width_column_wrapper<T>({ });
    auto poly_point_ys = fixed_width_column_wrapper<T>({ });

    auto expected = fixed_width_column_wrapper<int32_t>({ 0b0 });

    auto actual = cuspatial::point_in_polygon(xs,
                                              ys,
                                              poly_offsets,
                                              poly_ring_offsets,
                                              poly_point_xs,
                                              poly_point_ys);

    expect_columns_equal(expected, actual->view(), true);
}

TYPED_TEST(PointInPolygonTest, OnePolygonOneRing)
{
    using T = TypeParam;

    auto xs = fixed_width_column_wrapper<T>({ -2.0, 2.0,  0.0, 0.0, -0.5, 0.5,  0.0, 0.0 });
    auto ys = fixed_width_column_wrapper<T>({  0.0, 0.0, -2.0, 2.0,  0.0, 0.0, -0.5, 0.5 });
    auto poly_offsets = fixed_width_column_wrapper<cudf::size_type>({ 0 });
    auto poly_ring_offsets = fixed_width_column_wrapper<cudf::size_type>({ 0 });
    auto poly_point_ys = fixed_width_column_wrapper<T>({ -1.0,  1.0, 1.0, -1.0, -1.0 });
    auto poly_point_xs = fixed_width_column_wrapper<T>({ -1.0, -1.0, 1.0,  1.0, -1.0 });

    auto expected = fixed_width_column_wrapper<int32_t>({ false, false, false, false, true, true, true, true });

    auto actual = cuspatial::point_in_polygon(xs,
                                              ys,
                                              poly_offsets,
                                              poly_ring_offsets,
                                              poly_point_xs,
                                              poly_point_ys);

    expect_columns_equal(expected, actual->view(), true);
}

TYPED_TEST(PointInPolygonTest, TwoPolygonsOneRingEach)
{
    using T = TypeParam;

    auto xs = fixed_width_column_wrapper<T>({ -2.0, 2.0,  0.0, 0.0, -0.5, 0.5,  0.0, 0.0 });
    auto ys = fixed_width_column_wrapper<T>({  0.0, 0.0, -2.0, 2.0,  0.0, 0.0, -0.5, 0.5 });
    auto poly_offsets = fixed_width_column_wrapper<cudf::size_type>({ 0, 1 });
    auto poly_ring_offsets = fixed_width_column_wrapper<cudf::size_type>({ 0, 5 });
    auto poly_point_ys = fixed_width_column_wrapper<T>({ -1.0, -1.0, 1.0,  1.0, -1.0, 0.0, 1.0,  0.0, -1.0, 0.0 });
    auto poly_point_xs = fixed_width_column_wrapper<T>({ -1.0,  1.0, 1.0, -1.0, -1.0, 1.0, 0.0, -1.0,  0.0, 1.0 });

    auto expected = fixed_width_column_wrapper<int32_t>({ 0b00, 0b00, 0b00, 0b00, 0b11, 0b11, 0b11, 0b11 });

    auto actual = cuspatial::point_in_polygon(xs,
                                              ys,
                                              poly_offsets,
                                              poly_ring_offsets,
                                              poly_point_xs,
                                              poly_point_ys);

    expect_columns_equal(expected, actual->view(), true);
}

TYPED_TEST(PointInPolygonTest, OnePolygonTwoRings)
{
    using T = TypeParam;

    auto xs = fixed_width_column_wrapper<T>({ 0.0, -0.4, -0.6,  0.0,  0.0 });
    auto ys = fixed_width_column_wrapper<T>({ 0.0,  0.0,  0.0,  0.4, -0.6 });
    auto poly_offsets = fixed_width_column_wrapper<cudf::size_type>({ 0 });
    auto poly_ring_offsets = fixed_width_column_wrapper<cudf::size_type>({ 0, 5 });

    //   2x2 square, center  |  1x1 square, center
    auto poly_point_xs = fixed_width_column_wrapper<T>({
        -1.0, -1.0, 1.0,  1.0, -1.0, -0.5, -0.5, 0.5,  0.5, -0.5 });
    auto poly_point_ys = fixed_width_column_wrapper<T>({
        -1.0,  1.0, 1.0, -1.0, -1.0, -0.5,  0.5, 0.5, -0.5, -0.5 });

    auto expected = fixed_width_column_wrapper<int32_t>({ 0b0, 0b0, 0b1, 0b0, 0b1 });

    auto actual = cuspatial::point_in_polygon(xs,
                                              ys,
                                              poly_offsets,
                                              poly_ring_offsets,
                                              poly_point_xs,
                                              poly_point_ys);

    expect_columns_equal(expected, actual->view(), true);
}

TYPED_TEST(PointInPolygonTest, EdgesOfSquare)
{
    using T = TypeParam;

    auto xs = fixed_width_column_wrapper<T>({ 0.0 });
    auto ys = fixed_width_column_wrapper<T>({ 0.0 });
    auto poly_offsets = fixed_width_column_wrapper<cudf::size_type>({ 0, 1, 2, 3 });
    auto poly_ring_offsets = fixed_width_column_wrapper<cudf::size_type>({ 0, 5, 10, 15 });

    // 0: rect on min x side
    // 1: rect on max x side
    // 2: rect on min y side
    // 3: rect on max y side
    auto poly_point_xs = fixed_width_column_wrapper<T>({
        -1.0,  0.0, 0.0, -1.0, -1.0,  0.0,  1.0, 1.0, 0.0,  0.0, -1.0, -1.0, 1.0,  1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0 });
    auto poly_point_ys = fixed_width_column_wrapper<T>({
        -1.0, -1.0, 1.0,  1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0,  0.0, 0.0, -1.0, 1.0,   0.0,  1.0, 1.0, 0.0,  0.0 });

    // point is included in rects on min x and y sides, but not on max x or y sides.
    // this behavior is inconsistent, and not necessarily intentional.
    auto expected = fixed_width_column_wrapper<int32_t>({ 0b1010 });

    auto actual = cuspatial::point_in_polygon(xs,
                                              ys,
                                              poly_offsets,
                                              poly_ring_offsets,
                                              poly_point_xs,
                                              poly_point_ys);

    expect_columns_equal(expected, actual->view(), true);
}

TYPED_TEST(PointInPolygonTest, CornersOfSquare)
{
    using T = TypeParam;

    auto xs = fixed_width_column_wrapper<T>({ 0.0 });
    auto ys = fixed_width_column_wrapper<T>({ 0.0 });
    auto poly_offsets = fixed_width_column_wrapper<cudf::size_type>({ 0, 1, 2, 3 });
    auto poly_ring_offsets = fixed_width_column_wrapper<cudf::size_type>({ 0, 4, 8, 12 });

    // 0: min x min y corner
    // 1: min x max y corner
    // 2: max x min y corner
    // 3: max x max y corner
    auto poly_point_xs = fixed_width_column_wrapper<T>({
        -1.0, -1.0, 0.0,  0.0, -1.0, -1.0, -1.0, 0.0, -1.0, -1.0,  0.0, 0.0, 1.0,  1.0,  0.0, 0.0, 0.0, 1.0, 1.0, 0.0 });
    auto poly_point_ys = fixed_width_column_wrapper<T>({
        -1.0,  0.0, 0.0, -1.0, -1.0,  0.0,  1.0, 1.0,  0.0,  0.0, -1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0 });

    // point is only included on the max x max y corner.
    // this behavior is inconsistent, and not necessarily intentional.
    auto expected = fixed_width_column_wrapper<int32_t>({ 0b1000 });

    auto actual = cuspatial::point_in_polygon(xs,
                                              ys,
                                              poly_offsets,
                                              poly_ring_offsets,
                                              poly_point_xs,
                                              poly_point_ys);

    expect_columns_equal(expected, actual->view(), true);
}
