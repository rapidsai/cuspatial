/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/type_lists.hpp>
#include <cuspatial/haversine.hpp>
#include <cuspatial/error.hpp>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>

using namespace cudf::test;

template<typename T>
T reinterpret(std::conditional_t<std::is_same<double, T>::value, int64_t, int32_t> value)
{
    return *reinterpret_cast<T*>(&value);
}

template <typename T>
struct HaversineTest : public BaseFixture {};

// float and double are logically the same but would require seperate tests due to precision.
using TestTypes = Types<double>;
TYPED_TEST_CASE(HaversineTest, TestTypes);
   
TYPED_TEST(HaversineTest, Empty)
{
    using T = TypeParam;

    auto a_lon = fixed_width_column_wrapper<T>({});
    auto a_lat = fixed_width_column_wrapper<T>({});
    auto b_lon = fixed_width_column_wrapper<T>({});
    auto b_lat = fixed_width_column_wrapper<T>({});

    auto expected = fixed_width_column_wrapper<T>({});

    auto actual = cuspatial::haversine_distance(a_lon, a_lat, b_lon, b_lat);

    expect_columns_equal(expected, actual->view(), true);
}
   
TYPED_TEST(HaversineTest, Zero)
{
    using T = TypeParam;

    auto const count = 3;

    auto a_lon = fixed_width_column_wrapper<T>({ 0 });
    auto a_lat = fixed_width_column_wrapper<T>({ 0 });
    auto b_lon = fixed_width_column_wrapper<T>({ 0 });
    auto b_lat = fixed_width_column_wrapper<T>({ 0 });

    auto expected = fixed_width_column_wrapper<T>({ 0 });

    auto actual = cuspatial::haversine_distance(a_lon, a_lat, b_lon, b_lat);

    expect_columns_equal(expected, actual->view(), true);
}
   
TYPED_TEST(HaversineTest, EquivolentPoints)
{
    using T = TypeParam;

    auto const count = 3;

    auto a_lon = fixed_width_column_wrapper<T>({ -180 });
    auto a_lat = fixed_width_column_wrapper<T>({    0 });
    auto b_lon = fixed_width_column_wrapper<T>({  180 });
    auto b_lat = fixed_width_column_wrapper<T>({    0 });

    auto expected = fixed_width_column_wrapper<T>({ reinterpret<T>(4430261783215946468) /* nearly zero */ });

    auto actual = cuspatial::haversine_distance(a_lon, a_lat, b_lon, b_lat);

    expect_columns_equal(expected, actual->view(), true);
}
   
TYPED_TEST(HaversineTest, MismatchSize)
{
    using T = TypeParam;

    auto a_lon = fixed_width_column_wrapper<T>({ 0 });
    auto a_lat = fixed_width_column_wrapper<T>({ 0, 1 });
    auto b_lon = fixed_width_column_wrapper<T>({ 0 });
    auto b_lat = fixed_width_column_wrapper<T>({ 0 });

    EXPECT_THROW(cuspatial::haversine_distance(a_lon, a_lat, b_lon, b_lat),
                 cuspatial::logic_error);
}

template <typename T>
struct Haversine : public BaseFixture {};

using UnsupportedTypesTest = RemoveIf<ContainedIn<Types<float, double>>, AllTypes>;
TYPED_TEST_CASE(Haversine, UnsupportedTypesTest);

// TYPED_TEST(Haversine, MismatchSize)
// {
//     using T = TypeParam;
//     auto camera_lon = 0;
//     auto camera_lat = 0;
//     auto point_lon = fixed_width_column_wrapper<T>({ 0 });
//     auto point_lat = fixed_width_column_wrapper<T>({ 0 });

//     EXPECT_THROW(cuspatial::lonlat_to_cartesian(camera_lon, camera_lat, point_lon, point_lat),
//                  cuspatial::logic_error);
// }
