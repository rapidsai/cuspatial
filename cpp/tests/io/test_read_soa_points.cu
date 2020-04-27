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

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/soa_readers.hpp>
#include "utility/utility.hpp"

using namespace cudf::test;

template <typename T>
struct SoaTest : public BaseFixture {};

using TestTypes = Types<int64_t>;
TYPED_TEST_CASE(SoaTest, TestTypes);

TYPED_TEST(SoaTest, Empty)
{
    using T = TypeParam;

    TempDirTestEnvironment* const temp_env = static_cast<TempDirTestEnvironment*>(
        ::testing::AddGlobalTestEnvironment(new TempDirTestEnvironment));
    temp_env->SetUp();

    auto write_column = fixed_width_column_wrapper<T>({});
    auto to_host_result = to_host<T>(write_column).first;
    std::vector<T> h_write_column(to_host_result.size());
    thrust::copy(to_host_result.begin(), to_host_result.end(),
        h_write_column.begin());

    size_t write_result = cuspatial::write_field_from_vec(
        temp_env->get_temp_filepath("soa_its.tmp").c_str(), h_write_column);
    CUSPATIAL_EXPECTS(write_result==h_write_column.size(), "Wrote an empty vec");

    auto read_result = cuspatial::read_timestamp_soa(
        temp_env->get_temp_filepath("soa_its.tmp").c_str()
    );

	expect_columns_equal(read_result->view(), write_column, true);
}


TYPED_TEST(SoaTest, Single)
{
    using T = TypeParam;

    TempDirTestEnvironment* const temp_env = static_cast<TempDirTestEnvironment*>(
        ::testing::AddGlobalTestEnvironment(new TempDirTestEnvironment));
    temp_env->SetUp();

    auto write_column = fixed_width_column_wrapper<T>({0});
    auto to_host_result = to_host<T>(write_column).first;
    std::vector<T> h_write_column(to_host_result.size());
    thrust::copy(to_host_result.begin(), to_host_result.end(),
        h_write_column.begin());

    size_t write_result = cuspatial::write_field_from_vec(
        temp_env->get_temp_filepath("soa_its.tmp").c_str(), h_write_column);
    CUSPATIAL_EXPECTS(write_result==h_write_column.size(), "Wrote an empty vec");

    auto read_result = cuspatial::read_timestamp_soa(
        temp_env->get_temp_filepath("soa_its.tmp").c_str()
    );

	expect_columns_equal(read_result->view(), write_column, true);
}

TYPED_TEST(SoaTest, Triple)
{
    using T = TypeParam;

    TempDirTestEnvironment* const temp_env = static_cast<TempDirTestEnvironment*>(
        ::testing::AddGlobalTestEnvironment(new TempDirTestEnvironment));
    temp_env->SetUp();

    auto write_column = fixed_width_column_wrapper<T>({0, 1, 2});
    auto to_host_result = to_host<T>(write_column).first;
    std::vector<T> h_write_column(to_host_result.size());
    thrust::copy(to_host_result.begin(), to_host_result.end(),
        h_write_column.begin());

    size_t write_result = cuspatial::write_field_from_vec(
        temp_env->get_temp_filepath("soa_its.tmp").c_str(), h_write_column);
    CUSPATIAL_EXPECTS(write_result==h_write_column.size(), "Wrote an empty vec");

    auto read_result = cuspatial::read_timestamp_soa(
        temp_env->get_temp_filepath("soa_its.tmp").c_str()
    );

	expect_columns_equal(read_result->view(), write_column, true);
}

TYPED_TEST(SoaTest, Negative)
{
    using T = TypeParam;

    TempDirTestEnvironment* const temp_env = static_cast<TempDirTestEnvironment*>(
        ::testing::AddGlobalTestEnvironment(new TempDirTestEnvironment));
    temp_env->SetUp();

    auto write_column = fixed_width_column_wrapper<T>({-1, -2});
    auto to_host_result = to_host<T>(write_column).first;
    std::vector<T> h_write_column(to_host_result.size());
    thrust::copy(to_host_result.begin(), to_host_result.end(),
        h_write_column.begin());

    size_t write_result = cuspatial::write_field_from_vec(
        temp_env->get_temp_filepath("soa_its.tmp").c_str(), h_write_column);
    CUSPATIAL_EXPECTS(write_result==h_write_column.size(), "Wrote an empty vec");

    auto read_result = cuspatial::read_timestamp_soa(
        temp_env->get_temp_filepath("soa_its.tmp").c_str()
    );

	expect_columns_equal(read_result->view(), write_column, true);
}
