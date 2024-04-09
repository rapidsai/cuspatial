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

#pragma once

#include <cudf_test/cudf_gtest.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

namespace cuspatial {
namespace test {

/**
 * @brief Mixin to supply rmm resources for fixtures
 */
class RMMResourceMixin {
  rmm::device_async_resource_ref _mr{rmm::mr::get_current_device_resource()};
  rmm::cuda_stream_view _stream{rmm::cuda_stream_default};

 public:
  /**
   * @brief Returns pointer to `device_memory_resource` that should be used for
   * all tests inheriting from this fixture
   * @return pointer to memory resource
   */
  rmm::device_async_resource_ref mr() { return _mr; }

  /**
   * @brief Returns `cuda_stream_view` that should be used for computation in
   * tests inheriting from this fixture.
   * @return view to cuda stream
   */
  rmm::cuda_stream_view stream() { return _stream; }
};

/**
 * @brief Base test fixture class from which libcuspatial test with no parameterization or only with
 * type parameterization should inherit.
 *
 * Example:
 * ```
 * class MyTestFixture : public cuspatial::test::BaseFixture {};
 * ```
 */
class BaseFixture : public RMMResourceMixin, public ::testing::Test {};

/**
 * @brief Base test fixture class from which libcuspatial test with only value parameterization
 * should inherit.
 *
 * Example:
 * ```
 * template<int, int, int>
 * class MyTest : public cuspatial::test::BaseFixtureWithParam {};
 *
 * TEST_P(MyTest, TestParamterGet) {
 *  auto a = std::get<0>(GetParam());
 *  auto b = std::get<1>(GetParam());
 *  auto c = std::get<2>(GetParam());
 *  ...
 * }
 *
 * INSTANTIATE_TEST_SUITE_P(MyTests, MyTest, ::testing::Values(
 *    std::make_tuple(1, 2, 3),
 *    std::make_tuple(4, 5, 6, 9),
 *    std::make_tuple(7, 8)))
 * ```
 */
template <typename... Ts>
class BaseFixtureWithParam : public RMMResourceMixin,
                             public ::testing::TestWithParam<std::tuple<Ts...>> {};

/**
 * @brief Floating point types to be used in libcuspatial tests
 *
 */
using FloatingPointTypes = ::testing::Types<float, double>;

}  // namespace test
}  // namespace cuspatial
