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

#include <rmm/mr/device/per_device_resource.hpp>

#include <gtest/gtest.h>

namespace cuspatial {
namespace test {

/**
 * @brief Base test fixture class from which all libcuspatial tests should inherit.
 *
 * Example:
 * ```
 * class MyTestFixture : public cuspatial::test::BaseFixture {};
 * ```
 */
class BaseFixture : public ::testing::Test {
  rmm::mr::device_memory_resource* _mr{rmm::mr::get_current_device_resource()};

 public:
  /**
   * @brief Returns pointer to `device_memory_resource` that should be used for
   * all tests inheriting from this fixture
   * @return pointer to memory resource
   */
  rmm::mr::device_memory_resource* mr() { return _mr; }
};

}  // namespace test
}  // namespace cuspatial
