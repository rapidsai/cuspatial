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

#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <type_traits>

namespace cuspatial {
namespace test {

template <typename Vector1, typename Vector2>
inline void expect_vector_equivalent(Vector1 lhs, Vector2 rhs)
{
  using T = typename Vector1::value_type;
  static_assert(std::is_same_v<T, typename Vector2::value_type>, "Mismatch value type.");

  if constexpr (std::is_same_v<T, float>) {
    EXPECT_THAT(lhs, ::testing::Pointwise(::testing::FloatEq(), rhs));
  } else {
    EXPECT_THAT(lhs, ::testing::Pointwise(::testing::DoubleEq(), rhs));
  }
}

}  // namespace test
}  // namespace cuspatial
