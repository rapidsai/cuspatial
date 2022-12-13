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

#include <cuspatial/vec_2d.hpp>

namespace cuspatial {

/**
 * @addtogroup types
 * @{
 */

/**
 * @brief A generic axis-aligned box type.
 *
 * @tparam T the base type for the coordinates
 * @tparam Vertex the vector type to use for vertices, vec_2d<T> by default
 */
template <typename T, typename Vertex = cuspatial::vec_2d<T>>
class alignas(sizeof(Vertex)) box {
 public:
  using value_type = T;
  Vertex v1;
  Vertex v2;
};

// deduction guide, enables CTAD
template <typename T>
box(vec_2d<T> a, vec_2d<T> b) -> box<T, vec_2d<T>>;

/**
 * @} // end of doxygen group
 */

}  // namespace cuspatial
