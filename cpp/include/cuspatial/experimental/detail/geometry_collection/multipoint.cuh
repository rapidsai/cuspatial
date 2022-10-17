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
#include <cuspatial/cuda_utils.hpp>

namespace cuspatial {
namespace geometry_collection {

template <typename VecIterator>
CUSPATIAL_HOST_DEVICE multipoint<VecIterator>::multipoint(VecIterator begin, VecIterator end)
  : points_begin(begin), points_end(end)
{
}

template <typename VecIterator>
CUSPATIAL_HOST_DEVICE auto multipoint<VecIterator>::point_begin() const
{
  return points_begin;
}

template <typename VecIterator>
CUSPATIAL_HOST_DEVICE auto multipoint<VecIterator>::point_end() const
{
  return points_end;
}

}  // namespace geometry_collection
}  // namespace cuspatial
