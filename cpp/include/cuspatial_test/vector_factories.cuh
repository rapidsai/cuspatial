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

#include <rmm/device_vector.hpp>

#include <cuspatial/vec_2d.hpp>

#include <initializer_list>
#include <vector>

namespace cuspatial {
namespace test {

template <typename T>
auto make_device_vector(std::initializer_list<T> inl)
{
  return rmm::device_vector<T>(inl.begin(), inl.end());
}

template <typename GeometryArray, typename PartArray, typename CoordinateArray>
class multilinestring_array {
 public:
  multilinestring_array(GeometryArray geometry_offsets_array,
                        PartArray part_offsets_array,
                        CoordinateArray coordinate_offset_array)
    : _geometry_offset_array(geometry_offsets_array),
      _part_offset_array(part_offsets_array),
      _coordinate_offset_array(coordinate_offset_array)
  {
  }

  auto size() { return _geometry_offset_array.size() - 1; }

  auto range()
  {
    return multilinestring_range(_geometry_offset_array.begin(),
                                 _geometry_offset_array.end(),
                                 _part_offset_array.begin(),
                                 _part_offset_array.end(),
                                 _coordinate_offset_array.begin(),
                                 _coordinate_offset_array.end());
  }

 protected:
  GeometryArray _geometry_offset_array;
  PartArray _part_offset_array;
  CoordinateArray _coordinate_offset_array;
};

template <typename T>
auto make_multilinestring_array(std::initializer_list<std::size_t> geometry_inl,
                                std::initializer_list<std::size_t> part_inl,
                                std::initializer_list<vec_2d<T>> coord_inl)
{
  return multilinestring_array(
    rmm::device_vector<std::size_t>(
      std::vector<std::size_t>(geometry_inl.begin(), geometry_inl.end())),
    rmm::device_vector<std::size_t>(std::vector<unsigned int>(part_inl.begin(), part_inl.end())),
    rmm::device_vector<vec_2d<T>>(std::vector<vec_2d<T>>(coord_inl.begin(), coord_inl.end())));
}

}  // namespace test
}  // namespace cuspatial
