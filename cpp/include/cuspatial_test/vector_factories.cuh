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

#include <cuspatial/experimental/ranges/multilinestring_range.cuh>
#include <cuspatial/experimental/ranges/multipoint_range.cuh>
#include <cuspatial/experimental/ranges/range.cuh>
#include <cuspatial/traits.hpp>
#include <cuspatial/vec_2d.hpp>

#include <rmm/device_vector.hpp>

#include <initializer_list>
#include <numeric>
#include <vector>

namespace cuspatial {
namespace test {

template <typename T>
auto make_device_vector(std::initializer_list<T> inl)
{
  return rmm::device_vector<T>(inl.begin(), inl.end());
}

template <typename T>
auto make_device_uvector(std::initializer_list<T> inl,
                         rmm::cuda_stream_view stream,
                         rmm::mr::device_memory_resource* mr)
{
  std::vector<T> hvec(inl.begin(), inl.end());
  auto res = rmm::device_uvector<T>(inl.size(), stream, mr);
  cudaMemcpyAsync(res.data(),
                  hvec.data(),
                  hvec.size() * sizeof(T),
                  cudaMemcpyKind::cudaMemcpyHostToDevice,
                  stream.value());
  return res;
}

/**
 * @brief Owning object of a multilinestring array following geoarrow layout.
 *
 * @tparam GeometryArray Array type of geometry offset array
 * @tparam PartArray Array type of part offset array
 * @tparam CoordinateArray Array type of coordinate array
 */
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

  /// Return the number of multilinestrings
  auto size() { return _geometry_offset_array.size() - 1; }

  /// Return range object of the multilinestring array
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

/**
 * @brief Construct an owning object of a multilinestring array from initializer lists
 *
 * @tparam T Type of coordinate
 * @param geometry_inl Initializer list of geometry offsets
 * @param part_inl Initializer list of part offsets
 * @param coord_inl Initializer list of coordinate
 * @return multilinestring array object
 */
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

/**
 * @brief Owning object of a multipoint array following geoarrow format
 *
 * @tparam GeometryArray Array type of geometry offsets
 * @tparam CoordinateArray Array type of coordinates
 */
template <typename GeometryArray, typename CoordinateArray>
class multipoint_array {
 public:
  multipoint_array(GeometryArray geometry_offsets_array, CoordinateArray coordinate_array)
    : _geometry_offsets(geometry_offsets_array), _coordinates(coordinate_array)
  {
  }

  /// Return the number of multipoints
  auto size() { return _geometry_offsets.size() - 1; }

  /// Return range object of the multipoint array
  auto range()
  {
    return multipoint_range{
      _geometry_offsets.begin(), _geometry_offsets.end(), _coordinates.begin(), _coordinates.end()};
  }

  /// Release ownership
  auto release() { return std::pair{std::move(_geometry_offsets), std::move(_coordinates)}; }

 private:
  GeometryArray _geometry_offsets;
  CoordinateArray _coordinates;
};

/**
 * @brief Factory method to construct multipoint array from initializer list of multipoints.
 *
 * Example: Construct an array of 2 multipoints, each with 2, 0, 1 points:
 * using P = vec_2d<float>;
 * make_multipoints_array({{P{0.0, 1.0}, P{2.0, 0.0}}, {}, {P{3.0, 4.0}}});
 *
 * Example: Construct an empty multilinestring array:
 * make_multipoints_array<float>({}); // Explict parameter required to deduce type.
 *
 * @tparam T Type of coordinate
 * @param inl List of multipoints
 * @return multipoints_array object
 */
template <typename T>
auto make_multipoints_array(std::initializer_list<std::initializer_list<vec_2d<T>>> inl)
{
  std::vector<std::size_t> offsets{0};
  std::transform(inl.begin(), inl.end(), std::back_inserter(offsets), [](auto multipoint) {
    return multipoint.size();
  });
  std::inclusive_scan(offsets.begin(), offsets.end(), offsets.begin());

  std::vector<vec_2d<T>> coordinates = std::accumulate(
    inl.begin(), inl.end(), std::vector<vec_2d<T>>{}, [](std::vector<vec_2d<T>>& init, auto cur) {
      init.insert(init.end(), cur.begin(), cur.end());
      return init;
    });

  return multipoint_array{rmm::device_vector<std::size_t>(offsets),
                          rmm::device_vector<vec_2d<T>>(coordinates)};
}

}  // namespace test
}  // namespace cuspatial
