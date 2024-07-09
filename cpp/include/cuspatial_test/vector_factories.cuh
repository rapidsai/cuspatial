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

#include <cuspatial_test/test_util.cuh>

#include <cuspatial/geometry/vec_2d.hpp>
#include <cuspatial/range/multilinestring_range.cuh>
#include <cuspatial/range/multipoint_range.cuh>
#include <cuspatial/range/multipolygon_range.cuh>
#include <cuspatial/range/range.cuh>
#include <cuspatial/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <initializer_list>
#include <numeric>
#include <tuple>
#include <vector>

namespace cuspatial {
namespace test {

template <typename Range>
auto make_device_vector(Range rng)
{
  using T = typename Range::value_type;
  return rmm::device_vector<T>(rng.begin(), rng.end());
}

template <typename T>
auto make_device_vector(std::initializer_list<T> inl)
{
  return rmm::device_vector<T>(inl.begin(), inl.end());
}

template <typename T>
auto make_device_uvector(std::initializer_list<T> inl,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr)
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
 * @brief Owning object of a multipolygon array following geoarrow layout.
 *
 * @tparam GeometryArray Array type of geometry offset array
 * @tparam PartArray Array type of part offset array
 * @tparam RingArray Array type of ring offset array
 * @tparam CoordinateArray Array type of coordinate array
 */
template <typename GeometryArray, typename PartArray, typename RingArray, typename CoordinateArray>
class multipolygon_array {
 public:
  using geometry_t = typename GeometryArray::value_type;
  using part_t     = typename PartArray::value_type;
  using ring_t     = typename RingArray::value_type;
  using coord_t    = typename CoordinateArray::value_type;

  multipolygon_array(thrust::device_vector<geometry_t> geometry_offsets_array,
                     thrust::device_vector<part_t> part_offsets_array,
                     thrust::device_vector<ring_t> ring_offsets_array,
                     thrust::device_vector<coord_t> coordinates_array)
    : _geometry_offsets_array(geometry_offsets_array),
      _part_offsets_array(part_offsets_array),
      _ring_offsets_array(ring_offsets_array),
      _coordinates_array(coordinates_array)
  {
  }

  multipolygon_array(rmm::device_vector<geometry_t>&& geometry_offsets_array,
                     rmm::device_vector<part_t>&& part_offsets_array,
                     rmm::device_vector<ring_t>&& ring_offsets_array,
                     rmm::device_vector<coord_t>&& coordinates_array)
    : _geometry_offsets_array(std::move(geometry_offsets_array)),
      _part_offsets_array(std::move(part_offsets_array)),
      _ring_offsets_array(std::move(ring_offsets_array)),
      _coordinates_array(std::move(coordinates_array))
  {
  }

  multipolygon_array(rmm::device_uvector<geometry_t>&& geometry_offsets_array,
                     rmm::device_uvector<part_t>&& part_offsets_array,
                     rmm::device_uvector<ring_t>&& ring_offsets_array,
                     rmm::device_uvector<coord_t>&& coordinates_array)
    : _geometry_offsets_array(std::move(geometry_offsets_array)),
      _part_offsets_array(std::move(part_offsets_array)),
      _ring_offsets_array(std::move(ring_offsets_array)),
      _coordinates_array(std::move(coordinates_array))
  {
  }

  /// Return the number of multipolygons
  auto size() { return _geometry_offsets_array.size() - 1; }

  /// Return range object of the multipolygon array
  auto range()
  {
    return multipolygon_range(_geometry_offsets_array.begin(),
                              _geometry_offsets_array.end(),
                              _part_offsets_array.begin(),
                              _part_offsets_array.end(),
                              _ring_offsets_array.begin(),
                              _ring_offsets_array.end(),
                              _coordinates_array.begin(),
                              _coordinates_array.end());
  }

  /**
   * @brief Copy the offset arrays to host.
   */
  auto to_host() const
  {
    auto geometry_offsets   = cuspatial::test::to_host<geometry_t>(_geometry_offsets_array);
    auto part_offsets       = cuspatial::test::to_host<part_t>(_part_offsets_array);
    auto ring_offsets       = cuspatial::test::to_host<ring_t>(_ring_offsets_array);
    auto coordinate_offsets = cuspatial::test::to_host<coord_t>(_coordinates_array);

    return std::tuple{geometry_offsets, part_offsets, ring_offsets, coordinate_offsets};
  }

  auto release()
  {
    return std::tuple{std::move(_geometry_offsets_array),
                      std::move(_part_offsets_array),
                      std::move(_ring_offsets_array),
                      std::move(_coordinates_array)};
  }

  /**
   * @brief Output stream operator for `multipolygon_array` for human-readable formatting
   */
  friend std::ostream& operator<<(
    std::ostream& os,
    multipolygon_array<GeometryArray, PartArray, RingArray, CoordinateArray> const& arr)
  {
    auto [geometry_offsets, part_offsets, ring_offsets, coordinates] = arr.to_host();

    auto print_vector = [&](auto const& vec) {
      for (auto it = vec.begin(); it != vec.end(); it++) {
        os << *it;
        if (std::next(it) != vec.end()) { os << ", "; }
      }
    };

    os << "Geometry Offsets:\n\t{";
    print_vector(geometry_offsets);
    os << "}\n";
    os << "Part Offsets:\n\t{";
    print_vector(part_offsets);
    os << "}\n";
    os << "Ring Offsets: \n\t{";
    print_vector(ring_offsets);
    os << "}\n";
    os << "Coordinates: \n\t{";
    print_vector(coordinates);
    os << "}\n";
    return os;
  }

 protected:
  GeometryArray _geometry_offsets_array;
  PartArray _part_offsets_array;
  RingArray _ring_offsets_array;
  CoordinateArray _coordinates_array;
};

template <typename IndexRange,
          typename CoordRange,
          typename IndexType = typename IndexRange::value_type>
auto make_multipolygon_array(IndexRange geometry_inl,
                             IndexRange part_inl,
                             IndexRange ring_inl,
                             CoordRange coord_inl)
{
  using CoordType         = typename CoordRange::value_type;
  using DeviceIndexVector = thrust::device_vector<IndexType>;
  using DeviceCoordVector = thrust::device_vector<CoordType>;

  return multipolygon_array<DeviceIndexVector,
                            DeviceIndexVector,
                            DeviceIndexVector,
                            DeviceCoordVector>(make_device_vector(geometry_inl),
                                               make_device_vector(part_inl),
                                               make_device_vector(ring_inl),
                                               make_device_vector(coord_inl));
}

template <typename T>
auto make_multipolygon_array(std::initializer_list<std::size_t> geometry_inl,
                             std::initializer_list<std::size_t> part_inl,
                             std::initializer_list<std::size_t> ring_inl,
                             std::initializer_list<vec_2d<T>> coord_inl)
{
  return make_multipolygon_array(range(geometry_inl.begin(), geometry_inl.end()),
                                 range(part_inl.begin(), part_inl.end()),
                                 range(ring_inl.begin(), ring_inl.end()),
                                 range(coord_inl.begin(), coord_inl.end()));
}

template <typename IndexType, typename CoordType>
auto make_multipolygon_array(rmm::device_uvector<IndexType> geometry_inl,
                             rmm::device_uvector<IndexType> part_inl,
                             rmm::device_uvector<IndexType> ring_inl,
                             rmm::device_uvector<CoordType> coord_inl)
{
  return multipolygon_array<rmm::device_uvector<IndexType>,
                            rmm::device_uvector<IndexType>,
                            rmm::device_uvector<IndexType>,
                            rmm::device_uvector<CoordType>>(
    std::move(geometry_inl), std::move(part_inl), std::move(ring_inl), std::move(coord_inl));
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
                        CoordinateArray coordinate_array)
    : _geometry_offset_array(std::move(geometry_offsets_array)),
      _part_offset_array(std::move(part_offsets_array)),
      _coordinate_array(std::move(coordinate_array))
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
                                 _coordinate_array.begin(),
                                 _coordinate_array.end());
  }

  auto release()
  {
    return std::tuple{std::move(_geometry_offset_array),
                      std::move(_part_offset_array),
                      std::move(_coordinate_array)};
  }

 protected:
  GeometryArray _geometry_offset_array;
  PartArray _part_offset_array;
  CoordinateArray _coordinate_array;
};

/**
 * @brief Construct an owning object of a multilinestring array from `device_uvectors`
 *
 * @param geometry_inl Range of geometry offsets
 * @param part_inl Range of part offsets
 * @param coord_inl Ramge of coordinate
 * @return multilinestring array object
 */
template <typename IndexType, typename T>
auto make_multilinestring_array(rmm::device_uvector<IndexType>&& geometry_inl,
                                rmm::device_uvector<IndexType>&& part_inl,
                                rmm::device_uvector<vec_2d<T>>&& coord_inl)
{
  return multilinestring_array<rmm::device_uvector<IndexType>,
                               rmm::device_uvector<IndexType>,
                               rmm::device_uvector<vec_2d<T>>>(
    std::move(geometry_inl), std::move(part_inl), std::move(coord_inl));
}

/**
 * @brief Construct an owning object of a multilinestring array from `device_vectors`
 *
 * @param geometry_inl Range of geometry offsets
 * @param part_inl Range of part offsets
 * @param coord_inl Ramge of coordinate
 * @return multilinestring array object
 */
template <typename IndexType, typename T>
auto make_multilinestring_array(rmm::device_vector<IndexType>&& geometry_inl,
                                rmm::device_vector<IndexType>&& part_inl,
                                rmm::device_vector<vec_2d<T>>&& coord_inl)
{
  return multilinestring_array<rmm::device_vector<IndexType>,
                               rmm::device_vector<IndexType>,
                               rmm::device_vector<vec_2d<T>>>(
    std::move(geometry_inl), std::move(part_inl), std::move(coord_inl));
}

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
  using geometry_t = typename GeometryArray::value_type;
  using coord_t    = typename CoordinateArray::value_type;

  multipoint_array(thrust::device_vector<geometry_t> geometry_offsets_array,
                   thrust::device_vector<coord_t> coordinate_array)
    : _geometry_offsets(geometry_offsets_array), _coordinates(coordinate_array)
  {
  }

  multipoint_array(rmm::device_uvector<geometry_t>&& geometry_offsets_array,
                   rmm::device_uvector<coord_t>&& coordinate_array)
    : _geometry_offsets(std::move(geometry_offsets_array)),
      _coordinates(std::move(coordinate_array))
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

  /**
   * @brief Copy the offset arrays to host.
   */
  auto to_host() const
  {
    auto geometry_offsets   = cuspatial::test::to_host<geometry_t>(_geometry_offsets);
    auto coordinate_offsets = cuspatial::test::to_host<coord_t>(_coordinates);

    return std::tuple{geometry_offsets, coordinate_offsets};
  }

  /// Release ownership
  auto release() { return std::pair{std::move(_geometry_offsets), std::move(_coordinates)}; }

 private:
  GeometryArray _geometry_offsets;
  CoordinateArray _coordinates;
};

/**
 * @brief Factory method to construct multipoint array from ranges of geometry offsets and
 * coordinates
 */
template <typename GeometryRange, typename CoordRange>
auto make_multipoint_array(GeometryRange geometry_inl, CoordRange coordinates_inl)
{
  using IndexType         = typename GeometryRange::value_type;
  using CoordType         = typename CoordRange::value_type;
  using DeviceIndexVector = thrust::device_vector<IndexType>;
  using DeviceCoordVector = thrust::device_vector<CoordType>;

  return multipoint_array<DeviceIndexVector, DeviceCoordVector>{
    make_device_vector(geometry_inl), make_device_vector(coordinates_inl)};
}

/**
 * @brief Factory method to construct multipoint array from initializer list of multipoints.
 *
 * Example: Construct an array of 2 multipoints, each with 2, 0, 1 points:
 * using P = vec_2d<float>;
 * make_multipoint_array({{P{0.0, 1.0}, P{2.0, 0.0}}, {}, {P{3.0, 4.0}}});
 *
 * Example: Construct an empty multilinestring array:
 * make_multipoint_array<float>({}); // Explicit parameter required to deduce type.
 *
 * @tparam T Type of coordinate
 * @param inl List of multipoints
 * @return multipoints_array object
 */
template <typename T>
auto make_multipoint_array(std::initializer_list<std::initializer_list<vec_2d<T>>> inl)
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

  return multipoint_array<rmm::device_vector<std::size_t>, rmm::device_vector<vec_2d<T>>>{
    rmm::device_vector<std::size_t>(offsets), rmm::device_vector<vec_2d<T>>(coordinates)};
}

/**
 * @brief Factory method to construct multipoint array by moving the offsets and coordinates from
 * `rmm::device_uvector`.
 */
template <typename IndexType, typename T>
auto make_multipoint_array(rmm::device_uvector<IndexType> geometry_offsets,
                           rmm::device_uvector<vec_2d<T>> coords)
{
  return multipoint_array<rmm::device_uvector<std::size_t>, rmm::device_uvector<vec_2d<T>>>{
    std::move(geometry_offsets), std::move(coords)};
}

/**
 * @brief Factory method to construct multipoint array by moving the offsets and coordinates from
 * `rmm::device_vector`.
 */
template <typename IndexType, typename T>
auto make_multipoint_array(rmm::device_vector<IndexType> geometry_offsets,
                           rmm::device_vector<vec_2d<T>> coords)
{
  return multipoint_array<rmm::device_vector<std::size_t>, rmm::device_vector<vec_2d<T>>>{
    std::move(geometry_offsets), std::move(coords)};
}

}  // namespace test
}  // namespace cuspatial
