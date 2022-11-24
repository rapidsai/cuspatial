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

#include <cuspatial/experimental/bounding_box.cuh>
#include <cuspatial/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>

namespace cuspatial {

namespace detail {

template <class T,
          class OffsetIteratorA,
          class OffsetIteratorB,
          class VertexIterator,
          class BoundingBoxIterator>
BoundingBoxIterator polygon_bounding_boxes(OffsetIteratorA polygon_offsets_first,
                                           OffsetIteratorA polygon_offsets_last,
                                           OffsetIteratorB polygon_ring_offsets_first,
                                           OffsetIteratorB polygon_ring_offsets_last,
                                           VertexIterator polygon_vertices_first,
                                           VertexIterator polygon_vertices_last,
                                           BoundingBoxIterator bounding_boxes_first,
                                           T expansion_radius,
                                           rmm::cuda_stream_view stream)
{
  auto const num_polygons = std::distance(polygon_offsets_first, polygon_offsets_last);
  auto const num_rings    = std::distance(polygon_ring_offsets_first, polygon_ring_offsets_last);
  auto const num_poly_vertices = std::distance(polygon_vertices_first, polygon_vertices_last);

  // Wrapped in an IIFE so `first_ring_offsets` is freed on return
  auto vertex_ids = [&]() {
    // TODO: use device_uvector
    rmm::device_vector<int32_t> vertex_ids(num_poly_vertices);
    rmm::device_vector<int32_t> first_ring_offsets(num_polygons);

    // Gather the first ring offset for each polygon
    thrust::gather(rmm::exec_policy(stream),
                   polygon_offsets_first,
                   polygon_offsets_last,
                   polygon_ring_offsets_first,
                   first_ring_offsets.begin());

    // Scatter the first ring offset into a list of vertex_ids for reduction
    thrust::scatter(rmm::exec_policy(stream),
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(0) + num_polygons,
                    first_ring_offsets.begin(),
                    vertex_ids.begin());

    thrust::inclusive_scan(rmm::exec_policy(stream),
                           vertex_ids.begin(),
                           vertex_ids.end(),
                           vertex_ids.begin(),
                           thrust::maximum<int32_t>());

    return vertex_ids;
  }();

  return point_bounding_boxes(vertex_ids.begin(),
                              vertex_ids.end(),
                              polygon_vertices_first,
                              bounding_boxes_first,
                              expansion_radius,
                              stream);
}

}  // namespace detail

template <class OffsetIteratorA,
          class OffsetIteratorB,
          class VertexIterator,
          class BoundingBoxIterator,
          class T>
BoundingBoxIterator polygon_bounding_boxes(OffsetIteratorA polygon_offsets_first,
                                           OffsetIteratorA polygon_offsets_last,
                                           OffsetIteratorB polygon_ring_offsets_first,
                                           OffsetIteratorB polygon_ring_offsets_last,
                                           VertexIterator polygon_vertices_first,
                                           VertexIterator polygon_vertices_last,
                                           BoundingBoxIterator bounding_boxes_first,
                                           T expansion_radius,
                                           rmm::cuda_stream_view stream)
{
  static_assert(is_floating_point<T>(), "Only floating point polygon vertices supported");

  static_assert(is_same<vec_2d<T>, iterator_value_type<VertexIterator>>(),
                "Input vertices must be cuspatial::vec_2d");

  static_assert(cuspatial::is_integral<iterator_value_type<OffsetIteratorA>,
                                       iterator_value_type<OffsetIteratorB>>(),
                "OffsetIterators must have integral value type.");

  auto const num_polys = std::distance(polygon_offsets_first, polygon_offsets_last);
  auto const num_rings = std::distance(polygon_ring_offsets_first, polygon_ring_offsets_last);
  auto const num_poly_vertices = std::distance(polygon_vertices_first, polygon_vertices_last);

  CUSPATIAL_EXPECTS(num_rings >= num_polys, "Each polygon must have at least one ring");

  CUSPATIAL_EXPECTS(num_poly_vertices >= num_polys * 4,
                    "Each ring must have at least four vertices");

  // if (num_polys == 0 || num_rings == 0 || num_poly_vertices == 0) { return bounding_boxes_first;
  // }

  return detail::polygon_bounding_boxes<T>(polygon_offsets_first,
                                           polygon_offsets_last,
                                           polygon_ring_offsets_first,
                                           polygon_ring_offsets_last,
                                           polygon_vertices_first,
                                           polygon_vertices_last,
                                           bounding_boxes_first,
                                           expansion_radius,
                                           stream);
}
}  // namespace cuspatial
