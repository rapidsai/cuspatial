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
#include <cuspatial/experimental/iterator_factory.cuh>
#include <cuspatial/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>

namespace cuspatial {

template <class PolygonOffsetIterator,
          class RingOffsetIterator,
          class VertexIterator,
          class BoundingBoxIterator,
          class T,
          class IndexT>
BoundingBoxIterator polygon_bounding_boxes(PolygonOffsetIterator polygon_offsets_first,
                                           PolygonOffsetIterator polygon_offsets_last,
                                           RingOffsetIterator polygon_ring_offsets_first,
                                           RingOffsetIterator polygon_ring_offsets_last,
                                           VertexIterator polygon_vertices_first,
                                           VertexIterator polygon_vertices_last,
                                           BoundingBoxIterator bounding_boxes_first,
                                           T expansion_radius,
                                           rmm::cuda_stream_view stream)
{
  static_assert(is_floating_point<T>(), "Only floating point polygon vertices supported");

  static_assert(is_same<vec_2d<T>, iterator_value_type<VertexIterator>>(),
                "Input vertices must be cuspatial::vec_2d");

  static_assert(cuspatial::is_integral<iterator_value_type<PolygonOffsetIterator>,
                                       iterator_value_type<RingOffsetIterator>>(),
                "OffsetIterators must have integral value type.");

  auto const num_polys = std::distance(polygon_offsets_first, polygon_offsets_last) - 1;
  auto const num_rings = std::distance(polygon_ring_offsets_first, polygon_ring_offsets_last) - 1;
  auto const num_vertices = std::distance(polygon_vertices_first, polygon_vertices_last);

  CUSPATIAL_EXPECTS(num_rings >= num_polys, "Each polygon must have at least one ring");

  CUSPATIAL_EXPECTS(num_vertices >= num_polys * 3, "Each ring must have at least three vertices");

  if (num_polys == 0 || num_rings == 0 || num_vertices == 0) { return bounding_boxes_first; }

  auto vertex_ids_iter = make_geometry_id_iterator<IndexT>(
    polygon_offsets_first, polygon_offsets_last, polygon_ring_offsets_first);

  return point_bounding_boxes(vertex_ids_iter,
                              vertex_ids_iter + num_vertices,
                              polygon_vertices_first,
                              bounding_boxes_first,
                              expansion_radius,
                              stream);
}
}  // namespace cuspatial
