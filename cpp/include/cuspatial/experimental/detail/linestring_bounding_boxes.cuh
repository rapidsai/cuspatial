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

template <class LinestringOffsetIterator,
          class VertexIterator,
          class BoundingBoxIterator,
          class T,
          class IndexT>
BoundingBoxIterator linestring_bounding_boxes(LinestringOffsetIterator linestring_offsets_first,
                                              LinestringOffsetIterator linestring_offsets_last,
                                              VertexIterator linestring_vertices_first,
                                              VertexIterator linestring_vertices_last,
                                              BoundingBoxIterator bounding_boxes_first,
                                              T expansion_radius,
                                              rmm::cuda_stream_view stream)
{
  static_assert(is_floating_point<T>(), "Only floating point polygon vertices supported");

  static_assert(is_same<vec_2d<T>, iterator_value_type<VertexIterator>>(),
                "Input vertices must be cuspatial::vec_2d");

  static_assert(cuspatial::is_integral<iterator_value_type<LinestringOffsetIterator>>(),
                "Offset iterators must have integral value type.");

  // GeoArrow: Number of linestrings is number of offsets minus one.
  auto const num_linestrings = std::distance(linestring_offsets_first, linestring_offsets_last) - 1;
  auto const num_vertices    = std::distance(linestring_vertices_first, linestring_vertices_last);

  if (num_linestrings == 0 || num_vertices == 0) { return bounding_boxes_first; }

  auto vertex_ids_iter =
    make_geometry_id_iterator<IndexT>(linestring_offsets_first, linestring_offsets_last);

  return point_bounding_boxes(vertex_ids_iter,
                              vertex_ids_iter + num_vertices,
                              linestring_vertices_first,
                              bounding_boxes_first,
                              expansion_radius,
                              stream);
}
}  // namespace cuspatial
