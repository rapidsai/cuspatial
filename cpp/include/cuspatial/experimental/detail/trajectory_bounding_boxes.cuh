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

#include <cuspatial/experimental/detail/bounding_boxes.cuh>
#include <cuspatial/traits.hpp>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/zip_function.h>

namespace cuspatial {

template <typename IdInputIt, typename PointInputIt, typename BoundingBoxOutputIt>
BoundingBoxOutputIt trajectory_bounding_boxes(IdInputIt ids_first,
                                              IdInputIt ids_last,
                                              PointInputIt points_first,
                                              BoundingBoxOutputIt bounding_boxes_first,
                                              rmm::cuda_stream_view stream)
{
  return detail::point_bounding_boxes(
    ids_first, ids_last, points_first, bounding_boxes_first, 0, stream);
}

}  // namespace cuspatial
