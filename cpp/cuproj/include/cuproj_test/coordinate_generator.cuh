/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/tabulate.h>

namespace cuproj_test {

// Generate a grid of coordinates
template <typename Coord>
struct grid_generator {
  Coord min_corner{};
  Coord max_corner{};
  Coord spacing{};
  int num_points_x{};

  grid_generator(Coord const& min_corner,
                 Coord const& max_corner,
                 int num_points_x,
                 int num_points_y)
    : min_corner(min_corner), max_corner(max_corner), num_points_x(num_points_x)
  {
    spacing = Coord{(max_corner.x - min_corner.x) / num_points_x,
                    (max_corner.y - min_corner.y) / num_points_y};
  }

  __device__ Coord operator()(int i) const
  {
    return min_corner + Coord{(i % num_points_x) * spacing.x, (i / num_points_x) * spacing.y};
  }
};

// Create a Vector containing a grid of coordinates between the min and max corners
template <typename Coord, typename Vector>
auto make_grid_array(Coord const& min_corner,
                     Coord const& max_corner,
                     int num_points_x,
                     int num_points_y)
{
  auto gen = grid_generator(min_corner, max_corner, num_points_x, num_points_y);
  Vector grid(num_points_x * num_points_y);
  thrust::tabulate(rmm::exec_policy(), grid.begin(), grid.end(), gen);
  return grid;
}

}  // namespace cuproj_test
