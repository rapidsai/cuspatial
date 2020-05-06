/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>
#include <vector>

namespace cuspatial {

struct polygons
{
    std::vector<cudf::size_type> group_lengths;
    std::vector<cudf::size_type> feature_lengths;
    std::vector<cudf::size_type> ring_lengths;
    std::vector<double> xs;
    std::vector<double> ys;

    void shrink_to_fit()
    {
        group_lengths.shrink_to_fit();
        feature_lengths.shrink_to_fit();
        ring_lengths.shrink_to_fit();
        xs.shrink_to_fit();
        ys.shrink_to_fit();
    }
};

}
