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

#include <cuprojshim.hpp>

namespace cuprojshim {

// explicit template instantiations

// float

template cuproj::projection<cuproj::vec_2d<float>>* make_projection<float>(
  std::string const& src_epsg, std::string const& dst_epsg);

template void transform<float>(cuproj::projection<cuproj::vec_2d<float>> const& proj,
                               cuproj::vec_2d<float>* xy_in,
                               cuproj::vec_2d<float>* xy_out,
                               std::size_t n,
                               cuproj::direction dir);

template void transform<float>(cuproj::projection<cuproj::vec_2d<float>> const& proj,
                               float* x_in,
                               float* y_in,
                               float* x_out,
                               float* y_out,
                               std::size_t n,
                               cuproj::direction dir);

// double

template cuproj::projection<cuproj::vec_2d<double>>* make_projection<double>(
  std::string const& src_epsg, std::string const& dst_epsg);

template void transform<double>(cuproj::projection<cuproj::vec_2d<double>> const& proj,
                                cuproj::vec_2d<double>* xy_in,
                                cuproj::vec_2d<double>* xy_out,
                                std::size_t n,
                                cuproj::direction dir);

template void transform<double>(cuproj::projection<cuproj::vec_2d<double>> const& proj,
                                double* x_in,
                                double* y_in,
                                double* x_out,
                                double* y_out,
                                std::size_t n,
                                cuproj::direction dir);
}  // namespace cuprojshim
