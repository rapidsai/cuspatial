/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

namespace cuspatial {

/**
 * @brief .
 * @note: .
**/
std::unique_ptr<cudf::column> nested_column_test(cudf::column_view,cudf::column_view);

std::unique_ptr<cudf::experimental::table> quadtree_on_points(cudf::mutable_column_view,
	cudf::mutable_column_view,SBBox, double , int, int);

}// namespace cuspatial
