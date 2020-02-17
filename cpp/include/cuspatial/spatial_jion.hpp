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
std::unique_ptr<cudf::experimental::table> quad_bbox_join(cudf::table_view const&,
	cudf::table_view const&,double,double,double,double, double,uint32_t,uint32_t);

std::unique_ptr<cudf::experimental::table> pip_refine(const cudf::table_view&,
	const cudf::table_view&,const cudf::table_view&,
	const cudf::table_view&,const cudf::column_view&,const cudf::column_view&);

}// namespace cuspatial
