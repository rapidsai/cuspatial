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
 * @brief compute bounding boxes (bboxes) of a set of polygons
 *
 * @param[in] spos: polyline offset array to vertices
 *
 * @param[in] x: polygon x coordiante array.
 *
 * @param[in] y: polygon y coordiante array.
 *
 * @param[in] R: expansion radius
 *
 * @return experimental::table with four arrays of bounding boxes, x1,y1,x2,y2.
*/

std::unique_ptr<cudf::experimental::table> polyline_bbox(
    const cudf::column_view& spos,
    const cudf::column_view& x,const cudf::column_view& y,
    double R);

}  // namespace cuspatial
