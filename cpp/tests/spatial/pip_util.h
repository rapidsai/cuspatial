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

#include <cuspatial/types.hpp>
#include <utility/utility.hpp>

namespace cuspatial
{
/**
 * @brief sequential point-in-polygon test between a single point and a single polygon;
 *        basic unit for either sequential execution or parallellization in multi-point/multi-polygon test cases
 *
 * @param[in]  x    the x coordinates of the input points
 * @param[in]  y    the y coordinates of the input points
 * @param[in]  ply    complete metadata for a polygon dataset (with multiple polygons)
 * @param[in]  fid    index of the polygon dataset to identify the polyogn to be tested
 *
 * @return whehter the point is in the polygon
 */
template <typename T>
bool pip_test_sequential(const T& x, const T& y,
                         const struct polygons<T>& ply, int fid)
{
    uint32_t *f_pos=ply.feature_position;
    uint32_t *r_pos=ply.ring_position;
    T *poly_x=ply.x;
    T *poly_y=ply.y;
    uint32_t r_f = (0 == fid) ? 0 : f_pos[fid-1];
    uint32_t r_t=f_pos[fid];
    bool in_polygon = false;
    for (uint32_t r = r_f; r < r_t; r++) //for each ring
    {
        uint32_t m = (r==0)?0:r_pos[r-1];
        for (;m < r_pos[r]-1; m++) //for each line segment
        {
            T x0, x1, y0, y1;
            x0 = poly_x[m];
            y0 = poly_y[m];
            x1 = poly_x[m+1];
            y1 = poly_y[m+1];
            if ((((y0 <= y) && (y < y1)) ||
                    ((y1 <= y) && (y < y0))) &&
                    (x < (x1 - x0) * (y - y0) / (y1 - y0) + x0))
                in_polygon = !in_polygon;
        }
        }
    return (in_polygon);
}

template bool pip_test_sequential(const double& x, const double& y,
                                  const struct polygons<double>& ply, int fid);
template bool pip_test_sequential(const float& x, const float& y,
                                  const struct polygons<float>& ply, int fid);

/**
 * @brief multi-point/multi-polygon test on CPU with the same interface as the GPU implementation
 *  parallelization (e.g., OpenMP and Intel TBB) can be applied to the array/vector of points.
 *
 * @param[in]  num_pnt    number of points
 * @param[in]  x    pointer/array of x coodinates
 * @param[in]  x    pointer/array of y coodinates
 * @param[in]  ply  complete metadata for a polygon dataset (with multiple polygons)
 * @param[out]  res  pointer/array of unsinged integers; the jth bit of res[i] indicates whehter
 *                   a point of (x[i],y[i]) is in polygon j.
 *
 * @note The # of polygons, i.e., poly.f_num can not exceed sizeof(uint32_t)*8, i.e., 32.
 */
template <typename T>
std::vector<uint32_t> cpu_pip_loop(int num_pnt,const T* x, const T *y,
                               const struct polygons<T>& poly)
{
    std::vector<uint32_t> res;
    for(int i=0;i<num_pnt;i++)
    {
        uint32_t mask=0;
        for(size_t j=0;j<poly.num_feature;j++)
        {
            bool in_polygon =pip_test_sequential<T>(x[i],y[i],poly,j);
            if(in_polygon)
            {
                mask|=(0x01<<j);
            }
        }
        res.push_back(mask);
    }
    return res;
}

template std::vector<uint32_t> cpu_pip_loop(int num_pnt, const double *x,
                                        const double *y,
                                        const struct polygons<double>& poly);
template std::vector<uint32_t> cpu_pip_loop(int num_pnt, const float *x,
                                        const float *y,
                                        const struct polygons<float>& poly);

}//namespace cuspatial
