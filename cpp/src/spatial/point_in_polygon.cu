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

#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <utilities/legacy/cuda_utils.hpp>
#include <type_traits>
#include <utility/utility.hpp>
#include <cuspatial/point_in_polygon.hpp>

#include <cudf/legacy/column.hpp>

namespace {

template <typename T>
__global__ void pip_kernel(gdf_size_type pnt_size,const T* const __restrict__ pnt_x,const T* const __restrict__ pnt_y,
        gdf_size_type ply_size,const uint32_t* const __restrict__ ply_fpos,const uint32_t* const __restrict__ ply_rpos,
        const T* const __restrict__ ply_x,const T* const __restrict__ ply_y,
        uint32_t* const __restrict__ res_bm)
{
    uint32_t mask=0;
    //assuming 1D grid/block config
    uint32_t idx =blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=pnt_size) return;

    T x = pnt_x[idx];
    T y = pnt_y[idx];
    for (uint32_t j = 0; j < ply_size; j++) //for each polygon
    {
       uint32_t r_f = (0 == j) ? 0 : ply_fpos[j-1];
       uint32_t r_t=ply_fpos[j];
       bool in_polygon = false;
       for (uint32_t k = r_f; k < r_t; k++) //for each ring
       {
           uint32_t m = (k==0)?0:ply_rpos[k-1];
           for (;m < ply_rpos[k]-1; m++) //for each line segment
           {
              T x0, x1, y0, y1;
              x0 = ply_x[m];
              y0 = ply_y[m];
              x1 = ply_x[m+1];
              y1 = ply_y[m+1];

              if ((((y0 <= y) && (y < y1)) ||
                   ((y1 <= y) && (y < y0))) &&
                       (x < (x1 - x0) * (y - y0) / (y1 - y0) + x0))
                 in_polygon = !in_polygon;
            }
      }
      if(in_polygon)
      	mask|=(0x01<<j);
   }
   res_bm[idx]=mask;
}

struct pip_functor {
    template <typename T>
    static constexpr bool is_supported()
    {
        return std::is_floating_point<T>::value;
    }

    template <typename T, std::enable_if_t< is_supported<T>() >* = nullptr>
    gdf_column operator()(gdf_column const & pnt_x,gdf_column const & pnt_y,
 			  gdf_column const & ply_fpos,gdf_column const & ply_rpos,
			  gdf_column const & ply_x,gdf_column const & ply_y)
    {
        gdf_column res_bm;
        uint32_t* temp_bitmap{nullptr};

        cudaStream_t stream{0};
        RMM_TRY( RMM_ALLOC(&temp_bitmap, pnt_y.size * sizeof(uint32_t), stream) );

        gdf_size_type min_grid_size = 0, block_size = 0;
        CUDA_TRY( cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, pip_kernel<T>) );
        cudf::util::cuda::grid_config_1d grid{pnt_y.size, block_size, 1};

        pip_kernel<T> <<< grid.num_blocks, block_size >>> (pnt_x.size,
               	static_cast<T*>(pnt_x.data), static_cast<T*>(pnt_y.data),
        	ply_fpos.size,static_cast<uint32_t*>(ply_fpos.data),static_cast<uint32_t*>(ply_rpos.data),
        	static_cast<T*>(ply_x.data), static_cast<T*>(ply_y.data),
                temp_bitmap);
        CUDA_TRY( cudaDeviceSynchronize() );

        gdf_column_view(&res_bm, temp_bitmap, nullptr, pnt_x.size, GDF_INT32);

        return res_bm;
    }

    template <typename T, std::enable_if_t< !is_supported<T>() >* = nullptr>
    gdf_column operator()(gdf_column const & pnt_x,gdf_column const & pnt_y,
 			  gdf_column const & ply_fpos,gdf_column const & ply_rpos,
			  gdf_column const & ply_x,gdf_column const & ply_y)

    {
        CUDF_FAIL("Non-floating point operation is not supported");
    }
};

} // namespace anonymous

namespace cuspatial {

/*
 * Point-in-Polygon (PIP) tests among a column of points and a column of
 * polygons. See point_in_polygon.hpp
 */
gdf_column point_in_polygon_bitmap(const gdf_column& points_x,
                                   const gdf_column& points_y,
                                   const gdf_column& poly_fpos,
                                   const gdf_column& poly_rpos,
                                   const gdf_column& poly_x,
                                   const gdf_column& poly_y)
{

    CUDF_EXPECTS(points_y.data != nullptr && points_x.data != nullptr, "query point data cannot be empty");
    CUDF_EXPECTS(points_y.dtype == points_x.dtype, "polygon vertex and point temp_bitmap type mismatch for x array ");

    //future versions might allow pnt_(x/y) have null_count>0, which might be useful for taking query results as inputs
    CUDF_EXPECTS(points_x.null_count == 0 && points_y.null_count == 0, "this version does not support points_x/points_y contains nulls");

    CUDF_EXPECTS(poly_fpos.data != nullptr &&poly_rpos.data!=nullptr, "polygon index cannot be empty");
    CUDF_EXPECTS(poly_fpos.size >0 && (size_t)poly_fpos.size<=sizeof(uint32_t)*8, "#polygon of polygons can not exceed bitmap capacity (32 for unsigned int)");
    CUDF_EXPECTS(poly_y.data != nullptr && poly_x.data != nullptr, "polygon temp_bitmap cannot be empty");
    CUDF_EXPECTS(poly_fpos.size <=poly_rpos.size,"#of polygons must be equal or less than # of rings (one polygon has at least one ring");
    CUDF_EXPECTS(poly_y.size == poly_x.size, "polygon vertice sizes mismatch between x/y arrays");
    CUDF_EXPECTS(points_y.size == points_x.size, "query points size mismatch from between x/y arrays");
    CUDF_EXPECTS(poly_y.dtype == poly_x.dtype, "polygon vertex temp_bitmap type mismatch between x/y arrays");
    CUDF_EXPECTS(poly_y.dtype == points_y.dtype, "polygon vertex and point temp_bitmap type mismatch for y array");
    CUDF_EXPECTS(poly_x.null_count == 0 && poly_y.null_count == 0, "polygon should not contain nulls");

    gdf_column res_bm = cudf::type_dispatcher(points_x.dtype, pip_functor(),
                                              points_x, points_y, poly_fpos,
                                              poly_rpos,poly_x,poly_y);

    return res_bm;
  }//point_in_polygon_bitmap

}// namespace cuspatial
