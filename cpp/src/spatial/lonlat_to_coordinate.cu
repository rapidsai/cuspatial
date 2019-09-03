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
#include <utilities/cuda_utils.hpp>
#include <type_traits>
#include <utility>
#include <thrust/device_vector.h>
#include <sys/time.h>
#include <time.h>

#include <utility/utility.hpp>
#include <cuspatial/coordinate_transform.hpp>

template <typename T>
__global__ void coord_trans_kernel(gdf_size_type loc_size,
                                   double cam_lon, double cam_lat,
                                   const T* const __restrict__ in_lon,
                                   const T* const __restrict__ in_lat,
                                   T* const __restrict__ out_x,
                                   T* const __restrict__ out_y)
{
    //assuming 1D grid/block config
    uint32_t idx =blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=loc_size) return;    
    out_x[idx]=((cam_lon - in_lon[idx]) * 40000.0 *cos((cam_lat + in_lat[idx]) * M_PI / 360) / 360);
    out_y[idx]=(cam_lat - in_lat[idx]) * 40000.0 / 360;
}

struct ll2coord_functor {
    template <typename col_type>
    static constexpr bool is_supported()
    {
        return std::is_floating_point<col_type>::value;
    }

    template <typename col_type, std::enable_if_t< is_supported<col_type>() >* = nullptr>
    std::pair<gdf_column,gdf_column> operator()(const gdf_scalar  & cam_lon,const gdf_scalar  & cam_lat,
    	 const gdf_column  & in_lon,const gdf_column  & in_lat)
    	
    {
        gdf_column  out_x, out_y;
        int num_print=(in_lon.size<10)?in_lon.size:10;
        std::cout<<"ll2coord: showing the first "<< num_print<<" output records"<<std::endl;
        std::cout<<"in_lon"<<std::endl;
        thrust::device_ptr<col_type> in_lon_ptr=thrust::device_pointer_cast(static_cast<col_type*>(in_lon.data));
        thrust::copy(in_lon_ptr,in_lon_ptr+10,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl; 
        std::cout<<"in_lat"<<std::endl;
        thrust::device_ptr<col_type>in_lat_ptr=thrust::device_pointer_cast(static_cast<col_type*>(in_lat.data));
        thrust::copy(in_lat_ptr,in_lat_ptr+10,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl;  
 
 	out_x.dtype= in_lon.dtype;
  	out_x.col_name=(char *)malloc(strlen("x")+ 1);
	strcpy(out_x.col_name,"x");    
        RMM_TRY( RMM_ALLOC(&out_x.data, in_lon.size * sizeof(col_type), 0) );
     	out_x.size=in_lon.size;
     	out_x.valid=nullptr;
     	out_x.null_count=0;		

 	out_y.dtype= in_lat.dtype;
  	out_y.col_name=(char *)malloc(strlen("y")+ 1);
	strcpy(out_x.col_name,"x");    
        RMM_TRY( RMM_ALLOC(&out_y.data, in_lon.size * sizeof(col_type), 0) );
     	out_y.size=in_lat.size;
     	out_y.valid=nullptr;
     	out_y.null_count=0;	
        
        struct timeval t0,t1;
        gettimeofday(&t0, nullptr);
        
        gdf_size_type min_grid_size = 0, block_size = 0;
        CUDA_TRY( cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, coord_trans_kernel<col_type>) );
        cudf::util::cuda::grid_config_1d grid{in_lon.size, block_size, 1};
        std::cout<<"in_lon.size="<<in_lon.size<<" block_size="<<block_size<<std::endl;
       
        coord_trans_kernel<col_type> <<< grid.num_blocks, block_size >>> (in_lon.size,
        	*((double*)(&(cam_lon.data))),*((double*)(&(cam_lat.data))),
   	    	static_cast<col_type*>(in_lon.data), static_cast<col_type*>(in_lat.data),
   	    	static_cast<col_type*>(out_x.data), static_cast<col_type*>(out_y.data) );           
        CUDA_TRY( cudaDeviceSynchronize() );

        gettimeofday(&t1, nullptr);
        float ll2coord_kernel_time = cuspatial::calc_time("lon/lat to x/y conversion kernel time in ms=",t0,t1);

        num_print=(out_x.size<10)?out_x.size:10;
        std::cout<<"ll2coord: showing the first "<< num_print<<" output records"<<std::endl;
        thrust::device_ptr<col_type> outx_ptr=thrust::device_pointer_cast(static_cast<col_type*>(out_x.data));
        thrust::device_ptr<col_type> outy_ptr=thrust::device_pointer_cast(static_cast<col_type*>(out_y.data));
        std::cout<<"out_x"<<std::endl;     
        thrust::copy(outx_ptr,outx_ptr+num_print,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl;     
        std::cout<<"out_y"<<std::endl;     
 	thrust::copy(outy_ptr,outy_ptr+num_print,std::ostream_iterator<col_type>(std::cout, " "));std::cout<<std::endl;  
 	
 	return std::make_pair(out_x,out_y);
    }

    template <typename col_type, std::enable_if_t< !is_supported<col_type>() >* = nullptr>
    std::pair<gdf_column,gdf_column> operator()(const gdf_scalar  & cam_lon,const gdf_scalar  & cam_lat,
    	const gdf_column  & in_lon,const gdf_column  & in_lat)
    {
        CUDF_FAIL("Non-floating point operation is not supported");
    }
};
    

namespace cuspatial {

/**
 * @brief transforming in_lon/in_lat (lon/lat defined in coord_2d) to out_x/out_y relative to a camera origiin
 * see coordinate_transform.hpp
*/

std::pair<gdf_column,gdf_column> lonlat_to_coord(const gdf_scalar& cam_lon, const gdf_scalar& cam_lat,
	const gdf_column& in_lon, const gdf_column  & in_lat)

{       
    struct timeval t0,t1;
    gettimeofday(&t0, nullptr);
    
    double cx=*((double*)(&(cam_lon.data)));
    double cy=*((double*)(&(cam_lat.data)));
    CUDF_EXPECTS(cx >=-180 && cx <=180 && cy >=-90 && cy <=90,
    	"camera origin must have valid lat/lon values [-180,-90,180,90]");
    CUDF_EXPECTS(in_lon.data != nullptr &&in_lat.data!=nullptr, "input point cannot be empty");
    CUDF_EXPECTS(in_lon.size == in_lat.size, "input x and y arrays must have the same length");
    
    //future versions might allow in_(x/y) have null_count>0, which might be useful for taking query results as inputs 
    CUDF_EXPECTS(in_lon.null_count == 0 && in_lat.null_count == 0, "this version does not support point in_lon/in_lat contains nulls");
    
    auto res=cudf::type_dispatcher(in_lon.dtype, ll2coord_functor(), cam_lon,cam_lat,in_lon,in_lat);
    
    // handle null_count if needed 
     
    gettimeofday(&t1, nullptr);
    float ll2coord_end2end_time=calc_time("lon/lat to x/y conversion end2end time in ms=",t0,t1);
    return res;
  }//lonlat_to_coord 
    	
}// namespace cuspatial
