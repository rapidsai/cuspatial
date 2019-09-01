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

#include <time.h>
#include <sys/time.h>

#include <gtest/gtest.h>
#include <utilities/error_utils.hpp>

#include <cuspatial/soa_readers.hpp> 
#include <cuspatial/hausdorff.hpp> 
#include <cuspatial/shared_util.h> 
#include "hausdorff_util.h" 

#include <tests/utilities/cudf_test_utils.cuh>
#include <tests/utilities/cudf_test_fixtures.h>

using namespace cuspatial;

struct is_true
{
	__host__ __device__
	bool operator()(const thrust::tuple<double, double>& t)
	{
		double v1= thrust::get<0>(t);
		double v2= thrust::get<1>(t);
		return(fabs(v1-v2)>0.01);
	}
};


struct HausdorffTest : public GdfTest 
{
    
    gdf_column pnt_x,pnt_y,cnt;
    size_t free_mem = 0, total_mem = 0;
    
    void set_initialize(const char *point_fn, const char *cnt_fn)
    {
    
      cudaMemGetInfo(&free_mem, &total_mem);
      std::cout<<"GPU total_mem="<<total_mem<<std::endl;
      std::cout<<"beginning GPU free_mem="<<free_mem<<std::endl;
      
      struct timeval t0,t1;
      gettimeofday(&t0, nullptr);
      
      auto points=cuspatial::read_xy_points_soa(point_fn);
      pnt_x=points.first;
      pnt_y=points.second;
      cnt=cuspatial::read_uint32_soa(cnt_fn);
      
      gettimeofday(&t1, nullptr);
      float data_load_time=cuspatial::calc_time("point/cnt data loading time=", t0,t1);
      CUDF_EXPECTS(pnt_x.size>0 && pnt_y.size>0 && cnt.size>=0,"invalid # of points/trajectories");
      CUDF_EXPECTS(pnt_x.size==pnt_y.size, "x and y columns must have the same size");
      CUDF_EXPECTS(pnt_y.size >=cnt.size ,"a point set must have at least one point");      
    }
};

#if 0 // disable until data files are available
TEST_F(HausdorffTest, hausdorfftest)
{
    //currently using hard coded paths; to be updated
    std::string point_fn =std::string("/home/jianting/trajcode/locust256.coor");
    std::string cnt_fn =std::string("/home/jianting/trajcode/locust256.objcnt");
    
    //initializaiton
    this->set_initialize(point_fn.c_str(),cnt_fn.c_str());
    
    //run cuspatial::directed_hausdorff_distance twice 
    struct timeval t0,t1,t2;
    gettimeofday(&t0, nullptr);
    
    gdf_column dist1=cuspatial::directed_hausdorff_distance(this->pnt_x,this->pnt_y, this->cnt);         
    assert(dist1.data!=nullptr);
    gettimeofday(&t1, nullptr);
    float gpu_hausdorff_time1=cuspatial::calc_time("GPU Hausdorff Distance time 1......",t0,t1);
    
    gdf_column dist2=cuspatial::directed_hausdorff_distance(this->pnt_x,this->pnt_y, this->cnt);         
    assert(dist2.data!=nullptr);
    gettimeofday(&t2, nullptr);
    float gpu_hausdorff_time2=cuspatial::calc_time("GPU Hausdorff Distance time 2......",t1,t2);
  
    CUDF_EXPECTS(dist1.size==dist2.size ,"output of the two rounds needs to have the same size");
       
    int set_size=this->cnt.size;
    int num_pair=dist1.size;
    assert(num_pair==set_size*set_size);
    std::cout<<"num_pair="<<num_pair<<std::endl;
    
    //verify the results of two GPU runs are the same
    double *data1=nullptr,*data2=nullptr;
    RMM_TRY( RMM_ALLOC((void**)&data1, sizeof(double)*num_pair, 0) );
    RMM_TRY( RMM_ALLOC((void**)&data2, sizeof(double)*num_pair, 0) );
    assert(data1!=nullptr && data2!=nullptr);
    cudaMemcpy(data1,dist1.data ,num_pair*sizeof(double) , cudaMemcpyDeviceToDevice);
    cudaMemcpy(data2,dist2.data ,num_pair*sizeof(double) , cudaMemcpyDeviceToDevice);
    
    thrust::device_ptr<double> d_dist1_ptr=thrust::device_pointer_cast(data1);
    thrust::device_ptr<double> d_dist2_ptr=thrust::device_pointer_cast(data2);
    auto it=thrust::make_zip_iterator(thrust::make_tuple(d_dist1_ptr,d_dist2_ptr));
    	
    int this_cnt=thrust::copy_if(it,it+num_pair,it,is_true())-it;	
    thrust::copy(d_dist1_ptr,d_dist1_ptr+this_cnt,std::ostream_iterator<double>(std::cout, " "));
    std::cout<<std::endl<<std::endl;
    thrust::copy(d_dist2_ptr,d_dist2_ptr+this_cnt,std::ostream_iterator<double>(std::cout, " "));
    std::cout<<std::endl<<std::endl;
	
    if(this_cnt==0)
	std::cout<<"Two rounds GPU results are identical...................OK"<<std::endl;     	
    else
	std::cout<<"Two rounds GPU results diff="<<this_cnt<<std::endl;     	
    RMM_TRY( RMM_FREE(data1, 0) );
    RMM_TRY( RMM_FREE(data2, 0) );

    cudaMemGetInfo(&this->free_mem, &this->total_mem);
    std::cout<<"ending GPU free mem "<<this->free_mem<<std::endl;
}
#endif
