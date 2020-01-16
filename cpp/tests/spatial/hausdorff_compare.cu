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
#include <vector>
#include <string>
#include <iostream>

#include <gtest/gtest.h>
#include <utilities/legacy/error_utils.hpp>

#include <cuspatial/soa_readers.hpp> 
#include <cuspatial/hausdorff.hpp> 
#include <utility/utility.hpp> 
#include "hausdorff_util.h" 

#include <tests/utilities/legacy/cudf_test_utils.cuh>
#include <tests/utilities/legacy/cudf_test_fixtures.h>

struct HausdorffCompare : public GdfTest 
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
TEST_F(HausdorffCompare, hausdorfftest)
{
    //currently using hard coded paths; to be updated
    std::string point_fn =std::string("/home/jianting/trajcode/locust256.coor");
    std::string cnt_fn =std::string("/home/jianting/trajcode/locust256.objcnt");
    
    //initializaiton
    this->set_initialize(point_fn.c_str(),cnt_fn.c_str());
    
    //run cuspatial::directed_hausdorff_distance twice 
    struct timeval t0,t1;
    gettimeofday(&t0, nullptr);    
    gdf_column dist=cuspatial::directed_hausdorff_distance(this->pnt_x,this->pnt_y, this->cnt);         
    gettimeofday(&t1, nullptr);
    float gpu_hausdorff_time=cuspatial::calc_time("GPU Hausdorff Distance time......",t0,t1);
    
    int set_size=this->cnt.size;
    int num_pair=dist.size;
    assert(num_pair==set_size*set_size);
    std::cout<<"num_pair="<<num_pair<<std::endl;
    
	
    //transfer data to CPU and run on CPU 	
    int num_pnt=this->pnt_x.size;
    double *x_c=new double[num_pnt];
    double *y_c=new double[num_pnt];
    uint32_t *cnt_c=new uint32_t[set_size];
    assert(x_c!=nullptr && y_c!=nullptr && cnt_c!=nullptr);
    cudaMemcpy(x_c,this->pnt_x.data ,num_pnt*sizeof(double) , cudaMemcpyDeviceToHost);
    cudaMemcpy(y_c,this->pnt_y.data ,num_pnt*sizeof(double) , cudaMemcpyDeviceToHost);
    cudaMemcpy(cnt_c,this->cnt.data ,set_size*sizeof(uint32_t) , cudaMemcpyDeviceToHost);
    
    //test only the first subset_size pairs on CPUs
    int subset_size=100;
    double *dist_c=nullptr;
    hausdorff_test_sequential<double>(subset_size,x_c,y_c,cnt_c,dist_c);
    assert(dist_c!=nullptr);
    
    double *dist_h=new double[num_pair];    
    cudaMemcpy(dist_h,dist.data ,num_pair*sizeof(double) , cudaMemcpyDeviceToHost);
    
    //verify the CPU results are the same as the two GPU results
    int diff_cnt=0	;
    for(int i=0;i<subset_size;i++)
    {
    	for(int j=0;j<subset_size;j++)
    	{
    		int p1=i*subset_size+j;
    		int p2=i*set_size+j;
    		if(fabs(dist_c[p1]-dist_h[p2])>0.00001)
    		{
    			//std::cout<<"diff:("<<i<<","<<j<<") "<<dist_c[p1]<<"  "<<dist_h[p2]<<std::endl;
    			diff_cnt++;
    		}
   	}
    }
   
   if(diff_cnt==0)
	std::cout<<"GPU and CPU results are identical...................OK"<<std::endl;     	
    else
	std::cout<<"# of GPU and CPU diffs="<<diff_cnt<<std::endl;     
	
    cudaMemGetInfo(&this->free_mem, &this->total_mem);
    std::cout<<"ending GPU free mem "<<this->free_mem<<std::endl;
}
#endif
