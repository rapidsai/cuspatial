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

#include <utilities/error_utils.hpp>
#include <cuspatial/pip.hpp>
#include <gtest/gtest.h>
#include "pip_util.h"

#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/cudf_test_utils.cuh>
#include <tests/utilities/cudf_test_fixtures.h>

using namespace cuspatial;

template <typename T>
struct PIPTest : public GdfTest 
{
    T * x=nullptr, *y=nullptr;
    int point_len=-1;
    struct polygons<T> polygon;
    size_t free_mem = 0, total_mem = 0;
    
    int set_initialize(const char *poly_filename, const char *point_filename)
    {

     cudaMemGetInfo(&free_mem, &total_mem);
     std::cout<<"GPU total_mem="<<total_mem<<std::endl;
     std::cout<<"beginning GPU free_mem="<<free_mem<<std::endl;
     
     struct timeval t0,t1,t2;
     gettimeofday(&t0, nullptr);
     struct polygons<double> polygon;
     read_polygon_soa<double>(poly_filename,&polygon);
     gettimeofday(&t1, nullptr);
     float ply_load_time=cuspatial::calc_time("polygon data loading time=", t0,t1);
     CUDF_EXPECTS(polygon.num_feature>0 && polygon.num_ring>0,"invalid # of polygons/rings");
     CUDF_EXPECTS(polygon.num_feature<=polygon.num_ring,"a polygon must have at least one ring_length");
     CUDF_EXPECTS(polygon.y!=nullptr && polygon.y!=nullptr,"polygon vertex x/y array can not be nullptr");
    
     //from len to pos using inclusive scan (partial sum)
     //support both in-place (saving memory) and regular
     //for in-place prefix sum (pos and len arrays point to the same mem addresses)
     if(polygon.is_inplace)
     {
         polygon.group_position=polygon.group_length;
	 polygon.feature_position=polygon.feature_length;
	 polygon.ring_position=polygon.ring_length;
     }
     else
     {
	 polygon.group_position=new uint[polygon.num_group];
	 polygon.feature_position=new uint[polygon.num_feature];
	 polygon.ring_position=new uint[polygon.num_ring];
     }

     std::partial_sum(polygon.group_length,polygon.group_length+polygon.num_group,polygon.group_position,std::plus<uint>());
     std::partial_sum(polygon.feature_length,polygon.feature_length+polygon.num_feature,polygon.feature_position,std::plus<uint>());
     std::partial_sum(polygon.ring_length,polygon.ring_length+polygon.num_ring,polygon.ring_position,std::plus<uint>());

     /*printf("after partial sum\n");
     printf("num_group=%d\n",polygon.num_group);
     for(int i=0;i<polygon.num_group;i++)
	printf("(%u %u)",polygon.group_length[i],polygon.group_position[i]);
     printf("\n");

     printf("num_feature=%d\n",polygon.num_feature);
     for(int i=0;i<polygon.num_feature;i++)
	printf("(%u %u)",polygon.feature_length[i],polygon.feature_position[i]);
     printf("\n");
     printf("num_ring=%d\n",polygon.num_ring);
     for(int i=0;i<polygon.num_ring;i++)
		printf("(%u %u)",polygon.ring_length[i],polygon.ring_position[i]);
     printf("\n");*/
 
     point_len=read_point_lonlat<T>(point_filename,x,y);
     //std::cout<<"point_len="<<point_len<<std::endl;
     CUDF_EXPECTS(point_len>0,"# of points must be greater than 0");
     CUDF_EXPECTS(y!=nullptr && y!=nullptr,"point x/y array can not be nullptr");
      
     /*for(int i=0;i<10;i++)
     std::cerr<<i<<","<<x[i].longitude<<","<<y[i].lat<<std::endl;*/
      
     gettimeofday(&t2, nullptr);
     float pnt_load_time=cuspatial::calc_time("point data loading time ......",t1,t2);      
     return (0);
    }
    
    void exec_gpu_pip(uint *& gpu_pip_res)
    {
     
        //std::vector g_pos_v(polygon.group_position,polygon.group_position+polygon.num_group);
        std::vector<uint> f_pos_v(polygon.feature_position,polygon.feature_position+polygon.num_feature);
        std::vector<uint> r_pos_v(polygon.ring_position,polygon.ring_position+polygon.num_ring);
        std::vector<T> ply_x_v(polygon.x,polygon.x+polygon.num_vertex);
	std::vector<T> ply_y_v(polygon.y,polygon.y+polygon.num_vertex);
        std::vector<T> pnt_x_v(x,x+this->point_len);
	std::vector<T> pnt_y_v(y,y+this->point_len);   
        
        cudf::test::column_wrapper<uint> polygon_fpos_wrapp{f_pos_v};
        cudf::test::column_wrapper<uint> polygon_rpos_wrapp{r_pos_v};
        cudf::test::column_wrapper<T> polygon_x_wrapp{ply_x_v};
        cudf::test::column_wrapper<T> polygon_y_wrapp{ply_y_v};
        cudf::test::column_wrapper<T> point_x_wrapp{pnt_x_v};
        cudf::test::column_wrapper<T> point_y_wrapp{pnt_y_v};
        //std::cout<<"point_x_wrapp type="<<point_x_wrapp.get()->dtype<<std::endl;
        
        gdf_column res_bm1 = cuspatial::pip_bm( 
        	*(point_x_wrapp.get()), *(point_y_wrapp.get()),
        	*(polygon_fpos_wrapp.get()), *(polygon_rpos_wrapp.get()), 
        	*(polygon_x_wrapp.get()), *(polygon_y_wrapp.get()) );
    
        gpu_pip_res=new uint[this->point_len];
        CUDF_EXPECTS(gpu_pip_res!=nullptr,"error in allocating memory for results");
    	EXPECT_EQ(cudaMemcpy(gpu_pip_res, res_bm1.data, this->point_len * sizeof(uint), cudaMemcpyDeviceToHost), cudaSuccess);
    }
    
    void set_finalize()
    {
	delete [] polygon.group_length;
      	delete [] polygon.feature_length;
      	delete [] polygon.ring_length;
      	if(!polygon.is_inplace)
      	{
      		delete [] polygon.group_position;
      		delete [] polygon.feature_position;
      		delete [] polygon.ring_position;
      	}
      	delete [] polygon.x;
      	delete [] polygon.y;
      	
    	delete[] x;
    	delete[] y;
    	
    }
    
};

//typedef testing::Types<int16_t, int32_t, int64_t, float, double> NumericTypes;
typedef testing::Types<double> NumericTypes;

TYPED_TEST_CASE(PIPTest, NumericTypes);


TYPED_TEST(PIPTest, piptest)
{
    std::string pnt_filename =std::string("/home/jianting/cuspatial/data/locust.location");
    std::string ply_filename=std::string("/home/jianting/cuspatial/data/itsroi.ply"); 
    ASSERT_GE(this->set_initialize(ply_filename.c_str(),pnt_filename.c_str()),0);
    
    struct timeval t3,t4,t5,t6;
    gettimeofday(&t3, nullptr);
    uint* cpu_pip_res=new uint[this->point_len];
    CUDF_EXPECTS(cpu_pip_res!=nullptr,"failed allocating output bitmap memory");
    std::cout<<"beginning cpu_pip_loop.............."<<std::endl;
    cpu_pip_loop(this->point_len,this->x, this->y,this->polygon,cpu_pip_res);
    	
    gettimeofday(&t4, nullptr);
    cuspatial::calc_time("CPU PIP time......",t3,t4);

    //x,y and gpu_pip_res on CPU; pip code allocates memory
    //x and y will be uploaded and res will be downloaded automatically
    //cost include both computation and data transfer time
    
    uint* gpu_pip_res=nullptr;
    this->exec_gpu_pip(gpu_pip_res);
    assert(gpu_pip_res!=nullptr);
    
    gettimeofday(&t5, nullptr);
    float gpu_pip_time1=cuspatial::calc_time("GPU PIP time 1(including point data transfer and kernel time)......",t4,t5);

    //Testing asynchronous issues by 2nd call
    uint* gpu_pip_res2=nullptr;
    this->exec_gpu_pip(gpu_pip_res2);
    assert(gpu_pip_res2!=nullptr);
    
    gettimeofday(&t6, nullptr);
    float gpu_pip_time2=cuspatial::calc_time("GPU PIP time 2(including point data transfer and kernel time)......",t5,t6);
    delete[] gpu_pip_res2;

    int err_cnt=0,non_zero=0;
    for(int i=0;i<this->point_len;i++)
    {
	if(cpu_pip_res[i]!=gpu_pip_res[i])
	{
		/*printf("ERR: %d %d %d, G=%08x C=%08x\n",i,__builtin_popcount(cpu_pip_res[i]),
			__builtin_popcount(gpu_pip_res[i]), (unsigned int)(cpu_pip_res[i]),(unsigned int)(gpu_pip_res[i]));*/
		err_cnt++;
	}
	if(cpu_pip_res[i]!=0&&gpu_pip_res[i]!=0)
		non_zero++;
    }
    if(err_cnt==0)
	std::cout<<"GPU and CPU results are identical...................OK"<<std::endl;     	
    else
	std::cout<<"# of GPU and CPU diffs="<<err_cnt<<std::endl;     	
    std::cout<<"non zero results="<<non_zero<<std::endl;

    delete[] gpu_pip_res;
    
    this->set_finalize();
    
    cudaMemGetInfo(&this->free_mem, &this->total_mem);
    std::cout<<"ending GPU free mem "<<this->free_mem<<std::endl;
}
