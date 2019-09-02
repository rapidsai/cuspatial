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
#include <cuspatial/point_in_polygon.hpp>
#include "pip_util.h"

#include <tests/utilities/cudf_test_utils.cuh>
#include <tests/utilities/cudf_test_fixtures.h>

using namespace cuspatial;

template <typename T>
struct PIPCompare : public GdfTest 
{
    T * h_point_x=nullptr, *h_point_y=nullptr;
    int point_len=-1;
    struct polygons<T> h_polygon;
    gdf_column point_x,point_y,feature_position,ring_position,polygon_x,polygon_y;
    size_t free_mem = 0, total_mem = 0;
 
    int set_initialize(const char *poly_filename, const char *point_filename)
    {     
        struct timeval t0,t1,t2,t3;
        gettimeofday(&t0, nullptr);
        read_polygon_soa<T>(poly_filename,&h_polygon);
        gettimeofday(&t1, nullptr);
        float ply_load_time=cuspatial::calc_time("h_polygon data loading time=", t0,t1);
        CUDF_EXPECTS(h_polygon.num_feature>0 && h_polygon.num_ring>0,"invalid # of polygons/rings");
        CUDF_EXPECTS(h_polygon.num_feature<=h_polygon.num_ring,"a h_polygon must have at least one ring_length");
        CUDF_EXPECTS(h_polygon.x!=nullptr && h_polygon.y!=nullptr,"h_polygon vertex h_point_x/h_point_y array can not be nullptr");

        //from length to position using inclusive scan (partial sum) on CPU
        //support both in-place (saving memory) and regular
        //for in-place prefix sum (pos and len arrays point to the same mem addresses)
        if(h_polygon.is_inplace)
        {
            h_polygon.group_position=h_polygon.group_length;
            h_polygon.feature_position=h_polygon.feature_length;
            h_polygon.ring_position=h_polygon.ring_length;
        }
        else
        {
            h_polygon.group_position=new uint32_t[h_polygon.num_group];
            h_polygon.feature_position=new uint32_t[h_polygon.num_feature];
            h_polygon.ring_position=new uint32_t[h_polygon.num_ring];
        }
        std::partial_sum(h_polygon.group_length,h_polygon.group_length+h_polygon.num_group,h_polygon.group_position,std::plus<uint32_t>());
        std::partial_sum(h_polygon.feature_length,h_polygon.feature_length+h_polygon.num_feature,h_polygon.feature_position,std::plus<uint32_t>());
        std::partial_sum(h_polygon.ring_length,h_polygon.ring_length+h_polygon.num_ring,h_polygon.ring_position,std::plus<uint32_t>());
        gettimeofday(&t1, nullptr);
        float polygon_load_time=cuspatial::calc_time("h_polygon data loading time ......",t0,t1);    

        point_len=read_point_lonlat<T>(point_filename,h_point_x,h_point_y);
        std::cout<<"point_len="<<point_len<<std::endl;
        CUDF_EXPECTS(point_len>0,"# of points must be greater than 0");
        CUDF_EXPECTS(h_point_x!=nullptr && h_point_y!=nullptr,"point h_point_x/h_point_y array can not be nullptr");
        /*int print_len=(point_len<10)?point_len:10;
        for(int i=0;i<print_len;i++)
            std::cout<<i<<","<<h_point_x[i]<<","<<h_point_y[i]<<std::endl;*/
        gettimeofday(&t2, nullptr);
        float pnt_load_time=cuspatial::calc_time("point data loading time ......",t1,t2);    

        //cpu->gpu data transfer  
        memset(&point_x,0,sizeof(gdf_column));
        memset(&point_y,0,sizeof(gdf_column));
        memset(&feature_position,0,sizeof(gdf_column));
        memset(&ring_position,0,sizeof(gdf_column));
        memset(&polygon_x,0,sizeof(gdf_column));
        memset(&polygon_y,0,sizeof(gdf_column));
        
        point_x.dtype= cudf::gdf_dtype_of<T>();
        point_y.dtype= cudf::gdf_dtype_of<T>();
        feature_position.dtype= GDF_INT32;
        ring_position.dtype= GDF_INT32;
        polygon_x.dtype= cudf::gdf_dtype_of<T>();
        polygon_y.dtype= cudf::gdf_dtype_of<T>();
        
        point_x.size=point_len;
        point_y.size=point_len;
        feature_position.size= h_polygon.num_feature;
        ring_position.size= h_polygon.num_ring;
        polygon_x.size=h_polygon.num_vertex;
        polygon_y.size=h_polygon.num_vertex;
        
        RMM_TRY( RMM_ALLOC(&(point_x.data), point_len * sizeof(T), 0) );
        RMM_TRY( RMM_ALLOC(&(point_y.data), point_len * sizeof(T), 0) );
        RMM_TRY( RMM_ALLOC(&(feature_position.data), h_polygon.num_feature * sizeof(uint32_t), 0) );
        RMM_TRY( RMM_ALLOC(&(ring_position.data), h_polygon.num_ring * sizeof(uint32_t), 0) );
        RMM_TRY( RMM_ALLOC(&(polygon_x.data), h_polygon.num_vertex * sizeof(T), 0) );
        RMM_TRY( RMM_ALLOC(&(polygon_y.data), h_polygon.num_vertex* sizeof(T), 0) );
        
        cudaMemcpy(point_x.data, h_point_x,point_len * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(point_y.data, h_point_y,point_len * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(feature_position.data, h_polygon.feature_position,h_polygon.num_feature * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(ring_position.data, h_polygon.ring_position,h_polygon.num_ring * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(polygon_x.data, h_polygon.x, h_polygon.num_vertex* sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(polygon_y.data, h_polygon.y, h_polygon.num_vertex* sizeof(T), cudaMemcpyHostToDevice);
        
        point_x.valid=nullptr;
        point_y.valid=nullptr;
        feature_position.valid=nullptr;
        ring_position.valid=nullptr;
        polygon_x.valid=nullptr;
        polygon_y.valid=nullptr;        
        
        gettimeofday(&t3, nullptr);
        float data_transfer_time=cuspatial::calc_time("point data loading time ......",t2,t3);    

        return (0);
    }

    std::vector<uint32_t> exec_gpu_pip()
    {
        gdf_column result_bitmap =
            cuspatial::point_in_polygon_bitmap(this->point_x, this->point_y,
                                               this->feature_position,
                                               this->ring_position,
                                               this->polygon_x, this->polygon_y);
        std::vector<uint32_t> h_result(this->point_len);
        EXPECT_EQ(cudaMemcpy(h_result.data(), result_bitmap.data, 
                             this->point_len * sizeof(uint32_t),
                             cudaMemcpyDeviceToHost), cudaSuccess);
        gdf_column_free(&result_bitmap);
        return h_result;
    }

    void set_finalize()
    {
        delete [] h_polygon.group_length;
        delete [] h_polygon.feature_length;
        delete [] h_polygon.ring_length;
        if(!h_polygon.is_inplace)
        {
            delete [] h_polygon.group_position;
            delete [] h_polygon.feature_position;
            delete [] h_polygon.ring_position;
        }
        delete [] h_polygon.x;
        delete [] h_polygon.y;
        delete[] h_point_x;
        delete[] h_point_y;
    }
};

//typedef testing::Types<int16_t, int32_t, int64_t, float, double> NumericTypes;
typedef testing::Types<double> NumericTypes;

TYPED_TEST_CASE(PIPCompare, NumericTypes);

#if 0  // disable until data files are checked in
TYPED_TEST(PIPCompare, piptest)
{
    std::string pnt_filename =std::string("/home/jianting/cuspatial/data/locust.location");
    std::string ply_filename=std::string("/home/jianting/cuspatial/data/itsroi.ply"); 
    ASSERT_GE(this->set_initialize(ply_filename.c_str(),pnt_filename.c_str()),0);
    
    struct timeval t3,t4,t5;
    gettimeofday(&t3, nullptr);
    uint32_t* cpu_pip_res=new uint32_t[this->point_len];
    CUDF_EXPECTS(cpu_pip_res!=nullptr,"failed allocating output bitmap memory");
    std::cout<<"beginning cpu_pip_loop.............."<<std::endl;
    cpu_pip_loop(this->point_len,this->h_point_x, this->h_point_y,this->h_polygon,cpu_pip_res);
    	
    gettimeofday(&t4, nullptr);
    cuspatial::calc_time("CPU PIP time......",t3,t4);

    //h_point_x,h_point_y and gpu_pip_res on CPU; pip code allocates memory
    //h_point_x and h_point_y will be uploaded and res will be downloaded automatically
    //cost include both computation and data transfer time

    std::vector<uint32_t> gpu_pip_res = this->exec_gpu_pip();

    gettimeofday(&t5, nullptr);
    float gpu_pip_time=cuspatial::calc_time("GPU PIP time 1(including point data transfer and kernel time)......",t4,t5);

    int err_cnt=0,non_zero=0;
    for(int i=0;i<this->point_len;i++)
    {
        if(cpu_pip_res[i]!=gpu_pip_res[i])
        {
            //TODO: change from printf to cout with proper formatting on bitmap values
            //printf("ERR: %d %d %d, G=%08x C=%08x\n",i,__builtin_popcount(cpu_pip_res[i]),
            //	__builtin_popcount(gpu_pip_res[i]), (unsigned int)(cpu_pip_res[i]),(unsigned int)(gpu_pip_res[i]));
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

    this->set_finalize();

    cudaMemGetInfo(&this->free_mem, &this->total_mem);
    std::cout<<"ending GPU free mem "<<this->free_mem<<std::endl;
}
#endif