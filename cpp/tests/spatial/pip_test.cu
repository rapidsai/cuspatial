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
#include <utilities/legacy/error_utils.hpp>
#include <cuspatial/legacy/soa_readers.hpp>
#include <cuspatial/point_in_polygon.hpp>
#include "pip_util.h"

#include <tests/utilities/legacy/cudf_test_utils.cuh>
#include <tests/utilities/legacy/cudf_test_fixtures.h>

template <typename T>
struct PIPTest : public GdfTest 
{
    T * x=nullptr, *y=nullptr;
    int point_len=-1;
    gdf_column f_pos,r_pos,poly_x,poly_y,pnt_x,pnt_y;
    size_t free_mem = 0, total_mem = 0;
    
    int set_initialize(const char *poly_filename, const char *point_filename)
    {
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cout<<"GPU total_mem="<<total_mem<<std::endl;
        std::cout<<"beginning GPU free_mem="<<free_mem<<std::endl;
        
        struct timeval t0,t1,t2;
        gettimeofday(&t0, nullptr);
        cuspatial::read_polygon_soa(poly_filename,&f_pos,&r_pos,&poly_x,&poly_y);
        gettimeofday(&t1, nullptr);
        float ply_load_time=cuspatial::calc_time("polygon data loading time ......",t0,t1);      
        
        auto xy_pair=cuspatial::read_lonlat_points_soa(point_filename);   
        pnt_x=xy_pair.first;
        pnt_y=xy_pair.second;
        gettimeofday(&t2, nullptr);
        float pnt_load_time=cuspatial::calc_time("point data loading time ......",t1,t2); 
        point_len=pnt_x.size;
        return (0);
    }
    
    std::vector<uint32_t> exec_gpu_pip()
    {      
        gdf_column result_bitmap =
            cuspatial::point_in_polygon_bitmap(this->pnt_x, this->pnt_y,
                                               this->f_pos, this->r_pos,
                                               this->poly_x, this->poly_y); 
        std::vector<uint32_t> h_result(this->point_len);
        EXPECT_EQ(cudaMemcpy(h_result.data(), result_bitmap.data,
                             this->point_len * sizeof(uint32_t),
                             cudaMemcpyDeviceToHost), cudaSuccess);
        gdf_column_free(&result_bitmap);
        return h_result;
    }
};

//typedef testing::Types<int16_t, int32_t, int64_t, float, double> NumericTypes;
typedef testing::Types<double> NumericTypes;

TYPED_TEST_CASE(PIPTest, NumericTypes);

#if  0 // disable until data files are checked in
TYPED_TEST(PIPTest, piptest)
{
    std::string pnt_filename =std::string("../../data/locust.location");
    std::string ply_filename=std::string("../../data/itsroi.ply"); 
    ASSERT_GE(this->set_initialize(ply_filename.c_str(),pnt_filename.c_str()),0);

    struct timeval t0,t1,t2;

    gettimeofday(&t0, nullptr);
    std::vector<uint32_t> gpu_pip_res = this->exec_gpu_pip(gpu_pip_res);

    gettimeofday(&t1, nullptr);
    float gpu_pip_time1=cuspatial::calc_time("GPU PIP time 1(including point data transfer and kernel time)......",t0,t1);

    //Testing asynchronous issues by 2nd call
    std::vector<uint> gpu_pip_res2 = this->exec_gpu_pip();

    gettimeofday(&t2, nullptr);
    float gpu_pip_time2=cuspatial::calc_time("GPU PIP time 2(including point data transfer and kernel time)......",t1,t2);

    int err_cnt=0,non_zero=0;
    for(int i=0;i<this->point_len;i++)
    {
        if(gpu_pip_res[i]!=gpu_pip_res2[i])
        {
            /*printf("ERR: %d %d %d, G=%08x C=%08x\n",i,__builtin_popcount(gpu_pip_res[i]),
                __builtin_popcount(gpu_pip_res2[i]), (unsigned int)(gpu_pip_res[i]),(unsigned int)(gpu_pip_res2[i]));*/
            err_cnt++;
        }
        if(gpu_pip_res[i]!=0&&gpu_pip_res2[i]!=0)
            non_zero++;
    }
    if(err_cnt==0)
        std::cout<<"two rounds GPU results are identical...................OK"<<std::endl;     	
    else {
        std::cout<<"two rounds GPU results differ by: "<<err_cnt<<std::endl;     	
        std::cout<<"non zero results="<<non_zero<<std::endl;
    }

    this->set_finalize();

    cudaMemGetInfo(&this->free_mem, &this->total_mem);
    std::cout<<"ending GPU free mem "<<this->free_mem<<std::endl;
}
#endif
