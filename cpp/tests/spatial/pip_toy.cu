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

#include <vector>
#include <string>
#include <cuspatial/pip.hpp>
#include "pip_util.h"
#include <gtest/gtest.h>
#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/cudf_test_utils.cuh>
#include <tests/utilities/cudf_test_fixtures.h>

using namespace cuspatial;

struct PIPToy : public GdfTest 
{
    int point_len=3;

    double  *p_x=new double[point_len]{0,-8,6};
    double  *p_y=new double[point_len]{0,-8,6};
    
    struct PolyMeta<double> h_polygon;
    
    int set_initialize()
    {          
      h_polygon.num_g=1;
      h_polygon.num_f=2;
      h_polygon.num_r=2;
      h_polygon.num_v=10;
      h_polygon.p_f_pos=new uint[h_polygon.num_f]{1,2};
      h_polygon.p_r_pos=new uint[h_polygon.num_r]{5,10};
      h_polygon.p_x=new double[h_polygon.num_v]{-10,   5, 5, -10, -10,  0, 10, 10,  0, 0};
      h_polygon.p_y=new double[h_polygon.num_v]{-10, -10, 5,   5,  -10, 0,  0, 10, 10, 0};
  
      return 1;
    }

    
    void exec_gpu_pip(uint *& gpu_pip_res)
    {  
        //std::vector g_pos_v(h_polygon.p_g_pos,h_polygon.p_g_pos+h_polygon.num_g);
        std::vector<uint> f_pos_v(h_polygon.p_f_pos,h_polygon.p_f_pos+h_polygon.num_f);
        std::vector<uint> r_pos_v(h_polygon.p_r_pos,h_polygon.p_r_pos+h_polygon.num_r);
        std::vector<double> ply_x_v(h_polygon.p_x,h_polygon.p_x+h_polygon.num_v);
	std::vector<double> ply_y_v(h_polygon.p_y,h_polygon.p_y+h_polygon.num_v);
        std::vector<double> pnt_x_v(p_x,p_x+this->point_len);
	std::vector<double> pnt_y_v(p_y,p_y+this->point_len);   
        
        cudf::test::column_wrapper<uint> polygon_fpos_wrapp{f_pos_v};
        cudf::test::column_wrapper<uint> polygon_rpos_wrapp{r_pos_v};
        cudf::test::column_wrapper<double> polygon_x_wrapp{ply_x_v};
        cudf::test::column_wrapper<double> polygon_y_wrapp{ply_y_v};
        cudf::test::column_wrapper<double> point_x_wrapp{pnt_x_v};
        cudf::test::column_wrapper<double> point_y_wrapp{pnt_y_v};
         
        gdf_column res_bm1 = cuspatial::pip_bm( 
        	*(point_x_wrapp.get()), *(point_y_wrapp.get()),
        	*(polygon_fpos_wrapp.get()), *(polygon_rpos_wrapp.get()), 
        	*(polygon_x_wrapp.get()), *(polygon_y_wrapp.get()) );
    
        gpu_pip_res=new uint[this->point_len];
        assert(gpu_pip_res!=NULL);
    	EXPECT_EQ(cudaMemcpy(gpu_pip_res, res_bm1.data, this->point_len * sizeof(uint), cudaMemcpyDeviceToHost), cudaSuccess);
    }
    
    void set_finalize()
    {
	delete [] h_polygon.p_g_len;
      	delete [] h_polygon.p_f_len;
      	delete [] h_polygon.p_r_len;
      	if(!h_polygon.is_inplace)
      	{
      		delete [] h_polygon.p_g_pos;
      		delete [] h_polygon.p_f_pos;
      		delete [] h_polygon.p_r_pos;
      	}
      	delete [] h_polygon.p_x;
      	delete [] h_polygon.p_y;
      	
    	delete[] p_x;
    	delete[] p_y;
    	
    }
    
};

TEST_F(PIPToy, piptest)
{
    ASSERT_GE(this->set_initialize(),0);
    
    uint* cpu_pip_res=new uint[this->point_len];
    assert(cpu_pip_res!=NULL);  
    cpu_pip_loop(this->point_len,this->p_x,this->p_y, this->h_polygon,cpu_pip_res);
  
    uint* gpu_pip_res=NULL;
    this->exec_gpu_pip(gpu_pip_res);
    assert(gpu_pip_res!=NULL);
    
    int err_cnt=0,non_zero=0;
    for(int i=0;i<this->point_len;i++)
    {
	/*const char *sign=(cpu_pip_res[i]==gpu_pip_res[i])?"CORR":"ERR";
	printf("%s: %d %d %d, G=%08x C=%08x\n",sign,i,__builtin_popcount(cpu_pip_res[i]),
		__builtin_popcount(gpu_pip_res[i]), (unsigned int)(cpu_pip_res[i]),(unsigned int)(gpu_pip_res[i]));*/
	
	if(cpu_pip_res[i]!=gpu_pip_res[i])
			err_cnt++;
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
}
