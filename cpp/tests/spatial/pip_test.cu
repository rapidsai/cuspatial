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

using namespace cuSpatial;

template <typename T>
struct PIPTest : public GdfTest 
{
    T * p_x=NULL, *p_y=NULL;
    int point_len=-1;
    struct PolyMeta<T> polygon;
    size_t free_mem = 0, total_mem = 0;
    
    int set_initialize(const char *poly_fn, const char *point_fn)
    {

     cudaMemGetInfo(&free_mem, &total_mem);
     std::cout<<"GPU total_mem="<<total_mem<<std::endl;
     std::cout<<"beginning GPU free_mem="<<free_mem<<std::endl;
     
      struct timeval t0,t1,t2;
      gettimeofday(&t0, NULL);
      read_polygon_soa<T>(poly_fn,polygon);
      gettimeofday(&t1, NULL);
      float ply_load_time=cuSpatial::calc_time("polygon data loading time=", t0,t1);
      CUDF_EXPECTS(polygon.num_f>0 && polygon.num_r>0,"invalid # of polygons/rings");
      CUDF_EXPECTS(polygon.num_f<=polygon.num_r,"a polygon must have at least one ring");
      CUDF_EXPECTS(polygon.p_y!=NULL && polygon.p_y!=NULL,"polygon vertex x/y array can not be NULL");
    
      //from len to pos using inclusive scan (partial sum)
      //support both in-place (saving memory) and regular
      //for in-place prefix sum (pos and len arrays point to the same mem addresses)
      if(polygon.is_inplace)
      {
		polygon.p_g_pos=polygon.p_g_len;
		polygon.p_f_pos=polygon.p_f_len;
		polygon.p_r_pos=polygon.p_r_len;
      }
      else
      {
		polygon.p_g_pos=new uint[polygon.num_g];
		polygon.p_f_pos=new uint[polygon.num_f];
		polygon.p_r_pos=new uint[polygon.num_r];
      }

      std::partial_sum(polygon.p_g_len,polygon.p_g_len+polygon.num_g,polygon.p_g_pos,std::plus<uint>());
      std::partial_sum(polygon.p_f_len,polygon.p_f_len+polygon.num_f,polygon.p_f_pos,std::plus<uint>());
      std::partial_sum(polygon.p_r_len,polygon.p_r_len+polygon.num_r,polygon.p_r_pos,std::plus<uint>());

      /*printf("after partial sum\n");
      printf("num_g=%d\n",polygon.num_g);
      for(int i=0;i<polygon.num_g;i++)
		printf("(%u %u)",polygon.p_g_len[i],polygon.p_g_pos[i]);
      printf("\n");

      printf("num_f=%d\n",polygon.num_f);
      for(int i=0;i<polygon.num_f;i++)
		printf("(%u %u)",polygon.p_f_len[i],polygon.p_f_pos[i]);
      printf("\n");
      printf("num_r=%d\n",polygon.num_r);
      for(int i=0;i<polygon.num_r;i++)
		printf("(%u %u)",polygon.p_r_len[i],polygon.p_r_pos[i]);
      printf("\n");*/
 
      point_len=read_point_ll<T>(point_fn,p_x,p_y);
      std::cout<<"point_len="<<point_len<<std::endl;
      CUDF_EXPECTS(point_len>0,"# of points must be greater than 0");
      CUDF_EXPECTS(p_y!=NULL && p_y!=NULL,"point x/y array can not be NULL");
      
      /*for(int i=0;i<10;i++)
       std::cerr<<i<<","<<p_x[i].lon<<","<<p_y[i].lat<<std::endl;*/
      
      gettimeofday(&t2, NULL);
      float pnt_load_time=cuSpatial::calc_time("point data loading time ......",t1,t2);      
      return (0);
    }
    
    void exec_gpu_pip(uint *& gpu_pip_res)
    {
     
        //std::vector g_pos_v(polygon.p_g_pos,polygon.p_g_pos+polygon.num_g);
        std::vector<uint> f_pos_v(polygon.p_f_pos,polygon.p_f_pos+polygon.num_f);
        std::vector<uint> r_pos_v(polygon.p_r_pos,polygon.p_r_pos+polygon.num_r);
        std::vector<T> ply_x_v(polygon.p_x,polygon.p_x+polygon.num_v);
	std::vector<T> ply_y_v(polygon.p_y,polygon.p_y+polygon.num_v);
        std::vector<T> pnt_x_v(p_x,p_x+this->point_len);
	std::vector<T> pnt_y_v(p_y,p_y+this->point_len);   
        
        cudf::test::column_wrapper<uint> polygon_fpos_wrapp{f_pos_v};
        cudf::test::column_wrapper<uint> polygon_rpos_wrapp{r_pos_v};
        cudf::test::column_wrapper<T> polygon_x_wrapp{ply_x_v};
        cudf::test::column_wrapper<T> polygon_y_wrapp{ply_y_v};
        cudf::test::column_wrapper<T> point_x_wrapp{pnt_x_v};
        cudf::test::column_wrapper<T> point_y_wrapp{pnt_y_v};
        std::cout<<"point_x_wrapp type="<<point_x_wrapp.get()->dtype<<std::endl;
        
        gdf_column res_bm1 = cuSpatial::pip_bm( 
        	*(point_x_wrapp.get()), *(point_y_wrapp.get()),
        	*(polygon_fpos_wrapp.get()), *(polygon_rpos_wrapp.get()), 
        	*(polygon_x_wrapp.get()), *(polygon_y_wrapp.get()) );
    
        gpu_pip_res=new uint[this->point_len];
        CUDF_EXPECTS(gpu_pip_res!=NULL,"error in allocating memory for results");
    	EXPECT_EQ(cudaMemcpy(gpu_pip_res, res_bm1.data, this->point_len * sizeof(uint), cudaMemcpyDeviceToHost), cudaSuccess);
    }
    
    void set_finalize()
    {
	delete [] polygon.p_g_len;
      	delete [] polygon.p_f_len;
      	delete [] polygon.p_r_len;
      	if(!polygon.is_inplace)
      	{
      		delete [] polygon.p_g_pos;
      		delete [] polygon.p_f_pos;
      		delete [] polygon.p_r_pos;
      	}
      	delete [] polygon.p_x;
      	delete [] polygon.p_y;
      	
    	delete[] p_x;
    	delete[] p_y;
    	
    }
    
};

//typedef testing::Types<int16_t, int32_t, int64_t, float, double> NumericTypes;
typedef testing::Types<double> NumericTypes;

TYPED_TEST_CASE(PIPTest, NumericTypes);


TYPED_TEST(PIPTest, piptest)
{
    std::string pnt_fn =std::string("/home/jianting/cuspatial/data/locust.location");
    std::string ply_fn=std::string("/home/jianting/cuspatial/data/itsroi.ply"); 
    ASSERT_GE(this->set_initialize(ply_fn.c_str(),pnt_fn.c_str()),0);
    
    struct timeval t3,t4,t5,t6;
    gettimeofday(&t3, NULL);
    uint* cpu_pip_res=new uint[this->point_len];
    CUDF_EXPECTS(cpu_pip_res!=NULL,"failed allocating output bitmap memory");
    std::cout<<"beginning cpu_pip_loop.............."<<std::endl;
    cpu_pip_loop(this->point_len,this->p_x, this->p_y,this->polygon,cpu_pip_res);
    	
    gettimeofday(&t4, NULL);
    cuSpatial::calc_time("CPU PIP time......",t3,t4);

    //x,y and gpu_pip_res on CPU; pip code allocates memory
    //x and y will be uploaded and res will be downloaded automatically
    //cost include both computation and data transfer time
    
    uint* gpu_pip_res=NULL;
    this->exec_gpu_pip(gpu_pip_res);
    assert(gpu_pip_res!=NULL);
    
    gettimeofday(&t5, NULL);
    float gpu_pip_time1=cuSpatial::calc_time("GPU PIP time 1(including point data transfer and kernel time)......",t4,t5);

    //Testing asynchronous issues by 2nd call
    uint* gpu_pip_res2=NULL;
    this->exec_gpu_pip(gpu_pip_res2);
    assert(gpu_pip_res2!=NULL);
    
    gettimeofday(&t6, NULL);
    float gpu_pip_time2=cuSpatial::calc_time("GPU PIP time 2(including point data transfer and kernel time)......",t5,t6);
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
