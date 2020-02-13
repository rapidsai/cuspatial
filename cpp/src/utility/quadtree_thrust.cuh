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

#include <thrust/pair.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>
#include "z_order.cuh"
namespace 
{

static void HandleCudaError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );	
    }
}
#define HANDLE_CUDA_ERROR( err ) (HandleCudaError( err, __FILE__, __LINE__ ))

typedef unsigned char uchar;
typedef unsigned int  uint;
typedef thrust::pair<thrust::tuple<double,double>, thrust::tuple<double,double>> SBBox;

struct xytoz 
{
    
  SBBox bbox;
  uchar lev;
  double scale;

  xytoz(SBBox _bbox,uchar _lev,double _scale): bbox(_bbox),lev(_lev),scale(_scale) {}
   
    __device__
    uint operator()(thrust::tuple<uint,double,double> loc )
    {	
	double x=thrust::get<1>(loc);
	double y=thrust::get<2>(loc);
	if(x<thrust::get<0>(bbox.first)||x>thrust::get<0>(bbox.second)||y<thrust::get<1>(bbox.first)||y>thrust::get<1>(bbox.second))
		return (1<<(2*lev));
	else
	{	
		ushort a=(ushort)((x-thrust::get<0>(bbox.first))/scale);
		ushort b=(ushort)((y-thrust::get<1>(bbox.first))/scale);
		uint c= z_order(a,b);
		return c;
	}
    }
};

struct get_parent 
{
    uchar lev;
    get_parent(uchar _lev):lev(_lev){}
    
    __device__
    uint operator()(uint child )
    {
    	return (child>>lev);
    }
};

struct remove_discard
{
    uint *p_len,limit,end_pos;
    remove_discard(uint *_p_len,uint _limit): 
    	p_len(_p_len),limit(_limit){}
    
    __device__ 
    bool operator()(thrust::tuple<uint,uchar, uint,uint,uint> v)
    {
        //uint tid = threadIdx.x + blockDim.x*blockIdx.x;
        //printf("remove_discard tid=%d\n",tid);
        return (p_len[thrust::get<4>(v)]<=limit);
    }
};


struct what2output
{
    __device__ 
    uint operator()(thrust::tuple<uint, uint,bool> v)
    {
        return (thrust::get<2>(v)?(thrust::get<0>(v)):(thrust::get<1>(v)));
    }
};

struct flatten_z_code
{
        uint M;
        
        flatten_z_code(uint _M):M(_M){}
        
        __device__ 
	uint operator()(thrust::tuple<uint,uint,bool> v)
	{
		uint key=thrust::get<0>(v);
		uint lev=thrust::get<1>(v);
		uint ret=(thrust::get<2>(v))?0xFFFFFFFF:(key<<(2*(M-1-lev)));
		//uint tid = threadIdx.x + blockDim.x*blockIdx.x;
		//printf("tid=%d key=%d lev=%d ret=%d\n",tid,key,lev,ret);
		return (ret);
	}
};


} // namespace cuspatial
