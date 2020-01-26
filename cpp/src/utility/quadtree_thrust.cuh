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
typedef thrust::tuple<SBBox,double,uint,uint> quad_point_inputs;
typedef thrust::tuple<uint, uint *, bool *, uint*, uint*> quad_point_outputs;


struct xytoz 
{
    
  SBBox bbox;
  uchar lev;
  double scale;

  xytoz(SBBox _bbox,uchar _lev,double _scale): bbox(_bbox),lev(_lev),scale(_scale) {}
   
    __device__
    uint operator()(thrust::tuple<double,double> loc )
    {	
	double x=thrust::get<0>(loc);
	double y=thrust::get<1>(loc);
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
    bool operator()(thrust::tuple<uint, uint,uint,uint> v)
    {
        return (p_len[thrust::get<3>(v)]<=limit);
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

} // namespace cuspatial
