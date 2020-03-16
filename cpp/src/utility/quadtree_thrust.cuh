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
#include "bbox_thrust.cuh"
namespace 
{

template<typename T>
struct xytoz 
{

    SBBox<T> bbox;
    uint8_t lev;
    double scale;

    xytoz(SBBox<T> _bbox,uint8_t _lev,double _scale): bbox(_bbox),lev(_lev),scale(_scale) {}

    __device__
    uint32_t operator()(thrust::tuple<double,double> loc )
    {
        double x=thrust::get<0>(loc);
        double y=thrust::get<1>(loc);
        if(x<thrust::get<0>(bbox.first)||x>thrust::get<0>(bbox.second)||y<thrust::get<1>(bbox.first)||y>thrust::get<1>(bbox.second))
            return (1<<(2*lev)-1);
        else
        {
            uint16_t a=(uint16_t)((x-thrust::get<0>(bbox.first))/scale);
            uint16_t b=(uint16_t)((y-thrust::get<1>(bbox.first))/scale);
            uint32_t c= z_order(a,b);
            return c;
        }
    }
};

struct get_parent 
{
    uint8_t lev;
    get_parent(uint8_t _lev):lev(_lev){}

    __device__
    uint32_t operator()(uint32_t child )
    {
        return (child>>lev);
    }
};

struct remove_discard
{
    uint32_t *p_len,limit,end_pos;
    remove_discard(uint32_t *_p_len,uint32_t _limit): 
        p_len(_p_len),limit(_limit){}

    __device__ 
    bool operator()(thrust::tuple<uint32_t,uint8_t, uint32_t,uint32_t,uint32_t> v)
    {
        //uint32_t tid = threadIdx.x + blockDim.x*blockIdx.x;
        uint32_t key=thrust::get<0>(v);
        uint8_t lev=thrust::get<1>(v);
        uint32_t clen=thrust::get<2>(v);
        uint32_t nlen=thrust::get<3>(v);
        uint32_t ppos=thrust::get<4>(v);
        //uint32_t plen=p_len[ppos];
        //printf("remove_discard tid=%d key=%d lev=%d clen=%d nlen=%d ppos=%d plen=%d\n",tid,key,lev,clen,nlen,ppos,plen);
        return (p_len[thrust::get<4>(v)]<=limit);
    }
};


struct what2output
{
    __device__ 
    uint32_t operator()(thrust::tuple<uint32_t, uint32_t,bool> v)
    {
        return (thrust::get<2>(v)?(thrust::get<0>(v)):(thrust::get<1>(v)));
    }
};

template<typename T>
struct gen_quad_bbox
{
    uint32_t *d_p_key=NULL;
    double scale;
    SBBox<T> aoi_bbox;
    uint8_t *d_p_lev;
    uint32_t M;

    gen_quad_bbox(uint32_t _M,SBBox<T> _aoi_bbox,double _scale,uint32_t *_d_p_key,uint8_t *_d_p_lev):
        M(_M),aoi_bbox(_aoi_bbox),scale(_scale),d_p_key(_d_p_key),d_p_lev(_d_p_lev){}

    __device__
    SBBox<T> operator()(uint32_t p) const
    {
        double s=scale*pow(2.0,M-1-d_p_lev[p]);
        uint32_t zx=z_order_x(d_p_key[p]);
        uint32_t zy=z_order_y(d_p_key[p]);
        double x0=thrust::get<0>(aoi_bbox.first);;
        double y0=thrust::get<1>(aoi_bbox.first);
        double qx1=zx*s+x0;
        double qx2=(zx+1)*s+x0;   
        double qy1=zy*s+y0;
        double qy2=(zy+1)*s+y0;
        //printf("gen_quad_bbox: %5d %5d %5d %10.5f %10.5f %10.5f %10.5f (scale=%10.5f)\n",p,zx,zy,qx1,qy1,qx2,qy2,scale);
        SBBox<T> bbox(thrust::make_tuple(qx1,qy1),thrust::make_tuple(qx2,qy2));    
        return bbox;
    }

};

struct flatten_z_code
{
    uint32_t M;

    flatten_z_code(uint32_t _M):M(_M){}

    __device__ 
    uint32_t operator()(thrust::tuple<uint32_t,uint32_t,bool> v)
    {
        uint32_t key=thrust::get<0>(v);
        uint32_t lev=thrust::get<1>(v);
        uint32_t ret=(thrust::get<2>(v))?0xFFFFFFFF:(key<<(2*(M-1-lev)));
        //uint32_t tid = threadIdx.x + blockDim.x*blockIdx.x;
        //printf("flatten_z_code: tid=%d key=%d lev=%d ret=%d\n",tid,key,lev,ret);
        return (ret);
    }
};

struct qt_get_fpos
{
    uint32_t *d_p_qtfpos=NULL;
    qt_get_fpos(uint32_t *_d_p_qtfpos):d_p_qtfpos(_d_p_qtfpos){}

    __device__ 
    uint32_t operator()(uint32_t idx)
    {
        return d_p_qtfpos[idx];
    }
};

struct qt_is_type
{
    uint8_t type;
    qt_is_type(uint8_t _type):type(_type){}

    __device__ 
    bool operator()(thrust::tuple<uint8_t,uint8_t,uint32_t,uint32_t> v)
    {
        return thrust::get<1>(v)==type;
    }
};

struct qt_not_type
{
    uint8_t type;
    qt_not_type(uint8_t _type):type(_type){}
    
    __device__ 
    bool operator()(thrust::tuple<uint8_t,uint8_t,uint32_t,uint32_t> v)
    {
        return thrust::get<1>(v)!=type;
    }
};

struct update_quad
{
    const uint32_t *d_p_qtfpos=NULL,*d_seq_pos=NULL;

    update_quad(const uint32_t *_d_p_qtfpos,const uint32_t *_d_seq_pos):
        d_p_qtfpos(_d_p_qtfpos),d_seq_pos(_d_seq_pos){}

    __device__ 
    uint32_t operator()(thrust::tuple<uint32_t,uint32_t> v)
    {
        //assuming 1d grid
        uint32_t qid=thrust::get<0>(v);
        uint32_t sid=thrust::get<1>(v);
        uint32_t fpos=d_p_qtfpos[qid];
        uint32_t seq=d_seq_pos[sid];
        //uint32_t tid = threadIdx.x + blockDim.x*blockIdx.x;	   
        //printf("update_quad:tid=%d qid=%d sid=%d fpos=%d seq=%d\n",tid,qid,sid,fpos,seq);
        return(fpos+seq);
    }
};

} // namespace cuspatial
