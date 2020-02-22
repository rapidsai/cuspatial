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

#include <ostream>
#include <thrust/pair.h>
#include <thrust/functional.h>
#include <utility/z_order.cuh>

namespace
{

template<typename T>
using SBBox=thrust::pair<thrust::tuple<T,T>, thrust::tuple<T,T>> ;

template<typename T>
std::ostream& operator << (std::ostream& os, const SBBox<T> & bbox)
{
    T x1=thrust::get<0>(bbox.first);
    T y1=thrust::get<1>(bbox.first);
    T x2=thrust::get<0>(bbox.second);
    T y2=thrust::get<1>(bbox.second);

    os << "("<< x1 <<"," << y1 <<"," << x2 << "," << y2 <<std::endl;
    return os;
}

// reduce a pair of bounding boxes (a,b) to a bounding box containing a and b
template<typename T>
struct bbox_reduction
{
     __device__     
     SBBox<T> operator()(const SBBox<T>& a, const SBBox<T>& b)
     {
            double fx1=thrust::get<0>(a.first);
            double fx2=thrust::get<0>(b.first);
            double fy1=thrust::get<1>(a.first);
            double fy2=thrust::get<1>(b.first);

            double tx1=thrust::get<0>(a.second);
            double tx2=thrust::get<0>(b.second); 
            double ty1=thrust::get<1>(a.second);          
            double ty2=thrust::get<1>(b.second);
            
            //printf("%10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f\n",fx1,fy1,fx2,fy2,tx1,ty1,tx2,ty2);
            thrust::tuple<double,double> ll=thrust::make_tuple(min(fx1,tx1),min(fy1,ty1));
            thrust::tuple<double,double> ur=thrust::make_tuple(max(fx2,tx2),max(fy2,ty2));
            return SBBox<T>(ll, ur);
      }
};

template<typename T>
struct bbox_transformation
{
     __device__
     SBBox<T> operator()(const thrust::tuple<T,T>& t)
     {
         return SBBox<T>(t, t);
     }
};    

template<typename T>
struct bbox2tuple
{
     __device__
     thrust::tuple<T,T,T,T> operator()(const SBBox<T>& bbox)
     {
        T x1=thrust::get<0>(bbox.first);
        T x2=thrust::get<0>(bbox.second);
        T y1=thrust::get<1>(bbox.first);
        T y2=thrust::get<1>(bbox.second);
        //printf("bbox2tuple: %10.5f %10.5f %10.5f %10.5f\n",x1,y1,x2,y2);
        return thrust::make_tuple(x1,y1,x2,y2);
     }
};    

template<typename T>
struct tuple2bbox
{
     __device__
     SBBox<T> operator()(thrust::tuple<T,T,T,T> v)
     {
        T x1=thrust::get<0>(v);
        T y1=thrust::get<1>(v);
        T x2=thrust::get<2>(v);
        T y2=thrust::get<3>(v);
        //printf("tuple2bbox: %10.5f %10.5f %10.5f %10.5f\n",x1,y1,x2,y2);        
        return SBBox<T>(thrust::make_tuple(x1,y1), thrust::make_tuple(x2,y2));
     }
};    

template<typename T>
struct pairwise_test_intersection
{
  uint32_t M;
  uint32_t num_node;
  const uint32_t *d_p_key=NULL;
  const uint8_t *d_p_lev=NULL;
  const bool *d_p_qtsign=NULL;
  double scale;
  SBBox<double> aoi_bbox;
  const SBBox<T> *ply_bbox;
 
 pairwise_test_intersection(uint32_t _M,uint32_t _num_node,const SBBox<double>& _aoi_bbox,double _scale,
 	const uint32_t *_d_p_key,const uint8_t* _d_p_lev,const bool *_d_p_sign,const SBBox<T> *_ply_bbox):
  	M(_M),num_node(_num_node),aoi_bbox(_aoi_bbox),
  	d_p_key(_d_p_key),d_p_lev(_d_p_lev),d_p_qtsign(_d_p_sign),scale(_scale),ply_bbox(_ply_bbox)
 {
     //printf("pairwise_test_intersection: num_node=%d\n",num_node);
 }
  
  __device__
  thrust::tuple<uint8_t,uint8_t, uint32_t,uint32_t> operator()(uint32_t p) const
  {
    uint32_t poly_idx=p/num_node;
    uint32_t quad_idx=p%num_node;
  
    uint8_t lev=d_p_lev[quad_idx];   
    double s=scale*pow(2.0,M-1-lev);
    uint32_t zx=z_order_x(d_p_key[quad_idx]);
    uint32_t zy=z_order_y(d_p_key[quad_idx]);
       
    double x0=thrust::get<0>(aoi_bbox.first);;
    double y0=thrust::get<1>(aoi_bbox.first);
    double qx1=zx*s+x0;
    double qx2=(zx+1)*s+x0;   
    double qy1=zy*s+y0;
    double qy2=(zy+1)*s+y0;

    SBBox<T> box=ply_bbox[poly_idx];
    double px1=thrust::get<0>(box.first);
    double py1=thrust::get<1>(box.first);
    double px2=thrust::get<0>(box.second);
    double py2=thrust::get<1>(box.second);
    bool not_intersect=(qx1>px2||qx2<px1||qy1>py2||qy2<py1);
    uint8_t type=not_intersect?2:(d_p_qtsign[quad_idx]?1:0);
   
    //uint32_t tid = threadIdx.x + blockDim.x*blockIdx.x;
    //printf("init: tid=%d qx1=%10.5f qy1=%10.5f qx2=%10.5f qy2=%10.5f px1=%10.5f py1=%10.5f px2=%10.5f py2=%10.5f\n",tid,qx1,qy1,qx2,qy2,px1,py1,px2,py2);    
    //printf("init: tid=%d lev=%d sign=%d type=%d poly_idx=%d quad_id=%d\n",tid, lev,d_p_qtsign[quad_idx],type,poly_idx,quad_idx);
    return thrust::make_tuple(lev,type,poly_idx,quad_idx);
  }
};

template<typename T>
struct twolist_test_intersection
{
  uint32_t M;
  uint32_t num_node;
  const uint32_t *d_p_key=NULL;
  const uint8_t *d_p_lev=NULL;
  const bool *d_p_qtsign=NULL;
  double scale;
  SBBox<double> aoi_bbox;
  const SBBox<T> *ply_bbox;
 
 twolist_test_intersection(uint32_t _M,const SBBox<double>& _aoi_bbox,double _scale,const uint32_t *_d_p_key,
 	const uint8_t * _d_p_lev,const bool *_d_p_sign,const SBBox<T> *_ply_bbox):
  	M(_M),aoi_bbox(_aoi_bbox),d_p_key(_d_p_key),
  	d_p_lev(_d_p_lev),d_p_qtsign(_d_p_sign),scale(_scale),ply_bbox(_ply_bbox) {}
  	
  __device__
  thrust::tuple<uint8_t,uint8_t, uint32_t,uint32_t> operator()(thrust::tuple<uint32_t,uint32_t> v) const
  {
    uint32_t poly_idx=thrust::get<0>(v);
    uint32_t quad_idx=thrust::get<1>(v);
  
    uint8_t lev=d_p_lev[quad_idx];   
    double s=scale*pow(2.0,M-1-lev);
    uint32_t zx=z_order_x(d_p_key[quad_idx]);
    uint32_t zy=z_order_y(d_p_key[quad_idx]);
      
    double x0=thrust::get<0>(aoi_bbox.first);;
    double y0=thrust::get<1>(aoi_bbox.first);
    double qx1=zx*s+x0;
    double qx2=(zx+1)*s+x0;   
    double qy1=zy*s+y0;
    double qy2=(zy+1)*s+y0;
    
    SBBox<T> box=ply_bbox[poly_idx];
    double px1=thrust::get<0>(box.first);
    double py1=thrust::get<1>(box.first);
    double px2=thrust::get<0>(box.second);
    double py2=thrust::get<1>(box.second);
    bool not_intersect=(qx1>px2||qx2<px1||qy1>py2||qy2<py1);
    uint8_t type=not_intersect?2:(d_p_qtsign[quad_idx]?1:0);
    //uint32_t tid = threadIdx.x + blockDim.x*blockIdx.x;
    //printf("rest: tid=%d qx1=%10.5f qy1=%10.5f qx2=%10.5f qy2=%10.5f px1=%10.5f py1=%10.5f px2=%10.5f py2=%10.5f\n",tid,qx1,qy1,qx2,qy2,px1,py1,px2,py2);    
    //printf("rest: tid=%d lev=%d sign=%d type=%d poly_idx=%d quad_id=%d\n",tid, lev,d_p_qtsign[quad_idx],type,poly_idx,quad_idx);
    return thrust::make_tuple(lev,type,poly_idx,quad_idx);
  }
};

} // namespace cuspatial
