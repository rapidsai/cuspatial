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

namespace
{

typedef thrust::pair<thrust::tuple<double,double>, thrust::tuple<double,double>> SBBox;

std::ostream& operator << (std::ostream& os, const SBBox & bbox)
{
    double x1=thrust::get<0>(bbox.first);
    double y1=thrust::get<1>(bbox.first);
    double x2=thrust::get<0>(bbox.second);
    double y2=thrust::get<1>(bbox.second);

    os << "("<< x1 <<"," << y1 <<"," << x2 << "," << y2 <<std::endl;
    return os;
}

// reduce a pair of bounding boxes (a,b) to a bounding box containing a and b
struct bbox_reduction
{
     __device__
        SBBox operator()(SBBox a, SBBox b)
        {
            // lower left corner
            double fx1=thrust::get<0>(a.first);
            double fx2=thrust::get<0>(b.first);
            double fy1=thrust::get<1>(a.first);
            double fy2=thrust::get<1>(b.first);

            double tx1=thrust::get<0>(a.second);
            double tx2=thrust::get<0>(b.second);
            double ty1=thrust::get<1>(a.second);
            double ty2=thrust::get<1>(b.second);
            
            thrust::tuple<double,double> ll=thrust::make_tuple(min(fx1,tx1),min(fy1,ty1));
            thrust::tuple<double,double> ur=thrust::make_tuple(max(fx2,tx2),max(fy2,ty2));
            return SBBox(ll, ur);
        }
};

struct bbox_transformation
{
     __device__
        SBBox operator()(thrust::tuple<double,double> t)
        {
            return SBBox(t, t);
        }
};    

struct pairwise_test_intersection
{
  uint8_t M;
  uint32_t num_node;
  uint32_t *d_p_key=NULL;
  uint8_t *d_p_lev=NULL;
  bool *d_p_qtsign=NULL;
  double scale;
  SBBox *ply_bbox,aoi_bbox;
 
 pairwise_test_intersection(uint8_t _M,uint32_t _num_node,SBBox _aoi_bbox,double _scale,
 	uint32_t *_d_p_key,uint8_t* _d_p_lev,bool *_d_p_sign,SBBox *_ply_bbox):
  	M(_M),num_node(_num_node),aoi_bbox(_aoi_bbox),
  	d_p_key(_d_p_key),d_p_lev(_d_p_lev),d_p_qtsign(_d_p_sign),scale(_scale),ply_bbox(_ply_bbox)
 {
     //printf("pairwise_test_intersection: num_node=%d\n",num_node);
 }
  
  __device__
  thrust::tuple<uint8_t,uint8_t, uint32_t,uint32_t> operator()(uint32_t p) const
  {
    //uint32_t tid = threadIdx.x + blockDim.x*blockIdx.x;
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

    SBBox box=ply_bbox[poly_idx];
    double px1=thrust::get<0>(box.first);
    double py1=thrust::get<1>(box.first);
    double px2=thrust::get<0>(box.second);
    double py2=thrust::get<1>(box.second);
    bool not_intersect=(qx1>px2||qx2<px1||qy1>py2||qy2<py1);
    uint8_t type=not_intersect?2:(d_p_qtsign[quad_idx]?1:0);
    return thrust::make_tuple(lev,type,poly_idx,quad_idx);
  }
};

struct twolist_test_intersection
{
  uint8_t M;
  uint32_t num_node;
  uint32_t *d_p_key=NULL;
  uint8_t *d_p_lev=NULL;
  bool *d_p_qtsign=NULL;
  double scale;
  SBBox *ply_bbox,aoi_bbox;
 
 twolist_test_intersection(uint8_t _M,SBBox _aoi_bbox,double _scale,uint32_t *_d_p_key,
 	uint8_t * _d_p_lev,bool *_d_p_sign,SBBox *_ply_bbox):
  	M(_M),aoi_bbox(_aoi_bbox),d_p_key(_d_p_key),
  	d_p_lev(_d_p_lev),d_p_qtsign(_d_p_sign),scale(_scale),ply_bbox(_ply_bbox) {}
  	
  __device__
  thrust::tuple<uint8_t,uint8_t, uint32_t,uint32_t> operator()(thrust::tuple<uint32_t,uint32_t> v) const
  {
    //uint32_t tid = threadIdx.x + blockDim.x*blockIdx.x;
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
    
    SBBox box=ply_bbox[poly_idx];
    double px1=thrust::get<0>(box.first);
    double py1=thrust::get<1>(box.first);
    double px2=thrust::get<0>(box.second);
    double py2=thrust::get<1>(box.second);
    bool not_intersect=(qx1>px2||qx2<px1||qy1>py2||qy2<py1);
    uint8_t type=not_intersect?2:(d_p_qtsign[quad_idx]?1:0);
    //printf("lev=%d type=%d poly_idx=%d quad_id=%d\n",lev,type,poly_idx,quad_idx);
    return thrust::make_tuple(lev,type,poly_idx,quad_idx);
  }
};

} // namespace cuspatial
