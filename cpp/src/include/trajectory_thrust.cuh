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
#include <cuspatial/cuspatial.h>

namespace cuspatial
{

	/**
	 *@brief Thrust functor for comparing two its_timestamp variables; used in sorting based on timestamp
	 */
	__host__ __device__
	inline bool operator<(const its_timestamp & t1,const its_timestamp & t2)
	{
		//cout<<"in operator<"<<endl;
		if(t1.y<t2.y) return true;
		else if((t1.y==t2.y)&&(t1.m<t2.m)) return true;
		else if((t1.y==t2.y)&&(t1.m==t2.m)&&(t1.d<t2.d)) return true;
		else if((t1.y==t2.y)&&(t1.m==t2.m)&&(t1.d==t2.d)&&(t1.hh<t2.hh)) return true;
		else if((t1.y==t2.y)&&(t1.m==t2.m)&&(t1.d==t2.d)&&(t1.hh==t2.hh)&&(t1.mm<t2.mm)) return true;
		else if((t1.y==t2.y)&&(t1.m==t2.m)&&(t1.d==t2.d)&&(t1.hh==t2.hh)&&(t1.mm==t2.mm)&&(t1.ss<t2.ss)) return true;
		else if((t1.y==t2.y)&&(t1.m==t2.m)&&(t1.d==t2.d)&&(t1.hh==t2.hh)&&(t1.mm==t2.mm)&&(t1.ss==t2.ss)&&(t1.ms<t2.ms)) return true;
		return false;
	}

	typedef  thrust::pair<its_timestamp, its_timestamp> TBBox;

	/**
	 *@brief Thrust functor for lifting its_timestamp to an interval (1D box)
	 *
	 */
	struct TBBox_transformation : public thrust::unary_function<TBBox,TBBox>
	{
		__host__ __device__
			TBBox operator()(its_timestamp time)
			{
				return TBBox(time, time);
			}
	};

	/**
	 *@brief Thrust functor for generating an 1D bounding box from two 1D box
	 *
	 */
	struct TBBox_reduction : public thrust::binary_function<TBBox,TBBox,TBBox>
	{
		__host__ __device__
			TBBox operator()(TBBox a, TBBox b)
			{
				// lower left corner
				its_timestamp minT=(a.first<b.first)?a.first:b.first;
			its_timestamp maxT=(a.second<b.second)?b.second:a.second;
				return TBBox(minT, maxT);
			}
	};

	/**
	 *@brief Thrust functor for transforming lon/lat (location_3d) to x/y (coord_2d) relative to an origin
	 
	 Note: Both x and y are in the unit of kilometers (km)
	 */
        template <typename T>
	struct coord_transformation : public thrust::unary_function<location_3d<T>,coord_2d<T> >
	{
		location_3d<T> origin;
		__host__ __device__
		coord_transformation(location_3d<T> _origin): origin(_origin){}

		__host__ __device__
		coord_2d<T> operator()(location_3d<T> pt)
		{
		  coord_2d<T> c;
		  c.x = ((origin.longitude - pt.longitude) * 40000.0 *
			   cos((origin.latitude + pt.latitude) * M_PI / 360) / 360);
		  c.y = (origin.latitude - pt.latitude) * 40000.0 / 360;
		  return c;
		}
	};
} // namespace cuspatial
