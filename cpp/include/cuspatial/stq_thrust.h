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

/**
 *@brief Thrust functor for spatial window query on point data (x/y)
 */
template<typename T>
struct sw_functor_xy
{
	T x1,y1,x2,y2;

	__host__ __device__
	sw_functor_xy(T _x1,T _x2,T _y1,T _y2):
		x1(_x1),y1(_y1),x2(_x2),y2(_y2){};

	__host__ __device__
	bool operator()(const thrust::tuple<T, T>& t)
	{
		T x= thrust::get<0>(t);
		T y= thrust::get<1>(t);
		bool b1=x>x1 && x<x2;
		bool b2=y>x1 && y<x2;
		return(b1&&b2);
	}
};
