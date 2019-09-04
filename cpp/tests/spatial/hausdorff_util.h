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

#include <algorithm>
#include <cudf/cudf.h>


template <typename T>
void hausdorff_test_sequential(int set_size,const T* pnt_x, const T* pnt_y, const uint32_t * cnt ,T *& h_dist)
{
    int block_sz=set_size*set_size;
    h_dist=new T[block_sz];
    assert(h_dist!=NULL);
    uint32_t *pos=new uint32_t[set_size];
    assert(pos!=NULL);
    for(int i=0;i<set_size;i++)
    	pos[i]=(i==0)?cnt[i]:pos[i-1]+cnt[i];

    for (int i = 0; i < block_sz; i++)
    {
        if(i%1000==0)
        	printf("i=%d\n",i);
        int left = i/set_size;
        int right = i%set_size;

        int start_left = (left == 0) ? 0 : pos[left-1];
        int stop_left = pos[left];

        int start_right = (right == 0) ? 0 : pos[right-1];
        int stop_right = pos[right];

        T dist = -1;
        for (int m = start_left; m < stop_left; m++)
        {
            T min_dist = 1e20;
            for (int n = start_right; n < stop_right; n++)
            {
                T new_dist = (pnt_x[m]-pnt_x[n])*(pnt_x[m]-pnt_x[n])+(pnt_y[m]-pnt_y[n])*(pnt_y[m]-pnt_y[n]);
                min_dist = std::min(min_dist, new_dist);
            }
            dist = std::max(min_dist, dist);
        }
        h_dist[i] = sqrt(dist);
    }
}

template void hausdorff_test_sequential(int,const double*, const double*,const uint32_t *,double*&);
template void hausdorff_test_sequential(int,const float*, const float*,const uint32_t *,float*&);
