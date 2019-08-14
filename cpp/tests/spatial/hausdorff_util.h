#pragma once

#include <algorithm>
#include <cudf/cudf.h>


template <typename T>
void hausdorff_test_sequential(int set_size,const T* pnt_x, const T* pnt_y, const uint * cnt ,T *& h_dist)
{
    int block_sz=set_size*set_size;
    h_dist=new T[block_sz];
    assert(h_dist!=NULL);
    uint *pos=new uint[set_size];
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

template void hausdorff_test_sequential(int,const double*, const double*,const uint *,double*&);
template void hausdorff_test_sequential(int,const float*, const float*,const uint *,float*&);
