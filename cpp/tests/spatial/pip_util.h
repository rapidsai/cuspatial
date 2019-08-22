#pragma once

#include <cuspatial/cuspatial.h>
#include <cuspatial/shared_util.h>

namespace cuSpatial
{
	/**
	 * @brief sequential point-in-polygon test between a single point and a single polygon;
		basic unit for either sequential execution or parallellization in multi-point/multi-polygon test cases
	 *
	 * @param[in]  x    the x coordinates of the input points

	 * @param[in]  y    the y coordinates of the input points

	 * @param[in]  ply    complete metadata for a polygon dataset (with multiple polygons)

	 * @param[in]  fid    index of the polygon dataset to identify the polyogn to be tested
	 *
	 * @return whehter the point is in the polygon
	 */
	template <typename T>
	bool pip_test_sequential(const T& x, const T& y, const struct PolyMeta<T>& ply,int fid)
	{
		//printf("pip: x=%15.10f y=%15.10f\n",x,y);
		uint *f_pos=ply.p_f_pos;
		uint *r_pos=ply.p_r_pos;
		T *poly_x=ply.p_x;
		T *poly_y=ply.p_y;
		uint r_f = (0 == fid) ? 0 : f_pos[fid-1];
		uint r_t=f_pos[fid];
		//printf("(%d %d)=>\n",r_f,r_t);
		bool in_polygon = false;
		for (uint r = r_f; r < r_t; r++) //for each ring
		{
			uint m = (r==0)?0:r_pos[r-1];
			for (;m < r_pos[r]-1; m++) //for each line segment
			{
			   T x0, x1, y0, y1;
			   x0 = poly_x[m];
			   y0 = poly_y[m];
			   x1 = poly_x[m+1];
			   y1 = poly_y[m+1];
			   //printf("r=%d m=%d, x0=%15.10f y0=%15.10f x1=%15.10f y1=%15.10f\n",r, m,x0,y0,x1,y1);
			   if ((((y0 <= y) && (y < y1)) ||
					 ((y1 <= y) && (y < y0))) &&
					 (x < (x1 - x0) * (y - y0) / (y1 - y0) + x0))
				  in_polygon = !in_polygon;
			}
		 }
		return (in_polygon);
	}

	template bool pip_test_sequential(const double& x, const double& y, const struct PolyMeta<double>& ply,int fid);
	template bool pip_test_sequential(const float& x, const float& y, const struct PolyMeta<float>& ply,int fid);


	/**
	 * @brief multi-point/multi-polygon test on CUPs with the same interface as the GPU implementation
	   parallelization (e.g., OpenMP and Intel TBB) can be applied to the array/vector of points.

	   @param[in]  num_pnt    number of points

	   @param[in]  x    pointer/array of x coodinates

	   @param[in]  x    pointer/array of y coodinates

	   @param[in]  ply  complete metadata for a polygon dataset (with multiple polygons)

	   @param[out]  res  pointer/array of unsinged integers; the jth bit of res[i] indicates whehter
	   a point of (x[i],y[i]) is in polygon j.

	   Note: The # of polygons, i.e., poly.f_num can not exceed sizeof(uint)*8, i.e., 32.

	*/
	template <typename T>
	void cpu_pip_loop(int num_pnt,const T* x, const T *y ,const struct PolyMeta<T>& poly,uint* res)
	{
		//assert(res!=NULL);
		for(int i=0;i<num_pnt;i++)
		{
			uint mask=0;
			for(size_t j=0;j<poly.num_f;j++)
			{
				bool in_polygon =pip_test_sequential<T>(x[i],y[i],poly,j);
				if(in_polygon)
				{
					mask|=(0x01<<j);
				}
			}
			res[i]=mask;
		}
	}

	template void cpu_pip_loop(int num_pnt,const double *x, const double *y ,const struct PolyMeta<double>& poly,uint* res);
	template void cpu_pip_loop(int num_pnt,const float *x, const float *y ,const struct PolyMeta<float>& poly,uint* res);
}//namespace cuSpatial