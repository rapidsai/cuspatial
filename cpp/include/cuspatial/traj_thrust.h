#pragma once

#include <ostream>
#include <cuspatial/cuspatial.h>

/**
 *@brief Thrust functor for comparing two Time variables; used in sorting based on timestamp
 */
__host__ __device__
inline bool operator<(const Time & t1,const Time & t2)
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

typedef  thrust::pair<Time, Time> TBBox;

/**
 *@brief Thrust functor for lifting Time to an interval (1D box)
 */
struct TBBox_transformation : public thrust::unary_function<TBBox,TBBox>
{
    __host__ __device__
        TBBox operator()(Time time)
        {
            return TBBox(time, time);
        }
};

/**
 *@brief Thrust functor for generating an 1D bounding box from two 1D box
 */
struct TBBox_reduction : public thrust::binary_function<TBBox,TBBox,TBBox>
{
    __host__ __device__
        TBBox operator()(TBBox a, TBBox b)
        {
            // lower left corner
            Time minT=(a.first<b.first)?a.first:b.first;
	    Time maxT=(a.second<b.second)?b.second:a.second;
            return TBBox(minT, maxT);
        }
};

/**
 *@brief Thrust functor for transforming lon/lat (in a Coord) to x/y relative to an origin
 Note: Both x and y are in the unit of kilometers (km)
 */

struct coord_transformation : public thrust::unary_function<Location,Coord>
{
    Location origin;
    __host__ __device__
    coord_transformation(Location _origin): origin(_origin){}

    __host__ __device__
    Coord operator()(Location pt)
    {
      Coord c;
      c.x = ((origin.lon - pt.lon) * 40000.0 *
	       cos((origin.lat + pt.lat) * M_PI / 360) / 360);
      c.y = (origin.lat - pt.lat) * 40000.0 / 360;
      return c;
    }
};