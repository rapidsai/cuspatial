#pragma once

#include <iostream>
//for partial_sum
#include <numeric>
#include <map>
#include <vector>
#include <cuspatial/cuspatial.h>
#include <utilities/error_utils.hpp>

namespace cuSpatial
{
	/**
	 * @brief struct for SoA (Structure of Array) representation of a set of polygons with multiple rings
	 * The group level hierarchy variables (num_g, p_g_len and p_g_pos) are not exposed in cuSpatial but
	 * can be added later for applications that need to orgnaize polygons into groups for fast indexing
	 p_{g,f,r}_pos should be the the prefix-sum (inclusive) of p_{g,f,r}_len,
	 using e.g., std::partial_sum(...) and thrust::inclusive_scan (...)
	 If is_inplace is set to true, p_{g,f,r}_pos will be the same as p_{g,f,r}_len to reduce memory footprint
	 This is useful only when it can be sure that p_{g,f,r}_len are not needed anymore (e.g., in CPU for sequentail access)
	 is_inplace should be set to false (default) when p_{g,f,r}_len and p_{g,f,r}_pos are used together (on GPUs for efficiency)
	 *
	 **/

	template <typename T>
	struct PolyMeta
	{
		uint32_t num_g,num_f,num_r,num_v;
		uint32_t * p_g_len=NULL,*p_f_len=NULL,*p_r_len=NULL;
		uint32_t * p_g_pos=NULL,*p_f_pos=NULL,*p_r_pos=NULL;
		T *p_x=NULL,*p_y=NULL;
		bool is_inplace=false;
	};


	/**
	 * @brief output operator for TimeStamp
	 *
	 **/
	std::ostream& operator<<(std::ostream& os, const TimeStamp & t);

	/**
	 * @brief read a set of Location3D data (lon/lat/alt) into separate x,y arrays
	 *
	 **/
	template <typename T>
	int read_point_ll(const char *fn,T *& x, T *& y)
	{
		std::cout<<"reading point data from file"<<fn<<std::endl;
		FILE *fp=NULL;
		if((fp=fopen(fn,"rb"))==NULL)
		{
			std::cerr<<"can not open file "<<fn<<std::endl;
			return(-1);
		}
		fseek (fp , 0 , SEEK_END);
		size_t sz=ftell (fp);
		CUDF_EXPECTS(sz%sizeof(Location3D)==0,"sizeof(Location3D) does not divide file length");
		int num_rec = sz/sizeof(Location3D);
		std::cout<<"num_rec="<<num_rec<<std::endl;
		fseek (fp , 0 , SEEK_SET);

		x=new T[num_rec];
		y=new T[num_rec];
		CUDF_EXPECTS(x!=NULL && y!=NULL,"failed to allocation x/y arrays");
		Location3D loc;
		for(int i=0;i<num_rec;i++)
		{
			size_t t=fread(&loc,sizeof(Location3D),1,fp);
			x[i]=loc.lon;
			y[i]=loc.lat;
		}
		fclose(fp);
		return num_rec;
	}
	template int read_point_ll(const char *fn,double *& x, double *& y);
	template int read_point_ll(const char *fn,float *& x, float *& y);

	template <typename T>
	int read_point_xy(const char *fn,T *& x, T *& y)
	{
		std::cout<<"reading point data from file"<<fn<<std::endl;
		FILE *fp=NULL;
		if((fp=fopen(fn,"rb"))==NULL)
		{
			std::cerr<<"can not open file "<<fn<<std::endl;
			return(-1);
		}
		fseek (fp , 0 , SEEK_END);
		size_t sz=ftell (fp);
		CUDF_EXPECTS(sz%sizeof(Coord2D)==0,"sizeof(Location3D) does not divide file length");
		int num_rec = sz/sizeof(Coord2D);
		std::cout<<"num_rec="<<num_rec<<std::endl;
		fseek (fp , 0 , SEEK_SET);

		x=new T[num_rec];
		y=new T[num_rec];
		CUDF_EXPECTS(x!=NULL && y!=NULL,"failed to allocation x/y arrays");
		Coord2D coor;
		for(int i=0;i<num_rec;i++)
		{
			size_t t=fread(&coor,sizeof(Coord2D),1,fp);
			x[i]=coor.x;
			y[i]=coor.y;
		}
		fclose(fp);
		return num_rec;
	}
	template int read_point_xy(const char *fn,double *& x, double *& y);
	template int read_point_xy(const char *fn,float *& x, float *& y);

	/**
	 * @brief read polygon data into a PolyMeta struct
	 *
	 **/
	template <typename T>
	int read_polygon_soa(const char *poly_fn,struct PolyMeta<T>& ply)
	{
		FILE *fp=fopen(poly_fn,"rb");
		CUDF_EXPECTS(fp!=NULL,"can not open the input polygon file");

		//verify file integrity
		fseek(fp, 0L, SEEK_END);
		size_t sz=ftell(fp);
		fseek(fp, 0L, SEEK_SET);

		size_t ln=0;
		ln=fread(&(ply.num_g),sizeof(int),1,fp);
		CUDF_EXPECTS(ln==1,"expect reading an integer for num_g");
		ln=fread(&(ply.num_f),sizeof(int),1,fp);
		CUDF_EXPECTS(ln==1,"expect reading an integer for num_f");
		ln=fread(&(ply.num_r),sizeof(int),1,fp);
		CUDF_EXPECTS(ln==1,"expect reading an integer for num_r");
		ln=fread(&(ply.num_v),sizeof(int),1,fp);
		CUDF_EXPECTS(ln==1,"expect reading an integer for num_v");

		//std::cout<<"# of groups="<< ply.num_g<<std::endl;
		std::cout<<"# of features="<<ply.num_f<<std::endl;
		std::cout<<"# of rings="<< ply.num_r<<std::endl;
		std::cout<<"# of vertices="<< ply.num_v<<std::endl;
		size_t len=(4+ply.num_g+ply.num_f+ply.num_r)*sizeof(int)+2*ply.num_v*sizeof(double);
		CUDF_EXPECTS(len==sz,"expecting file size and read size are the same");

		ply.p_g_len=new uint32_t[ ply.num_g];
		ply.p_f_len=new uint32_t[ ply.num_f];
		ply.p_r_len=new uint32_t[ ply.num_r];
		CUDF_EXPECTS(ply.p_g_len!=NULL&&ply.p_f_len!=NULL&&ply.p_r_len!=NULL,"expecting p_{g,f,r}_len are non-zeron");
		ply.p_x=new T [ply.num_v];
		ply.p_y=new T [ply.num_v];
		CUDF_EXPECTS(ply.p_x!=NULL&&ply.p_y!=NULL,"expecting polygon x/y arrays are not NULL");

		size_t r_g=fread(ply.p_g_len,sizeof(int),ply.num_g,fp);
		size_t r_f=fread(ply.p_f_len,sizeof(int),ply.num_f,fp);
		size_t r_r=fread(ply.p_r_len,sizeof(int),ply.num_r,fp);
		size_t r_x=fread(ply.p_x,sizeof(T),ply.num_v,fp);
		size_t r_y=fread(ply.p_y,sizeof(T),ply.num_v,fp);
		CUDF_EXPECTS(r_g==ply.num_g && r_f==ply.num_f && r_r==ply.num_r && r_x==ply.num_v && r_y==ply.num_v,
			"wrong number of data items read for index or vertex arrays");

		return ply.num_f;
	}

	int read_polygon_soa(const char *poly_fn,struct PolyMeta<double>& ply);
	int read_polygon_soa(const char *poly_fn,struct PolyMeta<float>& ply);

	/**
	 * @brief read a CSV file into a map
	 *
	 **/
	int read_csv(const char *fn, std::vector<std::string> cols,int num_col,std::map<std::string,std::vector<std::string>>& df);

	/**
	 * @brief timing function to calaculate duration between t1 and t0 in milliseconds and output the duration proceeded by msg
	 *
	 **/
	float calc_time(const char *msg,timeval t0, timeval t1);


	/**
	 * @brief templated function to read data in SoA format from file to array
	 for the repective data type or user defined struct.
	 *
	 **/
	template<class T>
	size_t read_field(const char *fn,T*& field)
	{
		FILE *fp=NULL;
		if((fp=fopen(fn,"rb"))==NULL)
		CUDF_EXPECTS(fp!=NULL,"can not open the input point file");

		fseek (fp , 0 , SEEK_END);
		size_t sz=ftell (fp);
		CUDF_EXPECTS(sz%sizeof(T)==0,"sizeof(T) does not divide file length");
		size_t num_rec = sz/sizeof(T);
		std::cout<<"num_rec="<<num_rec<<std::endl;
		fseek (fp , 0 , SEEK_SET);

		field=new T[num_rec];
		CUDF_EXPECTS(field!=NULL,"data array allocaiton error");
		size_t t=fread(field,sizeof(T),num_rec,fp);
		CUDF_EXPECTS(t==num_rec,"wrong number of data items read from file");
		fclose(fp);
		return num_rec;
	}

	//materialization with three data types/structs: uint32_t, Location3D and TimeStamp

	template size_t read_field(const char *,uint32_t *&);
	template size_t read_field(const char *,Location3D*&);
	template size_t read_field(const char *,TimeStamp* &);
}
// namespace cuSpatial