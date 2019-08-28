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

#include <iostream>
//for partial_sum
#include <numeric>
#include <map>
#include <vector>
#include <cuspatial/cuspatial.h>
#include <utilities/error_utils.hpp>

namespace cuspatial
{
    /**
    * @brief  Data structure defines a set of polygons with mulitple rings

    * Please see Open Geospatial Consortium (OGC) Simple Feature Speficiation (SFS) for the definition with polygons

    * The structure allows to group polygons into multiple groups but the group level hierarchy variables,
    * i.e., num_group, group_length and group_position) are not exposed in the current version of cuSpatial.
    * But this can be changed for applications that need to orgnaize polygons into groups for fast indexing

    *{group,feature,ring_length}_position values should be the the prefix-sum (inclusive) of {group,feature,ring_length}_length values, respectively.
    *using e.g., std::partial_sum(...) and thrust::inclusive_scan (...)

    *If is_inplace is set to true, {group,feature,ring_length}_position will be the same as {group,feature,ring_length}_length to reduce memory footprint
    *This is useful only when it can be sure that {group,feature,ring_length}_length are not needed anymore (e.g., in CPU for sequentail access)
    *is_inplace should be set to false (default) when {group,feature,ring_length}_position and {group,feature,ring_length}_length are used together (on GPUs for efficiency)
    *
    **/

	template <typename T>
	struct polygons
	{
	    uint32_t num_group,num_feature,num_ring,num_vertex;//numbers of groups, features,rings and vertices
	    uint32_t *group_length{nullptr}; //pointer/array of numbers of features in all groups
	    uint32_t *feature_length{nullptr};//pointer/array of numbers of rings in all features
	    uint32_t *ring_length{nullptr};//pointer/array of numbers of vertices in all rings
	    uint32_t *group_position{nullptr};//pointer/array of offsets of features in all groups
	    uint32_t *feature_position{nullptr};//pointer/array of offsets of rings in all features
	    uint32_t *ring_position{nullptr};//pointer/array of offsets of vertices in all rings
	    T *x{nullptr};//pointer/array of x coordinates (for all groups/features/rings)
	    T *y{nullptr};//pointer/array of y coordinates (for all groups/features/rings)
	    bool is_inplace=false;
	};


    /**
    * @brief output operator for its_timestamp
    *
    **/
	std::ostream& operator<<(std::ostream& os, const its_timestamp & t);

    /**
    * @brief read a set of location_3d data (lon/lat/alt) into separate x,y arrays

    * @param[in] filename: name of a binary file storing an array of location_3d points
    * @param[out] lon: array of longitude values (initialized to nullptr)
    * @param[out] lat: array of latitude values (initialized to nullptr)

    * @returns number of points being read
    **/
	template <typename T>
	int read_point_lonlat(const char *filename,T *& lon, T *& lat)
	{
		FILE *fp=fopen(filename,"rb");
		CUDF_EXPECTS(fp!=nullptr,"can not open the input point file");
		fseek (fp , 0 , SEEK_END);
		size_t sz=ftell (fp);
		CUDF_EXPECTS(sz%sizeof(location_3d<T>)==0,"sizeof(location_3d) does not divide file length");
		int num_ringec = sz/sizeof(location_3d<T>);
		//std::cout<<"num_ringec="<<num_ringec<<std::endl;
		fseek (fp , 0 , SEEK_SET);

		lon=new T[num_ringec];
		lat=new T[num_ringec];
		CUDF_EXPECTS(lon!=nullptr && lat!=nullptr,"failed to allocation lon/lat arrays");
		location_3d<T> loc;
		for(int i=0;i<num_ringec;i++)
		{
			size_t t=fread(&loc,sizeof(location_3d<T>),1,fp);
			lon[i]=loc.longitude;
			lat[i]=loc.latitude;
		}
		fclose(fp);
		return num_ringec;
	}
	template int read_point_lonlat(const char *filename,double *& lon, double *& lat);
	template int read_point_lonlat(const char *filename,float *& lon, float *& lat);

    /**
    * @brief read a set of coordiante_2d data (x/y) into separate x,y arrays

    * @param[in] filename: name of a binary file storing an array of coordiante_2d points
	* @param[out] lon: array of x values (initialized to nullptr)
	* @param[out] lat: array of y values (initialized to nullptr)

    * @returns number of points being read
    **/

	template <typename T>
	int read_point_xy(const char *filename,T *& x, T *& y)
	{
		FILE *fp=fopen(filename,"rb");
		CUDF_EXPECTS(fp!=nullptr,"can not open the input point file");
		fseek (fp , 0 , SEEK_END);
		size_t sz=ftell (fp);
		CUDF_EXPECTS(sz%sizeof(coord_2d<T>)==0,"sizeof(coord_2d<T>) does not divide file length");
		int num_ringec = sz/sizeof(coord_2d<T>);
		//std::cout<<"num_rec="<<num_ringec<<std::endl;
		fseek (fp , 0 , SEEK_SET);

		x=new T[num_ringec];
		y=new T[num_ringec];
		CUDF_EXPECTS(x!=nullptr && y!=nullptr,"failed to allocation x/y arrays");
		coord_2d<T> coor;
		for(int i=0;i<num_ringec;i++)
		{
			size_t t=fread(&coor,sizeof(coord_2d<T>),1,fp);
			x[i]=coor.x;
			y[i]=coor.y;
		}
		fclose(fp);
		return num_ringec;
	}
	template int read_point_xy(const char *filename,double *& x, double *& y);
	template int read_point_xy(const char *filename,float *& x, float *& y);

    /**
    * @brief read polygon data into a polygons structure. Arrays in the structure,
    * initially nullptrs,are populated upon completion
    *
    * @param[in] poly_filename: name of a binary polygon file storing both vertices and indices
    * @param[out] ply: a polygons<T> structure holding vertex and indexing data

    *@returns number of points being read
    **/
	template <typename T>
	int read_polygon_soa(const char *poly_filename,struct polygons<T>& ply)
	{
		FILE *fp=fopen(poly_filename,"rb");
		CUDF_EXPECTS(fp!=nullptr,"can not open the input polygon file");

		//verify file integrity
		fseek(fp, 0L, SEEK_END);
		size_t sz=ftell(fp);
		fseek(fp, 0L, SEEK_SET);

		size_t ln=0;
		ln=fread(&(ply.num_group),sizeof(int),1,fp);
		CUDF_EXPECTS(ln==1,"expect reading an integer for num_group");
		ln=fread(&(ply.num_feature),sizeof(int),1,fp);
		CUDF_EXPECTS(ln==1,"expect reading an integer for num_feature");
		ln=fread(&(ply.num_ring),sizeof(int),1,fp);
		CUDF_EXPECTS(ln==1,"expect reading an integer for num_ring");
		ln=fread(&(ply.num_vertex),sizeof(int),1,fp);
		CUDF_EXPECTS(ln==1,"expect reading an integer for num_vertex");

		//brief outputs to check whether the numbers look reasonable or as expected
		//std::cout<<"# of groups="<< ply.num_group<<std::endl;
		std::cout<<"# of features="<<ply.num_feature<<std::endl;
		std::cout<<"# of rings="<< ply.num_ring<<std::endl;
		std::cout<<"# of vertices="<< ply.num_vertex<<std::endl;
		size_t len=(4+ply.num_group+ply.num_feature+ply.num_ring)*sizeof(int)+2*ply.num_vertex*sizeof(T);
		CUDF_EXPECTS(len==sz,"expecting file size and read size are the same");

		ply.group_length=new uint32_t[ ply.num_group];
		ply.feature_length=new uint32_t[ ply.num_feature];
		ply.ring_length=new uint32_t[ ply.num_ring];
		CUDF_EXPECTS(ply.group_length!=nullptr&&ply.feature_length!=nullptr&&ply.ring_length!=nullptr,"expecting p_{g,f,r}_len are non-zeron");
		ply.x=new T [ply.num_vertex];
		ply.y=new T [ply.num_vertex];
		CUDF_EXPECTS(ply.x!=nullptr&&ply.y!=nullptr,"expecting polygon x/y arrays are not nullptr");

		size_t r_g=fread(ply.group_length,sizeof(int),ply.num_group,fp);
		size_t r_f=fread(ply.feature_length,sizeof(int),ply.num_feature,fp);
		size_t r_r=fread(ply.ring_length,sizeof(int),ply.num_ring,fp);
		size_t r_x=fread(ply.x,sizeof(T),ply.num_vertex,fp);
		size_t r_y=fread(ply.y,sizeof(T),ply.num_vertex,fp);
		CUDF_EXPECTS(r_g==ply.num_group && r_f==ply.num_feature && r_r==ply.num_ring && r_x==ply.num_vertex && r_y==ply.num_vertex,
			"wrong number of data items read for index or vertex arrays");

		return ply.num_feature;
	}

	int read_polygon_soa(const char *poly_filename,struct polygons<double>& ply);
	int read_polygon_soa(const char *poly_filename,struct polygons<float>& ply);

    /**
    * @brief read a set of columns of a CSV file into a map

    * @param[in] filename: name of a CSV file to read from
    * @param[in] cols: specific column names to read
    * @param[in] num_col: number of columns to read
    * @param[out] df: a table storing rows of the specific columns (cols) as a map

    *@returns number of points being read
	 **/
	int read_csv(const char *filename, std::vector<std::string> cols,int num_col,std::map<std::string,std::vector<std::string>>& df);

	/**
	 * @brief timing function to calaculate duration between t1 and t0 in milliseconds and output the duration proceeded by msg
     * @param[in] msg: message to display
     * @param[in] t0: beginning timestamp
     * @param[in] t1: ending timestamp

	 *@returns time differnece between t0 and t1 in milliseconds
	 **/
	float calc_time(const char *msg,timeval t0, timeval t1);


	/**
	* @brief templated function to read data in SoA format from file to array
	*for the repective data type or user defined struct.

    * @param[in] filename: name of a binary file in SoA format to read
    * @param[out] field: pointer/array of values read from file

    *@returns number of records read from the file
	**/
	template<class T>
	size_t read_field(const char *filename,T*& field)
	{
		FILE *fp{nullptr};
		if((fp=fopen(filename,"rb"))==nullptr)
		CUDF_EXPECTS(fp!=nullptr,"can not open the input point file");

		fseek (fp , 0 , SEEK_END);
		size_t sz=ftell (fp);
		CUDF_EXPECTS(sz%sizeof(T)==0,"sizeof(T) does not divide file length");
		size_t num_ringec = sz/sizeof(T);
		//std::cout<<"num_rec="<<num_ringec<<std::endl;
		fseek (fp , 0 , SEEK_SET);

		field=new T[num_ringec];
		CUDF_EXPECTS(field!=nullptr,"data array allocaiton error");
		size_t t=fread(field,sizeof(T),num_ringec,fp);
		CUDF_EXPECTS(t==num_ringec,"wrong number of data items read from file");
		fclose(fp);
		return num_ringec;
	}

	//materialization with three data types/structs: uint32_t, location_3d and its_timestamp
	template size_t read_field(const char *,uint32_t *&);
	template size_t read_field(const char *,location_3d<double>*&);
	template size_t read_field(const char *,location_3d<float>*&);
	template size_t read_field(const char *,its_timestamp* &);
}
// namespace cuspatial