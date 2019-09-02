#include <string.h>
#include <algorithm>
#include <utility/utility.hpp>
#include "traj_util.h"

using namespace cuspatial;

/**
 * @brief retrive camera origin at a particular intersection from a configuration file
 *
 *
 * @return negative error code; 0 for success
 */

int get_camera_origin(const char *df_fn, const char * inter_name, location_3d<double> & camera_origin)
{
    const int num_col=39;
    const char *col_name[num_col]={"cameraIdString","ipaddress","gx0","gy0","gx1","gy1","gx2","gy2","gx3","gy3","cx0","cy0","cx1","cy1","cx2","cy2","cx3","cy3",
    "originLon","originLat","numberROI","roi_0_x0","roi_0_y0","roi_0_x1","roi_0_y1","roi_0_x2","roi_0_y2","roi_0_x3","roi_0_y3","roi_1_x0","roi_1_y0",
    "roi_1_x1","roi_1_y1","roi_1_x2","roi_1_y2","roi_1_x3","roi_1_y3","camera_resolutionX","camera_resolutionY"};
    std::vector<std::string> col_vec(col_name,col_name+num_col);
    std::map<std::string,std::vector<std::string>> camera_map;

    if(read_csv(df_fn,col_vec,num_col,camera_map)<0)
    {
		std::cout<<"can not read camera configurations................"<<std::endl;
		return -1;
	}
    else
    	std::cout<<"num cameras="<<camera_map.size()<<std::endl;

    int originLon_pos=std::find(col_vec.begin(),col_vec.end(),"originLon")-col_vec.begin();
    int originLat_pos=std::find(col_vec.begin(),col_vec.end(),"originLat")-col_vec.begin();
    if(!(originLon_pos>=0&&originLon_pos<num_col))
    {
		std::cout<<"can not locate originLon column"<<std::endl;
		return -2;
	}
    if(!(originLat_pos>=0&&originLat_pos<num_col))
    {
		std::cout<<"can not locate originLat column"<<std::endl;
		return -3;
	}

    auto hw20_locust_it=camera_map.find(inter_name);
    if(hw20_locust_it==camera_map.end())
    {
		std::cout<<"can not find intersection name"<< inter_name<<" from "<<df_fn<<std::endl;
		return (-4);
	}
    double originLon_val=std::stod((hw20_locust_it->second)[originLon_pos].c_str());
    double originLat_val=std::stod((hw20_locust_it->second)[originLat_pos].c_str());
    camera_origin.longitude=originLon_val;
    camera_origin.latitude=originLat_val;
    camera_origin.altitude=0;
    return (0);
}

/**
 * @brief retrive id, timestamp and location fields of a "raw" trajectory dataset,
 * i.e., a set of coordinates (lon/lat/alt) with a timestamp and an object (e.g., vehicle) identifier.
 *
 * @param[in]  root_fn: the root of the three files stored in columnar format,
 * with .objectid (uint32_t type),.time (its_timestamp type) and .location(location_3d type) extensions, respectively.
 * @param[out]  objid: out array for ID
 * @param[out]  time: out array for ID
 * @param[out]  location: out array for ID
 * @return the number of records (should be the same for all the three data files)
 */
size_t read_traj_soa(char *root_fn,int *& objid, its_timestamp *& time, location_3d<double>*&  location)
{
     enum FILEDS {objid_id=0,time_id,location_id};
	 const char * out_ext[]={".objectid",".time",".location"};

     objid=nullptr;
     location=nullptr;
     time=nullptr;
     char fn[100];
     strcpy(fn,root_fn);
     strcat(fn,out_ext[objid_id]);
     size_t objectid_len=read_field<int>(fn,objid);
     if(objid==nullptr) return 0;

     strcpy(fn,root_fn);
     strcat(fn,out_ext[time_id]);
     size_t time_len=read_field<its_timestamp>(fn,time);
     if(time==nullptr) return 0;

     strcpy(fn,root_fn);
     strcat(fn,out_ext[location_id]);
     size_t loc_len=read_field<location_3d<double> >(fn,location);
     if(location==nullptr) return 0;

     if((objectid_len!=loc_len||objectid_len!=time_len)) return 0;
     return objectid_len;
}
