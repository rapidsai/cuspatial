// g++ json2soa.cpp cJSON.c -o json2soa -O3
//./json2soa schema_HWY_20_AND_LOCUST-filtered.json locust -1

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <unistd.h>
#include <vector>
#include <map>
#include <string>
#include <string.h>
#include <map>
#include <iostream>
#include <sstream>

#include "cJSON.h"

using namespace std;
#define MAXLINE  	 4096
#define NUM_FIELDS 5

typedef unsigned int uint;
typedef unsigned short ushort;

typedef struct Time
{
    uint y : 6;
    uint m : 4;
    uint d : 5;
    uint hh : 5;
    uint mm : 6;
    uint ss : 6;
    uint wd: 3;
    uint yd: 9;
    uint mili: 10;
    uint pid:10;
}Time;

ostream& operator<<(ostream& os, const Time & t)
{
    os << t.y<<","<<t.m<<","<<t.d<<","<<t.hh<<","<<t.mm<<","<<t.ss<<","<<t.mili;
    return os;
}

bool operator<(const Time & t1,const Time & t2)
{
	if(t1.y<t2.y) return true;
	else if((t1.y==t2.y)&&(t1.m<t2.m)) return true;
	else if((t1.y==t2.y)&&(t1.m==t2.m)&&(t1.d<t2.d)) return true;
	else if((t1.y==t2.y)&&(t1.m==t2.m)&&(t1.d==t2.d)&&(t1.hh<t2.hh)) return true;
	else if((t1.y==t2.y)&&(t1.m==t2.m)&&(t1.d==t2.d)&&(t1.hh==t2.hh)&&(t1.mm<t2.mm)) return true;
	else if((t1.y==t2.y)&&(t1.m==t2.m)&&(t1.d==t2.d)&&(t1.hh==t2.hh)&&(t1.mm==t2.mm)&&(t1.ss<t2.ss)) return true;
	else if((t1.y==t2.y)&&(t1.m==t2.m)&&(t1.d==t2.d)&&(t1.hh==t2.hh)&&(t1.mm==t2.mm)&&(t1.ss==t2.ss)&&(t1.mili<t2.mili)) return true;
	return false;
}

bool operator==(const Time & t1,const Time & t2)
{
	return((t1.y==t2.y)&&(t1.m==t2.m)&&(t1.d==t2.d)&&(t1.hh==t2.hh)&&(t1.mm==t2.mm)&&(t1.ss==t2.ss)&&(t1.mili==t2.mili));
}

template <class T>
void append_map(map<T,int>& m, const T& key)
{
	typename map<T,int>::iterator it=m.find(key);
	if(it==m.end())
		m[key]=1;
	else
		it->second++;
}

template <class T>
int output_map(const map<T,int>& m)
{
	int cnt=0;
	typename map<T,int>::const_iterator it = m.begin();
	for(; it != m.end(); ++it)
	{
 	    std::cout<<"("<< it->first<<")==>"<< it->second<<"\n";
 	    cnt+=it->second;
	}
	return cnt;
}

int main(int argc, char *argv[])
{
    printf("sizeof(Time)=%ld\n",sizeof(Time));
    //std::map<Time,int> t_map;
    //std::map<int,int> p_map;
    std::map<int,int> oid_map;

    char line[MAXLINE];

    struct timeval t0,t1;
    gettimeofday(&t0, NULL);

 	if (argc!=4)
    {
        printf("USAGE: %s in_fn out_root run_num(-1 for all)\n", argv[0]);
        exit(1);
    }
    const char * in_name=argv[1];
    const char * out_root=argv[2];
 	enum FILEDS {time_id=0, objid_id,bbox_id,location_id,coordinate_id};
    const char * out_ext[NUM_FIELDS]={".time",".objectid",".bbox",".location",".coordinate"};

 	FILE *in_fp=fopen(in_name,"r");
	if(in_fp==NULL)
	{
		printf("can not open data file %s for input\n",in_name);
		return -1;
	}

	FILE *out_fp[NUM_FIELDS];
	for(int i=0;i<NUM_FIELDS;i++)
	{
		char out_name[100];
		strcpy(out_name,out_root);
		strcat(out_name,out_ext[i]);
		out_fp[i]=fopen(out_name,"wb");
		if(out_fp[i]==NULL)
		{
			printf("can not open data file %s for output\n",out_name);
			return -1;
		}
	}
	int run_num=atoi(argv[3]);
	printf("using run_num %d\n",run_num);
	size_t pos=0;
	while(!feof(in_fp))
	{
		//printf("processing #%d\n",pos);
		ssize_t n;
		char *lp=fgets(line,MAXLINE,in_fp);
        //printf("%s\n",line);
	    cJSON* root = cJSON_Parse(line);

		cJSON* timestamp = cJSON_GetObjectItem(root,"@timestamp");
		char * t_str=timestamp->valuestring;
		struct tm it;
	    strptime(t_str, "%Y-%m-%dT%H:%M:%S", &it);
	    char *p=strstr(t_str,".");
	    p++;
        char st[4];
        strncpy(st,p,3);
        st[3]='\n';
        int in_mili=atoi(st);
		//printf("s=%s t=%s:%3d %d\n",t_str,asctime(&it),in_mili,it.tm_year);

		Time ot;
        ot.y = it.tm_year - 100;//shifting starting year from 1900 to 2000, max 64 years allowd
        ot.m =it.tm_mon;
        ot.d =it.tm_mday;
        ot.hh=it.tm_hour;
        ot.mm=it.tm_min;
        ot.ss=it.tm_sec;
        ot.wd=it.tm_wday;
        ot.yd=it.tm_yday;
        ot.mili=in_mili;
        //append_map(t_map,ot);

		cJSON* place= cJSON_GetObjectItem(root,"place");
		string place_str=cJSON_GetObjectItem(place,"id")->valuestring;
		//cout<<place_str<<" ";
		int place_id=stoi(place_str);
		assert(place_id<1024);
		ot.pid=place_id;
		fwrite(&ot,sizeof(Time),1,out_fp[time_id]);
		//append_map(p_map,place_id);

	    cJSON* object = cJSON_GetObjectItem(root,"object");
	    string objid_str=cJSON_GetObjectItem(object,"id")->valuestring;
	    //cout<<objid_str<<" ";
	    int obj_id=stoi(objid_str);
        fwrite(&obj_id,sizeof(int),1,out_fp[objid_id]);
		append_map(oid_map,obj_id);

	    cJSON* bbox=cJSON_GetObjectItem(object,"bbox");
	    cJSON* location = cJSON_GetObjectItem(object,"location");
	    cJSON* coordinate = cJSON_GetObjectItem(object,"coordinate");

	    double topleftx=cJSON_GetObjectItem(bbox,"topleftx")->valuedouble;
	    double toplefty=cJSON_GetObjectItem(bbox,"toplefty")->valuedouble;
	    double bottomrightx=cJSON_GetObjectItem(bbox,"bottomrightx")->valuedouble;
  		double bottomrighty=cJSON_GetObjectItem(bbox,"bottomrighty")->valuedouble;
	    //printf("%15.10f %15.10f %15.10f %15.10f\n",topleftx, toplefty, bottomrightx,bottomrighty);
        fwrite(&topleftx,sizeof(double),1,out_fp[bbox_id]);
        fwrite(&toplefty,sizeof(double),1,out_fp[bbox_id]);
        fwrite(&bottomrightx,sizeof(double),1,out_fp[bbox_id]);
        fwrite(&bottomrighty,sizeof(double),1,out_fp[bbox_id]);

	    double lat=cJSON_GetObjectItem(location,"lat")->valuedouble;
	    double lon=cJSON_GetObjectItem(location,"lon")->valuedouble;
	    double alt=cJSON_GetObjectItem(location,"alt")->valuedouble;
        //printf("%15.10f %15.10f %15.10f\n",lat, lon, alt);
        fwrite(&lat,sizeof(double),1,out_fp[location_id]);
        fwrite(&lon,sizeof(double),1,out_fp[location_id]);
        fwrite(&alt,sizeof(double),1,out_fp[location_id]);

	    double x=cJSON_GetObjectItem(coordinate,"x")->valuedouble;
	    double y=cJSON_GetObjectItem(coordinate,"y")->valuedouble;
	    double z=cJSON_GetObjectItem(coordinate,"z")->valuedouble;
	    //printf("%15.10f %15.10f %15.10f\n",x, y, z);
        fwrite(&x,sizeof(double),1,out_fp[coordinate_id]);
        fwrite(&y,sizeof(double),1,out_fp[coordinate_id]);
        fwrite(&z,sizeof(double),1,out_fp[coordinate_id]);

        cJSON_Delete(root);
  		pos++;
		if(pos==run_num)
			break;

    }
    fclose(in_fp);
    for(int i=0;i<NUM_FIELDS;i++)
        fclose(out_fp[i]);
   //printf("output time map.................%d\n",output_map(t_map));
   //printf("output place id map.................%d\n",output_map(p_map));
   printf("output object id map.................%d\n",output_map(oid_map));
   printf("num_rec=%lu\n",pos);
   return 0;
}