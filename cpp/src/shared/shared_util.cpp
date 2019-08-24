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

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <cuspatial/shared_util.h>

namespace cuspatial {
	/**
	 * @brief output operator for TimeStamp
	 *
	 **/
	std::ostream& operator<<(std::ostream& os, const TimeStamp & t)
	{
		os <<"("<<t.y<<","<<t.m<<","<<t.d<<","<<t.hh<<","<<t.mm<<","<<t.ss<<","<<t.ms<<")";
		return os;
	}

	/**
	 * @brief timing function to calaculate duration between t1 and t0 in milliseconds and output the duration proceeded by msg
	 *
	 **/
	float calc_time(const char *msg,timeval t0, timeval t1)
	{
		long d = t1.tv_sec*1000000+t1.tv_usec - t0.tv_sec * 1000000-t0.tv_usec;
		float t=(float)d/1000;
		if(msg!=NULL)
			//printf("%s ...%10.3f\n",msg,t);
			std::cout<<msg<<t<<std::endl;
		return t;
	}

	/**
	 * @brief read a CSV file into a map
	 *
	 **/
	int read_csv(const char *fn, std::vector<std::string> cols,int num_col,std::map<std::string,std::vector<std::string>>& df)
	{
	   FILE *fp=fopen(fn,"r");
	   if(fp==NULL)
	   {
			//printf("can not open camera file %s\n",fn);
			std::cout<<"can not open camera file "<<fn<<std::endl;
			return(-1);
	   }
	   char line_str[3000];
	   std::vector<std::string> tokens;
	   char *tmp=fgets(line_str,3000,fp);
	   char *tok,*temp_str=line_str;
	   int cnt=0,ln=0;
	   while ((tok = strtok_r(temp_str, ",", &temp_str)))
	   {
		//printf("%d %s\n", cnt,tok);
		if(strcmp(tok,"")==0) break;
		if(cnt==num_col-1) break;
		assert(std::string(tok)==cols[cnt++]);
	   }
	   try
	   {
		while(!feof(fp))
		{
			tmp=fgets(line_str,3000,fp);
			//printf("(strlen(line_str)=%d\n",(strlen(line_str)));
			//printf("%s\n",line_str);
			if(strlen(line_str)==0) break;
			temp_str=line_str;
			tokens.clear();
			while ((tok = strtok_r(temp_str, ",", &temp_str)))
			{
				//printf("%s\n", tok);
				tokens.push_back(std::string(tok));
			}
			//printf("tokens.size()=%d\n",tokens.size());
			assert(tokens.size()==num_col);
			ln++;
			df[tokens[0]]=tokens;
		}
		}
		catch(const std::exception& e)
		{
			std::cout<<e.what()<<" at line"<<ln<<std::endl;
			return(-2);
		}
		fclose(fp);
		return(0);
	}//read_csv
} //namespace cuspatial