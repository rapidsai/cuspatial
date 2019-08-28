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
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <utilities/error_utils.hpp>
#include <rmm/rmm.h>
#include <cuspatial/soa_readers.hpp>
#include <cuspatial/shared_util.h>

namespace cuspatial
{
	/**
	 * @Brief read timestamp (ts: its_timestamp type) data from file as column
	 * see soa_readers.hpp
	*/
	void read_ts_soa(const char *ts_fn, gdf_column& ts)                             
	{
    		its_timestamp * time=NULL;
    		size_t num_t=read_field<its_timestamp>(ts_fn,time);
    		if(time==NULL) return;
    		
    		/*printf("1st (hex):%016llx\n", *((unsigned long long *)(&(time[0]))));
		printf("1st: y=%d m=%d d=%d hh=%d mm=%d ss=%d wd=%d yd=%d ms=%d pid=%d\n",
			time[0].y,time[0].m,time[0].d,time[0].hh,time[0].mm,time[0].ss,time[0].wd, time[0].yd,time[0].ms,time[0].pid);*/
 		
 		ts.dtype= GDF_INT64;
 		ts.col_name=(char *)malloc(strlen("ts")+ 1);
		strcpy(ts.col_name,"ts");
		//make sure sizeof(TIME)==sizeof(GDF_INT64)
		RMM_TRY( RMM_ALLOC(&ts.data, num_t * sizeof(its_timestamp), 0) );
		cudaMemcpy(ts.data,time ,num_t * sizeof(its_timestamp) , cudaMemcpyHostToDevice);		
		ts.size=num_t;
		ts.valid=nullptr;
		ts.null_count=0;		
		delete[] time;
	}
}