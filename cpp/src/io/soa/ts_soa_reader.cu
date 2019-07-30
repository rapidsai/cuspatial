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

namespace cuSpatial
{
	/**
	 * @Brief read timestamp (ts: Time type) data from file as column
	*
	*/
	void read_ts_soa(const char *ts_fn, gdf_column& ts)                             
	{
    		Time * time=NULL;
    		size_t num_t=read_field<Time>(ts_fn,time);
    		if(time==NULL) return;
    		
    		/*printf("1st (hex):%016llx\n", *((unsigned long long *)(&(time[0]))));
		printf("1st: y=%d m=%d d=%d hh=%d mm=%d ss=%d wd=%d yd=%d ms=%d pid=%d\n",
			time[0].y,time[0].m,time[0].d,time[0].hh,time[0].mm,time[0].ss,time[0].wd, time[0].yd,time[0].ms,time[0].pid);*/
 		
 		ts.dtype= GDF_INT64;
 		ts.col_name=(char *)malloc(strlen("ts")+ 1);
		strcpy(ts.col_name,"ts");
		//make sure sizeof(TIME)==sizeof(GDF_INT64)
		RMM_TRY( RMM_ALLOC(&ts.data, num_t * sizeof(Time), 0) );
		cudaMemcpy(ts.data,time ,num_t * sizeof(Time) , cudaMemcpyHostToDevice);		
		ts.size=num_t;
		ts.valid=nullptr;
		ts.null_count=0;		
		delete[] time;
	}
}