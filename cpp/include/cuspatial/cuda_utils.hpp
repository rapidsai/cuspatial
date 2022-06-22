#ifdef __CUDACC__
#define CUSPATIAL_HOST_DEVICE __host__ __device__
#else
#define CUSPATIAL_HOST_DEVICE
#endif
