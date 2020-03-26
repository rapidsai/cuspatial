namespace
{

static void HandleCudaError( cudaError_t err,const char *file,int line )
{
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_CUDA_ERROR( err ) (HandleCudaError( err, __FILE__, __LINE__ ))

template <typename T>
struct get_vec_element
{
    T *d_p_vec=nullptr;
    get_vec_element(T *_d_p_vec):d_p_vec(_d_p_vec){}

    __device__ 
    T operator()(uint32_t idx)
    {
        //uint32_t tid = threadIdx.x + blockDim.x*blockIdx.x;
        //printf("get_vec_element=%d %d %d\n",tid,idx,_d_p_vec[idx]);
        return d_p_vec[idx];
    }
};

}// namespace cuspatial