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

#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <utilities/cuda_utils.hpp>
#include <type_traits>
#include <thrust/device_vector.h>
#include <thrust/count.h>

#include <utility/utility.hpp>
#include <cuspatial/query.hpp>

namespace {

/**
 *@brief Thrust functor for spatial window query on point data (x/y)
 *
*/
template<typename T>
struct spatial_window_functor_xy
{
    T x1,y1,x2,y2;

    __device__
    spatial_window_functor_xy(T x1, T x2, T y1, T y2)
    : x1(x1), y1(y1), x2(x2), y2(y2) {};

    __device__
    bool operator()(const thrust::tuple<T, T>& t)
    {
        T x= thrust::get<0>(t);
        T y= thrust::get<1>(t);
        bool b1 = x > x1 && x < x2;
        bool b2 = y > x1 && y < x2;
        return(b1 && b2);
    }
};

struct sw_point_functor 
{
    template <typename T>
    static constexpr bool is_supported()
    {
            return std::is_floating_point<T>::value;
    }
     
    template <typename T, std::enable_if_t< is_supported<T>() >* = nullptr>
    std::pair<gdf_column,gdf_column> operator()(const gdf_scalar x1,
                                                const gdf_scalar y1,
                                                const gdf_scalar x2,
                                                const gdf_scalar y2,
                                                const gdf_column& in_x,
                                                const gdf_column& in_y)
    {        
        T q_x1=*((T*)(&(x1.data)));
        T q_y1=*((T*)(&(y1.data)));
        T q_x2=*((T*)(&(x2.data)));
        T q_y2=*((T*)(&(y2.data)));
  
        CUDF_EXPECTS(q_x1<q_x2,"x1 must be less than x2 in a spatial window query");
        CUDF_EXPECTS(q_y1<q_y2,"y1 must be less than y2 in a spatial window query");

        cudaStream_t stream{0};
        auto exec_policy = rmm::exec_policy(stream)->on(stream);

        auto in_it=thrust::make_zip_iterator(thrust::make_tuple(
                                   static_cast<T*>(in_x.data),
                                   static_cast<T*>(in_y.data)));
                                   
        int num_hits= thrust::count_if(exec_policy, in_it, in_it+in_x.size, 
                                       spatial_window_functor_xy<T>(q_x1,
                                                                           q_x2,
                                                                           q_y1,
                                                                           q_y2));            
        T* temp_x{nullptr};
        T* temp_y{nullptr};
        RMM_TRY( RMM_ALLOC(&temp_x, num_hits * sizeof(T), 0) );
        RMM_TRY( RMM_ALLOC(&temp_y, num_hits * sizeof(T), 0) );
            
        auto out_it=thrust::make_zip_iterator(thrust::make_tuple(temp_x,temp_y));
        thrust::copy_if(exec_policy, in_it, in_it+in_x.size,out_it, 
	                        spatial_window_functor_xy<T>(q_x1, q_x2,
	                                                            q_y1, q_y2));

        gdf_column out_x,out_y;
        memset(&out_x,0,sizeof(gdf_column));
        memset(&out_y,0,sizeof(gdf_column));
        
        gdf_column_view_augmented(&out_x, temp_x, nullptr, num_hits,
                          in_x.dtype, 0,
                              gdf_dtype_extra_info{TIME_UNIT_NONE}, "x");          
        gdf_column_view_augmented(&out_y, temp_y, nullptr, num_hits,
                              in_y.dtype, 0,
                              gdf_dtype_extra_info{TIME_UNIT_NONE}, "y");          
            
        return std::make_pair(out_x,out_y);
    }

    template <typename T, std::enable_if_t< !is_supported<T>() >* = nullptr>
    std::pair<gdf_column,gdf_column> operator()(const gdf_scalar x1,
                                                const gdf_scalar y1,
                                                const gdf_scalar x2,
                                                const gdf_scalar y2,
                                                const gdf_column& in_x,
                                                const gdf_column& in_y)
    {
       CUDF_FAIL("Non-floating point operation is not supported");
    }
};

} // namespace anonymous

namespace cuspatial {

/*
 * Return all points (x,y) that fall within a query window (x1,y1,x2,y2)
 * see query.hpp
 */
std::pair<gdf_column,gdf_column> spatial_window_points(const gdf_scalar& x1,
                                                       const gdf_scalar& y1,
                                                       const gdf_scalar& x2,
                                                       const gdf_scalar& y2,
                                                       const gdf_column& in_x,
                                                       const gdf_column& in_y)
{
    CUDF_EXPECTS(in_x.dtype == in_y.dtype, "point type mismatch between x/y arrays");
    CUDF_EXPECTS(in_x.size == in_y.size, "#of points mismatch between x/y arrays");

    CUDF_EXPECTS(in_x.null_count == 0 && in_y.null_count == 0, "this version does not support point data that contains nulls");

    std::pair<gdf_column,gdf_column> res = cudf::type_dispatcher( in_x.dtype, sw_point_functor(), x1,y1,x2,y2,in_x,in_y);		

    return res;
}
  
}// namespace cuspatial
