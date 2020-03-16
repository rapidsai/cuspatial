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

#include <time.h>
#include <sys/time.h>
#include <string>

#include <gtest/gtest.h>
#include <utilities/legacy/error_utils.hpp>
#include <tests/utilities/legacy/cudf_test_utils.cuh>
#include <tests/utilities/legacy/cudf_test_fixtures.h>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/table/table.hpp>
#include <cuspatial/quadtree.hpp>
#include <utility/helper_thrust.cuh>

struct QuadtreeOnPointIndexingTest : public GdfTest 
{
 
};

TEST_F(QuadtreeOnPointIndexingTest, test1)
{
    const int num_levels=3;
    int min_size=12;
    int point_len=71;

    double scale=1.0;
    double x1=0,x2=8,y1=0,y2=8;

    double xx[]={1.9804558865545805, 0.1895259128530169, 1.2591725716781235, 0.8178039499335275, 0.48171647380517046, 1.3890664414691907, 0.2536015260915061, 3.1907684812039956, 3.028362149164369, 3.918090468102582, 3.710910700915217, 3.0706987088385853, 3.572744183805594, 3.7080407833612004, 3.70669993057843, 3.3588457228653024, 2.0697434332621234, 2.5322042870739683, 2.175448214220591, 2.113652420701984, 2.520755151373394, 2.9909779614491687, 2.4613232527836137, 4.975578758530645, 4.07037627210835, 4.300706849071861, 4.5584381091040616, 4.822583857757069, 4.849847745942472, 4.75489831780737, 4.529792124514895, 4.732546857961497, 3.7622247877537456, 3.2648444465931474, 3.01954722322135, 3.7164018490892348, 3.7002781846945347, 2.493975723955388, 2.1807636574967466, 2.566986568683904, 2.2006520196663066, 2.5104987015171574, 2.8222482218882474, 2.241538022180476, 2.3007438625108882, 6.0821276168848994, 6.291790729917634, 6.109985464455084, 6.101327777646798, 6.325158445513714, 6.6793884701899, 6.4274219368674315, 6.444584786789386, 7.897735998643542, 7.079453687660189, 7.430677191305505, 7.5085184104988, 7.886010001346151, 7.250745898479374, 7.769497359206111, 1.8703303641352362, 1.7015273093278767, 2.7456295127617385, 2.2065031771469, 3.86008672302403, 1.9143371250907073, 3.7176098065039747, 0.059011873032214, 3.1162712022943757, 2.4264509160270813, 3.154282922203257};
    assert(sizeof(xx)/sizeof(double)==point_len);
    double yy[71]={1.3472225743317712, 0.5431061133894604, 0.1448705855995005, 0.8138440641113271, 1.9022922214961997, 1.5177694304735412, 1.8762161698642947, 0.2621847215928189, 0.027638405909631958, 0.3338651960183463, 0.9937713340192049, 0.9376313558467103, 0.33184908855075124, 0.09804238103130436, 0.7485845679979923, 0.2346381514128677, 1.1809465376402173, 1.419555755682142, 1.2372448404986038, 1.2774712415624014, 1.902015274420646, 1.2420487904041893, 1.0484414482621331, 0.9606291981013242, 1.9486902798139454, 0.021365525588281198, 1.8996548860019926, 0.3234041700489503, 1.9531893897409585, 0.7800065259479418, 1.942673409259531, 0.5659923375279095, 2.8709552313924487, 2.693039435509084, 2.57810040095543, 2.4612194182614333, 2.3345952955903906, 3.3999020934055837, 3.2296461832828114, 3.6607732238530897, 3.7672478678985257, 3.0668114607133137, 3.8159308233351266, 3.8812819070357545, 3.6045900851589048, 2.5470532680258002, 2.983311357415729, 2.2235950639628523, 2.5239201807166616, 2.8765450351723674, 2.5605928243991434, 2.9754616970668213, 2.174562817047202, 3.380784914178574, 3.063690547962938, 3.380489849365283, 3.623862886287816, 3.538128217886674, 3.4154469467473447, 3.253257011908445, 4.209727933188015, 7.478882372510933, 7.474216636277054, 6.896038613284851, 7.513564222799629, 6.885401350515916, 6.194330707468438, 5.823535317960799, 6.789029097334483, 5.188939408363776, 5.788316610960881};
    assert(sizeof(yy)/sizeof(double)==point_len);

    cudaStream_t stream=0;

    double *d_p_x=NULL,*d_p_y=NULL;
    RMM_TRY( RMM_ALLOC( &d_p_x,point_len* sizeof(double), stream));
    assert(d_p_x!=NULL);
    RMM_TRY( RMM_ALLOC( &d_p_y,point_len* sizeof(double), stream));
    assert(d_p_y!=NULL);    

    HANDLE_CUDA_ERROR( cudaMemcpy( d_p_x, xx, point_len * sizeof(double), cudaMemcpyHostToDevice ) );    
    HANDLE_CUDA_ERROR( cudaMemcpy( d_p_y, yy, point_len * sizeof(double), cudaMemcpyHostToDevice ) );     

    cudf::mutable_column_view x(cudf::data_type{cudf::FLOAT64},point_len,d_p_x);
    cudf::mutable_column_view y(cudf::data_type{cudf::FLOAT64},point_len,d_p_y);    

    std::unique_ptr<cudf::experimental::table> quadtree= cuspatial::quadtree_on_points(x,y,x1,y1,x2,y2, scale,num_levels, min_size);

    cudf::table_view quad_view=quadtree->view();
    std::cout<<"num cols="<<quad_view.num_columns()<<" num rows="<<quad_view.num_rows()<<std::endl;
    CUDF_EXPECTS(quad_view.num_columns()==5,"a quadtree table must have 5 columns (key,lev,sign,length,fpos)");
    
    const uint32_t *d_key=quad_view.column(0).data<uint32_t>();
    const uint8_t  *d_lev=quad_view.column(1).data<uint8_t>();
    const bool *d_sign=quad_view.column(2).data<bool>();
    const uint32_t *d_len=quad_view.column(3).data<uint32_t>();
    const uint32_t *d_fpos=quad_view.column(4).data<uint32_t>();

if(1)
{

    thrust::device_ptr<const uint32_t> d_key_ptr=thrust::device_pointer_cast(d_key);
    thrust::device_ptr<const uint8_t> d_lev_ptr=thrust::device_pointer_cast(d_lev);   
    thrust::device_ptr<const bool> d_sign_ptr=thrust::device_pointer_cast(d_sign);   
    thrust::device_ptr<const uint32_t> d_len_ptr=thrust::device_pointer_cast(d_len);	
    thrust::device_ptr<const uint32_t> d_fpos_ptr=thrust::device_pointer_cast(d_fpos);   

    printf("key\n");
    thrust::copy(d_key_ptr,d_key_ptr+quad_view.num_rows(),std::ostream_iterator<const uint32_t>(std::cout, " "));std::cout<<std::endl;

    printf("lev\n");
    //change from uint8_t to uint32_t in ostream_iterator to output numbers instead of special chars
    thrust::copy(d_lev_ptr,d_lev_ptr+quad_view.num_rows(),std::ostream_iterator<const uint32_t>(std::cout, " "));std::cout<<std::endl;

    printf("sign\n");
    thrust::copy(d_sign_ptr,d_sign_ptr+quad_view.num_rows(),std::ostream_iterator<const bool>(std::cout, " "));std::cout<<std::endl;

    printf("length\n");
    thrust::copy(d_len_ptr,d_len_ptr+quad_view.num_rows(),std::ostream_iterator<const uint32_t>(std::cout, " "));std::cout<<std::endl;

    printf("fpos\n");
    thrust::copy(d_fpos_ptr,d_fpos_ptr+quad_view.num_rows(),std::ostream_iterator<const uint32_t>(std::cout, " "));std::cout<<std::endl;
}
    
    uint32_t c_key[]={0, 1, 2, 0 ,1 ,3, 4, 7, 5, 6 ,13, 14, 28, 31};
    uint8_t  c_lev[]={0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2};
    bool c_sign[]={1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0};
    uint32_t c_len[]={3, 2, 11, 7, 2, 2, 9, 2, 9, 7, 5, 8, 8, 7};
    uint32_t c_fpos[]={3, 6, 60, 0, 8, 10, 36, 12, 7, 16, 23, 28, 45, 53};
    
    uint32_t n_key=sizeof(c_key)/sizeof(uint32_t);
    uint32_t n_lev=sizeof(c_lev)/sizeof(uint8_t);
    uint32_t n_sign=sizeof(c_sign)/sizeof(bool);
    uint32_t n_len=sizeof(c_len)/sizeof(uint32_t);
    uint32_t n_fpos=sizeof(c_fpos)/sizeof(uint32_t);    
    
    CUDF_EXPECTS(n_key==n_lev&& n_lev==n_sign && n_sign==n_len && n_len==n_fpos,"quadtree columns must have the same sizes");
    CUDF_EXPECTS(n_key==(uint32_t)(quad_view.num_rows()),"CPU and GPU results must agree on column sizes");

    uint32_t *h_key=new uint32_t[n_key];
    uint8_t  *h_lev=new uint8_t[n_key];
    bool     *h_sign=new bool[n_key];
    uint32_t *h_len=new uint32_t[n_key];
    uint32_t *h_fpos=new uint32_t[n_key];
    assert(h_key!=nullptr && h_lev!=nullptr && h_sign!=nullptr && h_len!=nullptr && h_fpos!=nullptr);

    EXPECT_EQ(cudaMemcpy(h_key,d_key,n_key*sizeof(uint32_t),cudaMemcpyDeviceToHost),cudaSuccess);
    EXPECT_EQ(cudaMemcpy(h_lev,d_lev,n_key*sizeof(uint8_t),cudaMemcpyDeviceToHost),cudaSuccess);
    EXPECT_EQ(cudaMemcpy(h_sign,d_sign,n_key*sizeof(bool),cudaMemcpyDeviceToHost),cudaSuccess);
    EXPECT_EQ(cudaMemcpy(h_len,d_len,n_key*sizeof(uint32_t),cudaMemcpyDeviceToHost),cudaSuccess);
    EXPECT_EQ(cudaMemcpy(h_fpos,d_fpos,n_key*sizeof(uint32_t),cudaMemcpyDeviceToHost),cudaSuccess);
    
    for(uint32_t i=0;i<n_key;i++)
    {
    	EXPECT_EQ(h_key[i],c_key[i]);
    	EXPECT_EQ(h_lev[i],c_lev[i]);
    	EXPECT_EQ(h_sign[i],c_sign[i]);
    	EXPECT_EQ(h_len[i],c_len[i]);
    	EXPECT_EQ(h_fpos[i],c_fpos[i]);
    }
    
    delete [] h_key;
    delete[] h_lev;
    delete[] h_sign;
    delete[] h_len;
    delete[] h_fpos;

}


