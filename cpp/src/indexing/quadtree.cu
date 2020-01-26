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

#include <vector>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cuspatial/quadtree.hpp>

namespace { //anonymous

std::vector<std::unique_ptr<cudf::column>> dowork(double *d_p_x,double *d_p_y,SBBox bbox, double scale,
	int point_len,int M, int MINSIZE, rmm::mr::device_memory_resource* mr, cudaStream_t stream)
	
                                         
{
    std::cout<<"point_len="<<point_len<<std::endl;
    std::cout<<"M="<<M<<std::endl;
    std::cout<<"MINSIZE="<<MINSIZE<<std::endl;
    
    auto d_pnt_iter=thrust::make_zip_iterator(thrust::make_tuple(d_p_x,d_p_y));       
    uint *d_p_pntkey=NULL,*d_p_runkey=NULL, *d_p_runlen=NULL;
    RMM_TRY( RMM_ALLOC( (void**)&(d_p_pntkey),point_len* sizeof(uint),0));
    RMM_TRY( RMM_ALLOC( (void**)&(d_p_runkey),point_len* sizeof(uint),0));
    RMM_TRY( RMM_ALLOC( (void**)&(d_p_runlen),point_len* sizeof(uint),0));
    assert(d_p_pntkey!=NULL & d_p_runkey!=NULL && d_p_runlen!=NULL);
    
    thrust::transform(thrust::device,d_pnt_iter,d_pnt_iter+point_len, d_p_pntkey,xytoz(bbox,M,scale));   
    thrust::stable_sort_by_key(thrust::device,d_p_pntkey, d_p_pntkey+point_len,d_pnt_iter);
    size_t num_run = thrust::reduce_by_key(thrust::device,d_p_pntkey,d_p_pntkey+point_len,
    	thrust::constant_iterator<int>(1),d_p_runkey,d_p_runlen).first -d_p_runkey;
    RMM_FREE(d_p_pntkey,0);d_p_pntkey=NULL;
    printf("num_run=%ld\n",num_run);

    uint *d_p_parentkey=NULL,*d_p_numchild=NULL,*d_p_pntlen=NULL;    
    RMM_TRY( RMM_ALLOC( (void**)&(d_p_parentkey),M*num_run* sizeof(uint),0));
    HANDLE_CUDA_ERROR( cudaMemcpy( (void *)d_p_parentkey, (void *)d_p_runkey, num_run * sizeof(uint), cudaMemcpyDeviceToDevice ) );
    assert(d_p_parentkey!=NULL);
    RMM_FREE(d_p_runkey,0);d_p_runkey=NULL;
    
    RMM_TRY( RMM_ALLOC( (void**)&(d_p_pntlen),M*num_run* sizeof(uint),0));    
    HANDLE_CUDA_ERROR( cudaMemcpy( (void *)d_p_pntlen, (void *)d_p_runlen, num_run * sizeof(uint), cudaMemcpyDeviceToDevice ) );
    assert(d_p_pntlen!=NULL);
    RMM_FREE(d_p_runlen,0);d_p_runlen=NULL;
     
    RMM_TRY( RMM_ALLOC( (void**)&(d_p_numchild),M*num_run* sizeof(uint),0));
    assert(d_p_numchild!=NULL);
    HANDLE_CUDA_ERROR( cudaMemset(d_p_numchild,0,num_run*sizeof(uint)) ); 
    
    int lev_num[M],lev_bpos[M],lev_epos[M];
    lev_num[M-1]=num_run;
    uint begin_pos=0, end_pos=num_run;
    for(int k=M-1;k>=0;k--)
    {  			        
        uint nk=thrust::reduce_by_key(thrust::device,
	    thrust::make_transform_iterator(d_p_parentkey+begin_pos,get_parent(2)),
	    thrust::make_transform_iterator(d_p_parentkey+end_pos,get_parent(2)),
	    thrust::constant_iterator<int>(1),
	    d_p_parentkey+end_pos,d_p_numchild+end_pos).first-(d_p_parentkey+end_pos);
        uint nn=thrust::reduce_by_key(thrust::device,
            thrust::make_transform_iterator(d_p_parentkey+begin_pos,get_parent(2)),
	    thrust::make_transform_iterator(d_p_parentkey+end_pos,get_parent(2)),
	    d_p_pntlen+begin_pos,
	    d_p_parentkey+end_pos,d_p_pntlen+end_pos).first-(d_p_parentkey+end_pos);
	assert(nk==nn);	
	printf("lev=%d cb=%d ce=%d nk=%d nn=%d\n",k,begin_pos,end_pos,nk,nn);
    	lev_num[k]=nk; lev_bpos[k]=begin_pos; lev_epos[k]=end_pos; 	  	
    	begin_pos=end_pos; end_pos+=nk;
    }  
    
    printf("begin_pos=%d   end_pos=%d\n",begin_pos,end_pos);
        
    std::unique_ptr<cudf::column> key_col = cudf::make_numeric_column(
       cudf::data_type(cudf::type_id::INT32), end_pos,cudf::mask_state::UNALLOCATED,  stream, mr);      
    uint *d_p_qtpkey=cudf::mutable_column_device_view::create(key_col->mutable_view(), stream)->data<uint>();
    assert(d_p_qtpkey!=NULL);

    uint *d_p_qtclen=NULL,*d_p_qtnlen=NULL;
    RMM_TRY( RMM_ALLOC( (void**)&(d_p_qtclen),end_pos* sizeof(uint),0));
    assert(d_p_qtclen!=NULL);
    RMM_TRY( RMM_ALLOC( (void**)&(d_p_qtnlen),end_pos* sizeof(uint),0));
    assert(d_p_qtnlen!=NULL);
    
    //reverse the order of quadtree nodes; skip the root node
    int num_count_nodes=0;
    for(int k=0;k<M;k++)
    {	
   	//printf("k=%d lev_bpos[k]=%d lev_epos[k]=%d\n",k,lev_bpos[k],lev_epos[k]);
   	int nq1=thrust::copy(thrust::device,d_p_parentkey+lev_bpos[k],d_p_parentkey+lev_epos[k],d_p_qtpkey+num_count_nodes)-(d_p_qtpkey+num_count_nodes);   	
   	int nq2=thrust::copy(thrust::device,d_p_numchild+lev_bpos[k],d_p_numchild+lev_epos[k],d_p_qtclen+num_count_nodes)-(d_p_qtclen+num_count_nodes); 
   	int nq3=thrust::copy(thrust::device,d_p_pntlen+lev_bpos[k],d_p_pntlen+lev_epos[k],d_p_qtnlen+num_count_nodes)-(d_p_qtnlen+num_count_nodes);   	
   	int nq4=thrust::reduce(thrust::device,d_p_pntlen+lev_bpos[k],d_p_pntlen+lev_epos[k]);
   	assert(nq1==nq2 && nq2==nq3 && nq4==point_len);
   	num_count_nodes+=nq1;
    } 
    assert(num_count_nodes==begin_pos);//root node not counted 
    
    //delete oversized nodes
    RMM_FREE(d_p_parentkey,0);d_p_parentkey=NULL;
    RMM_FREE(d_p_numchild,0);d_p_numchild=NULL;
    RMM_FREE(d_p_pntlen,0);d_p_pntlen=NULL;

    int num_parent_nodes=0;
    for(int k=1;k<M;k++) num_parent_nodes+=lev_num[k];
   
    //temporal device memory for vector expansion
    uint *d_p_tmppos=NULL;
    RMM_TRY( RMM_ALLOC( (void**)&(d_p_tmppos),num_parent_nodes* sizeof(uint),0));
    assert(d_p_tmppos!=NULL);
    thrust::exclusive_scan(thrust::device,d_p_qtclen,d_p_qtclen+num_parent_nodes,d_p_tmppos);
   
    size_t num_child_nodes=thrust::reduce(thrust::device,d_p_qtclen,d_p_qtclen+num_parent_nodes);   
    printf("num_child_nodes=%ld\n",num_child_nodes);
    uint *d_p_parentpos=NULL;
    RMM_TRY( RMM_ALLOC( (void**)&(d_p_parentpos),num_child_nodes* sizeof(uint),0));
    assert(d_p_parentpos!=NULL); 
    HANDLE_CUDA_ERROR( cudaMemset(d_p_parentpos,0,num_child_nodes*sizeof(uint)) );   
    thrust::scatter(thrust::device,thrust::make_counting_iterator(0),
  		thrust::make_counting_iterator(0)+num_parent_nodes,d_p_tmppos,d_p_parentpos);
    RMM_FREE(d_p_tmppos,0);d_p_tmppos=NULL;
   
    thrust::inclusive_scan(thrust::device,d_p_parentpos,d_p_parentpos+num_child_nodes,d_p_parentpos,thrust::maximum<int>()); 
    
   //counting the number of nodes whose children have numbers of points above MINSIZE;
   //note that we start at level 2 as level nodes (whose parents are the root node -level 0) need to be kept  
   auto iter_in=thrust::make_zip_iterator(thrust::make_tuple(d_p_qtpkey+lev_num[1],d_p_qtclen+lev_num[1],d_p_qtnlen+lev_num[1],d_p_parentpos));
   int num_invalid_parent_nodes = thrust::count_if(thrust::device,iter_in,iter_in+(num_parent_nodes-lev_num[1]),remove_discard(d_p_qtnlen,MINSIZE));  
   RMM_FREE(d_p_parentpos,0);d_p_parentpos=NULL;

   assert(num_invalid_parent_nodes<=num_parent_nodes);
   num_parent_nodes-=num_invalid_parent_nodes;
 
   uint *d_p_templen=NULL;
   RMM_TRY( RMM_ALLOC( (void**)&(d_p_templen),end_pos* sizeof(uint),0));
   assert(d_p_templen!=NULL);
   HANDLE_CUDA_ERROR( cudaMemcpy( (void *)d_p_templen, (void *)d_p_qtnlen, end_pos * sizeof(uint), cudaMemcpyDeviceToDevice ) );       
   int num_valid_nodes = thrust::remove_if(thrust::device,iter_in,iter_in+num_child_nodes,remove_discard(d_p_templen,MINSIZE))-iter_in;
   RMM_FREE(d_p_templen,0);d_p_templen=NULL;
   //add back level 1 nodes
   num_valid_nodes+=lev_num[1];
   printf("num_invalid_parent_nodes=%d, num_valid_nodes=%d\n",num_invalid_parent_nodes,num_valid_nodes);
     
   std::unique_ptr<cudf::column> sign_col = cudf::make_numeric_column(
           cudf::data_type(cudf::type_id::BOOL8), end_pos,cudf::mask_state::UNALLOCATED,  stream, mr);      
   bool *d_p_sign=cudf::mutable_column_device_view::create(sign_col->mutable_view(), stream)->data<bool>();
   assert(d_p_sign!=NULL);
   
    HANDLE_CUDA_ERROR( cudaMemset(d_p_sign,0,num_valid_nodes*sizeof(bool)) );
    thrust::transform(thrust::device,d_p_qtnlen,d_p_qtnlen+num_parent_nodes,d_p_sign,thrust::placeholders::_1 > MINSIZE);  
    thrust::replace_if(thrust::device,d_p_qtnlen,d_p_qtnlen+num_parent_nodes,d_p_sign,thrust::placeholders::_1,0);
 
    printf("total points=%d\n",thrust::reduce(thrust::device,d_p_qtnlen,d_p_qtnlen+num_valid_nodes));
    printf("non-last-level points=%d\n",thrust::reduce(thrust::device,d_p_qtnlen,d_p_qtnlen+num_parent_nodes));

   //assembling 
   uint *d_p_qtnpos=NULL,*d_p_qtcpos=NULL;
   RMM_TRY( RMM_ALLOC( (void**)&(d_p_qtnpos),num_valid_nodes* sizeof(uint),0));
   RMM_TRY( RMM_ALLOC(  (void**)&(d_p_qtcpos),num_valid_nodes* sizeof(uint),0));
   assert(d_p_qtcpos!=NULL && d_p_qtcpos!=NULL);
 
 
 
   thrust::exclusive_scan(thrust::device,d_p_qtnlen,d_p_qtnlen+num_valid_nodes,d_p_qtnpos);
   thrust::replace_if(thrust::device,d_p_qtclen,d_p_qtclen+num_valid_nodes,d_p_sign,!thrust::placeholders::_1,0);
   
   //
   thrust::exclusive_scan(thrust::device,d_p_qtclen,d_p_qtclen+num_valid_nodes,d_p_qtcpos,lev_num[1]);   
 
   thrust::device_ptr<uint> d_qtclen_ptr=thrust::device_pointer_cast(d_p_qtclen);
   thrust::device_ptr<uint> d_qtnlen_ptr=thrust::device_pointer_cast(d_p_qtnlen);
   thrust::copy(d_qtclen_ptr,d_qtclen_ptr+num_valid_nodes,std::ostream_iterator<uint>(std::cout, " "));std::cout<<std::endl;
   thrust::copy(d_qtnlen_ptr,d_qtnlen_ptr+num_valid_nodes,std::ostream_iterator<uint>(std::cout, " "));std::cout<<std::endl;

   printf("pos\n");
   thrust::device_ptr<uint> d_qtcpos_ptr=thrust::device_pointer_cast(d_p_qtcpos);
   thrust::device_ptr<uint> d_qtnpos_ptr=thrust::device_pointer_cast(d_p_qtnpos);
   thrust::copy(d_qtcpos_ptr,d_qtcpos_ptr+num_valid_nodes,std::ostream_iterator<uint>(std::cout, " "));std::cout<<std::endl;
   thrust::copy(d_qtnpos_ptr,d_qtnpos_ptr+num_valid_nodes,std::ostream_iterator<uint>(std::cout, " "));std::cout<<std::endl;

 
   auto iter_len_in=thrust::make_zip_iterator(thrust::make_tuple(d_p_qtclen,d_p_qtnlen,d_p_sign));
   auto iter_pos_in=thrust::make_zip_iterator(thrust::make_tuple(d_p_qtcpos,d_p_qtnpos,d_p_sign));
 
   std::unique_ptr<cudf::column> length_col = cudf::make_numeric_column(
   cudf::data_type(cudf::type_id::INT32), end_pos,cudf::mask_state::UNALLOCATED,  stream, mr);      
   uint *d_p_qtlength=cudf::mutable_column_device_view::create(length_col->mutable_view(), stream)->data<uint>();
   assert(d_p_qtlength!=NULL);

   std::unique_ptr<cudf::column> fpos_col = cudf::make_numeric_column(
   cudf::data_type(cudf::type_id::INT32), end_pos,cudf::mask_state::UNALLOCATED,  stream, mr);      
   uint *d_p_qtfpos=cudf::mutable_column_device_view::create(fpos_col->mutable_view(), stream)->data<uint>();
   assert(d_p_qtfpos!=NULL);   
  
   thrust::transform(thrust::device,iter_len_in,iter_len_in+num_valid_nodes,d_p_qtlength,what2output());
   thrust::transform(thrust::device,iter_pos_in,iter_pos_in+num_valid_nodes,d_p_qtfpos,what2output());
   
   RMM_FREE(d_p_qtnpos,0);d_p_qtnpos=NULL;
   RMM_FREE(d_p_qtcpos,0);d_p_qtcpos=NULL;
   RMM_FREE(d_p_qtnlen,0);d_p_qtnlen=NULL;
   RMM_FREE(d_p_qtclen,0);d_p_qtclen=NULL;

if(1)
{

    thrust::device_ptr<uint> d_qtpkey_ptr=thrust::device_pointer_cast(d_p_qtpkey);	
    thrust::device_ptr<bool> d_sign_ptr=thrust::device_pointer_cast(d_p_sign);   
    thrust::device_ptr<uint> d_qtlength_ptr=thrust::device_pointer_cast(d_p_qtlength);	
    thrust::device_ptr<uint> d_qtfpos_ptr=thrust::device_pointer_cast(d_p_qtfpos);   
 
    printf("key\n");
    thrust::copy(d_qtpkey_ptr,d_qtpkey_ptr+num_valid_nodes,std::ostream_iterator<uint>(std::cout, " "));std::cout<<std::endl;
    printf("sign\n");
    thrust::copy(d_sign_ptr,d_sign_ptr+num_valid_nodes,std::ostream_iterator<bool>(std::cout, " "));std::cout<<std::endl;
    printf("length\n");
    thrust::copy(d_qtlength_ptr,d_qtlength_ptr+num_valid_nodes,std::ostream_iterator<uint>(std::cout, " "));std::cout<<std::endl;
    printf("fpos\n");
    thrust::copy(d_qtfpos_ptr,d_qtfpos_ptr+num_valid_nodes,std::ostream_iterator<uint>(std::cout, " "));std::cout<<std::endl;
}   

   std::vector<std::unique_ptr<cudf::column>> quad_cols;
   quad_cols.push_back(std::move(key_col));
   quad_cols.push_back(std::move(sign_col));
   quad_cols.push_back(std::move(length_col));
   quad_cols.push_back(std::move(fpos_col));
   return quad_cols;
}

/*struct quadtree_point_processor {
 
  
  template<typename T, std::enable_if_t<std::is_floating_point<T>::value >* = nullptr>
  std::unique_ptr<cudf::experimental::table> operator()(cudf::mutable_column_view x,
 					  cudf::mutable_column_view y,
 					  quad_point_inputs qpi,
                                          rmm::mr::device_memory_resource* mr,
                                          cudaStream_t stream)
                                          {
    //double *d_p_x=cudf::mutable_column_device_view::create(x, stream)->data<double>();
    //double *d_p_y=cudf::mutable_column_device_view::create(y, stream)->data<double>();

    T *d_p_x=x.data<T>();
    T *d_p_y=y.data<T>();
   
    std::vector<std::unique_ptr<cudf::column>> quad_cols=
    	dowork(d_p_x,d_p_x,thrust::get<0>(qpi),thrust::get<1>(qpi), x.size(), 
    		thrust::get<1>(qpi), thrust::get<2>(qpi),mr,stream);
  
    std::unique_ptr<cudf::experimental::table> destination_table = std::make_unique<cudf::experimental::table>(std::move(quad_cols));      
    return destination_table;
    }
  };*/
} //end anonymous namespace

namespace cuspatial {

std::unique_ptr<cudf::column> nested_column_test(cudf::column_view x,cudf::column_view y)
{ 
    std::vector<std::unique_ptr<cudf::column>> children;
    
    std::unique_ptr<cudf::column> key_col=cudf::make_numeric_column(cudf::data_type{cudf::INT32}, 1);
    children.push_back(std::move(key_col));

    std::unique_ptr<cudf::column> indicator_col=cudf::make_numeric_column(cudf::data_type{cudf::BOOL8}, 1);
    children.push_back(std::move(indicator_col));

    std::unique_ptr<cudf::column> fpos_col=cudf::make_numeric_column(cudf::data_type{cudf::INT32}, 1);
    children.push_back(std::move(fpos_col));

    std::unique_ptr<cudf::column> len_col=cudf::make_numeric_column(cudf::data_type{cudf::INT32}, 1);
    children.push_back(std::move(len_col));

    //children.push_back(x);
    //children.push_back(y);
    
    //cudf::data_type type=cudf::data_type{cudf::EMPTY};
    cudf::data_type type=cudf::data_type{cudf::INT32};
    cudf::size_type size=1;
    cudf::mask_state state=cudf::mask_state::ALL_NULL;
    cudaStream_t stream=0;
    rmm::mr::device_memory_resource* mr=rmm::mr::get_default_resource();
    rmm::device_buffer  buffer{size * cudf::size_of(type), stream, mr};
    rmm::device_buffer nmask=create_null_mask(size, state, stream, mr);
    cudf::size_type ncount=state_null_count(state, size);
    
    std::unique_ptr<cudf::column> ret=std::make_unique<cudf::column>(type,size,buffer,nmask,ncount,std::move(children));
    return ret;
}

std::unique_ptr<cudf::experimental::table> quadtree_on_points(cudf::mutable_column_view x,cudf::mutable_column_view y,
	double x1,double y1,double x2,double y2, double scale, int M, int MINSIZE)
{   
    cudaStream_t stream=0;
    rmm::mr::device_memory_resource* mr=rmm::mr::get_default_resource();
    
    /*quad_point_inputs qpi=thrust::make_tuple(bbox,scale,M,MINSIZE);
    return cudf::experimental::type_dispatcher(x.type(),quadtree_point_processor{}, 
    	x,y, qpi, mr,stream); */
     
    double *d_p_x=x.data<double>();
    double *d_p_y=y.data<double>();
    SBBox bbox(thrust::make_tuple(x1,y1),thrust::make_tuple(x2,y2)); 
    int point_len=x.size();
    std::vector<std::unique_ptr<cudf::column>> quad_cols=
    	dowork(d_p_x,d_p_x,bbox,scale,point_len,M,MINSIZE,mr,stream);
  
    std::unique_ptr<cudf::experimental::table> destination_table = std::make_unique<cudf::experimental::table>(std::move(quad_cols));      
    return destination_table;
  	
}

}// namespace cuspatial
