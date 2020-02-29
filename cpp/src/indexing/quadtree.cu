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
#include <utility/helper_thrust.cuh>
#include <utility/quadtree_thrust.cuh>


typedef thrust::tuple<double, double,double,double,double,uint32_t,uint32_t> quad_point_parameters;

namespace { //anonymous

//quadtree indexing on points using the bottom-up algorithm described at ref.
//http://www.adms-conf.org/2019-camera-ready/zhang_adms19.pdf

template<typename T>
std::vector<std::unique_ptr<cudf::column>> dowork(cudf::size_type point_len,
	uint32_t* d_p_id,T *d_p_x,T *d_p_y,SBBox<double> bbox, double scale,
	uint32_t num_level, uint32_t min_size, rmm::mr::device_memory_resource* mr, cudaStream_t stream)	
                                         
{
    double x1=thrust::get<0>(bbox.first);
    double y1=thrust::get<1>(bbox.first);
    double x2=thrust::get<0>(bbox.second);
    double y2=thrust::get<1>(bbox.second);
  
    std::cout<<"bounding box(x1,y1,x2,y2)=("<<x1<<","<<y1<<","<<x2<<","<<x2<<","<<y2<<std::endl;
    std::cout<<"scale="<<scale<<std::endl;
    std::cout<<"point_len="<<point_len<<std::endl;
    std::cout<<"num_level="<<num_level<<std::endl;
    std::cout<<"min_size="<<min_size<<std::endl;
    
    auto exec_policy = rmm::exec_policy(stream)->on(stream);
    
//debugging: make sure the inputs are correct
if(0)
{
    thrust::device_ptr<T> d_x_ptr=thrust::device_pointer_cast(d_p_x);	
    thrust::device_ptr<T> d_y_ptr=thrust::device_pointer_cast(d_p_y);   
    
    std::cout<<"x:"<<std::endl;
    thrust::copy(d_x_ptr,d_x_ptr+point_len,std::ostream_iterator<T>(std::cout, " "));std::cout<<std::endl;
    std::cout<<"x:"<<std::endl;
    thrust::copy(d_y_ptr,d_y_ptr+point_len,std::ostream_iterator<T>(std::cout, " "));std::cout<<std::endl;

}    
    auto d_pnt_iter=thrust::make_zip_iterator(thrust::make_tuple(d_p_id,d_p_x,d_p_y));       
    uint32_t *d_p_pntkey=NULL,*d_p_runkey=NULL, *d_p_runlen=NULL;
    RMM_TRY( RMM_ALLOC( (void**)&(d_p_pntkey),point_len* sizeof(uint32_t),stream));
    RMM_TRY( RMM_ALLOC( (void**)&(d_p_runkey),point_len* sizeof(uint32_t),stream));
    RMM_TRY( RMM_ALLOC( (void**)&(d_p_runlen),point_len* sizeof(uint32_t),stream));
    assert(d_p_pntkey!=NULL & d_p_runkey!=NULL && d_p_runlen!=NULL);
    
    //computing Morton code (Z-order) 
    thrust::transform(exec_policy,d_pnt_iter,d_pnt_iter+point_len, d_p_pntkey,xytoz<T>(bbox,num_level,scale));   

if(0)
{
   
    thrust::device_ptr<uint32_t> d_pntkey_ptr=thrust::device_pointer_cast(d_p_pntkey);	
    thrust::copy(d_pntkey_ptr,d_pntkey_ptr+point_len,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
}

    //sorting points based on Morton code based on the first paragrah of Section 4.2 of ref. 
    thrust::sort_by_key(exec_policy,d_p_pntkey, d_p_pntkey+point_len,d_pnt_iter);
    size_t num_run = thrust::reduce_by_key(exec_policy,d_p_pntkey,d_p_pntkey+point_len,
    	thrust::constant_iterator<int>(1),d_p_runkey,d_p_runlen).first -d_p_runkey;
    RMM_FREE(d_p_pntkey,stream);d_p_pntkey=NULL;
    std::cout<<"num_run"<<num_run<<std::endl;

    //allocate sufficient GPU memory for "full quadrants" (Secection 4.1 of ref.)
    uint32_t *d_p_parentkey=NULL,*d_p_numchild=NULL,*d_p_pntlen=NULL;    
    RMM_TRY( RMM_ALLOC( (void**)&(d_p_parentkey),num_level*num_run* sizeof(uint32_t),stream));
    HANDLE_CUDA_ERROR( cudaMemcpy( (void *)d_p_parentkey, (void *)d_p_runkey, num_run * sizeof(uint32_t), cudaMemcpyDeviceToDevice ) );
    assert(d_p_parentkey!=NULL);
    RMM_FREE(d_p_runkey,stream);d_p_runkey=NULL;
    
    RMM_TRY( RMM_ALLOC( (void**)&(d_p_pntlen),num_level*num_run* sizeof(uint32_t),stream));    
    HANDLE_CUDA_ERROR( cudaMemcpy( (void *)d_p_pntlen, (void *)d_p_runlen, num_run * sizeof(uint32_t), cudaMemcpyDeviceToDevice ) );
    assert(d_p_pntlen!=NULL);
    RMM_FREE(d_p_runlen,stream);d_p_runlen=NULL;
     
    RMM_TRY( RMM_ALLOC( (void**)&(d_p_numchild),num_level*num_run* sizeof(uint32_t),stream));
    assert(d_p_numchild!=NULL);
    HANDLE_CUDA_ERROR( cudaMemset(d_p_numchild,0,num_run*sizeof(uint32_t)) ); 
    
    //generating keys of paraent quadrants and numbers of child quadrants of "full quadrants" 
    //based on the second of paragraph of Section 4.2 of ref. 
    //keeping track of the number of quadrants, their begining/ending positions for each level 
    int lev_num[num_level],lev_bpos[num_level],lev_epos[num_level];
    lev_num[num_level-1]=num_run;
    uint32_t begin_pos=0, end_pos=num_run;
    for(int k=num_level-1;k>=0;k--)
    {  			        
         uint32_t nk=thrust::reduce_by_key(exec_policy,
	    thrust::make_transform_iterator(d_p_parentkey+begin_pos,get_parent(2)),
	    thrust::make_transform_iterator(d_p_parentkey+end_pos,get_parent(2)),
	    thrust::constant_iterator<int>(1),
	    d_p_parentkey+end_pos,d_p_numchild+end_pos).first-(d_p_parentkey+end_pos);
        uint32_t nn=thrust::reduce_by_key(exec_policy,
            thrust::make_transform_iterator(d_p_parentkey+begin_pos,get_parent(2)),
	    thrust::make_transform_iterator(d_p_parentkey+end_pos,get_parent(2)),
	    d_p_pntlen+begin_pos,
	    d_p_parentkey+end_pos,d_p_pntlen+end_pos).first-(d_p_parentkey+end_pos);
	assert(nk==nn);	
	std::cout<<"lev="<<k<<" begin_pos="<<begin_pos<<" end_pos="<<end_pos<<" nk="<<nk<<" nn="<<nn<<std::endl;
    	lev_num[k]=nk; lev_bpos[k]=begin_pos; lev_epos[k]=end_pos; 	  	
    	begin_pos=end_pos; end_pos+=nk; 
 }  
            
    //allocate three temporal arrays for parent key,number of children,
    //and the number of points in each quadrant, respectively
    //d_p_fullkey will be copied to the data array of the key column after revmoing invlaid quadtree ndoes
    //d_p_qtclen and d_p_qtnlen will be combined to generate the final length array
    //see fig.1 of ref. 
    uint32_t *d_p_fullkey=NULL,*d_p_qtclen=NULL,*d_p_qtnlen=NULL;
    RMM_TRY( RMM_ALLOC( (void**)&(d_p_fullkey),end_pos* sizeof(uint32_t),stream));
    assert(d_p_fullkey!=NULL);
    RMM_TRY( RMM_ALLOC( (void**)&(d_p_qtclen),end_pos* sizeof(uint32_t),stream));
    assert(d_p_qtclen!=NULL);
    RMM_TRY( RMM_ALLOC( (void**)&(d_p_qtnlen),end_pos* sizeof(uint32_t),stream));
    assert(d_p_qtnlen!=NULL);
    uint8_t *d_p_fulllev=NULL;
    RMM_TRY( RMM_ALLOC( (void**)&(d_p_fulllev),end_pos* sizeof(uint8_t),0));
    assert(d_p_fulllev!=NULL);
   
    //reverse the order of quadtree nodes for easier manipulation; skip the root node 
    int num_count_nodes=0;
    for(uint32_t k=0;k<num_level;k++)
    {	
   	thrust::fill(thrust::device,d_p_fulllev+num_count_nodes,d_p_fulllev+num_count_nodes+(lev_epos[k]-lev_bpos[k]),k);
   	int nq1=thrust::copy(exec_policy,d_p_parentkey+lev_bpos[k],d_p_parentkey+lev_epos[k],d_p_fullkey+num_count_nodes)-(d_p_fullkey+num_count_nodes);   	
   	int nq2=thrust::copy(exec_policy,d_p_numchild+lev_bpos[k],d_p_numchild+lev_epos[k],d_p_qtclen+num_count_nodes)-(d_p_qtclen+num_count_nodes); 
   	int nq3=thrust::copy(exec_policy,d_p_pntlen+lev_bpos[k],d_p_pntlen+lev_epos[k],d_p_qtnlen+num_count_nodes)-(d_p_qtnlen+num_count_nodes);   	
   	int nq4=thrust::reduce(exec_policy,d_p_pntlen+lev_bpos[k],d_p_pntlen+lev_epos[k]);
   	assert(nq1==nq2 && nq2==nq3 && nq4==point_len);
   	num_count_nodes+=nq1;
    } 
    assert(num_count_nodes==begin_pos);//root node not counted 
    
    //delete oversized nodes for memroy efficiency
    //num_count_nodes should be typically much smaller than num_level*num_run 
    RMM_FREE(d_p_parentkey,stream);d_p_parentkey=NULL;
    RMM_FREE(d_p_numchild,stream);d_p_numchild=NULL;
    RMM_FREE(d_p_pntlen,stream);d_p_pntlen=NULL;

    int num_parent_nodes=0;
    for(uint32_t k=1;k<num_level;k++) num_parent_nodes+=lev_num[k];
   
    //temporal device memory for vector expansion
    uint32_t *d_p_tmppos=NULL;
    RMM_TRY( RMM_ALLOC( (void**)&(d_p_tmppos),num_parent_nodes* sizeof(uint32_t),stream));
    assert(d_p_tmppos!=NULL);
    //line 1 of algorithm in Fig. 5 in ref. 
    thrust::exclusive_scan(exec_policy,d_p_qtclen,d_p_qtclen+num_parent_nodes,d_p_tmppos);
   
    size_t num_child_nodes=thrust::reduce(exec_policy,d_p_qtclen,d_p_qtclen+num_parent_nodes);   
    std::cout<<"num_child_nodes="<<num_child_nodes<<std::endl;
    
    uint32_t *d_p_parentpos=NULL;
    RMM_TRY( RMM_ALLOC( (void**)&(d_p_parentpos),num_child_nodes* sizeof(uint32_t),stream));
    assert(d_p_parentpos!=NULL); 
    HANDLE_CUDA_ERROR( cudaMemset(d_p_parentpos,0,num_child_nodes*sizeof(uint32_t)) );   
    
    //line 2 of algorithm in Fig. 5 in ref. 
    thrust::scatter(exec_policy,thrust::make_counting_iterator(0),
  		thrust::make_counting_iterator(0)+num_parent_nodes,d_p_tmppos,d_p_parentpos);
    RMM_FREE(d_p_tmppos,stream);d_p_tmppos=NULL;
   
    //line 3 of algorithm in Fig. 5 in ref. 
    thrust::inclusive_scan(exec_policy,d_p_parentpos,d_p_parentpos+num_child_nodes,d_p_parentpos,thrust::maximum<int>()); 
    
    //counting the number of nodes whose children have numbers of points no less than min_size;
    //note that we start at level 2 as level nodes (whose parents are the root node -level 0) need to be kept  
    auto iter_in=thrust::make_zip_iterator(thrust::make_tuple(d_p_fullkey+lev_num[1],d_p_fulllev+lev_num[1],
   	d_p_qtclen+lev_num[1],d_p_qtnlen+lev_num[1],d_p_parentpos));
    int num_invalid_parent_nodes = thrust::count_if(exec_policy,iter_in,iter_in+(num_parent_nodes-lev_num[1]),
   	remove_discard(d_p_qtnlen,min_size));  

    assert(num_invalid_parent_nodes<=num_parent_nodes);
    num_parent_nodes-=num_invalid_parent_nodes;
 
    uint32_t *d_p_templen=NULL;
    RMM_TRY( RMM_ALLOC( (void**)&(d_p_templen),end_pos* sizeof(uint32_t),stream));
    assert(d_p_templen!=NULL);
   
    //line 4 of algorithm in Fig. 5 in ref. 
    HANDLE_CUDA_ERROR( cudaMemcpy( (void *)d_p_templen, (void *)d_p_qtnlen, end_pos * sizeof(uint32_t), cudaMemcpyDeviceToDevice ) );     
    //line 5 of algorithm in Fig. 5 in ref. 
    int num_valid_nodes = thrust::remove_if(exec_policy,iter_in,iter_in+num_child_nodes,remove_discard(d_p_templen,min_size))-iter_in;
    RMM_FREE(d_p_templen,stream);d_p_templen=NULL;
    RMM_FREE(d_p_parentpos,stream);d_p_parentpos=NULL;
   
    //add back level 1 nodes
    num_valid_nodes+=lev_num[1];
    std::cout<<"num_invalid_parent_nodes="<<num_invalid_parent_nodes<<std::endl;
    std::cout<<"num_valid_nodes="<<num_valid_nodes<<std::endl;
     
    //preparing the key column for output 
    //Note: only the first num_valid_nodes elements should in the output array
    std::unique_ptr<cudf::column> key_col = cudf::make_numeric_column(
       cudf::data_type(cudf::type_id::INT32), num_valid_nodes,cudf::mask_state::UNALLOCATED,  stream, mr);      
    uint32_t *d_p_qtkey=cudf::mutable_column_device_view::create(key_col->mutable_view(), stream)->data<uint32_t>();
    assert(d_p_qtkey!=NULL);
  
    thrust::copy(exec_policy,d_p_fullkey,d_p_fullkey+num_valid_nodes,d_p_qtkey);
    RMM_FREE(d_p_fullkey,stream);d_p_fullkey=NULL;

    std::unique_ptr<cudf::column> lev_col = cudf::make_numeric_column(
       cudf::data_type(cudf::type_id::INT8), num_valid_nodes,cudf::mask_state::UNALLOCATED,  stream, mr);      
    uint8_t *d_p_qtlev=cudf::mutable_column_device_view::create(lev_col->mutable_view(), stream)->data<uint8_t>();
    assert(d_p_qtlev!=NULL);
  
    thrust::copy(exec_policy,d_p_fulllev,d_p_fulllev+num_valid_nodes,d_p_qtlev);
    RMM_FREE(d_p_fulllev,stream);d_p_fulllev=NULL;   
   
    //preparing the indicator array for output
    std::unique_ptr<cudf::column> sign_col = cudf::make_numeric_column(
           cudf::data_type(cudf::type_id::BOOL8), num_valid_nodes,cudf::mask_state::UNALLOCATED,  stream, mr);      
    bool *d_p_qtsign=cudf::mutable_column_device_view::create(sign_col->mutable_view(), stream)->data<bool>();
    assert(d_p_qtsign!=NULL);
   
    HANDLE_CUDA_ERROR( cudaMemset(d_p_qtsign,0,num_valid_nodes*sizeof(bool)) );
    //line 6 of algorithm in Fig. 5 in ref. 
    thrust::transform(exec_policy,d_p_qtnlen,d_p_qtnlen+num_parent_nodes,d_p_qtsign,thrust::placeholders::_1 > min_size);  
    //line 7 of algorithm in Fig. 5 in ref. 
    thrust::replace_if(exec_policy,d_p_qtnlen,d_p_qtnlen+num_parent_nodes,d_p_qtsign,thrust::placeholders::_1,0);
 
    std::cout<<"total point"<<thrust::reduce(exec_policy,d_p_qtnlen,d_p_qtnlen+num_valid_nodes)<<std::endl;
    std::cout<<"non-last-level points="<<thrust::reduce(exec_policy,d_p_qtnlen,d_p_qtnlen+num_parent_nodes)<<std::endl;

    //allocating two temporal array for the first child position array and first point position array, respectively
    //later they will be used to generate the final position array 
    uint32_t *d_p_qtnpos=NULL,*d_p_qtcpos=NULL;
    RMM_TRY( RMM_ALLOC( (void**)&(d_p_qtnpos),num_valid_nodes* sizeof(uint32_t),stream));
    RMM_TRY( RMM_ALLOC(  (void**)&(d_p_qtcpos),num_valid_nodes* sizeof(uint32_t),stream));
    assert(d_p_qtnpos!=NULL && d_p_qtcpos!=NULL);
   
    //revision to line 8 of algorithm in Fig. 5 in ref. 
    //ajust nlen and npos based on last-level z-order code
    uint32_t *d_p_tmp_key=NULL;
    RMM_TRY( RMM_ALLOC( (void**)&(d_p_tmp_key),num_valid_nodes* sizeof(uint32_t),stream));
    assert(d_p_tmp_key!=NULL);
    HANDLE_CUDA_ERROR( cudaMemcpy( (void *)d_p_tmp_key, (void *)d_p_qtkey, num_valid_nodes * sizeof(uint32_t), cudaMemcpyDeviceToDevice ) );
    uint32_t *d_p_tmp_pos=NULL;
    RMM_TRY( RMM_ALLOC( (void**)&(d_p_tmp_pos),num_valid_nodes* sizeof(uint32_t),stream));
    assert(d_p_tmp_pos!=NULL);

    auto key_lev_iter=thrust::make_zip_iterator(thrust::make_tuple(d_p_qtkey,d_p_qtlev,d_p_qtsign));
    thrust::transform(exec_policy,key_lev_iter,key_lev_iter+num_valid_nodes,d_p_tmp_key,flatten_z_code(num_level));
    uint32_t num_leaf_nodes=thrust::copy_if(exec_policy,thrust::make_counting_iterator(0),
   	thrust::make_counting_iterator(0)+num_valid_nodes,d_p_qtsign,d_p_tmp_pos,!thrust::placeholders::_1)-d_p_tmp_pos;   

    uint32_t *d_p_tmp_seq=NULL;
    RMM_TRY( RMM_ALLOC( (void**)&(d_p_tmp_seq),num_valid_nodes* sizeof(uint32_t),stream));
    assert(d_p_tmp_seq!=NULL);

    uint32_t *d_p_tmp_neln=NULL;
    RMM_TRY( RMM_ALLOC( (void**)&(d_p_tmp_neln),num_leaf_nodes* sizeof(uint32_t),stream));
    assert(d_p_tmp_neln!=NULL);

    uint32_t *d_p_tmp_npos=NULL;
    RMM_TRY( RMM_ALLOC( (void**)&(d_p_tmp_npos),num_valid_nodes* sizeof(uint32_t),stream));
    assert(d_p_tmp_npos!=NULL);

    thrust::sequence(exec_policy,d_p_tmp_seq,d_p_tmp_seq+num_valid_nodes);
    thrust::copy(exec_policy,d_p_qtnlen,d_p_qtnlen+num_valid_nodes,d_p_tmp_neln);   
    auto seq_len_pos=thrust::make_zip_iterator(thrust::make_tuple(d_p_tmp_seq,d_p_tmp_neln));
    thrust::stable_sort_by_key(exec_policy,d_p_tmp_key,d_p_tmp_key+num_valid_nodes,seq_len_pos);    

if(0)
{
   printf("d_p_tmp_key:after sort\n");
   thrust::device_ptr<uint> d_tmpkey_ptr=thrust::device_pointer_cast(d_p_tmp_key);
   thrust::copy(d_tmpkey_ptr,d_tmpkey_ptr+num_valid_nodes,std::ostream_iterator<uint>(std::cout, " "));std::cout<<std::endl;
   
   printf("d_p_tmp_seq:after sort\n");
   thrust::device_ptr<uint> d_tmpseq_ptr=thrust::device_pointer_cast(d_p_tmp_seq);
   thrust::copy(d_tmpseq_ptr,d_tmpseq_ptr+num_valid_nodes,std::ostream_iterator<uint>(std::cout, " "));std::cout<<std::endl; 
   
   printf("d_p_tmp_neln:after sort\n");
   thrust::device_ptr<uint> d_tmplen_ptr=thrust::device_pointer_cast(d_p_tmp_neln);
   thrust::copy(d_tmplen_ptr,d_tmplen_ptr+num_valid_nodes,std::ostream_iterator<uint>(std::cout, " "));std::cout<<std::endl;  
}   
    thrust::remove_if(exec_policy,d_p_tmp_neln,d_p_tmp_neln+num_valid_nodes,d_p_tmp_neln,thrust::placeholders::_1==0);
    //only the first num_leaf_nodes are needed
    thrust::exclusive_scan(exec_policy,d_p_tmp_neln,d_p_tmp_neln+num_leaf_nodes,d_p_tmp_npos);
    auto len_pos_iter=thrust::make_zip_iterator(thrust::make_tuple(d_p_tmp_neln,d_p_tmp_npos));
    thrust::stable_sort_by_key(thrust::device,d_p_tmp_seq,d_p_tmp_seq+num_leaf_nodes,len_pos_iter);
    
    RMM_TRY(RMM_FREE(d_p_tmp_seq,stream));d_p_tmp_seq=NULL; 
    HANDLE_CUDA_ERROR( cudaMemset(d_p_qtnlen,0,num_valid_nodes*sizeof(uint32_t)) ); 
    HANDLE_CUDA_ERROR( cudaMemset(d_p_qtnpos,0,num_valid_nodes*sizeof(uint32_t)) ); 
   
    auto in_len_pos_iter=thrust::make_zip_iterator(thrust::make_tuple(d_p_tmp_neln,d_p_tmp_npos));
    auto out_len_pos_iter=thrust::make_zip_iterator(thrust::make_tuple(d_p_qtnlen,d_p_qtnpos));
    thrust::scatter(thrust::device,in_len_pos_iter,in_len_pos_iter+num_leaf_nodes,d_p_tmp_pos,out_len_pos_iter);
    
    RMM_TRY(RMM_FREE(d_p_tmp_pos,stream));d_p_tmp_pos=NULL;
    RMM_TRY(RMM_FREE(d_p_tmp_neln,stream));d_p_tmp_neln=NULL;
    RMM_TRY(RMM_FREE(d_p_tmp_npos,stream));d_p_tmp_npos=NULL;
  
  
    //line 9 of algorithm in Fig. 5 in ref. 
    thrust::replace_if(exec_policy,d_p_qtclen,d_p_qtclen+num_valid_nodes,d_p_qtsign,!thrust::placeholders::_1,0);
   
    //line 10 of algorithm in Fig. 5 in ref. 
    thrust::exclusive_scan(exec_policy,d_p_qtclen,d_p_qtclen+num_valid_nodes,d_p_qtcpos,lev_num[1]);   

if(0)
{
   std::cout<<"length0:"<<std::endl;
   thrust::device_ptr<uint32_t> d_qtclen_ptr=thrust::device_pointer_cast(d_p_qtclen);
   thrust::device_ptr<uint32_t> d_qtnlen_ptr=thrust::device_pointer_cast(d_p_qtnlen);
   thrust::copy(d_qtclen_ptr,d_qtclen_ptr+num_valid_nodes,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
   thrust::copy(d_qtnlen_ptr,d_qtnlen_ptr+num_valid_nodes,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;

   std::cout<<"pos0:"<<std::endl;
   thrust::device_ptr<uint32_t> d_qtcpos_ptr=thrust::device_pointer_cast(d_p_qtcpos);
   thrust::device_ptr<uint32_t> d_qtnpos_ptr=thrust::device_pointer_cast(d_p_qtnpos);
   thrust::copy(d_qtcpos_ptr,d_qtcpos_ptr+num_valid_nodes,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
   thrust::copy(d_qtnpos_ptr,d_qtnpos_ptr+num_valid_nodes,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
}
 
   //preparing the length and fpos array for output 
   std::unique_ptr<cudf::column> length_col = cudf::make_numeric_column(
   cudf::data_type(cudf::type_id::INT32), num_valid_nodes,cudf::mask_state::UNALLOCATED,  stream, mr);      
   uint32_t *d_p_qtlength=cudf::mutable_column_device_view::create(length_col->mutable_view(), stream)->data<uint32_t>();
   assert(d_p_qtlength!=NULL);

   std::unique_ptr<cudf::column> fpos_col = cudf::make_numeric_column(
   cudf::data_type(cudf::type_id::INT32), num_valid_nodes,cudf::mask_state::UNALLOCATED,  stream, mr);      
   uint32_t *d_p_qtfpos=cudf::mutable_column_device_view::create(fpos_col->mutable_view(), stream)->data<uint32_t>();
   assert(d_p_qtfpos!=NULL);   

   //line 11 of algorithm in Fig. 5 in ref. 
   auto iter_len_in=thrust::make_zip_iterator(thrust::make_tuple(d_p_qtclen,d_p_qtnlen,d_p_qtsign));
   auto iter_pos_in=thrust::make_zip_iterator(thrust::make_tuple(d_p_qtcpos,d_p_qtnpos,d_p_qtsign));
   thrust::transform(exec_policy,iter_len_in,iter_len_in+num_valid_nodes,d_p_qtlength,what2output());
   thrust::transform(exec_policy,iter_pos_in,iter_pos_in+num_valid_nodes,d_p_qtfpos,what2output());
   
   RMM_FREE(d_p_qtnpos,stream);d_p_qtnpos=NULL;
   RMM_FREE(d_p_qtcpos,stream);d_p_qtcpos=NULL;
   RMM_FREE(d_p_qtnlen,stream);d_p_qtnlen=NULL;
   RMM_FREE(d_p_qtclen,stream);d_p_qtclen=NULL;

if(0)
{

    thrust::device_ptr<uint32_t> d_key_ptr=thrust::device_pointer_cast(d_p_qtkey);
    thrust::device_ptr<uint8_t> d_lev_ptr=thrust::device_pointer_cast(d_p_qtlev);   
    thrust::device_ptr<bool> d_sign_ptr=thrust::device_pointer_cast(d_p_qtsign);   
    thrust::device_ptr<uint32_t> d_len_ptr=thrust::device_pointer_cast(d_p_qtlength);	
    thrust::device_ptr<uint32_t> d_fpos_ptr=thrust::device_pointer_cast(d_p_qtfpos);   
 
    printf("key\n");
    thrust::copy(d_key_ptr,d_key_ptr+num_valid_nodes,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
    
    printf("lev\n");
    //change from uint8_t to uint32_t in ostream_iterator to output numbers instead of special chars
    thrust::copy(d_lev_ptr,d_lev_ptr+num_valid_nodes,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
   
    printf("sign\n");
    thrust::copy(d_sign_ptr,d_sign_ptr+num_valid_nodes,std::ostream_iterator<bool>(std::cout, " "));std::cout<<std::endl;
    
    printf("length\n");
    thrust::copy(d_len_ptr,d_len_ptr+num_valid_nodes,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
    
    printf("fpos\n");
    thrust::copy(d_fpos_ptr,d_fpos_ptr+num_valid_nodes,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
}   

   std::vector<std::unique_ptr<cudf::column>> quad_cols;
   quad_cols.push_back(std::move(key_col));
   quad_cols.push_back(std::move(lev_col));
   quad_cols.push_back(std::move(sign_col));
   quad_cols.push_back(std::move(length_col));
   quad_cols.push_back(std::move(fpos_col));
   return quad_cols;
}

struct quadtree_point_processor {
  
  template<typename T, std::enable_if_t<std::is_floating_point<T>::value >* = nullptr>
  std::unique_ptr<cudf::experimental::table> operator()(
  					  cudf::mutable_column_view& id,
                                          cudf::mutable_column_view& x,
 					  cudf::mutable_column_view& y,
 					  quad_point_parameters qpi,
                                          rmm::mr::device_memory_resource* mr,
                                          cudaStream_t stream)
                                          {
    uint32_t *d_p_id=cudf::mutable_column_device_view::create(id, stream)->data<uint32_t>();
    T *d_p_x=cudf::mutable_column_device_view::create(x, stream)->data<T>();
    T *d_p_y=cudf::mutable_column_device_view::create(y, stream)->data<T>();
    double x1=thrust::get<0>(qpi);
    double y1=thrust::get<1>(qpi);
    double x2=thrust::get<2>(qpi);
    double y2=thrust::get<3>(qpi);
    SBBox<double> bbox(thrust::make_tuple(x1,y1),thrust::make_tuple(x2,y2));
    double scale=thrust::get<4>(qpi);
    uint32_t num_level=thrust::get<5>(qpi);
    uint32_t min_size=thrust::get<6>(qpi);
   
   
    std::vector<std::unique_ptr<cudf::column>> quad_cols=
    	dowork<T>(x.size(),d_p_id,d_p_x,d_p_y,bbox,scale, num_level,min_size,mr,stream); 
  
    std::unique_ptr<cudf::experimental::table> destination_table = 
    	std::make_unique<cudf::experimental::table>(std::move(quad_cols));      
    return destination_table;
    }
  
  
  template<typename T, std::enable_if_t<!std::is_floating_point<T>::value >* = nullptr>
  std::unique_ptr<cudf::experimental::table> operator()(
  					  cudf::mutable_column_view& id,
  					  cudf::mutable_column_view& x,
 					  cudf::mutable_column_view& y,
 					  quad_point_parameters qpi,
                                          rmm::mr::device_memory_resource* mr,
                                          cudaStream_t stream)
                                          {
 	CUDF_FAIL("Non-floating point operation is not supported");
   }  
      
  };
  
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

std::unique_ptr<cudf::experimental::table> quadtree_on_points(
	cudf::mutable_column_view& id,cudf::mutable_column_view& x,cudf::mutable_column_view& y,
	double x1,double y1,double x2,double y2, double scale, int num_level, int min_size)
{   
    cudaStream_t stream=0;
    rmm::mr::device_memory_resource* mr=rmm::mr::get_default_resource();
    
    quad_point_parameters qpi=thrust::make_tuple(x1,y1,x2,y2,scale,num_level,min_size);
    return cudf::experimental::type_dispatcher(x.type(),quadtree_point_processor{}, 
    	id,x,y, qpi, mr,stream);       	
}

}// namespace cuspatial
