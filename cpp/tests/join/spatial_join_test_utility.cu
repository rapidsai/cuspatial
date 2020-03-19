#include <thrust/sort.h>
#include <thrust/binary_search.h>

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/table/table.hpp>

#include <utility/utility.hpp>
#include <utility/helper_thrust.cuh>

#include <ogrsf_frmts.h>

#include "spatial_join_test_utility.hpp"

//
bool compute_mismatch(uint32_t num_pp_pairs,const std::vector<uint32_t>&  org_poly_idx_vec,
    const uint32_t *h_pnt_search_idx, const std::vector<uint32_t>& h_pnt_len_vec,const uint32_t * h_poly_search_idx,    
    uint32_t * d_pp_pnt_idx,uint32_t *d_pp_poly_idx,
    const double *h_pnt_x, const double * h_pnt_y,
    rmm::mr::device_memory_resource* mr, cudaStream_t stream)
{
    uint32_t num_search_pnt=h_pnt_len_vec.size();

if(0)
{
    std::cout<<"h_pnt_search_idx"<<std::endl;
    thrust::copy(h_pnt_search_idx,h_pnt_search_idx+num_search_pnt,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl; 
    std::cout<<"h_poly_search_idx"<<std::endl;
    thrust::copy(h_poly_search_idx,h_poly_search_idx+num_search_pnt,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl; 
}    
  
    thrust::sort_by_key(thrust::device, d_pp_pnt_idx,d_pp_pnt_idx+num_pp_pairs,d_pp_poly_idx);

    uint32_t *h_pp_poly_idx=NULL;
    h_pp_poly_idx=new uint32_t[num_pp_pairs];
    HANDLE_CUDA_ERROR( cudaMemcpy( h_pp_poly_idx, d_pp_poly_idx, num_pp_pairs * sizeof(uint32_t), cudaMemcpyDeviceToHost) ); 

    rmm::device_buffer *db_pnt_lb=new rmm::device_buffer(num_pp_pairs* sizeof(uint32_t),stream,mr);
    CUDF_EXPECTS(db_pnt_lb!=nullptr, "Error allocating memory for lower bounds array in serching polygon idx based on point idx");
    uint32_t *d_pnt_lb=static_cast<uint32_t *>(db_pnt_lb->data());

    rmm::device_buffer *db_pnt_ub=new rmm::device_buffer(num_pp_pairs* sizeof(uint32_t),stream,mr);
    CUDF_EXPECTS(db_pnt_ub!=nullptr, "Error allocating memory for upper bounds array in serching polygon idx based on point idx");
    uint32_t *d_pnt_ub=static_cast<uint32_t *>(db_pnt_ub->data());

    rmm::device_buffer *db_pnt_sign=new rmm::device_buffer(num_pp_pairs* sizeof(bool),stream,mr);
    CUDF_EXPECTS(db_pnt_sign!=nullptr, "Error allocating memory for sign array in serching polygon idx based on point idx");
    bool *d_pnt_sign=static_cast<bool *>(db_pnt_sign->data());

    rmm::device_buffer *db_pnt_search_idx=new rmm::device_buffer(num_search_pnt* sizeof(uint32_t),stream,mr);
    CUDF_EXPECTS(db_pnt_search_idx!=nullptr, "Error allocating memory for point idx array being used as search keys");
    uint32_t *d_pnt_search_idx=static_cast<uint32_t *>(db_pnt_search_idx->data());
    HANDLE_CUDA_ERROR( cudaMemcpy( d_pnt_search_idx, h_pnt_search_idx, num_search_pnt * sizeof(uint32_t), cudaMemcpyHostToDevice) ); 

    thrust::lower_bound(thrust::device,d_pp_pnt_idx,d_pp_pnt_idx+num_pp_pairs,d_pnt_search_idx,d_pnt_search_idx+num_search_pnt,d_pnt_lb);
    thrust::upper_bound(thrust::device,d_pp_pnt_idx,d_pp_pnt_idx+num_pp_pairs,d_pnt_search_idx,d_pnt_search_idx+num_search_pnt,d_pnt_ub);
    thrust::binary_search(thrust::device,d_pp_pnt_idx,d_pp_pnt_idx+num_pp_pairs,d_pnt_search_idx,d_pnt_search_idx+num_search_pnt,d_pnt_sign);

    uint32_t * h_pnt_lb=new uint32_t[num_search_pnt];
    uint32_t * h_pnt_ub=new uint32_t[num_search_pnt];
    bool *h_pnt_sign=new bool[num_search_pnt];
    assert(h_pnt_lb!=NULL && h_pnt_ub!=NULL && h_pnt_sign!=NULL);

    HANDLE_CUDA_ERROR( cudaMemcpy( h_pnt_lb, d_pnt_lb, num_search_pnt * sizeof(uint32_t), cudaMemcpyDeviceToHost) ); 
    HANDLE_CUDA_ERROR( cudaMemcpy( h_pnt_ub, d_pnt_ub, num_search_pnt * sizeof(uint32_t), cudaMemcpyDeviceToHost) );   
    HANDLE_CUDA_ERROR( cudaMemcpy( h_pnt_sign, d_pnt_sign, num_search_pnt * sizeof(bool), cudaMemcpyDeviceToHost) );

    delete db_pnt_lb; db_pnt_lb=NULL;
    delete db_pnt_ub; db_pnt_ub=NULL;
    delete db_pnt_sign; db_pnt_sign=NULL;
    delete db_pnt_search_idx; db_pnt_search_idx=NULL;
    std::cout<<"after H->D transfer.................."<<std::endl;

    //uncoment to write debug info to file
    /*FILE *fp=NULL;
    if((fp=fopen("debug.csv","w"))==NULL)
    {
        printf("can not open debug.txt for output");
        exit(-1);
    }*/

    FILE *fp=stdout;
    uint32_t bpos=0,epos=h_pnt_len_vec[0], num_mis_match=0,num_not_found=0;
    for(uint32_t i=0;i<num_search_pnt;i++)
    {
        //printf("i=%d idx=%d sign=%d lb=%d ub=%d\n",i,h_pnt_search_idx[i],h_pnt_sign[i],h_pnt_lb[i],h_pnt_ub[i]);
        if(!h_pnt_sign[i])
        {
        //printf("i=%d pntid=%d does not hit\n",i,h_pnt_search_idx[i]);
        uint32_t pntid=h_pnt_search_idx[i];
        uint32_t polyid=org_poly_idx_vec[h_poly_search_idx[i]];
        fprintf(fp,"%d, %10.2f, %10.2f, -1, %d\n",pntid,h_pnt_x[pntid],h_pnt_y[pntid],polyid);
        num_not_found++;
        }
        else
        {
        std::set<uint32_t> gpu_set;
        for(uint32_t j=h_pnt_lb[i];j<h_pnt_ub[i];j++)
            gpu_set.insert(org_poly_idx_vec[h_pp_poly_idx[j]]);
        std::set<uint32_t> cpu_set;
        for(uint32_t j=bpos;j<epos;j++)
            cpu_set.insert(org_poly_idx_vec[h_poly_search_idx[j]]);
        if(gpu_set!=cpu_set)
        {
            uint32_t pntid=h_pnt_search_idx[i];
if(0)
{
            printf("i=%d key=%d g_size=%lu c_size=%lu lb=%d ub=%d pointid=%d\n",
            i,h_pnt_search_idx[i],gpu_set.size(),cpu_set.size(),h_pnt_lb[i],h_pnt_ub[i],pntid);
            printf("gpu_set\n");
            thrust::copy(gpu_set.begin(),gpu_set.end(),std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl; 
            printf("cpu_set\n");
}           thrust::copy(cpu_set.begin(),cpu_set.end(),std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl; 

            fprintf(fp,"%d,%10.2f,%10.2f",pntid,h_pnt_x[pntid],h_pnt_y[pntid]);
            uint32_t gpu_len=h_pnt_ub[i]-h_pnt_lb[i];
            std::string ss="";
            if(gpu_len>0)
            {
            ss+=std::to_string(org_poly_idx_vec[h_pp_poly_idx[h_pnt_lb[i]]]);
            for(uint32_t j=1;j<gpu_len;j++)
                ss+=("|"+std::to_string(org_poly_idx_vec[h_pp_poly_idx[h_pnt_lb[i]+j]]));
            }
            else
               ss="-1";
            fprintf(fp,",%s",ss.c_str());
            ss="";
            if(h_pnt_len_vec[i]>0)
            {
            ss+=std::to_string(org_poly_idx_vec[h_poly_search_idx[bpos]]);
            for(uint32_t j=bpos+1;j<epos;j++)
                ss+=("|"+std::to_string(org_poly_idx_vec[h_poly_search_idx[j]]));
            }
            else 
               ss="-1";
            fprintf(fp,",%s\n",ss.c_str());
            num_mis_match++;
        }
        }
        if(i!=num_search_pnt-1)
        {
        bpos=epos;
        epos+=h_pnt_len_vec[i];
        }    
    }
    //fclose(fp);
    delete[] h_pnt_lb;
    delete[] h_pnt_ub;
    delete[] h_pnt_sign;
    delete[] h_pp_poly_idx;    

    std::cout<<"compute_mismatch: num_pp_pairs="<<num_pp_pairs<<std::endl;
    std::cout<<"compute_mismatch: num_search_pnt="<<num_search_pnt<<std::endl;
    std::cout<<"compute_mismatch: num_not_found="<<num_not_found<<std::endl;
    std::cout<<"compute_mismatch: num_mis_match="<<num_mis_match<<std::endl;    
    return (num_search_pnt==num_pp_pairs && num_not_found==0 && num_mis_match==0);
}    

std::unique_ptr<cudf::experimental::table> bbox_tbl_cpu(const std::vector<OGRGeometry *>& h_polygon_vec, 
    cudaStream_t stream,rmm::mr::device_memory_resource *mr)
{
    uint32_t num_poly=h_polygon_vec.size();
    std::unique_ptr<cudf::column> x1_col = cudf::make_numeric_column(
    cudf::data_type{cudf::experimental::type_to_id<double>()}, num_poly,cudf::mask_state::UNALLOCATED, stream, mr);
    double *d_x1=cudf::mutable_column_device_view::create(x1_col->mutable_view(), stream)->data<double>();
    assert(d_x1!=NULL);

    std::unique_ptr<cudf::column> y1_col = cudf::make_numeric_column(
    cudf::data_type{cudf::experimental::type_to_id<double>()}, num_poly,cudf::mask_state::UNALLOCATED, stream, mr);
    double *d_y1=cudf::mutable_column_device_view::create(y1_col->mutable_view(), stream)->data<double>();
    assert(d_y1!=NULL);

    std::unique_ptr<cudf::column> x2_col = cudf::make_numeric_column(
    cudf::data_type{cudf::experimental::type_to_id<double>()}, num_poly,cudf::mask_state::UNALLOCATED, stream, mr);
    double *d_x2=cudf::mutable_column_device_view::create(x2_col->mutable_view(), stream)->data<double>();
    assert(d_x2!=NULL);
 
    std::unique_ptr<cudf::column> y2_col = cudf::make_numeric_column(
    cudf::data_type{cudf::experimental::type_to_id<double>()}, num_poly,cudf::mask_state::UNALLOCATED, stream, mr);
    double *d_y2=cudf::mutable_column_device_view::create(y2_col->mutable_view(), stream)->data<double>();
    assert(d_y2!=NULL);

    double *h_x1=nullptr,*h_y1=nullptr,*h_x2=nullptr,*h_y2=nullptr;
    
    //polyvec_to_bbox(h_polygon_vec,"cpu_bbox.csv",h_x1,h_y1,h_x2,h_y2);
    
    polyvec_to_bbox(h_polygon_vec,nullptr,h_x1,h_y1,h_x2,h_y2);
    
    //write_shapefile("cpu_bbox.shp",num_poly,h_x1,h_y1,h_x2,h_y2);

    HANDLE_CUDA_ERROR( cudaMemcpy( (void *)d_x1, (void *)h_x1, num_poly * sizeof(double), cudaMemcpyHostToDevice ) );       
    HANDLE_CUDA_ERROR( cudaMemcpy( (void *)d_y1, (void *)h_y1, num_poly * sizeof(double), cudaMemcpyHostToDevice ) );
    HANDLE_CUDA_ERROR( cudaMemcpy( (void *)d_x2, (void *)h_x2, num_poly * sizeof(double), cudaMemcpyHostToDevice ) );
    HANDLE_CUDA_ERROR( cudaMemcpy( (void *)d_y2, (void *)h_y2, num_poly * sizeof(double), cudaMemcpyHostToDevice ) );

    std::vector<std::unique_ptr<cudf::column>> bbox_cols;
    bbox_cols.push_back(std::move(x1_col));
    bbox_cols.push_back(std::move(y1_col));
    bbox_cols.push_back(std::move(x2_col));
    bbox_cols.push_back(std::move(y2_col));
    std::unique_ptr<cudf::experimental::table> bbox_tbl = 
    std::make_unique<cudf::experimental::table>(std::move(bbox_cols));
    return bbox_tbl;
}

void bbox_table_to_csv(const std::unique_ptr<cudf::experimental::table>& bbox_tbl, const char * file_name,
     double * & h_x1,double * & h_y1,double * & h_x2,double * & h_y2)
{
    //output bbox coordiantes as a CSV file for examination/comparison

    const double *d_x1=bbox_tbl->view().column(0).data<double>();
    const double *d_y1=bbox_tbl->view().column(1).data<double>();
    const double *d_x2=bbox_tbl->view().column(2).data<double>();  
    const double *d_y2=bbox_tbl->view().column(3).data<double>();
    
    uint32_t num_poly=bbox_tbl->num_rows();

    h_x1=new double[num_poly];
    h_y1=new double[num_poly];
    h_x2=new double[num_poly];
    h_y2=new double[num_poly];
    assert(h_x1!=NULL && h_y1!=NULL && h_x2!=NULL && h_y2!=NULL);     

    HANDLE_CUDA_ERROR( cudaMemcpy( h_x1, d_x1, num_poly * sizeof(double), cudaMemcpyHostToDevice ) ); 
    HANDLE_CUDA_ERROR( cudaMemcpy( h_x2, d_x2, num_poly * sizeof(double), cudaMemcpyHostToDevice ) );
    HANDLE_CUDA_ERROR( cudaMemcpy( h_y1, d_y1, num_poly * sizeof(double), cudaMemcpyHostToDevice ) );
    HANDLE_CUDA_ERROR( cudaMemcpy( h_y2, d_y2, num_poly * sizeof(double), cudaMemcpyHostToDevice ) );

    FILE *fp=NULL;
    if((fp=fopen(file_name,"w"))==NULL)
    {
    std::cout<<"can not open "<<file_name<<" for output"<<std::endl;
     exit(-1);
    }
    
    for(uint32_t i=0;i<num_poly;i++)
       fprintf(fp,"%10d, %15.5f, %15.5f, %15.5f, %15.5f\n",i,h_x1[i],h_y1[i],h_x2[i],h_y2[i]);
   fclose(fp);
}