#include <cudf/table/table.hpp>

//internal helper function defintions, documentation TBD

int ReadLayer(const OGRLayerH layer,std::vector<int>& g_len_v,std::vector<int>&f_len_v,
        std::vector<int>& r_len_v,std::vector<double>& x_v, std::vector<double>& y_v,
        uint8_t type, std::vector<OGRGeometry *>& polygon_vec, std::vector<uint32_t>& idx_vec);

size_t read_point_binary(const char *fn,double*& h_pnt_x,double*& h_pnt_y);

void write_shapefile(const char * file_name,uint32_t num_poly,
    const double *x1,const double * y1,const double *x2,const double * y2);

void polyvec_to_bbox(const std::vector<OGRGeometry *>& h_polygon_vec,const char * file_name,
    double * & h_x1,double * & h_y1,double * & h_x2,double * & h_y2);

void bbox_table_to_csv(const std::unique_ptr<cudf::experimental::table>& bbox_tbl, const char * file_name,
     double * & h_x1,double * & h_y1,double * & h_x2,double * & h_y2);

std::unique_ptr<cudf::experimental::table> bbox_tbl_cpu(
    const std::vector<OGRGeometry *>& h_polygon_vec);

void gen_rand_idx(std::vector<uint32_t>& indices,uint32_t num_counts, uint32_t num_samples);

bool compute_mismatch(uint32_t num_pp_pairs,const std::vector<uint32_t>&  org_poly_idx_vec,
    const uint32_t *h_pnt_search_idx,const std::vector<uint32_t>& h_pnt_len_vec,const uint32_t * h_poly_search_idx,
    uint32_t * d_pp_pnt_idx,uint32_t *d_pp_poly_idx,
    const double *h_pnt_x, const double * h_pnt_y,
    rmm::mr::device_memory_resource* mr, cudaStream_t stream);


void rand_points_gdal_pip_test(uint32_t num_print_interval,const std::vector<uint32_t>& indices,
    const std::vector<OGRGeometry *>& h_polygon_vec, std::vector<uint32_t>& h_pnt_idx_vec,
    std::vector<uint32_t>& h_pnt_len_vec,std::vector<uint32_t>& h_poly_idx_vec,
    const double* h_pnt_x, const double* h_pnt_y);

void matched_pairs_gdal_pip_test(uint32_t num_print_interval,const std::vector<uint32_t>& indices,
    const uint32_t *h_pq_quad_idx,  const uint32_t *h_pq_poly_idx,
    const uint32_t *h_qt_length,  const uint32_t * h_qt_fpos,
    const std::vector<OGRGeometry *>& h_polygon_vec, std::vector<uint32_t>& h_pnt_idx_vec,
    std::vector<uint32_t>& h_pnt_len_vec,std::vector<uint32_t>& h_poly_idx_vec,
    const double* h_pnt_x, const double* h_pnt_y);

