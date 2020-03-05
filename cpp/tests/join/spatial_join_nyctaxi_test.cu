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
#include <random>
#include <algorithm>
#include <functional>

#include <thrust/sort.h>
#include <thrust/binary_search.h>

#include <gtest/gtest.h>
#include <utilities/legacy/error_utils.hpp>
#include <tests/utilities/legacy/cudf_test_utils.cuh>
#include <tests/utilities/legacy/cudf_test_fixtures.h>

#include <ogrsf_frmts.h>

#include <cudf/column/column_view.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/table/table.hpp>

#include <utility/helper_thrust.cuh>
#include <utility/quadtree_thrust.cuh>
#include <utility/bbox_thrust.cuh>

#include <cuspatial/quadtree.hpp>
#include <cuspatial/bounding_box.hpp>
#include <cuspatial/spatial_jion.hpp>

//==========================================================================================================
//to be incorparated into io module

struct rec_cnyc
{
    int xf,yf,xt,yt;
};

  void VertexFromLinearRing(OGRLinearRing const& poRing, std::vector<double> &aPointX, 
        std::vector<double> &aPointY,std::vector<int> &aPartSize )
    {
        int nCount = poRing.getNumPoints();
        int nNewCount = aPointX.size() + nCount;
 
        aPointX.reserve( nNewCount );
        aPointY.reserve( nNewCount );
        for (int i = nCount - 1; i >= 0; i-- )
        {
            aPointX.push_back( poRing.getX(i));
            aPointY.push_back( poRing.getY(i));
        }
        aPartSize.push_back( nCount );
    }
 
     /*
      * Read a Polygon (could be with multiple rings) into x/y/size vectors
     */

    void LinearRingFromPolygon(OGRPolygon const & poPolygon, std::vector<double> &aPointX, 
        std::vector<double> &aPointY,std::vector<int> &aPartSize )
    {
        
        VertexFromLinearRing( *(poPolygon.getExteriorRing()),
                                        aPointX, aPointY, aPartSize );

        for(int i = 0; i < poPolygon.getNumInteriorRings(); i++ )
            VertexFromLinearRing( *(poPolygon.getInteriorRing(i)),
                                            aPointX, aPointY, aPartSize );
    }

     /*
      * Read a Geometry (could be MultiPolygon/GeometryCollection) into x/y/size vectors
     */

    void PolygonFromGeometry(OGRGeometry const *poShape, std::vector<double> &aPointX, 
        std::vector<double> &aPointY,std::vector<int> &aPartSize )
    {
        OGRwkbGeometryType eFlatType = wkbFlatten(poShape->getGeometryType());

        if (eFlatType == wkbMultiPolygon || eFlatType == wkbGeometryCollection )             
        {      
            printf("warning: wkbMultiPolygon or wkbGeometryCollection..................\n");
            OGRGeometryCollection *poGC = (OGRGeometryCollection *) poShape;
            for(int i = 0; i < poGC->getNumGeometries(); i++ )
            {
                OGRGeometry *poGeom=poGC->getGeometryRef(i);
                PolygonFromGeometry(poGeom,aPointX, aPointY, aPartSize );
            }
        }
        else if (eFlatType == wkbPolygon)
            LinearRingFromPolygon(*((OGRPolygon *) poShape),aPointX, aPointY, aPartSize );
        else
        {
           printf("error: must be polygonal geometry.\n" );
           exit(-1);
        }
    }

    int ReadLayer(const OGRLayerH layer,std::vector<int>& g_len_v,std::vector<int>&f_len_v,
         std::vector<int>& r_len_v,std::vector<double>& x_v, std::vector<double>& y_v,
         	uint8_t type, std::vector<OGRGeometry *>& polygon_vec, std::vector<uint32_t>& idx_vec)         
    {
        uint32_t num_feature=0,num_seq=0;
        OGR_L_ResetReading(layer );
        OGRFeatureH hFeat;
        
        while( (hFeat = OGR_L_GetNextFeature( layer )) != NULL )
        {
            OGRGeometry *poShape=(OGRGeometry *)OGR_F_GetGeometryRef( hFeat );
            
            if(poShape==NULL)
            {
            	printf("Invalid Shape\n");
            	exit(-1);
            }
 	    if(type!=0&&type!=1&&type!=2)
	    {
	    	printf("unknown type to setup polygons (0 for all, 1 for simple polygon and 2 for multi-polygon): type=%d\n",type);
	    	exit(-1);
	    }
           
            if(type==1 ||type==2)
            {
		if(type==1)
		{
		    if(poShape->getGeometryType()!=wkbPolygon) //wkbGeometryType=3 
		    {
		       //printf("%d exp_type=%d geom_type==%d\n",num_feature,type,poShape->getGeometryType());
		       OGR_F_Destroy( hFeat );
		       num_seq++;
		       continue;
		    }
		}
		if(type==2)
		{
		    if(poShape->getGeometryType()!=wkbMultiPolygon)//wkbGeometryType =6
		    {
		       //printf("%d exp_type=%d geom_type==%d\n",num_feature,type,poShape->getGeometryType());
		       OGR_F_Destroy( hFeat );
		       num_seq++;
		       continue;
		    }
		}
	    }
            OGRGeometry *newShape;
            if(poShape->getGeometryType()==wkbPolygon)
           	newShape=new OGRPolygon(*((OGRPolygon *) poShape));
            else if(poShape->getGeometryType()==wkbMultiPolygon)
            	newShape=new OGRMultiPolygon(*((OGRMultiPolygon *) poShape));
            else
            {
            	printf("unsuported geometry type, exiting.........\n");
            	exit(-1);
            }
            polygon_vec.push_back(newShape);
            idx_vec.push_back(num_seq++);  
            std::vector<double> aPointX;
            std::vector<double> aPointY;
            std::vector<int> aPartSize;
            PolygonFromGeometry( poShape, aPointX, aPointY, aPartSize );

            x_v.insert(x_v.end(),aPointX.begin(),aPointX.end());
            y_v.insert(y_v.end(),aPointY.begin(),aPointY.end());
            r_len_v.insert(r_len_v.end(),aPartSize.begin(),aPartSize.end());
            f_len_v.push_back(aPartSize.size());
            OGR_F_Destroy( hFeat );
            num_feature++;
        }
        g_len_v.push_back(num_feature);
        return num_feature;
    }

size_t read_point_binary(const char *fn,double*& h_pnt_x,double*& h_pnt_y)
{
    FILE *fp=NULL;
    if((fp=fopen(fn,"rb"))==NULL)
    {
        printf("can not open %s\n",fn);
        exit(-1);
    }
    fseek (fp , 0 , SEEK_END);
    size_t sz=ftell (fp);
    assert(sz%sizeof(rec_cnyc)==0);
    size_t num_rec = sz/sizeof(rec_cnyc);
    printf("num_rec=%zd\n",num_rec);    
    fseek (fp , 0 , SEEK_SET);
    
    h_pnt_x=new double[num_rec];
    h_pnt_y=new double[num_rec];
    assert(h_pnt_x!=NULL && h_pnt_y!=NULL);
    struct rec_cnyc *temp=new rec_cnyc[num_rec];
 
    size_t t=fread(temp,sizeof(rec_cnyc),num_rec,fp);
    if(t!=num_rec)
    {
        printf("cny coord read error .....%10zd %10zd\n",t,num_rec);
        exit(-1);
    }
    for(uint32_t i=0;i<num_rec;i++)
    {
    	h_pnt_x[i]=temp[i].xf;
    	h_pnt_y[i]=temp[i].yf;
    } 	
    fclose(fp);
    delete[] temp;
    return num_rec;
    //timeval t0,t1;
    //gettimeofday(&t0, NULL);
}

/*void write_shapefile(const char * file_name,
	const thrust::host_vector<double>& x1,const thrust::host_vector<double>& y1,
	const thrust::host_vector<double>& x2,const thrust::host_vector<double>& y2)*/

void write_shapefile(const char * file_name,uint32_t num_poly,
	const double *x1,const double * y1,const double *x2,const double * y2)
{
   GDALAllRegister();
   const char *pszDriverName = "ESRI Shapefile";
   GDALDriver *poDriver= GetGDALDriverManager()->GetDriverByName(pszDriverName );

   if( poDriver == NULL )
   {
    printf( "%s driver not available.\n", pszDriverName );
    exit( 1 );
    }

   GDALDataset * poDS = poDriver->Create( file_name, 0, 0, 0, GDT_Unknown, NULL );
   if( poDS == NULL )
   {
    printf( "Creation of output file failed.\n" );
    exit( 1 );
   }

   OGRLayer *poLayer= poDS->CreateLayer( "bbox", NULL, wkbLineString, NULL );
   if( poLayer == NULL )
   {
     printf( "Layer creation failed.\n" );
     exit( 1 );
   }

   OGRFieldDefn oField0( "MID", OFTInteger );
   OGRFieldDefn oField1( "x1", OFTReal );
   OGRFieldDefn oField2( "y1", OFTReal );
   OGRFieldDefn oField3( "x2", OFTReal );
   OGRFieldDefn oField4( "y2", OFTReal );
  
   bool b0=(poLayer->CreateField( &oField0 ) != OGRERR_NONE);
   bool b1=(poLayer->CreateField( &oField1 ) != OGRERR_NONE);
   bool b2=(poLayer->CreateField( &oField2 ) != OGRERR_NONE);
   bool b3=(poLayer->CreateField( &oField3 ) != OGRERR_NONE);
   bool b4=(poLayer->CreateField( &oField4 ) != OGRERR_NONE);
   if(b0||b1||b2||b3||b4)
   {
       printf( "Creating Name field failed.\n" );
       exit( 1 );
   }
   uint32_t num_f=0;
   for(uint32_t i=0;i<num_poly;i++)
   {
	 OGRFeature *poFeature=OGRFeature::CreateFeature( poLayer->GetLayerDefn() );
	 assert(poFeature!=NULL);
	 poFeature->SetField( "MID",(int)i);
	 poFeature->SetField( "x1",x1[i]);
	 poFeature->SetField( "y1",y1[i]);
	 poFeature->SetField( "x2",x2[i]);
	 poFeature->SetField( "y2",y2[i]);
	 
	 OGRLineString *ls=(OGRLinearRing*)OGRGeometryFactory::createGeometry(wkbLinearRing);	 
	 ls->addPoint(x1[i],y1[i]);
	 ls->addPoint(x1[i],y2[i]);
	 ls->addPoint(x2[i],y2[i]);
	 ls->addPoint(x2[i],y1[i]);
	 ls->addPoint(x1[i],y1[i]);	 

	 OGRPolygon *polygon=(OGRPolygon*)OGRGeometryFactory::createGeometry(wkbPolygon);
	 polygon->addRing(ls);
	 poFeature->SetGeometry(polygon);

	 if( poLayer->CreateFeature( poFeature ) != OGRERR_NONE )
	 {
		 printf( "Failed to create feature in shapefile.\n" );
		 exit( 1 );
	 }
	 OGRFeature::DestroyFeature( poFeature );
	 num_f++;
      }
   GDALClose( poDS );
   printf("num_poly=%d num_f=%d\n",num_poly,num_f);
}  

float calc_time(const char *msg,timeval t0, timeval t1)
{
 	long d = t1.tv_sec*1000000+t1.tv_usec - t0.tv_sec * 1000000-t0.tv_usec;
 	float t=(float)d/1000;
 	if(msg!=NULL)
 		printf("%s ...%10.3f\n",msg,t);
 	return t;
}


//==========================================================================================================


struct SpatialJoinNYCTaxi : public GdfTest 
{        
    uint32_t num_pnt=0;
    uint32_t * d_pnt_id=NULL;
    double *d_pnt_x=NULL,*d_pnt_y=NULL;
    double *h_pnt_x=NULL,*h_pnt_y=NULL;
    std::unique_ptr<cudf::column> col_pnt_id,col_pnt_x,col_pnt_y;
    
    uint32_t num_poly=0,num_ring=0,num_vertex=0;
    uint32_t *d_poly_fpos=NULL,*d_poly_rpos=NULL;
    double *d_poly_x=NULL,*d_poly_y=NULL;
    uint32_t *h_poly_fpos=NULL,*h_poly_rpos=NULL;
    double *h_poly_x=NULL,*h_poly_y=NULL;
    
    uint32_t num_quadrants=0;
    uint32_t *h_qt_length=NULL,*h_qt_fpos=NULL;   
    
    uint32_t num_pq_pairs=0;
    uint32_t *h_pq_quad_idx=NULL,*h_pq_poly_idx=NULL;   

    uint32_t num_pp_pairs=0;
    uint32_t *d_pp_pnt_idx=NULL,*d_pp_poly_idx=NULL;
    
    std::vector<OGRGeometry *> h_polygon_vec;
    std::vector<uint32_t> org_poly_idx_vec;
    std::vector<uint32_t> h_pnt_idx_vec;
    std::vector<uint32_t> h_pnt_len_vec;
    std::vector<uint32_t> h_poly_idx_vec;
    
    uint32_t num_search_pnt=0,num_search_poly=0;
    uint32_t *h_pnt_search_idx=NULL,*h_poly_search_idx=NULL;
    
    std::unique_ptr<cudf::column> col_poly_fpos,col_poly_rpos,col_poly_x,col_poly_y;
    
    cudaStream_t stream=0;
    rmm::mr::device_memory_resource* mr=rmm::mr::get_default_resource();

    void setup_polygons(double& x1,double& y1,double& x2,double& y2,uint8_t type)
    {
        const char* env_p = std::getenv("CUSPATIAL_DATA");
        CUDF_EXPECTS(env_p!=NULL,"CUSPATIAL_DATA environmental variable must be set");
        
        //from https://s3.amazonaws.com/nyc-tlc/misc/taxi_zones.zip
        //std::string shape_filename=std::string(env_p)+std::string("taxi_zones.shp"); 
        
        //NYC Community Districts: 71 polygons
        //from https://www1.nyc.gov/assets/planning/download/zip/data-maps/open-data/nycd_11aav.zip
        //std::string shape_filename=std::string(env_p)+std::string("nycd.shp"); 
        
        //NYC Census Tract 2000 data: 2216 polygons
        //from: https://www1.nyc.gov/assets/planning/download/zip/data-maps/open-data/nyct2000_11aav.zip
  	std::string shape_filename=std::string(env_p)+std::string("nyct2000.shp");
        std::cout<<"Using shapefile "<<shape_filename<<std::endl;
	
        std::vector<int> g_len_v,f_len_v,r_len_v;
        std::vector<double> x_v, y_v;
        GDALAllRegister();
        const char *file_name=shape_filename.c_str();
        GDALDatasetH hDS = GDALOpenEx(file_name, GDAL_OF_VECTOR, NULL, NULL, NULL );
        if(hDS==NULL)
        {
	    printf("Failed to open ESRI Shapefile dataset %s\n",file_name);
	    exit(-1);
        }
        OGRLayerH hLayer = GDALDatasetGetLayer( hDS,0 );
        
        h_polygon_vec.clear();
        org_poly_idx_vec.clear();
        //type: 0 for all, 1 for simple polygon and 2 for multi-polygon
        int num_f=ReadLayer(hLayer,g_len_v,f_len_v,r_len_v,x_v,y_v,type,h_polygon_vec,org_poly_idx_vec);
        assert(num_f>0);

        //num_group=g_len_v.size();
        num_poly=f_len_v.size();
        num_ring=r_len_v.size();
        num_vertex=x_v.size();

        //uint32_t num_temp=std::accumulate(r_len_v.begin(), r_len_v.end(),0);
        //printf("num_temp=%d num_vertex=%d\n",num_temp,num_vertex);

        h_poly_fpos=new uint32_t[num_poly];
        h_poly_rpos=new uint32_t[num_ring];
        h_poly_x=new double [num_vertex];
        h_poly_y=new double [num_vertex];

        std::copy_n(f_len_v.begin(),num_poly,h_poly_fpos);
        std::copy_n(r_len_v.begin(),num_ring,h_poly_rpos);
        std::copy_n(x_v.begin(),num_vertex,h_poly_x);
        std::copy_n(y_v.begin(),num_vertex,h_poly_y);
        printf("num_poly=%d num_ring=%d num_vertex=%d\n",num_poly,num_ring,num_vertex);

        x1=*(std::min_element(x_v.begin(),x_v.end()));
        x2=*(std::max_element(x_v.begin(),x_v.end()));
        y1=*(std::min_element(y_v.begin(),y_v.end()));
        y2=*(std::max_element(y_v.begin(),y_v.end()));
        printf("read_polygon_shape: x_min=%10.5f y_min=%10.5f, x_max=%10.5f, y_max=%10.5f\n",x1,y1, x2,y2);

        col_poly_fpos = cudf::make_numeric_column( cudf::data_type{cudf::type_id::INT32}, 
    		num_poly, cudf::mask_state::UNALLOCATED, stream, mr );      
        d_poly_fpos=cudf::mutable_column_device_view::create(col_poly_fpos->mutable_view(), stream)->data<uint32_t>();
        assert(d_poly_fpos!=NULL);
        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_fpos, h_poly_fpos, num_poly * sizeof(uint32_t), cudaMemcpyHostToDevice ) ); 

        col_poly_rpos = cudf::make_numeric_column( cudf::data_type{cudf::type_id::INT32}, 
    		num_ring, cudf::mask_state::UNALLOCATED, stream, mr );      
        d_poly_rpos=cudf::mutable_column_device_view::create(col_poly_rpos->mutable_view(), stream)->data<uint32_t>();
        assert(d_poly_rpos!=NULL);
        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_rpos, h_poly_rpos, num_ring * sizeof(uint32_t), cudaMemcpyHostToDevice ) ); 

        col_poly_x = cudf::make_numeric_column( cudf::data_type{cudf::type_id::FLOAT64}, 
    		num_vertex, cudf::mask_state::UNALLOCATED, stream, mr );      
        d_poly_x=cudf::mutable_column_device_view::create(col_poly_x->mutable_view(), stream)->data<double>();
        assert(d_poly_x!=NULL);
        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_x, h_poly_x, num_vertex * sizeof(double), cudaMemcpyHostToDevice ) ); 

        col_poly_y = cudf::make_numeric_column( cudf::data_type{cudf::type_id::FLOAT64}, 
    		num_vertex, cudf::mask_state::UNALLOCATED, stream, mr );      
        d_poly_y=cudf::mutable_column_device_view::create(col_poly_y->mutable_view(), stream)->data<double>();
        assert(d_poly_y!=NULL);
        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_y, h_poly_y, num_vertex * sizeof(double), cudaMemcpyHostToDevice ) );   

        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_fpos, h_poly_fpos, num_poly * sizeof(uint32_t), cudaMemcpyHostToDevice ) );    
        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_rpos, h_poly_rpos, num_ring * sizeof(uint32_t), cudaMemcpyHostToDevice ) );     
        thrust::inclusive_scan(thrust::device,d_poly_fpos,d_poly_fpos+num_poly,d_poly_fpos);
        thrust::inclusive_scan(thrust::device,d_poly_rpos,d_poly_rpos+num_ring,d_poly_rpos);
        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_x, h_poly_x, num_vertex * sizeof(double), cudaMemcpyHostToDevice ) );    
        HANDLE_CUDA_ERROR( cudaMemcpy( d_poly_y, h_poly_y, num_vertex * sizeof(double), cudaMemcpyHostToDevice ) );     	
    }
  
    void setup_points(double& x1,double& y1,double& x2,double& y2, uint32_t first_n)
    {
        const char* env_p = std::getenv("CUSPATIAL_DATA");
        CUDF_EXPECTS(env_p!=NULL,"CUSPATIAL_DATA environmental variable must be set");
        
        //from https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page; 
        //pickup/drop-off locations are extracted and the lon/lat coordiates are converted to EPSG 2263 projection  
        std::string catalog_filename=std::string(env_p)+std::string("2009.cat"); 
        std::cout<<"Using catalog file "<<catalog_filename<<std::endl;
        
        std::vector<uint32_t> len_vec;
        std::vector<double *> x_vec;
        std::vector<double *> y_vec;
        uint32_t num=0;
        const char *file_name=catalog_filename.c_str();
        FILE *fp=NULL;
        if((fp=fopen(file_name,"r"))==NULL)
        {
   	    printf("Failed to open point catalog file %s\n",file_name);
	    exit(-2);      	
        }
        while(!feof(fp))
        {
             char str[500];
             int n1=fscanf(fp,"%s",str);
             printf("processing point data file %s\n",str);
             double *tmp_x=NULL,*tmp_y=NULL;
             size_t temp_len=read_point_binary(str,tmp_x,tmp_y);
             assert(tmp_x!=NULL && tmp_y!=NULL);
             num++;
             len_vec.push_back(temp_len);
             x_vec.push_back(tmp_x);
             y_vec.push_back(tmp_y);
             if(first_n>0 && num>=first_n) break;
        }    
        fclose(fp);

        for(uint32_t i=0;i<num;i++)
    	    num_pnt+=len_vec[i];
    
        uint32_t p=0;
        h_pnt_x=new double[num_pnt];
        h_pnt_y=new double[num_pnt];
        assert(h_pnt_x!=NULL && h_pnt_y!=NULL);
        for(uint32_t i=0;i<num;i++)
        {
            double *tmp_x=x_vec[i];
    	    double *tmp_y=y_vec[i];
    	    assert(tmp_x!=NULL && tmp_y!=NULL);
    	    int len=len_vec[i];
    	    std::copy(tmp_x,tmp_x+len,h_pnt_x+p);
    	    std::copy(tmp_y,tmp_y+len,h_pnt_y+p);
    	    p+=len;
    	    delete[] tmp_x;
    	    delete[] tmp_y;
        }
        assert(p==num_pnt);

        x1=*(std::min_element(h_pnt_x,h_pnt_x+num_pnt));
        x2=*(std::max_element(h_pnt_x,h_pnt_x+num_pnt));
        y1=*(std::min_element(h_pnt_y,h_pnt_y+num_pnt));
        y2=*(std::max_element(h_pnt_y,h_pnt_y+num_pnt));
        printf("read_point_catalog: x_min=%10.5f y_min=%10.5f, x_max=%10.5f, y_max=%10.5f\n",x1,y1, x2,y2);
        
        col_pnt_id = cudf::make_numeric_column( cudf::data_type{cudf::type_id::INT32}, 
         	num_pnt, cudf::mask_state::UNALLOCATED, stream, mr );      
        d_pnt_id=cudf::mutable_column_device_view::create(col_pnt_id->mutable_view(), stream)->data<uint32_t>();
        assert(d_pnt_id!=NULL);
        thrust::sequence(thrust::device,d_pnt_id,d_pnt_id+num_pnt);
      
        col_pnt_x = cudf::make_numeric_column( cudf::data_type{cudf::type_id::FLOAT64}, 
        	num_pnt, cudf::mask_state::UNALLOCATED, stream, mr );      
        d_pnt_x=cudf::mutable_column_device_view::create(col_pnt_x->mutable_view(), stream)->data<double>();
        assert(d_pnt_x!=NULL);
        HANDLE_CUDA_ERROR( cudaMemcpy( d_pnt_x, h_pnt_x, num_pnt * sizeof(double), cudaMemcpyHostToDevice ) );    
    
        col_pnt_y = cudf::make_numeric_column( cudf::data_type{cudf::type_id::FLOAT64}, 
        	num_pnt, cudf::mask_state::UNALLOCATED, stream, mr );      
        d_pnt_y=cudf::mutable_column_device_view::create(col_pnt_y->mutable_view(), stream)->data<double>();
        assert(d_pnt_y!=NULL);    
        HANDLE_CUDA_ERROR( cudaMemcpy( d_pnt_y, h_pnt_y, num_pnt * sizeof(double), cudaMemcpyHostToDevice ) );    
    } 
 
    void run_test(double x1,double y1,double x2,double y2,double scale,uint32_t num_level,uint32_t min_size)
    {       
        timeval t0,t1,t2,t3;
 
        gettimeofday(&t0, NULL); 
        cudf::mutable_column_view pnt_id_view=col_pnt_id->mutable_view();
        cudf::mutable_column_view pnt_x_view=col_pnt_x->mutable_view();
        cudf::mutable_column_view pnt_y_view=col_pnt_y->mutable_view();
        std::cout<<"run_test::num_pnt_view="<<pnt_id_view.size()<<std::endl;
        std::cout<<"run_test::num_pnt="<<col_pnt_id->size()<<std::endl;

       gettimeofday(&t2, NULL);
       std::unique_ptr<cudf::experimental::table> quadtree= 
      		cuspatial::quadtree_on_points(pnt_id_view,pnt_x_view,pnt_y_view,x1,y1,x2,y2, scale,num_level, min_size);
        num_quadrants=quadtree->view().num_rows();
        std::cout<<"# of quadrants="<<num_quadrants<<std::endl;
        gettimeofday(&t3, NULL);
        float quadtree_time=calc_time("quadtree constrution time=",t2,t3);
                
//alternatively derive polygon bboxes from GDAL and then create bbox table for subsequent steps
//also output bbox coordiantes as a CSV file for examination/comparison

if(0)
{
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
                
        double *h_x1=new double[num_poly];
        double *h_y1=new double[num_poly];
        double *h_x2=new double[num_poly];
        double *h_y2=new double[num_poly];
        assert(h_x1!=NULL && h_y1!=NULL && h_x2!=NULL && h_y2!=NULL);
        
        for(uint32_t i=0;i<num_poly;i++)
        {
        	OGREnvelope env;
        	h_polygon_vec[i]->getEnvelope(&env);
         	h_x1[i]=env.MinX;
        	h_y1[i]=env.MinY;
        	h_x2[i]=env.MaxX;
        	h_y2[i]=env.MaxY;
              	printf("%d %10.2f %10.2f %10.2f %10.2f\n",i,h_x1[i],h_y1[i],h_x2[i],h_y2[i]);
	}

	HANDLE_CUDA_ERROR( cudaMemcpy( (void *)d_x1, (void *)h_x1, num_poly * sizeof(double), cudaMemcpyHostToDevice ) );       
	HANDLE_CUDA_ERROR( cudaMemcpy( (void *)d_y1, (void *)h_y1, num_poly * sizeof(double), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( (void *)d_x2, (void *)h_x2, num_poly * sizeof(double), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( (void *)d_y2, (void *)h_y2, num_poly * sizeof(double), cudaMemcpyHostToDevice ) );
	
        FILE *fp=NULL;
        if((fp=fopen("poly_mbr.csv","w"))==NULL)
        {
             printf("can not open pp_pair.csv for output\n");
             exit(-1);
        }
        for(uint32_t i=0;i<num_poly;i++)
    	fprintf(fp,"%10d, %15.5f, %15.5f, %15.5f, %15.5f\n",i,h_x1[i],h_y1[i],h_x2[i],h_y2[i]);
        fclose(fp);
        write_shapefile("poly_mbr.shp",num_poly,h_x1,h_y1,h_x2,h_y2);


        std::vector<std::unique_ptr<cudf::column>> bbox_cols;
        bbox_cols.push_back(std::move(x1_col));
        bbox_cols.push_back(std::move(y1_col));
        bbox_cols.push_back(std::move(x2_col));
        bbox_cols.push_back(std::move(y2_col));
        std::unique_ptr<cudf::experimental::table> bbox_tbl = 
            std::make_unique<cudf::experimental::table>(std::move(bbox_cols));
 }        
        gettimeofday(&t2, NULL);
        std::unique_ptr<cudf::experimental::table> bbox_tbl=
            cuspatial::polygon_bbox(col_poly_fpos->view(),col_poly_rpos->view(),col_poly_x->view(),col_poly_y->view()); 
        gettimeofday(&t3, NULL);
	float polybbox_time=calc_time("compute polygon bbox time=",t2,t3);
        std::cout<<"# of polygon bboxes="<<bbox_tbl->view().num_rows()<<std::endl;
if(0)
{
	//output bbox coordiantes as a CSV file for examination/comparison

        const double *d_x1=bbox_tbl->view().column(0).data<double>();
        const double *d_y1=bbox_tbl->view().column(1).data<double>();
        const double *d_x2=bbox_tbl->view().column(2).data<double>();  
        const double *d_y2=bbox_tbl->view().column(3).data<double>();
        thrust::device_ptr<const double> x1_ptr=thrust::device_pointer_cast(d_x1);
        thrust::device_ptr<const double> y1_ptr=thrust::device_pointer_cast(d_y1);
        thrust::device_ptr<const double> x2_ptr=thrust::device_pointer_cast(d_x2);
        thrust::device_ptr<const double> y2_ptr=thrust::device_pointer_cast(d_y2);
      
        thrust::host_vector<double> h_x1(x1_ptr,x1_ptr+num_poly);
        thrust::host_vector<double> h_y1(y1_ptr,y1_ptr+num_poly);
        thrust::host_vector<double> h_x2(x2_ptr,x2_ptr+num_poly);
        thrust::host_vector<double> h_y2(y2_ptr,y2_ptr+num_poly);
    
        FILE *fp=NULL;
        if((fp=fopen("poly_mbr_old.csv","w"))==NULL)
        {
            printf("can not open pp_pair_old.csv for output\n");
  	    exit(-1);
        }
        for(uint32_t i=0;i<num_poly;i++)
            fprintf(fp,"%10d, %15.5f, %15.5f, %15.5f, %15.5f\n",i,h_x1[i],h_y1[i],h_x2[i],h_y2[i]);
        fclose(fp);    
 }  
        const cudf::table_view quad_view=quadtree->view();
        const cudf::table_view bbox_view=bbox_tbl->view();
  
        gettimeofday(&t2, NULL);
        std::unique_ptr<cudf::experimental::table> pq_pair_tbl=cuspatial::quad_bbox_join(
            quad_view,bbox_view,x1,y1,x2,y2, scale,num_level, min_size);   
 	gettimeofday(&t3, NULL);
 	float filtering_time=calc_time("spatial filtering time=",t2,t3);         
        std::cout<<"# of polygon/quad pairs="<<pq_pair_tbl->view().num_rows()<<std::endl;
 
        const cudf::table_view pq_pair_view=pq_pair_tbl->view();
        const cudf::table_view pnt_view({pnt_id_view,pnt_x_view,pnt_y_view});

        gettimeofday(&t2, NULL); 
        std::unique_ptr<cudf::experimental::table> pip_pair_tbl=cuspatial::pip_refine(
       	  pq_pair_view,quad_view,pnt_view,
         col_poly_fpos->view(),col_poly_rpos->view(),col_poly_x->view(),col_poly_y->view());   
  	gettimeofday(&t3, NULL);
 	float refinement_time=calc_time("spatial refinement time=",t2,t3);                
        std::cout<<"# of polygon/point pairs="<<pip_pair_tbl->view().num_rows()<<std::endl;      
        
        gettimeofday(&t1, NULL);
        float gpu_time=calc_time("gpu end-to-end computing time",t0,t1);
        float  runtimes[4]={quadtree_time,polybbox_time,filtering_time,refinement_time};
        float temp_time=0;
        for(uint32_t i=0;i<4;i++)
        {
        	printf("%10.3f ",runtimes[i]);
        	temp_time+=runtimes[i];
        }
        printf("\n%10.3f %10.3f\n",temp_time,gpu_time);
           
        const uint32_t *d_qt_length=quadtree->view().column(3).data<uint32_t>();
	const uint32_t *d_qt_fpos=quadtree->view().column(4).data<uint32_t>();
 	
  	h_qt_length=new uint32_t[num_quadrants];
        h_qt_fpos=new uint32_t[num_quadrants];
        assert(h_qt_length!=NULL && h_qt_fpos!=NULL);
        printf("num_quadrants=%d %p %p %p %p\n",num_quadrants,(void *)h_qt_length,(void *)d_qt_length,(void *)h_qt_fpos,(void *)d_qt_fpos);
          
  	HANDLE_CUDA_ERROR( cudaMemcpy( h_qt_length, d_qt_length, num_quadrants * sizeof(uint32_t), cudaMemcpyDeviceToHost) ); 
  	HANDLE_CUDA_ERROR( cudaMemcpy( h_qt_fpos, d_qt_fpos, num_quadrants * sizeof(uint32_t), cudaMemcpyDeviceToHost) );      
  
        num_pq_pairs=pq_pair_tbl->num_rows();
        const uint32_t * d_pq_poly_idx=pq_pair_tbl->view().column(0).data<uint32_t>();
        const uint32_t * d_pq_quad_idx=pq_pair_tbl->view().column(1).data<uint32_t>(); 
 
        h_pq_poly_idx=new uint32_t[num_pq_pairs];
        h_pq_quad_idx=new uint32_t[num_pq_pairs];
        assert(h_pq_poly_idx!=NULL && h_pq_quad_idx!=NULL);
         
        HANDLE_CUDA_ERROR( cudaMemcpy( h_pq_poly_idx, d_pq_poly_idx, num_pq_pairs * sizeof(uint32_t), cudaMemcpyDeviceToHost) ); 
  	HANDLE_CUDA_ERROR( cudaMemcpy( h_pq_quad_idx, d_pq_quad_idx, num_pq_pairs * sizeof(uint32_t), cudaMemcpyDeviceToHost) );       
 
 if(0)
 {
         printf("d_pq_quad_idx\n");
         thrust::device_ptr<const uint32_t> pq_quad_idx_ptr=thrust::device_pointer_cast(d_pq_quad_idx);
         thrust::copy(pq_quad_idx_ptr,pq_quad_idx_ptr+100,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
 
         printf("d_pq_poly_idx\n");
         thrust::device_ptr<const uint32_t> pq_poly_idx_ptr=thrust::device_pointer_cast(d_pq_poly_idx);
         thrust::copy(pq_poly_idx_ptr,pq_poly_idx_ptr+100,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;
 } 
	
        num_pp_pairs=pip_pair_tbl->num_rows();
   	//make a copy for d_pp_pnt_idx and d_pp_poly_idx as they will be sorted later
   	const uint32_t * d_temp_poly_idx=pip_pair_tbl->view().column(0).data<uint32_t>();
        const uint32_t * d_temp_pnt_idx=pip_pair_tbl->view().column(1).data<uint32_t>();
        RMM_TRY( RMM_ALLOC( (void**)&(d_pp_pnt_idx),num_pp_pairs*sizeof(uint32_t), 0));
        RMM_TRY( RMM_ALLOC( (void**)&(d_pp_poly_idx),num_pp_pairs*sizeof(uint32_t), 0));
        HANDLE_CUDA_ERROR( cudaMemcpy( d_pp_pnt_idx, d_temp_pnt_idx, num_pp_pairs * sizeof(uint32_t), cudaMemcpyDeviceToDevice) ); 
        HANDLE_CUDA_ERROR( cudaMemcpy( d_pp_poly_idx, d_temp_poly_idx, num_pp_pairs * sizeof(uint32_t), cudaMemcpyDeviceToDevice) );    

}

uint32_t compute_mismatch()
{
    printf("num_search_pnt=%d num_search_poly=%d\n",num_search_pnt,num_search_poly);

if(0)
{
 	printf("h_pnt_search_idx:\n");
 	thrust::copy(h_pnt_search_idx,h_pnt_search_idx+num_search_pnt,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl; 
 	printf("h_poly_search_idx:\n");
 	thrust::copy(h_poly_search_idx,h_poly_search_idx+num_search_pnt,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl; 
}	
  
  	thrust::sort_by_key(thrust::device, d_pp_pnt_idx,d_pp_pnt_idx+num_pp_pairs,d_pp_poly_idx);
        
        uint32_t *h_pp_pnt_idx=NULL,*h_pp_poly_idx=NULL;
        h_pp_poly_idx=new uint32_t[num_pp_pairs];
        h_pp_pnt_idx=new uint32_t[num_pp_pairs];
        assert(h_pp_poly_idx!=NULL && h_pp_pnt_idx!=NULL);
 	HANDLE_CUDA_ERROR( cudaMemcpy( h_pp_poly_idx, d_pp_poly_idx, num_pp_pairs * sizeof(uint32_t), cudaMemcpyDeviceToHost) ); 
 	HANDLE_CUDA_ERROR( cudaMemcpy( h_pp_pnt_idx, d_pp_pnt_idx, num_pp_pairs * sizeof(uint32_t), cudaMemcpyDeviceToHost) );   
 	
        uint32_t *d_pnt_lb=NULL,*d_pnt_ub=NULL;
        bool *d_pnt_sign=NULL;
        RMM_TRY( RMM_ALLOC( (void**)&(d_pnt_lb),num_pp_pairs*sizeof(uint32_t), 0));
        RMM_TRY( RMM_ALLOC( (void**)&(d_pnt_ub),num_pp_pairs*sizeof(uint32_t), 0));
        RMM_TRY( RMM_ALLOC( (void**)&(d_pnt_sign),num_pp_pairs*sizeof(bool), 0));
 	uint32_t *d_pnt_search_idx=NULL;
        RMM_TRY( RMM_ALLOC( (void**)&(d_pnt_search_idx),num_search_pnt*sizeof(uint32_t), 0));
   	HANDLE_CUDA_ERROR( cudaMemcpy( d_pnt_search_idx, h_pnt_search_idx, num_search_pnt * sizeof(uint32_t), cudaMemcpyHostToDevice) ); 
        printf("after H->D transfer..................\n");
       
        thrust::lower_bound(thrust::device,d_pp_pnt_idx,d_pp_pnt_idx+num_pp_pairs,d_pnt_search_idx,d_pnt_search_idx+num_search_pnt,d_pnt_lb);
        thrust::upper_bound(thrust::device,d_pp_pnt_idx,d_pp_pnt_idx+num_pp_pairs,d_pnt_search_idx,d_pnt_search_idx+num_search_pnt,d_pnt_ub);
        thrust::binary_search(thrust::device,d_pp_pnt_idx,d_pp_pnt_idx+num_pp_pairs,d_pnt_search_idx,d_pnt_search_idx+num_search_pnt,d_pnt_sign);
        printf("after GPU search...................\n");
        
   	uint32_t * h_pnt_lb=new uint32_t[num_search_pnt];
  	uint32_t * h_pnt_ub=new uint32_t[num_search_pnt];
  	bool *h_pnt_sign=new bool[num_search_pnt];
  	assert(h_pnt_lb!=NULL && h_pnt_ub!=NULL && h_pnt_sign!=NULL);
  	
  	HANDLE_CUDA_ERROR( cudaMemcpy( h_pnt_lb, d_pnt_lb, num_search_pnt * sizeof(uint32_t), cudaMemcpyDeviceToHost) ); 
  	HANDLE_CUDA_ERROR( cudaMemcpy( h_pnt_ub, d_pnt_ub, num_search_pnt * sizeof(uint32_t), cudaMemcpyDeviceToHost) );   
  	HANDLE_CUDA_ERROR( cudaMemcpy( h_pnt_sign, d_pnt_sign, num_search_pnt * sizeof(bool), cudaMemcpyDeviceToHost) );

        RMM_TRY(RMM_FREE(d_pnt_lb,0));d_pnt_lb=NULL;
        RMM_TRY(RMM_FREE(d_pnt_ub,0));d_pnt_ub=NULL;
        RMM_TRY(RMM_FREE(d_pnt_sign,0));d_pnt_sign=NULL;
        printf("after H->D transfer..................\n");
 
    	FILE *fp=NULL;
      	if((fp=fopen("debug.csv","w"))==NULL)
      	{
      		printf("can not open debug.txt for output");
      		exit(-1);
     	}
 	uint32_t bpos=0,epos=h_pnt_len_vec[0], num_mis_match=0,num_not_found=0;
 	for(uint32_t i=0;i<num_search_pnt;i++)
 	{
 		//printf("i=%d idx=%d sign=%d lb=%d ub=%d\n",i,h_pnt_search_idx[i],h_pnt_sign[i],h_pnt_lb[i],h_pnt_ub[i]);
 		if(!h_pnt_sign[i])
 		{
 			printf("i=%d pntid=%d does not hit\n",i,h_pnt_search_idx[i]);
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
if(1)
{
				printf("i=%d key=%d g_size=%lu c_size=%lu lb=%d ub=%d pointid=%d\n",
					i,h_pnt_search_idx[i],gpu_set.size(),cpu_set.size(),h_pnt_lb[i],h_pnt_ub[i],pntid);
				printf("gpu_set\n");
				thrust::copy(gpu_set.begin(),gpu_set.end(),std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl; 
				printf("cpu_set\n");
}				thrust::copy(cpu_set.begin(),cpu_set.end(),std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl; 
				
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
 			bpos=epos;epos+=h_pnt_len_vec[i];
 		}	
 	}
 	fclose(fp);
 	delete[] h_pnt_lb;
 	delete[] h_pnt_ub;
 	delete[] h_pnt_sign;
        delete[] h_pp_pnt_idx;
	delete[] h_pp_poly_idx;	
 	
 	printf("num_search_pnt=%d num_not_found=%d num_mis_match=%d\n",num_search_pnt,num_not_found,num_mis_match);
 	return num_mis_match;
 }	
    
    void compare_full_random(uint32_t num_samples,uint32_t num_print_interval)
    {
        h_pnt_idx_vec.clear();
        h_pnt_len_vec.clear();
        h_poly_idx_vec.clear();        

	printf("compare_full_random: num_quadrants=%d num_pq_pairs=%d num_pp_pair=%d num_samples=%d \n",
		num_quadrants,num_pq_pairs,num_pp_pairs,num_samples);
        uint32_t *nums=NULL;
        if(num_samples<num_pnt)
        {       	
		std::seed_seq seed{2};
		std::mt19937 g(seed);
		std::uniform_int_distribution<> dist_rand (0,num_pnt-1);
		nums=new uint32_t[num_samples];
		assert(nums!=NULL);
		std::generate(nums, nums+num_samples, [&] () mutable { return dist_rand(g); });	
        }
        else if(num_samples==num_pnt)
        {
        	nums=new uint32_t[num_samples];
        	std::generate(nums, nums+num_pnt, [n = 0] () mutable { return n++; });
        }
        else
             printf("num_samples=%d must be less or equal to num_pnt=%d\n",num_samples,num_pnt);
        assert(nums!=NULL);

	timeval t0,t1;
	gettimeofday(&t0, NULL);
        
        char  msg[100];
	timeval t2,t3;
	gettimeofday(&t2, NULL);
	for(uint32_t k=0;k<num_samples;k++)
	{
	    uint32_t pntid=nums[k];	
	    OGRPoint pnt(h_pnt_x[pntid],h_pnt_y[pntid]);
	    std::vector<uint32_t> temp_vec;
	    for(uint32_t j=0;j<h_polygon_vec.size();j++)
	    {
		if(h_polygon_vec[j]->Contains(&pnt))
		  temp_vec.push_back(j);
	    }
	    if(temp_vec.size()>0)
	    {
	    	h_pnt_len_vec.push_back(temp_vec.size());
	    	h_pnt_idx_vec.push_back(pntid);
	    	h_poly_idx_vec.insert(h_poly_idx_vec.end(),temp_vec.begin(),temp_vec.end());
	    }
            if(k>0 && k%num_print_interval==0)
            {
	    	    gettimeofday(&t3, NULL);    
 	            sprintf(msg,"loop=%d runtime for the last %d iterations is\n",k,num_print_interval);
 	            float cpu_time_per_interval=calc_time(msg,t2,t3);
 	            t2=t3;
 	    }
	}
	delete[] nums;
        num_search_pnt=h_pnt_idx_vec.size();
        num_search_poly=h_poly_idx_vec.size();
	printf("h_pnt_idx_vec.size()=%d\n",num_search_pnt);
	printf("h_poly_idx_vec.size()=%d\n",num_search_poly);
	printf("num_pp_pairs=%d\n",num_pp_pairs);	
 
        gettimeofday(&t1, NULL);
        float cpu_time=calc_time("cpu all-pair computing time",t0,t1);
        
        //global vectors, use their data pointers
        h_pnt_search_idx=&h_pnt_idx_vec[0];
        h_poly_search_idx=&h_poly_idx_vec[0];
        assert(h_pnt_search_idx!=NULL && h_poly_search_idx!=NULL); 
        printf("h_pnt_search_idx[0]=%d h_poly_search_idx[0]=%d\n",h_pnt_search_idx[0],h_poly_search_idx[0]);
     }
  
    void compare_matched_pairs(uint32_t num_samples,uint32_t num_print_interval)
    {
        h_pnt_idx_vec.clear();
        h_pnt_len_vec.clear();
        h_poly_idx_vec.clear();        

	printf("compare_matched_pairs: num_quadrants=%d num_pq_pairs=%d num_pp_pair=%d num_samples=%d \n",
		num_quadrants,num_pq_pairs,num_pp_pairs,num_samples);


        uint32_t *nums=NULL;
        //random quadrants
        if(num_samples<num_pq_pairs)
        {       	
		std::seed_seq seed{0};
		std::mt19937 g(seed);
		std::uniform_int_distribution<> dist_rand (0,num_pq_pairs-1);
		nums=new uint32_t[num_samples];
		assert(nums!=NULL);
		std::generate(nums, nums+num_samples, [&] () mutable { return dist_rand(g); });	
        }
        else if(num_samples==num_pq_pairs)
        {
        	nums=new uint32_t[num_samples];
        	std::generate(nums, nums+num_pq_pairs, [n = 0] () mutable { return n++; });
        }
        else
             printf("compare_matched_pairs: num_samples=%d must be less or equal to num_pq_pairs=%d\n",num_samples,num_pq_pairs);
        assert(nums!=NULL);
 
 	timeval t0,t1;
	gettimeofday(&t0, NULL);
        
        char  msg[100];
	timeval t2,t3;
	uint32_t p=0;
	gettimeofday(&t2, NULL);
	for(uint32_t k=0;k<num_samples;k++)
	{
	    uint32_t qid=h_pq_quad_idx[nums[k]];
	    uint32_t qlen=h_qt_length[qid];
	    uint32_t fpos=h_qt_fpos[qid];
	    //printf("k=%d qid=%u qlen=%u fpos=%u\n",k,qid,qlen,fpos);
	    for(uint32_t i=0;i<qlen;i++)
	    {
		    assert(fpos+i<num_pnt);
		    OGRPoint pnt(h_pnt_x[fpos+i],h_pnt_y[fpos+i]);
		    std::vector<uint32_t> temp_vec;
		    for(uint32_t j=0;j<h_polygon_vec.size();j++)
		    {
			if(h_polygon_vec[j]->Contains(&pnt))
			  temp_vec.push_back(j);
		    }
		    if(temp_vec.size()>0)
		    {
			h_pnt_len_vec.push_back(temp_vec.size());
			uint32_t pntid=fpos+i;
			h_pnt_idx_vec.push_back(pntid);
			h_poly_idx_vec.insert(h_poly_idx_vec.end(),temp_vec.begin(),temp_vec.end());
		    }
		    if(p>0 && p%num_print_interval==0)
		    {
			    gettimeofday(&t3, NULL);    
			    sprintf(msg,"loop=%d quad=%d runtime for the last %d iterations is\n",p,k,num_print_interval);
			    float cpu_time_per_interval=calc_time(msg,t2,t3);
			    t2=t3;
		    }
		    p++;
		}
	}
        gettimeofday(&t1, NULL);
        float cpu_time=calc_time("cpu matched-pair computing time",t0,t1);          
        
        num_search_pnt=h_pnt_idx_vec.size();
        num_search_poly=h_poly_idx_vec.size();
	printf("h_pnt_idx_vec.size()=%d\n",num_search_pnt);
	printf("h_poly_idx_vec.size()=%d\n",num_search_poly);
	printf("num_pp_pairs=%d\n",num_pp_pairs);    
        h_pnt_search_idx=&h_pnt_idx_vec[0];
        h_poly_search_idx=&h_poly_idx_vec[0];
        assert(h_pnt_search_idx!=NULL && h_poly_search_idx!=NULL); 
        printf("h_pnt_search_idx[0]=%d h_poly_search_idx[0]=%d\n",h_pnt_search_idx[0],h_poly_search_idx[0]);
    }
    
    
    void tear_down()
    {
        delete[] h_poly_fpos;h_poly_fpos=NULL;
        delete[] h_poly_rpos;h_poly_rpos=NULL;
        delete[] h_poly_x; h_poly_x=NULL;
        delete[] h_poly_y; h_poly_y=NULL;
        
        delete[] h_pnt_x; h_pnt_x=NULL;
        delete[] h_pnt_y; h_pnt_y=NULL;         
    }
 
};

TEST_F(SpatialJoinNYCTaxi, test)
{
    const uint32_t num_level=15;
    const uint32_t min_size=512;
    const uint32_t first_n=12;
    
    std::cout<<"loading NYC taxi zone shapefile data..........."<<std::endl;  
    double poly_x1,poly_y1,poly_x2,poly_y2;
      
    //uint8_t type=2; //multi-polygons only  
    uint8_t type=0; //all polygons
    this->setup_polygons(poly_x1,poly_y1,poly_x2,poly_y2,type);
        
    std::cout<<"loading NYC taxi pickup locations..........."<<std::endl;  
    double pnt_x1,pnt_y1,pnt_x2,pnt_y2;
    this->setup_points(pnt_x1,pnt_y1,pnt_x2,pnt_y2,first_n);
 
    double width=poly_x2-poly_x1;
    double height=poly_y2-poly_y1;
    double length=(width>height)?width:height;
    double scale=length/((1<<num_level)+2);
    double bbox_x1=poly_x1-scale; 
    double bbox_y1=poly_y1-scale;	
    double bbox_x2=poly_x2+scale; 
    double bbox_y2=poly_y2+scale;
    printf("Area of Interests: length=%15.10f scale=%15.10f\n",length,scale);

    std::cout<<"running test on NYC taxi trip data..........."<<std::endl;
    
    this->run_test(bbox_x1,bbox_y1,bbox_x2,bbox_y2,scale,num_level,min_size);
 
 if(1)
 {
    std::cout<<"running GDAL CPU code for comparison..........."<<std::endl;
  
    uint32_t num_print_interval=1000;
  
    uint32_t num_pnt_samples=10000;
    this->compare_full_random(num_pnt_samples,num_print_interval);   
    
    //uint32_t num_quad_samples=100;
    //this->compare_matched_pairs(num_quad_samples,num_print_interval);
    
    this->compute_mismatch();
    
    this->tear_down();
 }
}

