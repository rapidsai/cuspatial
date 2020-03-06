#include <cassert>
#include <ogrsf_frmts.h>

//placeholder for structus and functions needed to run NYC taxi experiments
//they will be incorporated into the new io module of cuspatial in a later release

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

//==========================================================================================================
