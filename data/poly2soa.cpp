// g++ -I /usr/local/include -L /usr/local/lib poly2soa.cpp -lgdal -o poly2soa
// ./poly2soa its.cat itsroi.ply

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <string>
#include <cassert>
#include <iterator>
#include<sys/time.h>
#include<time.h>

#include "gdal.h"
#include "gdal_priv.h"
#include "cpl_conv.h"
#include "ogr_api.h"
#include "ogr_srs_api.h"
#include "cpl_string.h"
#include "ogrsf_frmts.h"
#include "gdal_alg.h"
#include "ogr_geometry.h"

using namespace std;

void GDALCollectRingsFromGeometry(
    OGRGeometry *poShape,
    std::vector<double> &aPointX, std::vector<double> &aPointY,
    std::vector<int> &aPartSize )

{
    if( poShape == NULL )
        return;

    OGRwkbGeometryType eFlatType = wkbFlatten(poShape->getGeometryType());
    int i;

    if ( eFlatType == wkbPoint )
    {
        OGRPoint    *poPoint = (OGRPoint *) poShape;
        int nNewCount = aPointX.size() + 1;

        aPointX.reserve( nNewCount );
        aPointY.reserve( nNewCount );
        aPointX.push_back( poPoint->getX());
        aPointY.push_back( poPoint->getY());
        aPartSize.push_back( 1 );
    }
    else if ( eFlatType == wkbLineString )
    {
        OGRLineString   *poLine = (OGRLineString *) poShape;
        int nCount = poLine->getNumPoints();
        int nNewCount = aPointX.size() + nCount;

        aPointX.reserve( nNewCount );
        aPointY.reserve( nNewCount );
        for ( i = nCount - 1; i >= 0; i-- )
        {
            aPointX.push_back( poLine->getX(i));
            aPointY.push_back( poLine->getY(i));
        }
        aPartSize.push_back( nCount );
    }
    else if ( EQUAL(poShape->getGeometryName(),"LINEARRING") )
    {
        OGRLinearRing *poRing = (OGRLinearRing *) poShape;
        int nCount = poRing->getNumPoints();
        int nNewCount = aPointX.size() + nCount;

        aPointX.reserve( nNewCount );
        aPointY.reserve( nNewCount );
        for ( i = nCount - 1; i >= 0; i-- )
        {
            aPointX.push_back( poRing->getX(i));
            aPointY.push_back( poRing->getY(i));
        }
        aPartSize.push_back( nCount );
    }
    else if( eFlatType == wkbPolygon )
    {
        OGRPolygon *poPolygon = (OGRPolygon *) poShape;

        GDALCollectRingsFromGeometry( poPolygon->getExteriorRing(),
                                      aPointX, aPointY, aPartSize );

        for( i = 0; i < poPolygon->getNumInteriorRings(); i++ )
            GDALCollectRingsFromGeometry( poPolygon->getInteriorRing(i),
                                          aPointX, aPointY, aPartSize );
    }

    else if( eFlatType == wkbMultiPoint
             || eFlatType == wkbMultiLineString
             || eFlatType == wkbMultiPolygon
             || eFlatType == wkbGeometryCollection )
    {
        OGRGeometryCollection *poGC = (OGRGeometryCollection *) poShape;

        for( i = 0; i < poGC->getNumGeometries(); i++ )
            GDALCollectRingsFromGeometry( poGC->getGeometryRef(i),
                                          aPointX, aPointY, aPartSize );
    }
    else
    {
        CPLDebug( "GDAL", "Rasterizer ignoring non-polygonal geometry." );
    }
}

int addData(const OGRLayerH layer,vector<int>& g_len_v,vector<int>&f_len_v,vector<int>&r_len_v,vector<double>&xx_v, vector<double>&yy_v)
{
	int num_feature=0;
	OGR_L_ResetReading( layer );
	OGRFeatureH hFeat;
	int this_rings=0,this_points=0;
    while( (hFeat = OGR_L_GetNextFeature( layer )) != NULL )
    {
		OGRGeometry *poShape=(OGRGeometry *)OGR_F_GetGeometryRef( hFeat );
		if(poShape==NULL)
		{
			cout<<"error:............shape is NULL"<<endl;
			num_feature++;
			continue;
		}
	    OGRwkbGeometryType eFlatType = wkbFlatten(poShape->getGeometryType());
		if( eFlatType == wkbPolygon )
		{
			OGRPolygon *poPolygon = (OGRPolygon *) poShape;
			this_rings+=(poPolygon->getNumInteriorRings()+1);
		}
        else
        {
        }
		std::vector<double> aPointX;
		std::vector<double> aPointY;
		std::vector<int> aPartSize;
		GDALCollectRingsFromGeometry( poShape, aPointX, aPointY, aPartSize );
		if(aPartSize.size()==0)
		{
			printf("warning: aPartSize.size()==0\n");
			//num_feature++;
		}
		xx_v.insert(xx_v.end(),	aPointX.begin(),aPointX.end());
		yy_v.insert(yy_v.end(),	aPointY.begin(),aPointY.end());
        r_len_v.insert(r_len_v.end(),aPartSize.begin(),aPartSize.end());
		f_len_v.push_back(aPartSize.size());
		OGR_F_Destroy( hFeat );
		num_feature++;
	}
	g_len_v.push_back(num_feature);
	return num_feature;
}

void process_coll(char *catfn,vector<int>& g_len_v,vector<int>&f_len_v,vector<int>&r_len_v,vector<double>&xx_v, vector<double>&yy_v)
{
	FILE *fp;
	if((fp=fopen(catfn,"r"))==NULL)
	{
		printf("can not open catalog file\n");
		exit(-1);
	}
	int this_seq=0;
	//while(!feof(fp))
	for(int i=0;i<1;i++)
	{
		char fn[100];
		fscanf(fp,"%s",fn);
		GDALDatasetH hDS = GDALOpenEx( fn, GDAL_OF_VECTOR, NULL, NULL, NULL );
		if(hDS==NULL)
		{
			  printf("hDS is NULL, skipping 1......\n");
			  //skiplist.push_back(fn);
			  continue;
		}

		OGRLayerH hLayer = GDALDatasetGetLayer( hDS,0 );
		if( hLayer == NULL )
		{
			  printf( "Unable to find layer 0, skipping 2......\n");
			  //skiplist.push_back(fn);
			  continue;
		}
		printf("%d %s \n",this_seq,fn);
		int num0=addData(hLayer,g_len_v,f_len_v,r_len_v,xx_v,yy_v);
		if(num0==0)
		{
			  printf("zero features, skipping 3......\n");
			  //skiplist.push_back(fn);
		}
		this_seq++;
	}
}

int main(int argc,char** argv)
{
	if(argc!=3)
	{
		printf("EXE cat_fn out_fn \n");
		exit(-1);
	}
	vector<int> g_len_v,f_len_v,r_len_v;
	vector<double> xx_v, yy_v;

	GDALAllRegister();
	char *inc=argv[1];
	timeval start, end;
	gettimeofday(&start, NULL);
	printf("catalog=%s  output=%s\n",argv[1],argv[2]);
	process_coll(inc,g_len_v,f_len_v,r_len_v,xx_v, yy_v);
	printf("skip list.............\n");
	gettimeofday(&end, NULL);
	long diff = end.tv_sec*1000000+end.tv_usec - start.tv_sec * 1000000-start.tv_usec;
	printf("CPU Processing time.......%10.2f\n",diff/(float)1000);
	printf("%lu %lu %lu %lu\n",g_len_v.size(),f_len_v.size(),r_len_v.size(),xx_v.size());

	int gc=g_len_v.size();
	int fc=0,rc=0,vc=0;
	printf("#of groups(datasets)=%d\n",gc);
	for(int g=0;g<gc;g++)
	{
		printf("#of features: (%d,%d,%d)\n",g,fc,g_len_v[g]);
		for(int f=fc;f<fc+g_len_v[g];f++)
		{
			printf("#of rings= (%d,%d,%d)\n",f,fc,f_len_v[f]);
			for(int r=rc;r<rc+f_len_v[f];r++)
			{
				printf("#of vertices (%d,%d,%d)\n",r,rc,r_len_v[r]);
				printf("...v:");
				for(int v=vc;v<rc+r_len_v[r];v++) printf("%5d",v);
				printf("\n");
				vc+=r_len_v[r];
			}
			rc+=f_len_v[f];
		}
		fc+=g_len_v[g];
	 }

	printf("%d %d %d %d\n",gc,fc,rc,vc);

	FILE *fp=fopen(argv[2],"wb");
	assert(fp!=NULL);

    fwrite(&gc,sizeof(int),1,fp);
    fwrite(&fc,sizeof(int),1,fp);
    fwrite(&rc,sizeof(int),1,fp);
    fwrite(&vc,sizeof(int),1,fp);

    int *g_p=&(g_len_v[0]);
    int *f_p=&(f_len_v[0]);
    int *r_p=&(r_len_v[0]);
    double *x_p=&(xx_v[0]);
    double *y_p=&(yy_v[0]);

    fwrite(g_p,sizeof(int),gc,fp);
    fwrite(f_p,sizeof(int),fc,fp);
    fwrite(r_p,sizeof(int),rc,fp);
    fwrite(x_p,sizeof(double),vc,fp);
    fwrite(y_p,sizeof(double),vc,fp);

    fclose(fp);

}
