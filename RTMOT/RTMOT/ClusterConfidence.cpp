 // Copyright 2012 by Thanuja Mallikarachchi, Chammi Dilhari,  Chathurangi Kumarasinghe, Tharindu Bandaragoda
 //
 // This file is part of RTMOT.
 //
 // TLDCUDA is free software: you can redistribute it and/or modify
 // it under the terms of the GNU General Public License as published by
 // the Free Software Foundation, either version 3 of the License, or
 // (at your option) any later version.
 //
 // TLDCUDA is distributed in the hope that it will be useful,
 // but WITHOUT ANY WARRANTY; without even the implied warranty of
 // MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 // GNU General Public License for more details.
 //
 // You should have received a copy of the GNU General Public License
 // along with TLDCUDA.  If not, see <http://www.gnu.org/licenses/>.
#include "stdafx.h"
#include "ClusterConfidence.h"

void ClusterConfidence::bb_cluster_confidance(float2DArrayStruct *iBB, floatArrayStruct *iConf, float2DArrayStruct *oBB, floatArrayStruct *oConf, intArrayStruct *oSize)
{


	float SPACE_THR = 0.5;
	intArrayStruct T;
	float2DArrayStruct bbd;

	if (iBB->cols<1 || iBB->rows<1)
	{
		T.size=0;
		T.ptr=NULL;

		return;
	}

	switch (iBB->cols)
	{
		case 0: 
			T.size=0;
			T.ptr=NULL;
			break;

		case 1: 
			T.size = 1;
			T.ptr=(int*)malloc(T.size*sizeof(int)) ;
			T.ptr[0] = 1;
			break;

		case 2:
			T.size=2;
			T.ptr=(int*)malloc(T.size*sizeof(int)) ;
			T.ptr[0]=1;
			T.ptr[1]=1;
			
			bbd=bb_distance(iBB);
			if (bbd.ptr[0] > SPACE_THR)
			{ 
				T.ptr[0] = 2;
			}
			break;

		default:
			bbd = bb_distance(iBB);			
			float2DArrayStruct Z;
			link.mexLinkageTEMPLATE(&Z,&bbd,"si",false);	
			clus.getCluster(&Z, SPACE_THR,&T);
			break;
	}
	
	
	intArrayStruct idx_cluster; 
	get_unique_T(&T,&idx_cluster);
	int num_clusters = idx_cluster.size;

	oBB->cols=num_clusters  ;  
	oBB->rows=4;
	oBB->ptr=(float*)malloc(sizeof(float)*oBB->cols*oBB->rows);

	oConf->size=num_clusters;
	oConf->ptr=(float*)malloc(sizeof(float)*oConf->size);

	oSize->size=num_clusters;
	oSize->ptr=(int*) malloc(sizeof(int)*num_clusters);

	for (int i = 0;i<num_clusters;i++){
		float sum1_1=0;
		float sum1_2=0;
		float sum1_3=0;
		float sum1_4=0;
		
		float sum2=0;
		float sum3=0;
		 
		for (int j = 0;j<T.size;j++){
			if(T.ptr[j]==idx_cluster.ptr[i]){
				
					sum1_1=sum1_1+iBB->ptr[0+j*iBB->rows];
					sum1_2=sum1_2+iBB->ptr[1+j*iBB->rows];
					sum1_3=sum1_3+iBB->ptr[2+j*iBB->rows];
					sum1_4=sum1_4+iBB->ptr[3+j*iBB->rows];

					sum2=sum2+iConf->ptr[j];
					sum3=sum3+1;
				
			}
			
		}


		oBB->ptr[i*oBB->rows+0] = sum1_1/sum3;
		oBB->ptr[i*oBB->rows+1] = sum1_2/sum3;
		oBB->ptr[i*oBB->rows+2] = sum1_3/sum3;
		oBB->ptr[i*oBB->rows+3] = sum1_4/sum3;
			
		oConf->ptr[i]  = sum2/sum3;
		oSize->ptr[i]  = sum3;

	}

	if(T.size!=0)
		free(T.ptr);

	if(idx_cluster.size!=0)
		free(idx_cluster.ptr);
}

float2DArrayStruct ClusterConfidence::bb_distance(float2DArrayStruct *bb1)
{

	float2DArrayStruct d;
	d=bb_overlap1(bb1);
	for(int i=0;i<d.cols;i++)
	{
		for(int j=0;j<d.rows;j++)
		{
			d.ptr[i*d.rows+j]=1-d.ptr[i*d.rows+j];
			
		}

	}


	return d;


}
float2DArrayStruct ClusterConfidence::bb_overlap1(float2DArrayStruct* bb)
{

			int nBB = bb->cols;//cols
			int mBB = bb->rows;//rows

			// Output
			float2DArrayStruct output;
			output.cols=(int)nBB*(nBB-1)/2;
			output.rows=1;
			output.ptr = (float*)malloc(output.rows*output.cols*sizeof(float));

			float* out=output.ptr;
	

			for (int i = 0; i < nBB-1; i++) {
				for (int j = i+1; j < nBB; j++) {
					*out++ = bb_overlap_calc(bb->ptr + mBB*i, bb->ptr + mBB*j);
				}
			}
			return output;

}

float ClusterConfidence::bb_overlap_calc(float *bb1, float *bb2) 
{
	
	if (bb1[0] > bb2[2]) { return 0.0; }
	if (bb1[1] > bb2[3]) { return 0.0; }
	if (bb1[2] < bb2[0]) { return 0.0; }
	if (bb1[3] < bb2[1]) { return 0.0; }
	
	float colInt =  std::min(bb1[2], bb2[2]) - std::max(bb1[0], bb2[0]) + 1;
	float rowInt =  std::min(bb1[3], bb2[3]) - std::max(bb1[1], bb2[1]) + 1;

	float intersection = colInt * rowInt;
	float area1 = (bb1[2]-bb1[0]+1)*(bb1[3]-bb1[1]+1);
	float area2 = (bb2[2]-bb2[0]+1)*(bb2[3]-bb2[1]+1);
	return intersection / (area1 + area2 - intersection);
}

void ClusterConfidence::get_unique_T(intArrayStruct* pp,intArrayStruct* idx)
{
	
	std::vector<int> w(pp->ptr,pp->ptr+pp->size);
	std::vector<int>::iterator it,it2;
	std::sort(w.begin(),w.end());
	it=std::unique(w.begin(),w.end()); 
	int value=0;

	for (it2=w.begin(); it2!=it; ++it2){
		value=value+1;
	}
	  
	idx->size=value;
	idx->ptr=new int[idx->size];
	memmove( idx->ptr,&w.at(0), idx->size*sizeof(int) );
	
}
