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
#include "tldDetection.h"

void tldDetection::doTldDetection(tld_Structure *tld, CudaFunctions *fernObj, float2DArrayStruct *dBB, floatArrayStruct *dConf,intArrayStruct *confIdx, bool status)
{

	int num_detections =0;
	/*if the there is a bounding box to track this function fernDetection 
	is executed inside the tracking code using streams to gain more performance and if there is no 
	bounding box(object is lost) tracking is not called so the fernDetection detection function is called here*/
	if(!status){  
		fernObj->fernDetection(tld->tmp.patt,tld->tmp.conf,confIdx,tld->model.thr_fern*10.0f,tld->var);
	}
	num_detections = confIdx->size;


	if(num_detections == 0)
	{
		dBB->cols=0;
		dBB->rows=0;
		dBB->ptr=0;		
		dConf->size=0;
		dConf->ptr = 0;
	
		tld->dt.bb =0;
		tld->dt.conf1 =0;
		tld->dt.conf2 =0;
		tld->dt.num_detections =0;
		tld->dt.patch =0;
		tld->dt.patt =0;

		free(confIdx->ptr);
		return;
	}

	float* bb	=(float*)malloc(4*num_detections*sizeof(float));
	float* bbtemp = bb;

	float* patch   =(float*)malloc(num_detections*tld->model.patchsize[0]*tld->model.patchsize[1]*sizeof(float)); 
	float* patchTemp = patch;

	float* pattDet = (float*)malloc(num_detections*10*sizeof(float));
	float* conf1=(float*)malloc(num_detections*sizeof(float));
	float* conf2=(float*)malloc(num_detections*sizeof(float));
	int* isin=(int*)malloc(num_detections*3*sizeof(int));

	//float thr_nn = 0.65; //NN confidence threshhold
	float* BB	=(float*)malloc(4*num_detections*sizeof(float));
	float* CONF = (float*)malloc(num_detections*sizeof(float));

	for(int i=0;i<num_detections;i++)
	{
	
		for(int j=0; j<4;j++)
			*(bbtemp+j)  = tld->grid.ptr[6*confIdx->ptr[i]+j];

		for(int j=0;j<10;j++)
			pattDet[i*10+j]=tld->tmp.patt.ptr[confIdx->ptr[i]*10+j];

		bbtemp +=4;
	
	}

	fernObj->cudaGetPattern(num_detections,patch,bb);  //get resized(45x45) and normalized image patches for the bounding boxes detected by FERN 


	/*Get the confidences from the NN classifier for the patches detected by FERN*/
	tnn.dotldNN(patch,num_detections,tld->model.patchsize[0]*tld->model.patchsize[1],tld->pex.ptr,tld->pex.cols,tld->nex.ptr,tld->nex.cols,conf1,conf2,isin);


	int nn_idx=0;
	for(int i=0;i<num_detections;i++)
	{
		if(conf1[i] > tld->model.thr_nn)//select bounding boxes which have higher confidence
		{
			BB[4*nn_idx]   =bb[4*i];
			BB[4*nn_idx+1] =bb[4*i+1];
			BB[4*nn_idx+2] =bb[4*i+2];
			BB[4*nn_idx+3] =bb[4*i+3];
			CONF[nn_idx++] = conf2[i];
		}
	}

	if(nn_idx>0)
	{
		BB = (float*)realloc(BB,4*nn_idx*sizeof(float));
		CONF = (float*)realloc((float*)CONF,nn_idx*sizeof(float));
	}
	else
	{

		BB = (float*)realloc(BB,4*1*sizeof(float));
		CONF =(float*)realloc((float*)CONF,1*sizeof(float));
	}

	dBB->ptr=BB;
	dBB->cols=nn_idx;
	dBB->rows=4;

	dConf->ptr=CONF;
	dConf->size=nn_idx;

	tld->dt.bb		= bb;
	tld->dt.patch	= patch;
	tld->dt.patt	= pattDet;
	tld->dt.conf1	= conf1;
	tld->dt.conf2	= conf2;
	tld->dt.isin = isin;
	tld->dt.num_detections = num_detections;
	free(confIdx->ptr);
	

}


