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
#include "tldTracking.h"

void tldTracking::doTldTracking(tld_Structure *tld, floatArrayStruct *BB2, floatArrayStruct *Conf1, int *Valid,int objectID, CudaFunctions* cudaObj,intArrayStruct *confIdx)
{
	if(tld->bb.rows ==0) //if bounding box is not defined
	{
		return;
	}

	float medianFB; //median foward backward error
	
	cudaObj->doLK((unsigned char*)tld->currentFrames.iplimage_ptr.rgb_ipl->imageData,tld->bb.ptr,BB2->ptr,&medianFB,confIdx,tld->tmp.patt,tld->tmp.conf,tld->model.thr_fern*10.0f,tld->var);

	bool isinbox = ((BB2->ptr[0]>0)?1:0) && ((BB2->ptr[1]>0)?1:0) && ((BB2->ptr[2]< tld->currentFrames.iplimage_ptr.input_ipl->width)?1:0) && ((BB2->ptr[3]<tld->currentFrames.iplimage_ptr.input_ipl->height)?1:0);
	if(!isinbox)
	{
		BB2->ptr[0] = 0.0;
		BB2->ptr[1] = 0.0;
		BB2->ptr[2] = 0.0;
		BB2->ptr[3] = 0.0;

		return;
	}
	if(medianFB > 10)
	{ 
		BB2->ptr[0] = 0.0;
		BB2->ptr[1] = 0.0;
		BB2->ptr[2] = 0.0;
		BB2->ptr[3] = 0.0;

		return;
	}

	// esimate confidence and validity..
	int n=1;
	float* patch = (float*)malloc(tld->model.patchsize[0]*tld->model.patchsize[1]*sizeof(float));
	cudaObj->cudaGetPattern(n,patch,BB2->ptr);
	

	
	Conf1->size=n;//patch width
	Conf1->ptr=(float*) malloc(Conf1->size*sizeof(float));
	
	floatArrayStruct conf2;
	conf2.size=n;//patch width
	conf2.ptr=(float*) malloc(conf2.size*sizeof(float));
	
	int2DArrayStruct isin;
	isin.rows=3;
	isin.cols=n;
	isin.ptr=(int*) malloc(isin.rows*isin.cols*sizeof(int));
	

	tnn.dotldNN(patch,n,tld->model.patchsize[0]*tld->model.patchsize[1],tld->pex.ptr,tld->pex.cols,tld->nex.ptr,tld->nex.cols,Conf1->ptr,conf2.ptr,isin.ptr);

	(*Valid) = tld->valid.ptr[0];// I is the frame number..
	
	if(conf2.ptr[0]> tld->model.thr_nn_valid)
	{
		(*Valid)=1;
	}
		
	free(conf2.ptr);
	free(isin.ptr);			
	free(patch);
}
