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
#include "TldLearning.h"


float TldLearning::varience(const float* dataArr,int size)
{
	float sum=0;
	float sqSum = 0;
	float mean;

	for(int i =0;i<size;i++)
		sum += dataArr[i];
	mean = sum/(float)(size);

	for(int i=0;i < size;i++)
		sqSum += (dataArr[i] - mean)*(dataArr[i] - mean);

	return sqSum/(float)(size-1);
}



void TldLearning::doLearning(tld_Structure* tld,CudaFunctions* fernObj)
{

	float* bb= tld->bb.ptr;
	float* pPatt = (float*)malloc(tld->model.patchsize[0]*tld->model.patchsize[1]*sizeof(float));
	fernObj->cudaGetPattern(1,pPatt,bb);//get the image patch for current bounding box
		
	float* pConf1=(float*) malloc(1*sizeof(float));
	float* pConf2=(float*) malloc(1*sizeof(float));
	int* pIsin=(int*) malloc(3*1*sizeof(int));

	//using NN clissifier check confidence for the image patch for current bounding box
	tnn.dotldNN(pPatt,1,tld->model.patchsize[0]*tld->model.patchsize[1],tld->pex.ptr,tld->pex.cols,tld->nex.ptr,tld->nex.cols,pConf1,pConf2,pIsin);


	int pConf1Valid=1;



	if(pConf1[0]>=0.5)
	{
		pConf1Valid=0;
	}

	if (pConf1Valid == 1)
	{ //too fast change of appearance
		std::cout<<"\nFast change";
		tld->valid.ptr[0] = 0;
		return;
	} 
	else if(varience(pPatt,1*tld->model.patchsize[0]*tld->model.patchsize[1])< tld->var)
	{
		std::cout<<"\nLow variance"; //too low variance of the patch
		tld->valid.ptr[0] = 0;
		return;
	}
	else if(pIsin[2] == 1)
	{
		std::cout<<"patch is in negative data";
		tld->valid.ptr[0] = 0;
		return;
	}

	floatArrayStruct overlap;
	overlap.ptr = (float*)malloc(sizeof(float)*tld->grid.cols);
	overlap.size = tld->grid.cols;

	//calculate fractional overlap between bounding box which has the object and other boxes in the grid
	fernObj->calcBBOverLapWithGrid(bb,1,overlap.ptr);


	int2DArrayStruct nX;
	int2DArrayStruct pX;
	float2DArrayStruct pEx;
	floatArrayStruct bbP;
	
	int* pattPtr =tld->tmp.patt.ptr;


	nX.ptr = (int*)malloc(tld->grid.cols*10*sizeof(int));
	nX.cols =0;
	nX.rows = 10;


	/*Negative Learning sample generation for FERN: 
	select the boxes with lower fractional overlap and higher confidence to be the object*/
	for(int i=0;i<tld->grid.cols;i++)
	{
		if((overlap.ptr[i] < tld->n_par.overlap) && (tld->tmp.conf.ptr[i]>=1))
		{
			memcpy((nX.ptr+nX.cols*nX.rows),pattPtr,10*sizeof(int));	
			nX.cols++;
		}
		pattPtr += 10;
	}
	nX.ptr = (int*)realloc(nX.ptr,nX.cols*10*sizeof(int));	 
	
	//generate positive samples for FERN learning
	genPos.genPositiveData(tld,&overlap,&pX,&pEx,&bbP,fernObj);
	free(bbP.ptr);

	//training the FERN with selected positive and negative samples
	fernObj->fernLearning(pX,nX,tld->model.thr_fern*10.0f,5.0,2); 

		
	free(overlap.ptr);
	free(nX.ptr);
	free(pX.ptr); 


	floatArrayStruct overlap2;
	/*measure fractional overlap between current bounding box and boxes in detections*/
	overlap2 = bb_lap.bb_overlap(tld->bb.ptr,4,1,tld->dt.bb,4,tld->dt.num_detections);

	int nEx_size = 0;
	float* nEx =(float*)malloc(tld->dt.num_detections*tld->model.patchsize[0]*tld->model.patchsize[1]*sizeof(float));
	float* nEXptr = nEx;
	float* patchPtr=tld->dt.patch;	

	/*Select low fractional overlap patches in the detected boxes as Negative Learning Samples for NN classifier*/
	for(int i =0 ;i<tld->dt.num_detections ; i++)
	{
		if(overlap2.ptr[i] < tld->n_par.overlap)
		{
			memcpy(nEXptr,patchPtr,tld->model.patchsize[0]*tld->model.patchsize[1]*sizeof(float));
			nEx_size++;
			nEXptr += tld->model.patchsize[0]*tld->model.patchsize[1];
		}
		patchPtr += tld->model.patchsize[0]*tld->model.patchsize[1];
	}

	nEx = (float*)realloc(nEx,nEx_size*tld->model.patchsize[0]*tld->model.patchsize[1]*sizeof(float));

	/*Traing NN classifier*/
	train.doTrainNN(pEx.ptr,pEx.rows,pEx.cols,nEx,nEx_size,tld);

	free(overlap2.ptr);
	free(pEx.ptr);
	
	free(pPatt);
	free(pConf1);
	free(pConf2);
	free(pIsin);

	if(nEx_size>0)
	{
		free(nEx);
	}
}