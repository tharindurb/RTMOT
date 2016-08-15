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
#include "GenPositiveData.h"

//find the minimum or maximum value(according to the flag set) of a dataset
float GenPositiveData::findMinMax(float arr[],int end,int type)
{
	float minMax=arr[0];
	if(type == 1)
	{
		for(int i=1;i<end;i++)
		{
			if(minMax < arr[i])
				minMax = arr[i];
		}
	}
	else{
		for(int i=1;i<end;i++)
		{
			if(minMax > arr[i])
				minMax = arr[i];
		}
	}
	return minMax;
}

//find the Hull coordinates including all 10 boxes
float* GenPositiveData::bb_hull(float** bbP,int num_closest_Found)
{
	float* bbP1 = (float*)malloc(4*sizeof(float));
	bbP1[0] = (float)findMinMax(bbP[0],num_closest_Found,FIND_MIN);
	bbP1[1] = (float)findMinMax(bbP[1],num_closest_Found,FIND_MIN);
	bbP1[2] = (float)findMinMax(bbP[2],num_closest_Found,FIND_MAX);
	bbP1[3] = (float)findMinMax(bbP[3],num_closest_Found,FIND_MAX);
    return bbP1;
}


int GenPositiveData::higestValueIndexes(float dataArray[],int indx[],int size,float thresh,int limit)
{
	int id;
	float val;
	float* tempDataArray = (float*)malloc(size*sizeof(float));
	int tempDataArraySize = 0;
	
	/*Get the overlap values that are greater than the threshold*/
	for(int i =0; i<size; i++){
		val = dataArray[i];
		if(val >= thresh){ 
			tempDataArray[tempDataArraySize] = val; indx[tempDataArraySize++] = i;
		}
	}

	/*Get highest 10 overlap values and it's indexes*/
	for (int i=0;i<tempDataArraySize;i++)
	{
		for(int j=i+1;j<tempDataArraySize;j++)
		{
			if(tempDataArray[i]< tempDataArray[j])
			{
				id= indx[j]; 
				indx[j] = indx[i];	
				indx[i] = id;

				val= tempDataArray[i]; 
				tempDataArray[i]=tempDataArray[j]; 
				tempDataArray[j]=val;
			}
		}
		if(tempDataArray[i] < thresh || i==(limit-1))
		{
				free(tempDataArray);
				return i+1;
		}
	}
	free(tempDataArray);
	return limit;

}


void GenPositiveData::genPositiveData(tld_Structure *tld, floatArrayStruct *overlap, int2DArrayStruct *pX, float2DArrayStruct *pEx, floatArrayStruct *bbP_out, CudaFunctions* fernObj, bool first)
{

	intArrayStruct idxBoxes;
	idxBoxes.ptr = (int*)malloc(overlap->size*sizeof(int)); /*Holds the indexes for grid boxes.*/ 
	
	//find the most 10 overlapped boxes in sorted order
	int num_closest_Found = higestValueIndexes(overlap->ptr,idxBoxes.ptr,overlap->size,0.6,tld->p_par_update.num_closest);

	idxBoxes.ptr = (int*)realloc(idxBoxes.ptr,num_closest_Found*sizeof(int));
	idxBoxes.size = num_closest_Found;

	//Get closest bbox --> box that has the maximum overlap
	int maxOverlap  = idxBoxes.ptr[0];
	float* bbP0  = (float*)calloc(4,sizeof(float));
	bbP0[0] = tld->grid.ptr[6*maxOverlap];
	bbP0[1] = tld->grid.ptr[6*maxOverlap+1];
	bbP0[2] = tld->grid.ptr[6*maxOverlap+2];
	bbP0[3] = tld->grid.ptr[6*maxOverlap+3];
	
	bbP_out->ptr=bbP0; /*The grid box that has maximum overlap is returned */
	bbP_out->size=4;

	float** bbP  =(float**)malloc(4*sizeof(float*));
	
	for(int i=0;i<4;i++) /*Corresponsding grid boxes are found for highest overlap values*/ 
	{
		bbP[i]  =(float*)malloc(num_closest_Found*sizeof(float));

		for(int j=0;j<num_closest_Found;j++)
			bbP[i][j] = tld->grid.ptr[6*idxBoxes.ptr[j]+i];
	}
	
	//A Hull is created including all 10 maximum overlapped boxes
	float* bbP1= bb_hull(bbP,num_closest_Found);
	

	

	pEx->ptr = (float*)malloc(tld->model.patchsize[0]*tld->model.patchsize[1]*sizeof(float));
	fernObj->cudaGetPattern(1,pEx->ptr,bbP0);

	pEx->rows = tld->model.patchsize[0]*tld->model.patchsize[1]; /* 45 x 45 =  2025 */ 
	pEx->cols = 1;


	free(bbP[0]);
	free(bbP[1]);
	free(bbP[2]);
	free(bbP[3]);
	free(bbP);

	int sizeIdx = tld->model.num_trees*num_closest_Found; /* 10 x 100 */
	int numWarpsLimit = 0;
	int* temInc;

	
	if(first)
	{
		//for the first marked frame 20 warps are done to the HULL and 20(20 warps) x 10(10 max overlaped boxes) features were taken
		pX->ptr = (int*)malloc(tld->p_par_init.num_warps*sizeIdx*sizeof(int));
		pX->cols = tld->p_par_init.num_warps*sizeIdx/10;	
		pX->rows = 10;
		temInc = pX->ptr;
		numWarpsLimit = tld->p_par_init.num_warps;
	}
	else
	{
		//for other frames 10(10 warps) x 10(10 max overlaped boxes) features were taken
		pX->ptr = (int*)malloc(tld->p_par_update.num_warps*sizeIdx*sizeof(int));
		pX->cols = tld->p_par_update.num_warps*sizeIdx/10;	
		pX->rows = 10;
		temInc = pX->ptr;
		numWarpsLimit = tld->p_par_update.num_warps;
	}
	
	if(num_closest_Found > 0)
	{
		CvRandState RandState; 
		cvRandInit(&RandState, -0.5 , 0.5 ,cvGetTickCount(),CV_RAND_UNI);
		
		float2DArrayStruct matrix;
		matrix.cols = (numWarpsLimit-1);
		matrix.rows = 9;

		
		matrix.ptr = (float*)malloc(sizeof(float)*matrix.cols*matrix.rows);

		for(int i=0;i<(numWarpsLimit-1);i++) 
			getMatrix(bbP1,(matrix.ptr + 9*i),RandState); //10(20 for initial frame)affine warping matrixs is created with random values
		
		//generate Positive Samples for learning
		fernObj->fern5Positive(pX,bbP1,matrix,idxBoxes); 
		free(matrix.ptr);
		

	}
	free(idxBoxes.ptr);
	free(bbP1);
	
}

void GenPositiveData::getMatrix(float* bb,float* matrix,CvRandState RandState){
	int ANGLE = 20;
	float SCALE = 0.02;
	float SHIFTT = 0.02;
	float randN;
		 
        float Sh1[9] = {1, 0, -0.5*(bb[0]+bb[2]), 0, 1, -0.5*(bb[1]+bb[3]), 0, 0, 1};
		 
		 //calculate warping Scale
		 cvbRand(&RandState,&randN,1);
		 float sca = 1-SCALE*randN;
		 float Sca[9] = {sca,0.0,0.0,0.0,sca,0.0,0.0,0.0,1};
		 
		 //Calculate warping Angle
		 cvbRand(&RandState,&randN,1);
		 float ang = 2*CV_PI*ANGLE*randN/360;
		 float Ang[9] = {cos(ang), -sin(ang),0.0,sin(ang),cos(ang),0.0,0.0,0.0,1.0};
		 
		 //calculate warping Shifts in X and Y direction
		 cvbRand(&RandState,&randN,1);
		 float shR = SHIFTT * (bb[3]-bb[1]+1) * randN;		 
		 cvbRand(&RandState,&randN,1);
		 float shC = SHIFTT * (bb[2]-bb[0]+1) * randN;	
		 float Sh2[9] = {1.0,0.0,shC,0.0,1.0,shR,0.0,0.0,1.0};

		 float tem1[9];
		 float tem2[9];


		 //generates affine transform matrix
		 mulMat(tem2,Sca,Sh1);
		 mulMat(tem1,Ang,tem2);
		 mulMat(tem2,Sh2,tem1);
	     
		 CvMat H = cvMat(3,3,CV_MAT32F,(float*)tem2);
		 float dataInv[] = {0,0,0,0,0,0,0,0,0};
		CvMat Hinv = cvMat(3,3,CV_MAT32F,(float*)dataInv);
		cvInv(&H,&Hinv,CV_LU);

		float* temp = matrix;
		for(int i=0;i<3;i++){
			 for(int j=0;j<3;j++){
				*temp++ = CV_MAT_ELEM(Hinv,float,i,j);
			}
		}
		
}

//multiply 2 3x3 matrices
void GenPositiveData::mulMat(float out[],float data1[],float data2[]){
   int k=0;
   for(int i=0;i<3;i++){
	   for(int j=0;j<3;j++){
		   out[k++] = data1[3*i]*data2[j]  + data1[3*i+1]*data2[j+3] + data1[3*i+2]*data2[j+6];
	   }
   }
}
