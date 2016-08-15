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
#ifndef CUDA_FUNC_H
#define CUDA_FUNC_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include <cutil_inline.h> 
#include <time.h>
#include "DataStructs.h"
#include "timingMicro.h"

#include <iostream>
#include <fstream>

using namespace std;


#define PYRAMID_LEVELS 3


class CudaFunctions
{
public:

	cudaStream_t tracking_stream;
	cudaStream_t fern_stream;

	CudaFunctions();
	void freeMemory();

	void initMemTrack(int _w,int _h);
	void doLK(unsigned char *cur,float* bbIn,float* bbOut,float *medFB,intArrayStruct *confIdx, int2DArrayStruct patt,floatArrayStruct conf,float thresh,float var);
	void run(intArrayStruct *confIdx,int2DArrayStruct patt,floatArrayStruct conf,float thresh,float var,float* ptsI,int nBoxes = 1);
	void run_FB(int nBoxes = 1);
	void euclidianDistance(int nPts = 100);
	void doNormalizedCrossCorr(int nPts = 100,int Winsize = 9);

    //void cutilCheckMsg(const char *msg);
	
	float* bb_points(float* bb,int numM,int numN,float margin);
	void bb_predict(float* bb0,float* bb1, int nPts,float *medFB);
	void sort(float arr[], int end);
	void sortMedian(float arr[], int end);
	float median(float arr[],int nPts);
	float pdist(float pts0[],float pts1[],int nPts);

    int scaledWidths[PYRAMID_LEVELS], scaledHeights[PYRAMID_LEVELS];
	void freePointMemory(int nBoxes);
	void loadImageTrack(unsigned char *prev);
	void initPointSetMemoryTrack(int nBoxes);

    int w, h;
	
    char *status;
	float *ptsI, *ptsJ, *ncc, *fb;
	float *ptsIIN, *ptsJIN, *fbIN, *nccIN;
		
   
	unsigned char *gpuPreviousRGBImage;
    unsigned char *gpuCurrentRGBImage;

    float *gpuImagePyramidPrevious[PYRAMID_LEVELS];
    float *gpuImagePyramidCurrent[PYRAMID_LEVELS];
    float *gpuSmoothedXPrevious;
    float *gpuSmoothedXCurrent;
    float *gpuSmoothedPrevious;
    float *gpuSmoothedCurrent;

	cudaArray *gpuArrayPyramidPrevious;
	cudaArray *gpuArrayPyramidCurrent;
	cudaArray *gpuArrayPyramid;
	

    float *gpu_ptsFB, *gpu_ptsJ, *gpu_ptsI;
    char *gpu_status;


	///---------------CUDA FERN ELEMENTS-------------------/////
	int objectID;	
	int width, height;

	float* weights;
	int* nP;
	int* nN;

	float2DArrayStruct gpu_grid;
	float2DArrayStruct gpu_featuresOffsets;

	size_t gpu_image_pitch;
	size_t gpu_blurImageRow_pitch;
	size_t gpu_blurImage_pitch; 
	size_t gpu_warpedImage_pitch; 
	size_t gpu_IntregalImage_pitch; 
	size_t gpu_IntregalSQImage_pitch; 

	unsigned char* gpu_image;
	unsigned char* gpu_blurImageRow;
	unsigned char* gpu_blurImage;
	unsigned char* gpu_warpedImage;
	float* gpu_rowImage;
	float* gpu_IntregalImage;
	float* gpu_rowSQImage;
	float* gpu_IntregalSQImage;

	CudaFunctions(int oId)
	{
		objectID = oId;
	}

	void initializeFern(float2DArrayStruct grid,float2DArrayStruct featureOffsets,int w,int h);  
	void fern5Positive(int2DArrayStruct* patt,float* bb,float2DArrayStruct matrix,intArrayStruct idxBoxs);
	void fern5Negative(int2DArrayStruct patt,char* status, int *idxBoxs, float varienceThresh=0);
	void fernLearning(int2DArrayStruct pX, int2DArrayStruct nX,float threshPositive,float threshNegative,int bootStrap =1);
	void fern3(int2DArrayStruct nX,float* confidences);
	void fernDetection(int2DArrayStruct patt,floatArrayStruct conf,intArrayStruct* tconfidx,float pthresh,float varienceThresh=0);

	void initMem_fern(int w,int h);	
	void loadGrid(float2DArrayStruct grid);
	void loadFeatureOffsets(float2DArrayStruct featureOffsets);
	void loadImage(unsigned char* imageData);
	//-------Image Blur CUDA implementation-------//
	void doImageBlurRow(int w, int h);
	void doImageBlurCol(int w, int h);

	//----------Integral Image ------------------//
	void doIntregal();
	void createIntregalImages();

	//-----------BB Overlap Calculate --------------//
	void calcBBOverLapWithGrid(float* bb,int sizeBB,float* ovelap);
	//------Sort CUDA implementation------------//
	void sort_data(float* data,intArrayStruct *tconfidx,int input_size,float threshold );

	//------- NN Cuda Implementation--------------//
	void calc_tldNN(float* x,int n,int M,float* pex,int N1,float* nex,int N2,float* conf1,float* conf2,int* isin);

	//------- getPattern Cuda Implementation--------------//
	void cudaGetPattern(int num,float* pattern,float *bb);

	int** calcntuples(int x1[],int x2[],int num_x1_col,int num_x2_col);
};

#endif
