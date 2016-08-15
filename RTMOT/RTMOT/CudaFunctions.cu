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
#include "CudaFunctions.h"
#define patchSize 45
#define steps 15
#define SmoothYNoOfThreads 1
#define BlurNoOfThreads 1

//scales for each pyramid 
const float pyramidScales[] = {1, 0.5f, 0.25f, 0.125f, 0.0625f, 0.03125f, 0.015625f, 0.0078125f};

texture<float, 2, cudaReadModeElementType> texturePreviousFrame;
texture<float, 2, cudaReadModeElementType> textureCurrentFrame;
texture<float, 2, cudaReadModeElementType> texturePyramid;

texture<unsigned char, 2, cudaReadModeElementType> textureImage;
texture<unsigned char, 2, cudaReadModeElementType> textureBlurredImage;
texture<unsigned char, 2, cudaReadModeElementType> textureWarpedImage;
texture<float, 2, cudaReadModeElementType> textureIntregalImage;
texture<float, 2, cudaReadModeElementType> textureIntregalSQImage;
texture<unsigned char, 2, cudaReadModeElementType> textureRowBlurredImage;

//-------------------n tuples ------------------kernel//
__global__ void calc_grid_kernel(int* x1,int* x2,int* x3,int x1_cols,int x2_cols)
{
	int idx_x = blockDim.x*blockIdx.x + threadIdx.x;
	int numDim=x1_cols*x2_cols;
	int idx = numDim*idx_x;

	int idx_x1_col =(int)floorf(idx_x/x2_cols);
	int idx_x2_col = idx_x-idx_x1_col*x2_cols;

	x3[idx_x]=x1[idx_x1_col];
	x3[x1_cols*x2_cols+idx_x]=x2[idx_x2_col];
}


//////////////-------------------gpu IntregalImage and Square Intregal Image Generation -----------------------------------/////////////////////
/*Intregal Image Calculation is Implementation of paper "Efficient Integral Image Computation on the GPU" see the paper for clarifications*/
__global__ void rowIntegralKernel(float *gpu_rowImage,int w,int h) {

  const int rowSize = 1024;		
__shared__ float input[rowSize];
	int idxN = threadIdx.x;
	int idx = 2*threadIdx.x;

	if (idx <w){
		input[idx] = tex2D(textureImage,(float)idx,(float)blockIdx.x);
		input[idx+1] = tex2D(textureImage,(float)(idx+1),(float)blockIdx.x);
	}
	else {
		input[idx] = 0.0f;
		input[idx+1] = 0.0f;
	}
		
	
	int offset = 1;
	
	for(int d = rowSize>>1; d > 0; d >>= 1){
			
		__syncthreads();
				
		if(idxN < d){
			int ai = offset*(idx+1)-1;
			int bi = offset*(idx+2)-1;
			input[bi] += input[ai];
		}
		offset *= 2;
	}
		
	if(idxN == 0) input[rowSize - 1] = 0.0f;
	
	for(int d = 1; d < rowSize; d *= 2){
		offset >>= 1; __syncthreads();
			if(idxN < d){
				int ai = offset*(idx+1)-1;
				int bi = offset*(idx+2)-1;
				float t = input[ai];
				input[ai] = input[bi];
				input[bi] += t;
			}
	}
	__syncthreads();

	if((idx+1) < w){
		gpu_rowImage[blockIdx.x*w+idx]=input[idx+1];
		gpu_rowImage[blockIdx.x*w+idx+1]=input[idx+2];
	}
}


__global__ void colIntegralKernel(float *gpu_rowImage,float *gpu_intregalImage,int w,int h,int pitch) {
	const int colSize = 512;	
	__shared__ float input[colSize];
	int idxN = threadIdx.x;
	int idx = 2*threadIdx.x;

	if (idx<h){
		input[idx] = gpu_rowImage[idx*w + blockIdx.x];
		input[idx+1] = gpu_rowImage[(idx+1)*w + blockIdx.x];
	}
	else {
	input[idx] = 0.0f;
	input[idx+1] = 0.0f;
	}
		
	
	int offset = 1;
	
	for(int d = colSize>>1; d > 0; d >>= 1){
			
		__syncthreads();
				
		if(idxN < d){
			int ai = offset*(idx+1)-1;
			int bi = offset*(idx+2)-1;
			input[bi] += input[ai];
		}
		offset *= 2;
	}
		
	if(idxN == 0) input[colSize - 1] = 0.0f;
	
	for(int d = 1; d < colSize; d *= 2){
		offset >>= 1; __syncthreads();
			if(idxN < d){
				int ai = offset*(idx+1)-1;
				int bi = offset*(idx+2)-1;
				float t = input[ai];
				input[ai] = input[bi];
				input[bi] += t;
			}
	}
	__syncthreads();

	if((idx+1) < h){
		gpu_intregalImage[blockIdx.x + idx*pitch]=input[idx+1];
		gpu_intregalImage[blockIdx.x + (idx+1)*pitch]=input[idx+2];
	}
}

__global__ void rowIntegralSQKernel(float *gpu_rowSQImage,int w,int h) {

	const int rowSize = 1024;		
	__shared__ float input[rowSize];
	int idxN = threadIdx.x;
	int idx = 2*threadIdx.x;

	if (idx <w){
		float i = tex2D(textureImage,(float)idx,(float)blockIdx.x);
		float j = tex2D(textureImage,(float)(idx+1),(float)blockIdx.x);
		input[idx] = i * i;
		input[idx+1] = j * j;
	}
	else {
		input[idx] = 0.0f;
		input[idx+1] = 0.0f;
	}
		
	int offset = 1;
	
	for(int d = rowSize>>1; d > 0; d >>= 1){
			
		__syncthreads();
				
		if(idxN < d){
			int ai = offset*(idx+1)-1;
			int bi = offset*(idx+2)-1;
			input[bi] += input[ai];
		}
		offset *= 2;
	}
		
	if(idxN == 0) input[rowSize - 1] = 0.0f;
	
	for(int d = 1; d < rowSize; d *= 2){
		offset >>= 1; __syncthreads();
			if(idxN < d){
				int ai = offset*(idx+1)-1;
				int bi = offset*(idx+2)-1;
				float t = input[ai];
				input[ai] = input[bi];
				input[bi] += t;
			}
	}
	__syncthreads();

	if((idx+1) < w){
		gpu_rowSQImage[blockIdx.x*w+idx]=input[idx+1];
		gpu_rowSQImage[blockIdx.x*w+idx+1]=input[idx+2];
	}
}


__global__ void colIntegralSQKernel(float *gpu_rowSQImage,float *gpu_intregalSQImage,int w,int h,int pitch) {
	const int colSize = 512;	
__shared__ float input[colSize];
	int idxN = threadIdx.x;
	int idx = 2*threadIdx.x;

	if (idx<h){
		input[idx] = gpu_rowSQImage[idx*w + blockIdx.x];
		input[idx+1] = gpu_rowSQImage[(idx+1)*w + blockIdx.x];		
	}
	else {
	input[idx] = 0.0f;
	input[idx+1] = 0.0f;
	}
		
	
	int offset = 1;
	
	for(int d = colSize>>1; d > 0; d >>= 1){
			
		__syncthreads();
				
		if(idxN < d){
			int ai = offset*(idx+1)-1;
			int bi = offset*(idx+2)-1;
			input[bi] += input[ai];
		}
		offset *= 2;
	}
		
	if(idxN == 0) input[colSize - 1] = 0.0f;
	
	for(int d = 1; d < colSize; d *= 2){
		offset >>= 1; __syncthreads();
			if(idxN < d){
				int ai = offset*(idx+1)-1;
				int bi = offset*(idx+2)-1;
				float t = input[ai];
				input[ai] = input[bi];
				input[bi] += t;
			}
	}
	__syncthreads();

	if((idx+1) < h){
		gpu_intregalSQImage[blockIdx.x + idx*pitch]=input[idx+1];
		gpu_intregalSQImage[blockIdx.x + (idx+1)*pitch]=input[idx+2];
	}
}

///////////// --------------------GETPATTERN-------------------------------------/////////////////////
__global__ void floatToCharKernal(float* imageIn, unsigned char* imageOut,int height,int width, int imageOut_pitch){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if((x < width) && (y < height)) imageOut[y*imageOut_pitch + x] = imageIn[y*width + x];
}
__global__ void getPatternKernel( float* patchout,float* bb) {	
	
__shared__ float patch[patchSize][patchSize];
	__shared__ float total[steps][steps];
	__shared__ float rowSum[steps];
	__shared__ float sum;
	
	int boxWidth,boxHeight;
	int boxX,boxY;
	total[threadIdx.y][threadIdx.x]=0.0f;
	boxX = bb[4*blockIdx.x+0];
	boxY = bb[4*blockIdx.x+1];
	boxWidth	= bb[4*blockIdx.x+2] - boxX;
	boxHeight	= bb[4*blockIdx.x+3] - boxY;
	int threadw=patchSize/steps;
		
	int x0,y0;	
	float px,py;
	float p0,p1,p2,p3;
	float px0,py0;
	float dx,dy;

	for(int i=0;i<threadw;i++){
		y0=threadIdx.y*threadw+i;
		py = (float)boxY + (float)y0*((float)boxHeight/(float)patchSize);
		
			for(int j=0;j<threadw;j++){
				x0= threadIdx.x*threadw+j;
				px =(float) boxX + (float)x0*((float)boxWidth/(float)patchSize);

				px0=floorf(px);
				py0=floorf(py);				
				  dx=px-px0;
				  dy=py-py0;

				 p0 = tex2D(textureWarpedImage,px0,py0);
				 p1 = tex2D(textureWarpedImage,px0+1.0f,py0);
				 p2 = tex2D(textureWarpedImage,px0+1.0f,py0+1.0f);
				 p3 = tex2D(textureWarpedImage,px0,py0+1.0f);
 	
				 patch[y0][x0]=(p0*(1.0f-dx)+p1*dx)*(1.0f-dy)+(p3*(1.0f-dx)+p2*dx)*dy;				
				 total[threadIdx.y][threadIdx.x]  += patch[y0][x0];
			}
	}
	
	__syncthreads();
		
	if(threadIdx.x==0){
		rowSum[threadIdx.y]=0.0f;
		
		for(int i=0;i<steps;i++){
			rowSum[threadIdx.y] += total[threadIdx.y][i];
		}
	}
	
	__syncthreads();

    if(threadIdx.x==0 && threadIdx.y==0){
		sum=0.0f;
		
		for(int i=0;i<steps;i++){
			sum += rowSum[i];
		}
	}
   	__syncthreads();
	sum /= (float)(patchSize*patchSize);


	for(int i=0;i<threadw;i++){
		y0=threadIdx.y*threadw+i;
        
		for(int j=0;j<threadw;j++){
			x0=threadIdx.x*threadw+j;
			patchout[blockIdx.x*2025+y0*patchSize+x0]= patch[y0][x0] - sum;
			}
		}
}

///////////// -------------------- TRACKER-------------------------------------/////////////////////

/*Calculate Normalized Crosscorelation between given data sets: for the calculation a 13 by 13 pixel window is taken around each pixel for the crosscorelation calculation*/
__global__ void find_ncc_kernel(int nPts,int Winsize,float* d_Ipts,float* d_Jpts,char* d_status,float* d_ncc,int w,int h)
{

	__shared__ float ixy[64][96];
	__shared__ float jxy[64][96];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(idx > nPts-1)
		return;

	int iSum = 0;
	int jSum = 0;
	
	float ix = d_Ipts[2*idx  ];
	float iy = d_Ipts[2*idx+1];
	float jx = d_Jpts[2*idx  ];
	float jy = d_Jpts[2*idx+1];
	int halfWinSize  = (Winsize-1)*0.5f;

	float jNew = 0.0f;
	float iNew = 0.0f;
	
	float numerator = 0.0f;
	float denominator = 0.0f;

	float jsqSum = 0.0f;
	float isqSum = 0.0f;

	if(d_status[idx] == 1) //corelation is calculated only if there is a valid optical flow calculation for that point
	{
		// get the summation of all points inside the window
		for(int y = -halfWinSize ; y <= halfWinSize ; y++)
		{
			for(int x = -halfWinSize ; x <= halfWinSize ; x++)
			{
				ixy[threadIdx.x][(y+halfWinSize)*Winsize + x+halfWinSize] = tex2D(textureCurrentFrame, ix + x, iy + y);
				jxy[threadIdx.x][(y+halfWinSize)*Winsize + x+halfWinSize] = tex2D(texturePreviousFrame, jx + x, jy + y);				
				iSum += ixy[threadIdx.x][(y+halfWinSize)*Winsize + x+halfWinSize];
				jSum += jxy[threadIdx.x][(y+halfWinSize)*Winsize + x+halfWinSize];

			}
		}


		// find T' and I' and 
		iSum = iSum/(float)(w*h);
		jSum = jSum/(float)(w*h);
		// find T' and I' and 

		for(int x = -halfWinSize ; x<= halfWinSize ; x++)
		{
			for(int y = -halfWinSize ; y <= halfWinSize ; y++)
			{			
				iNew = ixy[threadIdx.x][(y+halfWinSize)*Winsize + x+halfWinSize] - iSum; 

				jNew = jxy[threadIdx.x][(y+halfWinSize)*Winsize + x+halfWinSize] - jSum;

				numerator += (iNew * jNew); //find numerator

				isqSum += iNew*iNew;
				jsqSum += jNew*jNew;
			}
		}

		// find denominator
		float val = isqSum*jsqSum;
		denominator = sqrtf(val);


		d_ncc[idx] = (float)(numerator / denominator); //normalized crosscorelation
		
	}
	else
	{
		d_ncc[idx] = 0.0f; //if status is zero no point of calculating crosscorelation
	}
	__syncthreads();

}

/*Calculate Euclidian distance between 2 datasets*/
__global__ void calEuDistance(const int nPts,const float* ptsI,const float* ptsJ,float* distances){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx > nPts-1)
		return;

	float x = (ptsJ[2*idx]-ptsI[2*idx])*(ptsJ[2*idx]-ptsI[2*idx]);
	float y = (ptsJ[2*idx+1]-ptsI[2*idx+1])*(ptsJ[2*idx+1]-ptsI[2*idx+1]);
	distances[idx]  = sqrtf(x+y);
}


/*Convert image from RGB to Greay*/
__global__ void convertToGrey(unsigned char *d_in, float *d_out, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx < N) 
        d_out[idx] = d_in[idx*3]*0.1144f + d_in[idx*3+1]*0.5867f + d_in[idx*3+2]*0.2989f;
}

/*Downsampling the image to 0.5 size*/
__global__ void pyrDownsample(int w1, int h1, float *out, int w2, int h2)
{
    // Input has to be greyscale
    int x2 = blockIdx.x*blockDim.x + threadIdx.x;
    int y2 = blockIdx.y*blockDim.y + threadIdx.y;

    if( (x2 < w2) && (y2 < h2) ) {    
        int x = x2*2;
        int y = y2*2;
		
		float val = 0.25f*tex2D(texturePyramid,x,y);
			  val += 0.125f*tex2D(texturePyramid,x-1,y);
			  val += 0.125f*tex2D(texturePyramid,x-2,y);
			  val += 0.125f*tex2D(texturePyramid,x,y-1);
			  val += 0.125f*tex2D(texturePyramid,x,y-2);
			  val += 0.0625f*tex2D(texturePyramid,x-1,y-1);
			  val += 0.0625f*tex2D(texturePyramid,x-1,y-2);
			  val += 0.0625f*tex2D(texturePyramid,x-2,y-1);
			  val += 0.0625f*tex2D(texturePyramid,x-2,y-2);
        out[y2*w2 + x2] = val;
    }
} 

/*Smoothing row wise*/
__global__ void smoothX( float* imageout,int width,int height) {	

	const int Blocksize=252; 
	int x = blockIdx.x*Blocksize+threadIdx.x;
	int pX = threadIdx.x;
	int y = blockIdx.y;
	
	__shared__ unsigned char imageBlock[Blocksize+4];	
	 
	imageBlock[threadIdx.x]= tex2D(texturePyramid,(x-2),y);
	__syncthreads();

	if( x >= width) return;
	
	if( pX < 252){
		imageout[ y*width + x]= 
		0.0625f*imageBlock[pX++]+
		0.25f*imageBlock[pX++]+
		0.375f*imageBlock[pX++]+
		0.25f*imageBlock[pX++]+
		0.0625f*imageBlock[pX];	
		}		
}

/*Smoothing column wise*/
__global__ void smoothY( float* imageout,int width,int height) {

	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y =  blockIdx.y;
	
	if((x < width) && (y < height)){

		for(int i=0; i<SmoothYNoOfThreads; i++){
			int yIndex = SmoothYNoOfThreads*y+i-2;		 
			float val = 0.0625f*tex2D(texturePyramid,x,yIndex++);
			val += 0.25f  *tex2D(texturePyramid,x,yIndex++);
			val += 0.375f *tex2D(texturePyramid,x,yIndex++);
			val += 0.25f  *tex2D(texturePyramid,x,yIndex++);
			val += 0.0625f*tex2D(texturePyramid,x,yIndex);
			imageout[yIndex * width + x ] = val;
		}
	}		
}


/*Calculate Optical Flow*/
__global__ void track(const int nBoxes,const int width, const int scaledHeights, 
                      float pyramidScales, int level, char initialGuess, 
                      float *ptsJ, char *status,float* ptsI)
{  
	__shared__ float Ix[256];
	__shared__ float Iy[256];
	__shared__ float Ixy[256];
	__shared__ float SumIxx[256];
	__shared__ float SumIyy[256];
	__shared__ float SumIxy[256];
	__shared__ float SumIxt[256];
	__shared__ float SumIyt[256];
	__shared__ float tempX;
	__shared__ float tempY;
    __shared__ float predictedX;
    __shared__ float predictedY;
	__shared__ float determinant;
	__shared__ float inverseDeterminant;
	__shared__ int breakCondition;
	
	int x   = threadIdx.x;
	int y   = threadIdx.y;
	int idx = blockIdx.x;
	int index = y*blockDim.x + x;
	
	if((index <169) && (status[blockIdx.x] != 0)){

			float previousX = ptsI[2*idx]  *pyramidScales;
			float previousY = ptsI[2*idx+1]*pyramidScales;
			
			if(index == 0){
				if(initialGuess) {
					tempX = 0.0f;
					tempY = 0.0f;
					predictedX = previousX;
					predictedY = previousY;
				}
				else {
					predictedX = ptsJ[2*idx];
					predictedY = ptsJ[2*idx+1];
					tempX = predictedX - previousX;
					tempY = predictedY - previousY;
				}
			}
			
			if((index > 0) &&(index < 88)){
				SumIxx[index + 168] = 0.0f;
				SumIyy[index + 168] = 0.0f;
				SumIxy[index + 168] = 0.0f;
				SumIxt[index + 168] = 0.0f;
				SumIyt[index + 168] = 0.0f;
			}
			
			
			
			Ix[index]  = (tex2D(texturePreviousFrame, previousX + x-5, previousY + y-6) - tex2D(texturePreviousFrame, previousX + x-7, previousY + y-6)) * 0.5f;
			Iy[index]  = (tex2D(texturePreviousFrame, previousX + x-6, previousY + y-5) - tex2D(texturePreviousFrame, previousX + x-6, previousY + y-7)) * 0.5f;
			Ixy[index] = tex2D(texturePreviousFrame, previousX + x-6, previousY + y-6);
			
			SumIxx[index] = Ix[index] * Ix[index];
			SumIyy[index] = Iy[index] * Iy[index];
			SumIxy[index] = Ix[index] * Iy[index];
			
			__syncthreads();
			
			
			if(index < 128){
				SumIxx[index] += SumIxx[index+128];
				SumIxy[index] += SumIxy[index+128];
				SumIyy[index] += SumIyy[index+128];
			}
			
			__syncthreads();
						
			if(index  < 64){
				SumIxx[index] += SumIxx[index+64];
				SumIxy[index] += SumIxy[index+64];
				SumIyy[index] += SumIyy[index+64];
			}
			
			__syncthreads();

			if(index < 32){
				SumIxx[index] += SumIxx[index+32];
				SumIxy[index] += SumIxy[index+32];
				SumIyy[index] += SumIyy[index+32];
			}
			
			__syncthreads();

			if(index < 16){
				SumIxx[index] += SumIxx[index+16];
				SumIxy[index] += SumIxy[index+16];
				SumIyy[index] += SumIyy[index+16];
			}
			
			__syncthreads();
			
			if(index < 8){
				SumIxx[index] += SumIxx[index+8];
				SumIxy[index] += SumIxy[index+8];
				SumIyy[index] += SumIyy[index+8];
			}
			
			__syncthreads();
			
			if(index < 4){
				SumIxx[index] += SumIxx[index+4];
				SumIxy[index] += SumIxy[index+4];
				SumIyy[index] += SumIyy[index+4];
			}
			
			__syncthreads();
			
			if(index < 2){
				SumIxx[index] += SumIxx[index+2];
				SumIxy[index] += SumIxy[index+2];
				SumIyy[index] += SumIyy[index+2];
			}
			
			__syncthreads();
			
			if(index == 0){
				SumIxx[index] += SumIxx[index+1];
				SumIxy[index] += SumIxy[index+1];
				SumIyy[index] += SumIyy[index+1];
				determinant = SumIxx[index] * SumIyy[index] - SumIxy[index] * SumIxy[index];
				inverseDeterminant = 1.0f/determinant;
				
				breakCondition = 0;
				if(determinant < 0.00001f) {
					status[idx] = 0;
					breakCondition = 1;
				}
			}
			
			__syncthreads();
			
			if(breakCondition == 1) {
				return;
			}
			
			
			for(int i = 0; i<10; i++){
				if(index == 0){
					if(predictedX < 0 || predictedX > width || predictedY < 0 || predictedY > scaledHeights) {
						status[idx] = 0;
						 breakCondition = 1;
					}
				}
				
				__syncthreads();

				if( breakCondition == 1) return;
				
				float It = tex2D(textureCurrentFrame, predictedX + x - 6 , predictedY + y - 6) - Ixy[index];
				SumIxt[index] = Ix[index] * It;
				SumIyt[index] = Iy[index] * It;
					
				__syncthreads();
				
				
				if(index < 128){
					SumIxt[index] += SumIxt[index+128];
					SumIyt[index] += SumIyt[index+128];
				}
				
				__syncthreads();
							
				if(index  < 64){
					SumIxt[index] += SumIxt[index+64];
					SumIyt[index] += SumIyt[index+64];
				}
				
				__syncthreads();

				if(index < 32){
					SumIxt[index] += SumIxt[index+32];
					SumIyt[index] += SumIyt[index+32];
				}

				__syncthreads();

				if(index < 16){
					SumIxt[index] += SumIxt[index+16];
					SumIyt[index] += SumIyt[index+16];
				}
				
				__syncthreads();
				
				if(index < 8){
					SumIxt[index] += SumIxt[index+8];
					SumIyt[index] += SumIyt[index+8];
				}
				
				__syncthreads();
				
				if(index < 4){
					SumIxt[index] += SumIxt[index+4];
					SumIyt[index] += SumIyt[index+4];
				}
				
				__syncthreads();
				
				if(index < 2){
					SumIxt[index] += SumIxt[index+2];
					SumIyt[index] += SumIyt[index+2];
				}
				
				__syncthreads();
				
				if(index == 0){
					SumIxt[index] += SumIxt[index+1];
					SumIyt[index] += SumIyt[index+1];
					float vx = inverseDeterminant * ( -SumIyy[index] * SumIxt[index] + SumIxy[index] * SumIyt[index]);
					float vy = inverseDeterminant * (  SumIxy[index] * SumIxt[index] - SumIxx[index] * SumIyt[index]);

					tempX += vx;
					tempY += vy;
					predictedX += vx;
					predictedY += vy;
					
					breakCondition = 0;
					if(fabsf(vx) < 0.01f && fabsf(vy) < 0.01f) breakCondition = 1;
				}
				__syncthreads();	
				
				if(breakCondition == 1) break;
				
			}
			
			if(index == 0){
				    if(level != 0) {
						predictedX += predictedX;
						predictedY += predictedY;

						tempX += tempX;
						tempY += tempY;
					}

				ptsJ[2*idx]	  = predictedX;
				ptsJ[2*idx+1] = predictedY;
			}
		}
	}



//---------7x7 Image Blur Kernal Implementation----------------------------

//Image Blur Row wise
__global__ void imageBlurRowKernel( unsigned char* imageout,int width,int pitch) 
{	

	const int Blocksize=248;
	int x = blockIdx.x*Blocksize+threadIdx.x -4;
	int pX = threadIdx.x;
	unsigned int y = blockIdx.y;
	
	__shared__ unsigned char imageBlock[Blocksize+8];	
	
	
	if((x >=0) || (x < width)) imageBlock[threadIdx.x]= tex2D(textureImage,x,y);
	
	else if((( x >= -4) && (x < 0)) || ((x >= width)) && (x < (width+4))) imageBlock[threadIdx.x]=0.0; 
	
	if( x >= (width-4)) return;

	__syncthreads();
	
	if( pX < 248 ){
		imageout[ y*pitch + x + 4]= 
		0.0076f*imageBlock[pX++]+
		0.0361f*imageBlock[pX++]+
		0.1096f*imageBlock[pX++]+
		0.2134f*imageBlock[pX++]+
		0.2666f*imageBlock[pX++]+
		0.2134f*imageBlock[pX++]+
		0.1096f*imageBlock[pX++]+
		0.0361f*imageBlock[pX++]+
		0.0076f*imageBlock[pX];	
		}
		
}

//Image Blur Column wise
__global__ void imageBlurColKernel( unsigned char* imageout,int width,int height,int pitch) 
{

    int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y =  blockIdx.y;
	
	if((x < width) && (BlurNoOfThreads*y < height)){
		for(int i=0; i<BlurNoOfThreads; i++){
			int yIndex = BlurNoOfThreads*y+i-4;
			float val =  0.0076f*tex2D(textureRowBlurredImage,x,yIndex++);
				  val += 0.0361f*tex2D(textureRowBlurredImage,x,yIndex++);
				  val += 0.1096f*tex2D(textureRowBlurredImage,x,yIndex++);
				  val += 0.2134f*tex2D(textureRowBlurredImage,x,yIndex++);
				  val += 0.2666f*tex2D(textureRowBlurredImage,x,yIndex++);
				  val += 0.2134f*tex2D(textureRowBlurredImage,x,yIndex++);
				  val += 0.1096f*tex2D(textureRowBlurredImage,x,yIndex++);
				  val += 0.0361f*tex2D(textureRowBlurredImage,x,yIndex++);
				  val += 0.0076f*tex2D(textureRowBlurredImage,x,yIndex);
				imageout[ (BlurNoOfThreads*y+i) * pitch + x ] = val;									
		}
	}
		
}

/*--------------BB Overlap Kernel-----------------------------
Calculate fraction of overlap between the boxes of the grid and the bounding box 
*/
__global__ void bbOverLapKernal(float* bb,float* grid,int rowBBGrid,int colBBGrid,float* overlap)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx >= colBBGrid) return;

	float x11 = grid[rowBBGrid*idx   ];
	float y11 = grid[rowBBGrid*idx +1];
	float x12 = grid[rowBBGrid*idx +2];
	float y12 = grid[rowBBGrid*idx +3];

	float x21 = bb[0];
	float y21 = bb[1];
	float x22 = bb[2];
	float y22 = bb[3];

	float interSectWidth  = max((min(x12,x22) - max(x11,x21) + 1.0f),0.0f);
	float interSectHeight = max((min(y12,y22) - max(y11,y21) + 1.0f),0.0f);

	float intersection = interSectHeight * interSectWidth ; 
	float area1 = (x22 - x21 + 1.0f) * (y22 - y21 + 1.0f);
	float area2 = (x12 - x11 + 1.0f) * (y12 - y11 + 1.0f);

	overlap[idx] = intersection / (area1 + area2 - intersection);
   
}

/*Warp Image using a affine transform matrix*/
__global__ void transformKernel(unsigned char* warpedImage, int width, int height,int warpedImage_pitch, float* matrix, float bbPX,float bbPY, int bbH, int bbW) 
{	
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	if((x > bbW) || (y > bbH)) return;


	float x1 = bbPX + (float)x;
	float y1 = bbPY + (float)y;

	float xShift = matrix[2];
	float yShift = matrix[5];

	float u = x1 - xShift;
	float v = y1 - yShift;

	float x0 = (matrix[0]*u + matrix[1]*v + xShift);
	float y0 = (matrix[3]*u + matrix[4]*v + yShift);

	// read from texture and write to global memory
	warpedImage[ (int)(y1*warpedImage_pitch) + (int)x1] = tex2D(textureBlurredImage, x0, y0);// + randNoice[(int)y1*width + (int)x1];
}

//---------------Fern Implementation------------------------------------------------------

/*calculate feature patterns for positive samples*/
__global__ void calculatePatternsPositiveKernal(int* idxBoxs,int* patt,float* grid,float* featuresOffsets,int featuresOffsetStep,
										int noOfBoxes,int gridStep){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx >= noOfBoxes) return;

	float x = grid[idxBoxs[idx]*gridStep + 0];
	float y = grid[idxBoxs[idx]*gridStep + 1];
	float fx0,fx1,fy0,fy1;
	int scaleIndex  = grid[idxBoxs[idx]*gridStep + 4] * featuresOffsetStep;
	unsigned char f0,f1;
	int index;


	for(int tree=0; tree<10;tree++){
		index =0;
		int treeIndex = scaleIndex + 52*tree;
		for(int feature = 0;feature< 13;feature++){
			int featureIndex = treeIndex + 4*feature;
			fx0 = featuresOffsets[featureIndex    ];
			fy0 = featuresOffsets[featureIndex + 1];
			fx1 = featuresOffsets[featureIndex + 2];
			fy1 = featuresOffsets[featureIndex + 3];

			f0 = tex2D(textureWarpedImage,x+fx0,y+fy0); 
			f1 = tex2D(textureWarpedImage,x+fx1,y+fy1);

			index<<=1;
			if(f0>f1){
				index |= 1;
			}
		}
		patt[10*idx + tree] = index;
	}
}

/*Check the Varience of a given patch and if it is lower than a threshold status is set to zero*/
__global__ void calBoxVarienceKernal(int* idxBoxs,char* status,float* grid,int gridStep,float varienceThresh,int noOfBoxes){
	
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx >= noOfBoxes) return;

	int gridIndex = idxBoxs[idx];

	float x1 = grid[gridIndex*gridStep + 0];
	float y1 = grid[gridIndex*gridStep + 1];
	float x2 = grid[gridIndex*gridStep + 2];
	float y2 = grid[gridIndex*gridStep + 3];
	float area = (y2-y1+1)*(x2-x1+1);

	
	float mX = (tex2D(textureIntregalImage,x2,y2)-tex2D(textureIntregalImage,x2,y1)
				-tex2D(textureIntregalImage,x1,y2)+tex2D(textureIntregalImage,x1,y1))/area;
	
	float mX2 = (tex2D(textureIntregalSQImage,x2,y2)-tex2D(textureIntregalSQImage,x2,y1)
				-tex2D(textureIntregalSQImage,x1,y2)+tex2D(textureIntregalSQImage,x1,y1))/area;

	if((mX2- mX*mX) >= varienceThresh)
	{
		status[idx] = 1;
	} 
}

/*Calculate Feature Patterens for Negative Samples*/
__global__ void calculatePatternsNegatveKernal(int* idxBoxs,char* status,int* patt,float* grid,float* featuresOffsets,int featuresOffsetStep,
										int noOfBoxes,int gridStep){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx >= noOfBoxes) return;

	if(status[idx] == 0) {
		for(int tree=0; tree<10;tree++){
			patt[10*idx + tree] = 0; }
		return;}

	float x = grid[idxBoxs[idx]*gridStep + 0];
	float y = grid[idxBoxs[idx]*gridStep + 1];
	float fx0,fx1,fy0,fy1;
	int scaleIndex  = grid[idxBoxs[idx]*gridStep + 4] * featuresOffsetStep;
	unsigned char f0,f1;
	int index;


	for(int tree=0; tree<10;tree++){
		index =0;
		int treeIndex = scaleIndex + 52*tree;
		for(int feature = 0;feature< 13;feature++){
			int featureIndex = treeIndex + 4*feature;		
			fx0 = featuresOffsets[featureIndex    ];
			fy0 = featuresOffsets[featureIndex + 1];
			fx1 = featuresOffsets[featureIndex + 2];
			fy1 = featuresOffsets[featureIndex + 3];

			f0 = tex2D(textureWarpedImage,x+fx0,y+fy0);
			f1 = tex2D(textureWarpedImage,x+fx1,y+fy1);

			index<<=1;
			if(f0>f1){
				index |= 1;
			}
		}
		patt[10*idx + tree] = index;
	}
}

/*Update weights using Negative samples*/
__global__ void updateNegativeKernal(float* weights,int* nP,int* nN,int* nX,int pattStep,
											 int noOfPats,float updateThreshNegative){
	
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= noOfPats) return;
	
	int pattern[10];
	float confidenceVal =0.0f; 

	for(int tree =0; tree<pattStep; tree++){
		pattern[tree] = nX[idx*pattStep + tree];
		confidenceVal += weights[pattern[tree]*pattStep + tree];
	}
	if(confidenceVal >= updateThreshNegative){
		for(int tree =0; tree<pattStep; tree++){		
				nN[pattern[tree]*pattStep + tree]++;
				weights[pattern[tree]*pattStep + tree] = 
				((float)nP[pattern[tree]*pattStep + tree]/(float)(nP[pattern[tree]*pattStep + tree] + nN[pattern[tree]*pattStep + tree]));
		}
	}
}

/*Update weights using Positive samples*/
__global__ void updatePositiveKernal(float* weights,int* nP,int* nN,int* pX,int pattStep,
											 int noOfPats,float updateThreshPositive){
	
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= noOfPats) return;
	
	int pattern[10];
	float confidenceVal =0.0f; 

	for(int tree =0; tree<pattStep; tree++){
		pattern[tree] = pX[idx*pattStep + tree];
		confidenceVal += weights[pattern[tree]*pattStep + tree];
	}
	if(confidenceVal <= updateThreshPositive){
		for(int tree =0; tree<pattStep; tree++){
			nP[pattern[tree]*pattStep + tree]++;
			weights[pattern[tree]*pattStep + tree] = 
				((float)nP[pattern[tree]*pattStep + tree]/(float)(nP[pattern[tree]*pattStep + tree] + nN[pattern[tree]*pattStep + tree]));
		}
	}
}

__global__ void getConfidences_fern3_Kernal(float* confidence,float* weights,int* nX,int pattStep,int pattColSize)
{
	
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= pattColSize) return;
	
	int pattern;
	float confidenceVal = 0.0f; 

	for(int tree =0; tree<pattStep; tree++){
		pattern = nX[idx*pattStep + tree];
		confidenceVal += weights[pattern*pattStep + tree];
	}

	confidence[idx] = confidenceVal;
}

/*Calculate the feature vectors for each bounding box in Grid*/
__global__ void calculatePatternsKernal(int* patt,float* grid,float* featuresOffsets,int featuresOffsetStep,
										int gridColSize,int gridStep){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx >= gridColSize) return; 

	float x = grid[idx*gridStep + 0];
	float y = grid[idx*gridStep + 1];
	float fx0,fx1,fy0,fy1;
	int scaleIndex  = (int)(grid[idx*gridStep + 4]) * featuresOffsetStep;
	unsigned char f0,f1;
	int index;


	for(int tree=0; tree<10;tree++){
		int treeIndex = scaleIndex + 52*tree;
		index =0;

		for(int feature = 0;feature< 13;feature++){
			int featureOndex = treeIndex + 4*feature;
			fx0 = featuresOffsets[featureOndex    ];
			fy0 = featuresOffsets[featureOndex + 1];
			fx1 = featuresOffsets[featureOndex + 2];
			fy1 = featuresOffsets[featureOndex + 3];

			f0 = tex2D(textureWarpedImage,x+fx0,y+fy0);//get pixel value of Feature Points 
			f1 = tex2D(textureWarpedImage,x+fx1,y+fy1);

			index<<=1;

			if(f0 > f1){
				index |= 1;
			}
		}
		patt[10*idx + tree] = index;
		/*This is a Integer value of 13 bit binary number xxxxxxxxxxxxx where x =0 if f0 > f1 and x =1 otherwise*/
	}

}

/*Calculate confidence value for each bounding box in the grid based on generated 13bit feature pattern*/
__global__ void getConfidencesFromTreeKernal(float* confidence,float* grid,float* weights,int* patt,int pattStep,
											 int gridColSize,int gridStep,float varienceThresh){
	
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= gridColSize) return;
	
	int pattern;
	float x1,x2,y1,y2;
	float confidenceVal =0.0f;
	x1 = grid[idx*gridStep + 0];
	y1 = grid[idx*gridStep + 1];
	x2 = grid[idx*gridStep + 2];
	y2 = grid[idx*gridStep + 3];
	float area = (y2-y1+1)*(x2-x1+1);

	
	float mX = (tex2D(textureIntregalImage,x2,y2)-tex2D(textureIntregalImage,x2,y1)
				-tex2D(textureIntregalImage,x1,y2)+tex2D(textureIntregalImage,x1,y1))/area;
	
	float mX2 = (tex2D(textureIntregalSQImage,x2,y2)-tex2D(textureIntregalSQImage,x2,y1)
				-tex2D(textureIntregalSQImage,x1,y2)+tex2D(textureIntregalSQImage,x1,y1))/area;

	if((mX2- mX*mX) < varienceThresh) //Check whether the intensity of the box has significant varience(mX2- mX*mX) did to eliminate plain patches
	{
		confidence[idx] =0.0;
		return;
	}
	

	for(int tree =0; tree<10; tree++){
		pattern = patt[idx*pattStep + tree];
		confidenceVal += weights[pattern*10 + tree]; //confidence for the ten feature vectors are summed to get the confidence
	}

	confidence[idx] = confidenceVal;
}

//--------------- Sorting Result Kernel--------------------------------------------------//
__global__ void sort_fern(float* input,int* input_index,int size,int* sorted_index,int limit,float threshold)
{


    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(idx>=size)
	{
        return;
    }
   
    float val_2,val_1;
    val_1=input[idx];
   
    int tot=0;
	
	#pragma unroll
    for(int i=0;i<size;i++)
	{
        val_2=input[i];
        if(val_2>val_1 || (idx>i && val_2==val_1))
		{           
            tot++;
        }
        if(tot >= limit )
		{
            return;
        }
    }
    
	sorted_index[tot]=input_index[idx];
}

//-----------Normalised Cross Corelation Kernels ---------------------------------//

__global__ void ccorr_normed_nn(float *f1,float *f2,float *f3,int row_size,int f1_col,int f2_col)
{   
  int tid=threadIdx.y;
    __shared__ float f1_shared[1024];
    __shared__ float f2_shared[1024];   
    __shared__ float f1_f2_shared[1024];   

    if(tid<1012){
        int idx_2=floorf(blockIdx.y/f1_col)*row_size+2*tid;
        int idx_1=(blockIdx.y-floorf(blockIdx.y/f1_col)*f1_col)*row_size+2*tid;


        float a1=f1[idx_1];
        float b1=f2[idx_2];
        float a2=f1[idx_1+1];
        float b2=f2[idx_2+1];
   
        f1_f2_shared[tid]=a1*b1 +a2*b2;   
        f1_shared[tid]=a1*a1 +a2*a2 ;
        f2_shared[tid]=b1*b1 + b2*b2;
       
    }
    else if(tid==1012){
        int idx_2=floorf(blockIdx.y/f1_col)*row_size+2*tid;
        int idx_1=(blockIdx.y-floorf(blockIdx.y/f1_col)*f1_col)*row_size+2*tid;


        float a1=f1[idx_1];
        float b1=f2[idx_2];

        f1_f2_shared[tid]=a1*b1 ;   
        f1_shared[tid]=a1*a1 ;
        f2_shared[tid]=b1*b1 ;
    }

    else{

        f1_f2_shared[tid]=0.0;
        f2_shared[tid]=0.0;
        f1_shared[tid]=0.0;
    }

            __syncthreads();

			if(tid<512){
				f1_f2_shared[tid]=f1_f2_shared[tid]+f1_f2_shared[tid+512];
				f1_shared[tid]=f1_shared[tid]+f1_shared[tid+512];
				f2_shared[tid]=f2_shared[tid]+f2_shared[tid+512];
			}

			__syncthreads();
			if(tid<256){
				f1_f2_shared[tid]=f1_f2_shared[tid]+f1_f2_shared[tid+256];
				f1_shared[tid]=f1_shared[tid]+f1_shared[tid+256];
				f2_shared[tid]=f2_shared[tid]+f2_shared[tid+256];
			}

			__syncthreads();

			if(tid<128){
				f1_f2_shared[tid]=f1_f2_shared[tid]+f1_f2_shared[tid+128];
				f1_shared[tid]=f1_shared[tid]+f1_shared[tid+128];
				f2_shared[tid]=f2_shared[tid]+f2_shared[tid+128];
			}

			__syncthreads();

            if(tid<64){
                f1_f2_shared[tid]=f1_f2_shared[tid]+f1_f2_shared[tid+64];
                f1_shared[tid]=f1_shared[tid]+f1_shared[tid+64];
                f2_shared[tid]=f2_shared[tid]+f2_shared[tid+64];
            }

            __syncthreads();

           
            if(tid<32){
                f1_f2_shared[tid]=f1_f2_shared[tid]+f1_f2_shared[tid+32];
                f1_shared[tid]=f1_shared[tid]+f1_shared[tid+32];
                f2_shared[tid]=f2_shared[tid]+f2_shared[tid+32];
            }
            __syncthreads();
           
            if(tid<16){
                f1_f2_shared[tid]=f1_f2_shared[tid]+f1_f2_shared[tid+16];
                f1_shared[tid]=f1_shared[tid]+f1_shared[tid+16];
                f2_shared[tid]=f2_shared[tid]+f2_shared[tid+16];
            }
            __syncthreads();
            if(tid<8){
                f1_f2_shared[tid]=f1_f2_shared[tid]+f1_f2_shared[tid+8];
                f1_shared[tid]=f1_shared[tid]+f1_shared[tid+8];
                f2_shared[tid]=f2_shared[tid]+f2_shared[tid+8];
            }
            __syncthreads();
            if(tid<4){
                f1_f2_shared[tid]=f1_f2_shared[tid]+f1_f2_shared[tid+4];
                f1_shared[tid]=f1_shared[tid]+f1_shared[tid+4];
                f2_shared[tid]=f2_shared[tid]+f2_shared[tid+4];
            }
            __syncthreads();
            if(tid<2){
                f1_f2_shared[tid]=f1_f2_shared[tid]+f1_f2_shared[tid+2];
                f1_shared[tid]=f1_shared[tid]+f1_shared[tid+2];
                f2_shared[tid]=f2_shared[tid]+f2_shared[tid+2];
            }
            __syncthreads();
           
            if(tid==0 ){
                f3[blockIdx.y]=(((f1_f2_shared[0]+f1_f2_shared[1])/sqrt((f1_shared[0]+f1_shared[1])*(f2_shared[0]+f2_shared[1])))+1)*0.5;
            }
   
}

//-----------Confidance checking Kernel ---------------------------------//
__global__ void calc_ConfIsin(int n,float* maxP,int* iP,float* maxPP,float* maxN,int* isin,float* conf1,float* conf2){
	
int idx = blockDim.y*blockIdx.y + threadIdx.y;
    if(idx>=n){
        return;
    }

    if(maxP[idx]>0.95){
            isin[3*idx]=1;
    }
    else{
        isin[3*idx]=-1;
    }
    isin[3*idx+1]=iP[idx];

    if(maxN[idx]>0.95){
        isin[3*idx+2]=1;
    }
    else{
        isin[3*idx+2]=-1;
    }

    // measure Relative Similarity
        float dN = 1 - maxN[idx];
        float dP = 1 - maxP[idx];
        conf1[idx] = dN / (dN + dP);

     // measure Conservative Similarity        ???
        dP = 1 - maxPP[idx];
        conf2[idx] = dN / (dN + dP);

}




//-----------Finding maximum of negative data Kernel ---------------------------------//
__global__ void get_maxN(float* input,int size,float* maxN){


   int idx = blockDim.y*blockIdx.y + threadIdx.y;
    if(idx>=size){
        return;
    }
    float val_1,val_2;
    val_1=input[idx];

    for(int i=0;i<size;i++){
        val_2=input[i];
       
        if(!(val_1>val_2 || (val_2==val_1 && i>=idx))){
            return;
        }       
    }

   
    maxN[0]=val_1;
}




//-----------Finding maximum of the total and first half of positive data Kernel ---------------------------------//
__global__ void get_maxP_maxPP(float* input,int size,float* maxP,float* maxPP,int* iP){

    int idx = blockDim.y*blockIdx.y + threadIdx.y;
    if(idx>=size){
        return;
    }
    float val_1,val_2,val_3;
    val_1=input[idx];
    val_3=input[idx];
    bool set1=1,set2=1;

    for(int i=0;i<size;i++){
        val_2=input[i];
        if(i<ceilf(size*0.5) ){
            if(!((val_3>val_2 || (val_2==val_3 && i>=idx)))){
                set1=0;
            }
        }
        if(!(val_1>val_2 || (val_2==val_1 && i>=idx))){
            set2=0;
        }
        if(!set1 && !set2){
            return;
        }
       

    }

    if(set1==1 && idx<ceilf(size*0.5)){
        maxPP[0]=val_3;
    }
    if(set2==1){
    maxP[0]=val_1;
    iP[0]=idx;
    }
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CudaFunctions::CudaFunctions()
{

}

void CudaFunctions::freeMemory()
{
    for(int i=0; i < PYRAMID_LEVELS; i++) {
        cudaFree(gpuImagePyramidPrevious[i]);
        cudaFree(gpuImagePyramidCurrent[i]);
    }

    cudaFree(gpuSmoothedXPrevious);
    cudaFree(gpuSmoothedXCurrent);
    cudaFree(gpuSmoothedPrevious);
    cudaFree(gpuSmoothedCurrent);

    cudaFreeArray(gpuArrayPyramidPrevious);
    cudaFreeArray(gpuArrayPyramidCurrent);

    cudaFree(gpuPreviousRGBImage);
    cudaFree(gpuCurrentRGBImage);
    cudaFree(gpu_status);


    delete [] status;
}

void CudaFunctions:: cudaGetPattern(int num,float* pattern,float *bb){

	int threadX = steps;
	int threadY = steps;

	int blockX = num;
	int blockY = 1;
	dim3 blocks(blockX,blockY);
	dim3 threads(threadX,threadY);
	
	float* gpu_patchout;
	float* gpu_bb;

	cudaMalloc((void**)&gpu_patchout,patchSize*patchSize*num*sizeof(float));
	cudaMalloc((void**)&gpu_bb,4*num*sizeof(float));
	cutilCheckMsg("Error in cudaMalloc ...");
	
	cudaMemcpy(gpu_bb,bb,4*num*sizeof(float),cudaMemcpyHostToDevice);
	cutilCheckMsg("Error in cudaMemcpyToDevice...");	

	cudaBindTexture2D(0,&textureWarpedImage, gpu_warpedImage,&textureWarpedImage.channelDesc,width,height,gpu_warpedImage_pitch);
	cutilCheckMsg("Error in warpedImage binding...");
	getPatternKernel<<<blocks,threads>>>(gpu_patchout,gpu_bb);
	cutilCheckMsg("Error in getPatternKernel...");
	cudaMemcpy(pattern,gpu_patchout,patchSize*patchSize*num*sizeof(float),cudaMemcpyDeviceToHost);
	cutilCheckMsg("Error in cudaMemcpyToHost...");

	cudaFree(gpu_patchout);
	cudaFree(gpu_bb);

}

/*Generate Intregal Images*/
void CudaFunctions::doIntregal(){
	
	cudaBindTexture2D(0,&textureImage, gpu_image,&textureImage.channelDesc,width,height,gpu_image_pitch);
	cutilCheckMsg("Error in image binding...");

	rowIntegralKernel<<<h,512>>>(gpu_rowImage,w,h);
	cutilCheckMsg("Error in rowIntegralKernel...");

	rowIntegralSQKernel<<<h,512>>>(gpu_rowSQImage,w,h);
	cutilCheckMsg("Error in rowIntegralSQKernel...");
	
	colIntegralKernel<<<w,256>>>(gpu_rowImage,gpu_IntregalImage,w,h,gpu_IntregalImage_pitch/sizeof(float));
	cutilCheckMsg("Error in colIntegralKernel...");

	colIntegralSQKernel<<<w,256>>>(gpu_rowSQImage,gpu_IntregalSQImage,w,h,gpu_IntregalSQImage_pitch/sizeof(float));
	cutilCheckMsg("Error in colIntegralSQKernel...");

}


void CudaFunctions::freePointMemory(int nBoxes)
{
}


void CudaFunctions::initMemTrack(int _w,int _h)
{
	w = _w;
    h = _h;

    cudaMalloc((void**)&gpuPreviousRGBImage, sizeof(char)*w*h*3);
    cudaMalloc((void**)&gpuCurrentRGBImage, sizeof(char)*w*h*3);
    cudaMalloc((void**)&gpuImagePyramidPrevious[0], sizeof(float)*w*h);
    cudaMalloc((void**)&gpuImagePyramidCurrent[0], sizeof(float)*w*h);

    cudaMalloc((void**)&gpuSmoothedXPrevious, sizeof(float)*w*h);
    cudaMalloc((void**)&gpuSmoothedXCurrent, sizeof(float)*w*h);
    cudaMalloc((void**)&gpuSmoothedPrevious, sizeof(float)*w*h);
    cudaMalloc((void**)&gpuSmoothedCurrent, sizeof(float)*w*h);
	
	cudaMallocArray(&gpuArrayPyramidPrevious, &texturePreviousFrame.channelDesc, w, h);
    cudaMallocArray(&gpuArrayPyramidCurrent, &textureCurrentFrame.channelDesc, w, h);
	cudaMallocArray(&gpuArrayPyramid, &textureCurrentFrame.channelDesc, w, h);

    texturePreviousFrame.normalized = 0;
    texturePreviousFrame.filterMode = cudaFilterModeLinear;
    texturePreviousFrame.addressMode[0] = cudaAddressModeClamp;
    texturePreviousFrame.addressMode[1] = cudaAddressModeClamp;

    textureCurrentFrame.normalized = 0;
    textureCurrentFrame.filterMode = cudaFilterModeLinear;
    textureCurrentFrame.addressMode[0] = cudaAddressModeClamp;
    textureCurrentFrame.addressMode[1] = cudaAddressModeClamp;

	texturePyramid.normalized = 0;
    texturePyramid.filterMode = cudaFilterModeLinear;
    texturePyramid.addressMode[0] = cudaAddressModeClamp;
    texturePyramid.addressMode[1] = cudaAddressModeClamp;
	
	

    scaledWidths[0] = w;
    scaledHeights[0] = h;

    for(int i=1; i < PYRAMID_LEVELS; i++) {
        _w /= 2;
        _h /= 2;
        scaledWidths[i] = _w;
        scaledHeights[i] = _h;

        cudaMalloc((void**)&gpuImagePyramidPrevious[i], sizeof(float)*_w*_h);
        cudaMalloc((void**)&gpuImagePyramidCurrent[i], sizeof(float)*_w*_h);
    }
}

/*Process the initial Image frame for Tracking*/
void CudaFunctions::loadImageTrack(unsigned char *prev)
{
	
    int blocks1D = (w*h)/256 + (w*h % 256?1:0); // for greyscale
	
	cudaMemcpy(gpuPreviousRGBImage, prev, w*h*3, cudaMemcpyHostToDevice); 
	
	convertToGrey<<<blocks1D, 256>>>(gpuPreviousRGBImage, gpuImagePyramidPrevious[0], w*h);
	cudaThreadSynchronize();
    cutilCheckMsg("convertToGrey");
	
	for(int i=0; i < PYRAMID_LEVELS-1; i++) {
		
		//smoothX
		int length=252;	
		int blockSmoothXX = (scaledWidths[i]/length) + ((scaledWidths[i]%length)?1:0);
		int blockSmoothXY = scaledHeights[i];
		dim3 blocksSmoothX(blockSmoothXX,blockSmoothXY);

		cudaMemcpy2DToArray(gpuArrayPyramid, 0, 0, gpuImagePyramidPrevious[i], 
                           sizeof(float)*scaledWidths[i], sizeof(float)*scaledWidths[i], scaledHeights[i], cudaMemcpyDeviceToDevice);
		cudaBindTextureToArray(texturePyramid, gpuArrayPyramid);
        smoothX<<<blocksSmoothX, 256>>>(gpuSmoothedXPrevious, scaledWidths[i], scaledHeights[i]);
        cudaThreadSynchronize();
		cutilCheckMsg("smoothX here");
       
		//smoothY
		int blockSmoothYX = ( scaledWidths[i]/256) + (( scaledWidths[i]%256)?1:0);
		int blockSmoothYY = (scaledHeights[i]/SmoothYNoOfThreads) + ((scaledHeights[i]%SmoothYNoOfThreads)?1:0);
		dim3 blocksSmoothY(blockSmoothYX,blockSmoothYY);
		
		cudaMemcpy2DToArray(gpuArrayPyramid, 0, 0, gpuSmoothedXPrevious, 
                           sizeof(float)*scaledWidths[i], sizeof(float)*scaledWidths[i], scaledHeights[i], cudaMemcpyDeviceToDevice);		
		cudaBindTextureToArray(texturePyramid, gpuArrayPyramid);
		smoothY<<<blocksSmoothY, 256>>>(gpuSmoothedPrevious, scaledWidths[i], scaledHeights[i]);
        cudaThreadSynchronize();
		cutilCheckMsg("smoothY here");
		
		/////pyramid
		int nthreadsPyramid = 16;
		int blocksPyramidW = scaledWidths[i+1]/nthreadsPyramid + ((scaledWidths[i+1] % nthreadsPyramid)?1:0);
		int blocksPyramidH = scaledHeights[i+1]/nthreadsPyramid + ((scaledHeights[i+1] % nthreadsPyramid )?1:0);
		dim3 blocksPyramid(blocksPyramidW, blocksPyramidH);
		dim3 threadsPyramid(nthreadsPyramid, nthreadsPyramid);
		
		cudaMemcpy2DToArray(gpuArrayPyramid, 0, 0, gpuSmoothedPrevious, 
                           sizeof(float)*scaledWidths[i], sizeof(float)*scaledWidths[i], scaledHeights[i], cudaMemcpyDeviceToDevice);	
        cudaBindTextureToArray(texturePyramid, gpuArrayPyramid);
		pyrDownsample<<<blocksPyramid, threadsPyramid>>>(scaledWidths[i], scaledHeights[i], gpuImagePyramidPrevious[i+1], scaledWidths[i+1], scaledHeights[i+1]);
        cudaThreadSynchronize();
        cutilCheckMsg("pyrDownsample here");  
    }
}

void CudaFunctions::initPointSetMemoryTrack(int nBoxes)
{
	cudaMalloc((void**)&gpu_ptsFB, sizeof(float)*nBoxes*200);
	cudaMalloc((void**)&gpu_ptsJ, sizeof(float)*nBoxes*200);
	cudaMalloc((void**)&gpu_ptsI, sizeof(float)*nBoxes*200);
    cudaMalloc((void**)&gpu_status, sizeof(char)*nBoxes*100);
}

void CudaFunctions::run(intArrayStruct *confIdx,int2DArrayStruct patt,floatArrayStruct conf,float thresh,float var,float* ptsI,int nBoxes)
{	

    /*2 CUDA streams are created to carry on optical flow calculation and FERN detection functions simultaniously*/
	cudaStreamCreate(&tracking_stream);
	cudaStreamCreate(&fern_stream);

	int* gpu_patt;
	float* gpu_conf;
	cudaMalloc((void**)&gpu_patt,patt.cols*patt.rows*sizeof(int));
	cudaMalloc((void**)&gpu_conf,conf.size*sizeof(float));

	cudaMemsetAsync(gpu_conf,0,conf.size*sizeof(float),fern_stream);

	//page locking data in main memory for cudaMemcpyAsync 
	VirtualLock(patt.ptr, patt.cols*patt.rows*sizeof(int));
	VirtualLock(conf.ptr, conf.size*sizeof(float));
	VirtualLock(ptsI, 200*nBoxes*sizeof(float));
	VirtualLock(ptsJ,200*nBoxes*sizeof(float));
	VirtualLock(status,100*nBoxes*sizeof(char));

	cudaMemcpyAsync(gpu_ptsI, ptsI, 200*nBoxes*sizeof(float), cudaMemcpyHostToDevice,tracking_stream); 

	//tracking
	dim3 threadsTracking(13,13);
	int blocksTracking = 100;
   
////////////////////////////////fernDetection Detection/////////////////////////////////////////////////////////////////////////
	int threadsPatternsKernal = 256; 
	int blocksPatternsKernal = (gpu_grid.cols)/threadsPatternsKernal + (gpu_grid.cols % threadsPatternsKernal?1:0);
	int threadsConfidenceKernal = 256;
	int blocksConfidenceKernal = (gpu_grid.cols)/threadsConfidenceKernal + (gpu_grid.cols % threadsConfidenceKernal?1:0);

	cudaBindTexture2D(0,&textureWarpedImage, gpu_warpedImage,&textureWarpedImage.channelDesc,width,height,gpu_warpedImage_pitch);
	cutilCheckMsg("Error in warpedImage binding..."); 
 	calculatePatternsKernal<<<blocksPatternsKernal,threadsPatternsKernal,0,fern_stream>>>(gpu_patt, gpu_grid.ptr, gpu_featuresOffsets.ptr,gpu_featuresOffsets.cols, gpu_grid.cols, gpu_grid.rows);
	cutilCheckMsg("Error in calculatePatternsKernal"); 

    cudaStreamSynchronize(fern_stream);
	
	cudaBindTexture2D(0,&textureIntregalImage,gpu_IntregalImage,&textureIntregalImage.channelDesc,width,height,gpu_IntregalImage_pitch);
	cutilCheckMsg("Error in BindTexture from gpu_IntregalImage...");
	cudaBindTexture2D(0,&textureIntregalSQImage,gpu_IntregalSQImage,&textureIntregalSQImage.channelDesc,width,height,gpu_IntregalSQImage_pitch);
	cutilCheckMsg("Error in BindTexture from gpu_IntregalSQImage...");

	getConfidencesFromTreeKernal<<<blocksConfidenceKernal,threadsConfidenceKernal,0,fern_stream>>>(gpu_conf,gpu_grid.ptr,weights,gpu_patt,patt.rows,gpu_grid.cols,gpu_grid.rows,var);
	cutilCheckMsg("Error in getConfidencesFromTreeKernal(in Run)...");	
	
	cudaMemcpyAsync(conf.ptr,gpu_conf, conf.size*sizeof(float),cudaMemcpyDeviceToHost,fern_stream);
	cudaMemcpyAsync(patt.ptr,gpu_patt, patt.cols*patt.rows*sizeof(int),cudaMemcpyDeviceToHost,fern_stream);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 
    for(int i=0; i < PYRAMID_LEVELS-1; i++) {
		
		//Image is blured before optical flow calculation and the bluring matrix is decomposed to 2 vectors and smoothing is done in X and Y directions respectively 
		
		//X direction Smoothing
		int length=252;	
		int blockSmoothXX = (scaledWidths[i]/length) + ((scaledWidths[i]%length)?1:0);
		int blockSmoothXY = scaledHeights[i];
		dim3 blocksSmoothX(blockSmoothXX,blockSmoothXY);

		cudaMemcpy2DToArrayAsync(gpuArrayPyramid, 0, 0, gpuImagePyramidCurrent[i], 
                           sizeof(float)*scaledWidths[i], sizeof(float)*scaledWidths[i], scaledHeights[i], cudaMemcpyDeviceToDevice,tracking_stream);	
		cudaBindTextureToArray(texturePyramid, gpuArrayPyramid);
		smoothX<<<blocksSmoothX, 256,0,tracking_stream>>>(gpuSmoothedXCurrent, scaledWidths[i], scaledHeights[i]);
		cudaStreamSynchronize(tracking_stream);
		cutilCheckMsg("smoothX ERROR");
       
		//Y direction Smoothing
		int blockSmoothYX = ( scaledWidths[i]/256) + (( scaledWidths[i]%256)?1:0);
		int blockSmoothYY = (scaledHeights[i]/SmoothYNoOfThreads) + ((scaledHeights[i]%SmoothYNoOfThreads)?1:0);
		dim3 blocksSmoothY(blockSmoothYX,blockSmoothYY);
		
		cudaMemcpy2DToArrayAsync(gpuArrayPyramid, 0, 0, gpuSmoothedXCurrent, 
                           sizeof(float)*scaledWidths[i], sizeof(float)*scaledWidths[i], scaledHeights[i], cudaMemcpyDeviceToDevice,tracking_stream);	
		cudaBindTextureToArray(texturePyramid, gpuArrayPyramid);
		smoothY<<<blocksSmoothY, 256,0,tracking_stream>>>( gpuSmoothedCurrent, scaledWidths[i], scaledHeights[i]);
		cudaStreamSynchronize(tracking_stream);
		cutilCheckMsg("smoothY ERROR");
		
		
		//Create Image Pyramids
		int nthreadsPyramid = 16;
		int blocksPyramidW = scaledWidths[i+1]/nthreadsPyramid + ((scaledWidths[i+1] % nthreadsPyramid)?1:0);
		int blocksPyramidH = scaledHeights[i+1]/nthreadsPyramid + ((scaledHeights[i+1] % nthreadsPyramid )?1:0);
		dim3 blocksPyramid(blocksPyramidW, blocksPyramidH);
		dim3 threadsPyramid(nthreadsPyramid, nthreadsPyramid);

		cudaMemcpy2DToArrayAsync(gpuArrayPyramid, 0, 0, gpuSmoothedCurrent, 
                           sizeof(float)*scaledWidths[i], sizeof(float)*scaledWidths[i], scaledHeights[i], cudaMemcpyDeviceToDevice,tracking_stream);	
		cudaBindTextureToArray(texturePyramid, gpuArrayPyramid);


		pyrDownsample<<<blocksPyramid, threadsPyramid,0,tracking_stream>>>(scaledWidths[i], scaledHeights[i], gpuImagePyramidCurrent[i+1],  scaledWidths[i+1], scaledHeights[i+1]);
		cudaStreamSynchronize(tracking_stream);
        cutilCheckMsg("pyrDownsample ERROR");  
    }
	
	
    //set initial status of all points to Valid
    cudaMemsetAsync(gpu_status, 1, sizeof(char)*100*nBoxes,tracking_stream);
	float *tempSwap;



    // Do the Optical Flow Calculation For the Image Pyramids from coarse to fine 	
    for(int l=PYRAMID_LEVELS-1; l >= 0; l--) {
        cudaMemcpy2DToArrayAsync(gpuArrayPyramidPrevious, 0, 0, gpuImagePyramidPrevious[l], 
                           sizeof(float)*scaledWidths[l], sizeof(float)*scaledWidths[l], scaledHeights[l], cudaMemcpyDeviceToDevice,tracking_stream);
        cudaMemcpy2DToArrayAsync(gpuArrayPyramidCurrent, 0, 0, gpuImagePyramidCurrent[l], 
                            sizeof(float)*scaledWidths[l], sizeof(float)*scaledWidths[l], scaledHeights[l], cudaMemcpyDeviceToDevice,tracking_stream);

		cudaBindTextureToArray(texturePreviousFrame, gpuArrayPyramidPrevious);
		cudaBindTextureToArray(textureCurrentFrame, gpuArrayPyramidCurrent);
        track<<<blocksTracking, threadsTracking,0,tracking_stream>>>(nBoxes, scaledWidths[l], scaledWidths[l], pyramidScales[l], l, (l == PYRAMID_LEVELS-1), gpu_ptsJ, gpu_status,gpu_ptsI);
		cutilCheckMsg("ERROR in RUN(Optical Flow Calculation)");  
		cudaStreamSynchronize(tracking_stream);

		//swap Images: New Image are copied to gpuImagePyramidCurrent and after optical flow calculation 
		//it's pointers are swapped to make it gpuImagePyramidPrevious 
		tempSwap = gpuImagePyramidPrevious[l];
		gpuImagePyramidPrevious[l] = gpuImagePyramidCurrent[l];
		gpuImagePyramidCurrent[l] = tempSwap;
    }

	//copying predicted point set and status of validity to main memory
	cudaMemcpyAsync(ptsJ, gpu_ptsJ, sizeof(float)*200*nBoxes, cudaMemcpyDeviceToHost,tracking_stream);  
    cudaMemcpyAsync(status, gpu_status, sizeof(char)*100*nBoxes, cudaMemcpyDeviceToHost,tracking_stream);

	cudaStreamSynchronize(tracking_stream);
	cudaStreamDestroy(tracking_stream);

	cudaStreamSynchronize(fern_stream);
	cudaStreamDestroy(fern_stream);

	sort_data(conf.ptr,confIdx,conf.size,thresh);
	
	//unlocking locked memory
	VirtualUnlock(ptsI, 200*nBoxes*sizeof(float));
	VirtualUnlock(ptsJ,200*nBoxes*sizeof(float));
	VirtualUnlock(status,100*nBoxes*sizeof(char));
	VirtualUnlock(patt.ptr, patt.cols*patt.rows*sizeof(int));
	VirtualUnlock(conf.ptr, conf.size*sizeof(float));

	cudaFree(gpu_patt);
	cudaFree(gpu_conf);
}

/*calculate Optical flow in reverse direction*/
void CudaFunctions::run_FB(int nBoxes)
{
	dim3 threadsTracking(13,13);
	int blocksTracking = 100;

    for(int l=PYRAMID_LEVELS-1; l >= 0; l--) {

        cudaMemcpy2DToArray(gpuArrayPyramidPrevious, 0, 0, gpuImagePyramidPrevious[l], 
                           sizeof(float)*scaledWidths[l], sizeof(float)*scaledWidths[l], scaledHeights[l], cudaMemcpyDeviceToDevice);
        cudaMemcpy2DToArray(gpuArrayPyramidCurrent, 0, 0, gpuImagePyramidCurrent[l], 
                            sizeof(float)*scaledWidths[l], sizeof(float)*scaledWidths[l], scaledHeights[l], cudaMemcpyDeviceToDevice);
        
		cudaBindTextureToArray(texturePreviousFrame, gpuArrayPyramidPrevious);
		cudaBindTextureToArray(textureCurrentFrame, gpuArrayPyramidCurrent);

		//calculate optical flow in reverse due to previous swapping now the gpuImagePyramidCurrent points to the previous frame 
		track<<<blocksTracking, threadsTracking>>>(nBoxes, scaledWidths[l], scaledWidths[l], pyramidScales[l], l, (l == PYRAMID_LEVELS-1), gpu_ptsFB, gpu_status,gpu_ptsJ);
		cutilCheckMsg("ERROR in RUN FB(Optical Flow Calculation)");  
        cudaThreadSynchronize();
    }

}

/*Calculate Euclidian distance between actual point set(gpu_ptsI) and point
set predicted by backward optical flow calculatio (gpu_ptsFB)*/
void CudaFunctions::euclidianDistance(int nPts)
{
	float *gpu_distance;
	cudaMalloc((void**)&gpu_distance,nPts*sizeof(float));

	int nThread = 256;
	int nBlock = nPts/nThread + ((nPts % nThread)? 1:0);

	calEuDistance<<<nBlock,nThread>>>(nPts,gpu_ptsI,gpu_ptsFB,gpu_distance);
	cutilCheckMsg("error in EUDISTANCE"); 

	cudaMemcpy(fb,gpu_distance,nPts*sizeof(float),cudaMemcpyDeviceToHost);
	cudaFree(gpu_distance);

}

/*Calculate Normalized Crosscorelation between actual point set(ptsI) 
and point set predicted by Optical Flow calculation (ptsJ)*/
void CudaFunctions::doNormalizedCrossCorr(int nPts, int Winsize)
{
	float *d_Ipts;
	float *d_Jpts;
	
	char* d_status;
	float* d_ncc;

	// allocate meory for points in device..	
	cudaMalloc((void**)&d_Ipts,nPts*2*sizeof(float));
	cudaMalloc((void**)&d_Jpts,nPts*2*sizeof(float));

	// copy points to device..
	cudaMemcpy(d_Ipts,ptsI,nPts*2*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_Jpts,ptsJ,nPts*2*sizeof(float),cudaMemcpyHostToDevice);

	// allocate memory for status in deivce..
	size_t statusSize = nPts*sizeof(char);
	cudaMalloc((void**)&d_status,statusSize);

	// copy status to device..
	cudaMemcpy(d_status,status,statusSize,cudaMemcpyHostToDevice);


	// allocate memory for ncc in device..
	size_t nccSize = nPts*sizeof(float);
	cudaMalloc((void**)&d_ncc,nccSize);

	// create threadblock..
	int nccThread = 64;
	int nccBlock = nPts/nccThread + ((nPts % nccThread)? 1:0);

	find_ncc_kernel<<<nccBlock,nccThread>>>(nPts,Winsize,d_Ipts,d_Jpts,d_status,d_ncc,w, h);
	cutilCheckMsg("NCC here");
	cudaMemcpy(ncc,d_ncc,nccSize,cudaMemcpyDeviceToHost);

	cudaFree(d_Ipts);
	cudaFree(d_Jpts);
	cudaFree(d_status);
	cudaFree(d_ncc);
}

/*Calculte Pointset(ptsI) for bounding Box*/
float* CudaFunctions::bb_points(float* bb,int numM,int numN,float margin)
{
	float tempX=bb[0]+margin;
	float tempY=bb[1]+margin;

	float stepW = (bb[2] - bb[0] - 2*margin) / (float)(numN - 1);
	float stepH = (bb[3] - bb[1] - 2*margin) / (float)(numM - 1);

	float* bbP = (float*)malloc(2*numN*numM*sizeof(float));
	float* tempbbP = bbP; 
	for(int i=0 ;i<numN;i++)
	{
		for(int j=0;j<numM;j++)
		{
			*tempbbP++ = tempX+i*stepW;
			*tempbbP++ = tempY+j*stepH;

		}
	}
	return bbP;

}

/*Calculate median of Given Dataset of size nPts*/
float CudaFunctions::median(float arr[],int nPts)
{
	if(nPts == 0)
		return 0;

	sortMedian(arr,nPts);
	int med=nPts/2;
	if(nPts%2==0)
		return (arr[med-1]+arr[med])/2;
	else
		return arr[med];

}

/*Calculate Pairwise Distance of given 2 datasets of size nPts*/
float CudaFunctions::pdist(float pts0[],float pts1[],int nPts)
{
	if(nPts == 0) return 1;

	int size = nPts*(nPts-1)/2;
	float *pdistArr= (float*)malloc(size*sizeof(float));
	int k=0;
	for(int i=0;i<nPts;i++){
		for(int j =i+1 ;j<nPts;j++){
			pdistArr[k]=sqrt(pow((pts1[2*i]-pts1[2*j]),2)+pow((pts1[2*i+1]-pts1[2*j+1]),2))/sqrt(pow((pts0[2*i]-pts0[2*j]),2)+pow((pts0[2*i+1]-pts0[2*j+1]),2));
			k++;
		}
	}
	float pdistMedian=median(pdistArr,k);
	free(pdistArr);
	return pdistMedian;	

}

/*Sort only to midpoint using bubble sort sufficent to get the mediant*/
void CudaFunctions::sortMedian(float arr[], int end)
{
	float t;
	for (int i=0;i<(end/2+1);i++){
		for(int j=i+1;j<end;j++){
			if(arr[i]>arr[j]){
				t=arr[i]; arr[i]=arr[j]; arr[j]=t;
			}
		}
	}

}

/*Predict the Next Bounding Box*/
void CudaFunctions::bb_predict(float* bb0,float* bb1, int nPts,float *medFB){
	
	float *pt0Crop = (float*)malloc(2*nPts*sizeof(float));
	float *pt1Crop = (float*)malloc(2*nPts*sizeof(float));
	float *temFb = (float*)malloc(nPts*sizeof(float));
	float *temNcc = (float*)malloc(nPts*sizeof(float));
	float *xArr = (float*)malloc(nPts*sizeof(float));
	float *yArr = (float*)malloc(nPts*sizeof(float));
   
	memcpy(temFb,fbIN,nPts*sizeof(float));
	memcpy(temNcc,nccIN,nPts*sizeof(float));

	float medianFbJ=median(temFb,nPts); //meidan of Foward Backward ERROR
    float medianNccJ=median(temNcc,nPts); //median of Normalized Crosscorelation
	
	*medFB = medianFbJ;
    
	//Choose Point wich have a less Foward Backward Error than medianFbJ and High Normalized Crosscorelation than medianNccJ
	int j=0;
	for (int i = 0; i < nPts; i++) {
		if(fbIN[i] <= medianFbJ && nccIN[i] >= medianNccJ){
			xArr[j] = ptsJIN[2*i]   - ptsIIN[2*i];
			yArr[j] = ptsJIN[2*i+1] - ptsIIN[2*i+1];
			pt0Crop[2*j] = ptsIIN[2*i];  pt0Crop[2*j+1] = ptsIIN[2*i+1];
			pt1Crop[2*j] = ptsJIN[2*i];  pt1Crop[2*j+1] = ptsJIN[2*i+1];
			j++;
		}
	}
	nPts = j;

	float medianDist = pdist(pt0Crop,pt1Crop,nPts); //pairwise didtance of pointset I and point set J (measure of scale change)
	float medianX = median(xArr,nPts); //median of X shift
    float medianY = median(yArr,nPts); //median of Y shift

    
	free(xArr);
	free(yArr);
	free(pt0Crop);
	free(pt1Crop);
	free(temFb);
	free(temNcc);
   
    float s1  = 0.5*(medianDist-1)*(float)(bb0[2] - bb0[0]); //scale change in X direction
    float s2  = 0.5*(medianDist-1)*(float)(bb0[3] - bb0[1]); //scale change in Y direction	
	
	/*Predicting Next Bounding Box 0.5 is added to compensate for the loss of precision in Float to Int conversion*/
	bb1[0]      = floorf(bb0[0] +medianX + 0.5 -s1);
	bb1[1]     	= floorf(bb0[1] +medianY + 0.5 -s2);
	bb1[2]  	= floorf(bb0[2] +medianX + 0.5 +s1);
	bb1[3]		= floorf(bb0[3] +medianY + 0.5 +s2);
}


void CudaFunctions::doLK(unsigned char *cur,float* bbIn,float* bbOut,float *medFB,intArrayStruct *confIdx, int2DArrayStruct patt,floatArrayStruct conf,float thresh,float var)
{
	/*100 (10 x 10) sample points from the bounding box is taken*/
	int numM =10; 
	int numN = 10;
	float margin =5.0;
	int nPts = 100;

	ptsI = bb_points(bbIn,numM,numN,margin); //get cordinates of sample points x(n) = ptsI[2n], y[n] = ptsI[2n+1]
	ptsJ = (float*) malloc(2*nPts*sizeof(float));

	status = (char*)  malloc(nPts*sizeof(char));
	ncc    = (float*) malloc(nPts*sizeof(float));
	fb     = (float*) calloc(nPts,sizeof(float));
	memset(status,1,nPts*sizeof(char));

	ptsIIN = (float*) malloc(2*nPts*sizeof(float));
	ptsJIN = (float*) malloc(2*nPts*sizeof(float));
	fbIN   = (float*) malloc(nPts*sizeof(float));
	nccIN  = (float*) malloc(nPts*sizeof(float));


	run(confIdx,patt,conf,thresh,var,ptsI); //calculate optical flow
	run_FB(); //Calculate optical flow backward
	doNormalizedCrossCorr(nPts); //calculate Normalize Crosscorelation	
	euclidianDistance(nPts); //Calculate Euclidian Distance 

	
	

	int j =0; 
	for(int i=0;i<100;i++)
	{
		if (status[i] == 1) 
		{

			ptsIIN[2*j]	= ptsI[2*i];	// x val
			ptsIIN[2*j+1]= ptsI[2*i+1];	// y val

			ptsJIN[2*j]		= ptsJ[2*i];	// x val
			ptsJIN[2*j+1]	= ptsJ[2*i+1];	// y val

			fbIN[j]  =  fb[i];
			nccIN[j++] = ncc[i];
		} 
	}
	nPts = j;

	free(ptsI);
	free(ptsJ);
	free(fb);
	free(ncc);
	free(status);
	

	bb_predict(bbIn, bbOut ,nPts,medFB); //predict the next bounding box


	free(ptsIIN);
	free(ptsJIN);
	free(fbIN);
	free(nccIN);	
}

///////////////////////////-----------CUDA FERN Definitions---------------------//////////////////////////////


void CudaFunctions::initializeFern(float2DArrayStruct grid, float2DArrayStruct featureOffsets, int w, int h)
{	
	initMem_fern(w,h);
	loadGrid(grid);
	loadFeatureOffsets(featureOffsets);
}

/*Initialize Fern's Memory*/
void CudaFunctions::initMem_fern(int w,int h)
{
	width  = w;
	height = h;
	
	cudaMalloc((void**)&weights,81920*sizeof(float));
	cutilCheckMsg("Error in creating weights...");
	cudaMalloc((void**)&nP,81920*sizeof(int));
	cutilCheckMsg("Error in creating nP...");
	cudaMalloc((void**)&nN,81920*sizeof(int));
	cutilCheckMsg("Error in creating nN...");

	cudaMemset(weights,0,81920*sizeof(float));
	cutilCheckMsg("Error in initializing weights...");
	cudaMemset(nP,0,81920*sizeof(int));
	cutilCheckMsg("Error in initializing nP...");
	cudaMemset(nN,0,81920*sizeof(int));
	cutilCheckMsg("Error in initializing nN...");


	cudaMallocPitch((void**)&gpu_image,&gpu_image_pitch,width*sizeof(unsigned char),height);
	cudaMallocPitch((void**)&gpu_blurImageRow,&gpu_blurImageRow_pitch,width*sizeof(unsigned char),height);
	cudaMallocPitch((void**)&gpu_blurImage,&gpu_blurImage_pitch,width*sizeof(unsigned char),height);
	cudaMallocPitch((void**)&gpu_warpedImage,&gpu_warpedImage_pitch,width*sizeof(unsigned char),height);
	cudaMalloc((void**)&gpu_rowImage,width*height*sizeof(float));
	cudaMallocPitch((void**)&gpu_IntregalImage,&gpu_IntregalImage_pitch,width*sizeof(float),height);
	cudaMalloc((void**)&gpu_rowSQImage,width*height*sizeof(float));
	cudaMallocPitch((void**)&gpu_IntregalSQImage,&gpu_IntregalSQImage_pitch,width*sizeof(float),height);
	cutilCheckMsg("Error in cudaMalloc ...");

	textureImage.normalized = 0;
	textureImage.filterMode = cudaFilterModePoint;
    textureImage.addressMode[0] = cudaAddressModeClamp;
    textureImage.addressMode[1] = cudaAddressModeClamp;
	
	textureBlurredImage.normalized = 0;
	textureBlurredImage.filterMode = cudaFilterModePoint;
    textureBlurredImage.addressMode[0] = cudaAddressModeBorder;
    textureBlurredImage.addressMode[1] = cudaAddressModeBorder;

	textureWarpedImage.normalized = 0;
	textureWarpedImage.filterMode = cudaFilterModePoint;
    textureWarpedImage.addressMode[0] = cudaAddressModeClamp;
    textureWarpedImage.addressMode[1] = cudaAddressModeClamp;

	textureRowBlurredImage.normalized = 0;
	textureRowBlurredImage.filterMode = cudaFilterModePoint;
	textureRowBlurredImage.addressMode[0] = cudaAddressModeClamp;
	textureRowBlurredImage.addressMode[1] = cudaAddressModeClamp;

	textureIntregalImage.normalized = 0;
	textureIntregalImage.filterMode = cudaFilterModePoint;
    textureIntregalImage.addressMode[0] = cudaAddressModeClamp;
    textureIntregalImage.addressMode[1] = cudaAddressModeClamp;

	textureIntregalSQImage.normalized = 0;
	textureIntregalSQImage.filterMode = cudaFilterModePoint;
    textureIntregalSQImage.addressMode[0] = cudaAddressModeClamp;
    textureIntregalSQImage.addressMode[1] = cudaAddressModeClamp;

}

/*Load Grid Points from Main Memory to GPU Global memory*/
void CudaFunctions::loadGrid(float2DArrayStruct grid)
{

	gpu_grid.cols = grid.cols;
	gpu_grid.rows = grid.rows;

	cudaMalloc((void**)&gpu_grid.ptr,grid.cols*grid.rows*sizeof(float));
	cudaMemcpy(gpu_grid.ptr, grid.ptr, grid.cols*grid.rows*sizeof(float), cudaMemcpyHostToDevice);	
	cutilCheckMsg("Error in loading Grid...");
	
}

/*Copy Generated Feature Offsets to GPU Memory*/
void CudaFunctions::loadFeatureOffsets(float2DArrayStruct featureOffsets)
{
	gpu_featuresOffsets.cols = featureOffsets.cols; //520 = 10 Feature Vectors x 13 Features per Vector x 4 Cordinates(2 points) per Feature
	gpu_featuresOffsets.rows = featureOffsets.rows; //num of bounding box sizes

	cudaMalloc((void**)&gpu_featuresOffsets.ptr,featureOffsets.cols*featureOffsets.rows*sizeof(float));
	cudaMemcpy(gpu_featuresOffsets.ptr, featureOffsets.ptr, featureOffsets.cols*featureOffsets.rows*sizeof(float), cudaMemcpyHostToDevice);
	cutilCheckMsg("Error in loading featureOffsets...");

}

/*Load Image initialy*/
void CudaFunctions::loadImage(unsigned char* imageData)
{
	int blocks1D = (width*h)/256 + (width*height % 256?1:0); //threads for greyscale

	
	cudaMemcpy(gpuCurrentRGBImage, imageData, width*height*3, cudaMemcpyHostToDevice); //copy RGB image to GPU main memory

	convertToGrey<<<blocks1D, 256>>>(gpuCurrentRGBImage, gpuImagePyramidCurrent[0], width*height);
	cudaThreadSynchronize();
    cutilCheckMsg("convertToGrey");

	dim3 threadFloatToCharKernal(16,16);
	int blockY = (height/16) + ((height%16)?1:0);
	int blockX = (width/16) + ((width%16)?1:0);;
	dim3 blockFloatToCharKernal(blockX,blockY);
	floatToCharKernal<<<blockFloatToCharKernal, threadFloatToCharKernal>>>(gpuImagePyramidCurrent[0],gpu_image,height,width,gpu_image_pitch);
	cutilCheckMsg("convert FloatToChar");

	doImageBlurRow(width,  height);
	cutilCheckMsg("Error in gpu_blurImage binding...");

	doImageBlurCol(width, height);

	cudaMemcpy2D(gpu_warpedImage,gpu_warpedImage_pitch,gpu_blurImage,gpu_blurImage_pitch,sizeof(unsigned char)*width,
				height,cudaMemcpyDeviceToDevice);
	cutilCheckMsg("Error in warpedImage loading...");
	
	doIntregal();

}

void CudaFunctions:: doImageBlurRow( int w, int h)
{

	int length=248;
	int threadX = 256;
	int threadY = 1;
	
	int blockX = (w/length) + ((w%length)?1:0);
	int blockY = h;
	
	dim3 blocks(blockX,blockY);
	dim3 threads(threadX,threadY);	

	cudaBindTexture2D(0,&textureImage, gpu_image,&textureImage.channelDesc,width,height,gpu_image_pitch);
	cutilCheckMsg("Error in image binding...");
	imageBlurRowKernel<<<blocks,threads>>>(gpu_blurImageRow,w,gpu_blurImageRow_pitch/sizeof(unsigned char));
	cutilCheckMsg("Error in imageBlurRowKernel...");

}

void CudaFunctions:: doImageBlurCol(int w, int h)
{


	int length=256;
	int blockX = (w/length) + ((w%length)?1:0);
	int blockY = (h/BlurNoOfThreads) + ((h%BlurNoOfThreads)?1:0);
	dim3 blocks(blockX,blockY);

	cudaBindTexture2D(0,&textureRowBlurredImage, gpu_blurImageRow,&textureRowBlurredImage.channelDesc,width,height,gpu_blurImageRow_pitch);	
	imageBlurColKernel<<<blocks,length>>>(gpu_blurImage,w,h,gpu_blurImage_pitch/sizeof(unsigned char));
	cutilCheckMsg("Error in imageBlurColKernel...");

}


void CudaFunctions::calcBBOverLapWithGrid(float* bb,int sizeBB,float* overlap)
{

	float *gpu_bb;
	float *gpu_overlap;

	cudaMalloc((void**)&gpu_bb, sizeof(float)*sizeBB*4);
	cudaMemcpy(gpu_bb, bb, 4*sizeBB*sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&gpu_overlap, sizeof(float)*gpu_grid.cols*sizeBB);

	
	int threadbbOverLapKernal = 256;
	int blockbbOverLapKernal = (gpu_grid.cols)/threadbbOverLapKernal + (gpu_grid.cols % threadbbOverLapKernal?1:0); // for greyscale

	bbOverLapKernal<<<blockbbOverLapKernal,threadbbOverLapKernal>>>(gpu_bb,gpu_grid.ptr,gpu_grid.rows,gpu_grid.cols,gpu_overlap);
	cutilCheckMsg("Error in BBOVERLAP Kernal...");
	
	cudaMemcpy( overlap,gpu_overlap,gpu_grid.cols*sizeBB*sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(gpu_bb);
	cudaFree(gpu_overlap);		
   
}

void CudaFunctions::fern5Positive(int2DArrayStruct* patt,float* bb,float2DArrayStruct matrix,intArrayStruct idxBoxs)
{

	int* gpu_patt;
	int* gpu_idxBoxs;
	int numWarps = matrix.cols;
	float* gpu_matrix;
	int bbW,bbH;
	bbW = (int)(bb[2] - bb[0] + 1.0);
	bbH = (int)(bb[3] - bb[1] + 1.0);

	cudaMalloc((void**)&gpu_matrix,matrix.cols*matrix.rows*sizeof(float));
	cudaMalloc((void**)&gpu_patt,patt->cols*patt->rows*sizeof(int));
	cudaMalloc((void**)&gpu_idxBoxs,idxBoxs.size*sizeof(int));
	cutilCheckMsg("Error in cudaMalloc fern5Positive...");

	cudaMemcpy(gpu_idxBoxs,idxBoxs.ptr,idxBoxs.size*sizeof(int),cudaMemcpyHostToDevice); 	
	cutilCheckMsg("Error in memcpy fern5Positive idxBoxs...");

	cudaMemcpy(gpu_matrix,matrix.ptr,matrix.cols*matrix.rows*sizeof(float),cudaMemcpyHostToDevice);
	cutilCheckMsg("Error in memcpy fern5Positive matrix...");

	int threadWarpX = 16;
	int threadWarpY = 16;

	int blockWarpX = (bbW/threadWarpX) + ((bbW%threadWarpX)?1:0);
	int blockWarpY = (bbH/threadWarpY) + ((bbH%threadWarpY)?1:0);

	dim3 blocks(blockWarpX,blockWarpY);
	dim3 threads(threadWarpX,threadWarpY);

	int threadsPatternsKernal = 256;
	int blocksPatternsKernal = (idxBoxs.size)/threadsPatternsKernal + (idxBoxs.size % threadsPatternsKernal?1:0);

	cudaBindTexture2D(0,&textureWarpedImage, gpu_warpedImage,&textureWarpedImage.channelDesc,width,height,gpu_warpedImage_pitch);
	cutilCheckMsg("Error in warpedImage binding...");
	calculatePatternsPositiveKernal<<<blocksPatternsKernal,threadsPatternsKernal>>>(gpu_idxBoxs, gpu_patt, gpu_grid.ptr, gpu_featuresOffsets.ptr,gpu_featuresOffsets.cols,idxBoxs.size, gpu_grid.rows);
	cutilCheckMsg("Error in calculatePatternsPositiveKernal...");

	for(int i=0;i<numWarps;i++)
	{

		cudaBindTexture2D(0,&textureBlurredImage, gpu_blurImage,&textureBlurredImage.channelDesc,width,height,gpu_blurImage_pitch);
		cutilCheckMsg("Error in blurImage binding...");
		transformKernel<<<blocks,threads>>>(gpu_warpedImage,width,height,gpu_warpedImage_pitch/sizeof(unsigned char),(gpu_matrix+i*9),bb[0],bb[1],bbW,bbH);
		cudaThreadSynchronize();
		cutilCheckMsg("Error in WARP Kernal...");

		cudaBindTexture2D(0,&textureWarpedImage, gpu_warpedImage,&textureWarpedImage.channelDesc,width,height,gpu_warpedImage_pitch);
		cutilCheckMsg("Error in warpedImage binding...");
		calculatePatternsPositiveKernal<<<blocksPatternsKernal,threadsPatternsKernal>>>(gpu_idxBoxs, (gpu_patt+(i+1)*10*idxBoxs.size), gpu_grid.ptr, gpu_featuresOffsets.ptr,gpu_featuresOffsets.cols,idxBoxs.size, gpu_grid.rows);
		cutilCheckMsg("Error in calculatePatternsPositiveKernal...");
	}



	cudaMemcpy(patt->ptr,gpu_patt,patt->cols*patt->rows*sizeof(int),cudaMemcpyDeviceToHost);
	cutilCheckMsg("Error in memcpy fern5Positive patt...");
		
	cudaFree(gpu_patt);
	cudaFree(gpu_idxBoxs);
	cudaFree(gpu_matrix);

}

void CudaFunctions::fern5Negative(int2DArrayStruct patt,char* status, int *idxBoxs, float varienceThresh)
{
	int* gpu_patt;
	int* gpu_idxBoxs;
	char* gpu_status;
	
	cudaMalloc((void**)&gpu_patt,patt.cols*patt.rows*sizeof(int));
	cutilCheckMsg("Error in cudaMalloc gpu_patt fern5Negative...");
	cudaMalloc((void**)&gpu_idxBoxs,patt.cols*sizeof(int));
	cutilCheckMsg("Error in cudaMalloc gpu_idxBoxs fern5Negative...");
	cudaMalloc((void**)&gpu_status,patt.cols*sizeof(char));
	cutilCheckMsg("Error in cudaMalloc gpu_status fern5Negative...");

	cudaMemset(gpu_status,0,patt.cols*sizeof(char));
	cutilCheckMsg("Error in memset gpu_status fern5Negative...");

	cudaMemcpy(gpu_idxBoxs,idxBoxs,patt.cols*sizeof(int),cudaMemcpyHostToDevice); 
	cutilCheckMsg("Error in memcpy fern5Negative idxBoxs...");

	int threadsPatternsKernal = 256;
	int blocksPatternsKernal = (patt.cols)/threadsPatternsKernal + (patt.cols % threadsPatternsKernal?1:0);
	int threadsBoxVarienceKernal = 256;
	int blocksBoxVarienceKernal = (patt.cols)/threadsBoxVarienceKernal + (patt.cols % threadsBoxVarienceKernal?1:0);

	cudaBindTexture2D(0,&textureIntregalImage,gpu_IntregalImage,&textureIntregalImage.channelDesc,width,height,gpu_IntregalImage_pitch);
	cutilCheckMsg("Error in BindTexture from gpu_IntregalImage...");
	cudaBindTexture2D(0,&textureIntregalSQImage,gpu_IntregalSQImage,&textureIntregalSQImage.channelDesc,width,height,gpu_IntregalSQImage_pitch);
	cutilCheckMsg("Error in BindTexture from gpu_IntregalSQImage...");
	calBoxVarienceKernal<<<blocksBoxVarienceKernal,threadsBoxVarienceKernal>>>(gpu_idxBoxs,gpu_status,gpu_grid.ptr,gpu_grid.rows,varienceThresh, patt.cols);
	cutilCheckMsg("Error in calBoxVarienceKernal...");

	cudaBindTexture2D(0,&textureWarpedImage, gpu_warpedImage,&textureWarpedImage.channelDesc,width,height,gpu_warpedImage_pitch);
	cutilCheckMsg("Error in warpedImage binding...");
	calculatePatternsNegatveKernal<<<blocksPatternsKernal,threadsPatternsKernal>>>(gpu_idxBoxs,gpu_status,gpu_patt, gpu_grid.ptr, gpu_featuresOffsets.ptr,gpu_featuresOffsets.cols, patt.cols, gpu_grid.rows);
	cutilCheckMsg("Error in calculatePatternsNegatveKernal...");
	
	cudaMemcpy(patt.ptr,gpu_patt,patt.cols*patt.rows*sizeof(int),cudaMemcpyDeviceToHost);
	cutilCheckMsg("Error in memcpy fern5Positive patt...");
	cudaMemcpy(status,gpu_status,patt.cols*sizeof(char),cudaMemcpyDeviceToHost);
	cutilCheckMsg("Error in memcpy fern5Positive status...");	
	

	cudaFree(gpu_patt);
	cudaFree(gpu_idxBoxs);
	cudaFree(gpu_status);	
}

void CudaFunctions::fernLearning(int2DArrayStruct pX, int2DArrayStruct nX,float threshPositive,float threshNegative,int bootStrap)
{
	int threadsUpdate = 256;
	if(pX.cols > 0)
	{
		int *gpu_pX;
		int blocksUpdatePositive = (pX.cols)/threadsUpdate + (pX.cols % threadsUpdate?1:0);

		cudaMalloc((void**)&gpu_pX,pX.cols*pX.rows*sizeof(int));

		cudaMemcpy(gpu_pX, pX.ptr, pX.cols*pX.rows*sizeof(int),cudaMemcpyHostToDevice);
		cutilCheckMsg("Error in memcpy pX fernLearning...");
	
		for(int i=0; i<bootStrap; i++)
		{
			updatePositiveKernal<<<blocksUpdatePositive,threadsUpdate>>>(weights,nP,nN,gpu_pX,pX.rows,pX.cols,threshPositive);
			cudaThreadSynchronize();
			cutilCheckMsg("Error in updatePositiveKernal...");
		}
		cudaFree(gpu_pX);
	}


	if(nX.cols > 0)
	{
		int *gpu_nX;

		int blocksUpdateNegative = (nX.cols)/threadsUpdate + (nX.cols % threadsUpdate?1:0);

		cudaMalloc((void**)&gpu_nX, nX.cols*nX.rows*sizeof(int));	

		cudaMemcpy(gpu_nX,nX.ptr, nX.cols*nX.rows*sizeof(int),cudaMemcpyHostToDevice);
		cutilCheckMsg("Error in memcpy nX fernLearning...");

		for(int i=0; i<bootStrap; i++)
		{
			updateNegativeKernal<<<blocksUpdateNegative,threadsUpdate>>>(weights,nP,nN,gpu_nX,nX.rows,nX.cols,threshNegative);
			cudaThreadSynchronize();
			cutilCheckMsg("Error in updateNegativeKernal...");
			
		}
		cudaFree(gpu_nX);

	}
}

void CudaFunctions::fern3(int2DArrayStruct nX,float* confidences)
{
	int threadsfern3 = 256;
	int *gpu_nX;
	float* gpu_confidences;

	cudaMalloc((void**)&gpu_confidences,nX.cols*sizeof(float));
	cudaMalloc((void**)&gpu_nX,nX.cols*nX.rows*sizeof(int));

	int blocksfern3 = (nX.cols)/threadsfern3 + (nX.cols % threadsfern3?1:0);

	//std::cout<<nX.cols<<"\t"<<nX.rows<<"\n";
	cudaMemcpy(gpu_nX,nX.ptr,nX.cols*nX.rows*sizeof(int),cudaMemcpyHostToDevice);
	cutilCheckMsg("Error in memcpy nX fern3...");

	getConfidences_fern3_Kernal<<<blocksfern3,threadsfern3>>>(gpu_confidences,weights,gpu_nX,nX.rows,nX.cols);
	cutilCheckMsg("Error in getConfidences_fern3_Kernal...");

	cudaMemcpy(confidences,gpu_confidences,nX.cols*sizeof(float),cudaMemcpyDeviceToHost);
	cutilCheckMsg("Error in memcpy confidences fern3...");
	
	cudaFree(gpu_confidences);		

}

/*Evaluate all the boxes generated by the grid using FERN and find confidences for them to be the object*/
void CudaFunctions::fernDetection(int2DArrayStruct patt,floatArrayStruct conf,intArrayStruct* tconfidx,float pthresh,float varienceThresh)
{
	int* gpu_patt;
	float* gpu_conf;
	

	cudaMalloc((void**)&gpu_patt,patt.cols*patt.rows*sizeof(int));
	cudaMalloc((void**)&gpu_conf,conf.size*sizeof(float));
	cudaMemset(gpu_conf,0,conf.size*sizeof(float));	

	int threadsPatternsKernal = 192;
	int blocksPatternsKernal = (gpu_grid.cols)/threadsPatternsKernal + (gpu_grid.cols % threadsPatternsKernal?1:0);
	int threadsConfidenceKernal = 256;
	int blocksConfidenceKernal = (gpu_grid.cols)/threadsConfidenceKernal + (gpu_grid.cols % threadsConfidenceKernal?1:0);


	cudaBindTexture2D(0,&textureWarpedImage, gpu_warpedImage,&textureWarpedImage.channelDesc,width,height,gpu_warpedImage_pitch);
	cutilCheckMsg("Error in warpedImage binding...");
	calculatePatternsKernal<<<blocksPatternsKernal,threadsPatternsKernal>>>(gpu_patt, gpu_grid.ptr, gpu_featuresOffsets.ptr,gpu_featuresOffsets.cols, gpu_grid.cols, gpu_grid.rows);
	cutilCheckMsg("Error in calculatePatternsKernal...");
	cudaThreadSynchronize();

	cudaBindTexture2D(0,&textureIntregalImage,gpu_IntregalImage,&textureIntregalImage.channelDesc,width,height,gpu_IntregalImage_pitch);
	cutilCheckMsg("Error in BindTexture from gpu_IntregalImage...");
	cudaBindTexture2D(0,&textureIntregalSQImage,gpu_IntregalSQImage,&textureIntregalSQImage.channelDesc,width,height,gpu_IntregalSQImage_pitch);
	cutilCheckMsg("Error in BindTexture from gpu_IntregalSQImage...");
	getConfidencesFromTreeKernal<<<blocksConfidenceKernal,threadsConfidenceKernal>>>(gpu_conf,gpu_grid.ptr,weights,gpu_patt,patt.rows,gpu_grid.cols,gpu_grid.rows,varienceThresh);
	cutilCheckMsg("Error in getConfidencesFromTree Kernal...");


	cudaMemcpy(patt.ptr,gpu_patt, patt.cols*patt.rows*sizeof(int),cudaMemcpyDeviceToHost);
	cutilCheckMsg("Error in memecpy patt...");
	cudaMemcpy(conf.ptr,gpu_conf, conf.size*sizeof(float),cudaMemcpyDeviceToHost);
	cutilCheckMsg("Error in memecpy conf...");

	sort_data(conf.ptr,tconfidx,conf.size,pthresh);

	cudaFree(gpu_patt);
	cudaFree(gpu_conf);

}

void CudaFunctions::sort_data(float *data,intArrayStruct *out, int input_size, float threshold)
{

	float *cuda_in_array;
    int *cuda_in_index;
    int *cuda_out_index;

    float* data_final=(float*)malloc(sizeof(float)*input_size);
    int* index_final=(int*)malloc(sizeof(int)*input_size);

	int limit = out->size;
    int a=0;
    for(int i=0;i<input_size;i++)
	{
        if(data[i]>threshold)
		{
            data_final[a]=data[i];
            index_final[a]=i;

			a++;
        }
    }
   
	if(a<limit)
		limit=a;

	if(limit==0)
	{
		out->size = 0;
		return;
	}

    cudaMalloc( (void**)&cuda_in_array, a*sizeof(float));
    cudaMemcpy( cuda_in_array, data_final,a*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc( (void**)&cuda_in_index, a*sizeof(int));
    cudaMemcpy( cuda_in_index, index_final,a*sizeof(int), cudaMemcpyHostToDevice);


    cudaMalloc( (void**)&cuda_out_index, (limit)*sizeof(int));
   
    int block_size =256;
    dim3 threads(block_size,1);
    dim3 grid((input_size+a-1) / threads.x,1/ threads.y);

    sort_fern<<< grid, threads>>>(cuda_in_array,cuda_in_index,a,cuda_out_index,limit,threshold);
    cudaThreadSynchronize();
	
	cudaMemcpyAsync(out->ptr, cuda_out_index,limit* sizeof(int), cudaMemcpyDeviceToHost);
	
	
	cudaFree(cuda_out_index);
    cudaFree(cuda_in_array);
    cudaFree(cuda_in_index);
	
	free(data_final);
	free(index_final);
	out->size = limit;

	
}

//_________Nearest Neighbor Confidance Calculation___________
//Distance between given set of patches and existing ‘pex’ and ‘nex’ data stored in TLD structure are calculated to find out 
//which of the given patches have the closest resemblance to ‘pex’ and ‘nex’.
void CudaFunctions::calc_tldNN(float* x,int n,int M,float* pex,int N1,float* nex,int N2,float* conf1,float* conf2,int* isin)
{
    for (int i=0;i<3*n;i++)
        isin[i]= -1;

     
    if(N1==0)
    {// IF positive examples in the model are not defined THEN everything is negative
        for (int i=0;i<n;i++)
        {
            conf1[i]=0;
            conf2[i]=0; 
        }
        return;
    }

    if(N2==0)
    {// IF negative examples in the model are not defined THEN everything is positive
        for (int i=0;i<n;i++)
        {
            conf1[i]=1;
            conf2[i]=1;
        }
        return;
    }


    float *x_c, *x_h;
    float *pex_c, *pex_h;
    float *nex_c, *nex_h;
    float *nccP_c;
    float *nccN_c;

	int *isin_h;
	float *conf1_h;
	float *conf2_h;


    cutilSafeCall(cudaMalloc( (void**)&x_c, n*M*sizeof(float)));
    cutilSafeCall(cudaMemcpy( x_c, x, n*M*sizeof(float), cudaMemcpyHostToDevice ));


    cutilSafeCall(cudaMalloc( (void**)&pex_c, N1*M*sizeof(float)));
	cutilSafeCall(cudaMemcpy( pex_c, pex,N1*M*sizeof(float), cudaMemcpyHostToDevice ));
	
    cutilSafeCall(cudaMalloc( (void**)&nex_c, N2*M*sizeof(float)));
    cutilSafeCall(cudaMemcpy( nex_c, nex,N2*M*sizeof(float), cudaMemcpyHostToDevice ));
	

    cutilSafeCall(cudaMalloc( (void**)&nccP_c, n*N1* sizeof(float) ));
    cutilSafeCall(cudaMalloc( (void**)&nccN_c, n*N2* sizeof(float) ));   
	
    cutilCheckMsg("Error in ccorr_normed Kernal before pex...");
    int block_size = 128;

    dim3 threads(1,block_size); 
	dim3 threads_new(1,1024); 
   
    dim3 grid1(1/ threads_new.x,n*N1);
    ccorr_normed_nn<<< grid1, threads_new>>>(pex_c,x_c,nccP_c,M,N1,n); //,0,c_stream 
    cutilCheckMsg("Error in ccorr_normed Kernal after pex...");
    cudaThreadSynchronize();
		
    dim3 grid2(1/ threads_new.x,n*N2);
    ccorr_normed_nn<<< grid2, threads_new>>>(nex_c,x_c,nccN_c,M,N2,n);
    cutilCheckMsg("Error in ccorr_normed Kernal after nex...");
    cudaThreadSynchronize();		

    float *maxP_c;
    float *maxN_c;
    float *maxPP_c;
    int *iP_c;

    dim3 grid3(1/ threads.x,(N1+block_size-1)/block_size);
    dim3 grid4(1/ threads.x,(N2+block_size-1)/block_size);

    cutilSafeCall(cudaMalloc((void**)&maxP_c, n*sizeof(float)));
    cutilSafeCall(cudaMalloc( (void**)&maxPP_c, n*sizeof(float)));
    cutilSafeCall(cudaMalloc( (void**)&iP_c, n*sizeof(int)));
    cutilSafeCall(cudaMalloc((void**)&maxN_c, n*sizeof(float)));
   
    for(int i=0;i<n;i++)
	{
		get_maxP_maxPP<<<grid3,threads >>>(nccP_c+i*N1,N1,maxP_c+i,maxPP_c+i,iP_c+i);
		cudaThreadSynchronize();
	}

    for(int i=0;i<n;i++)
	{
		get_maxN<<<grid4,threads>>>(nccN_c+i*N2,N2,maxN_c+i);
	    cudaThreadSynchronize();		
	}   


    dim3 grid5(1/ threads.x,(n+block_size-1)/block_size);
    int* isIn_c;
    float* conf1_c;
    float* conf2_c;
    cutilSafeCall(cudaMalloc( (void**)&isIn_c, 3*n*sizeof(int)));
    cutilSafeCall(cudaMalloc( (void**)&conf1_c, n*sizeof(float)));
    cutilSafeCall(cudaMalloc( (void**)&conf2_c, n*sizeof(float)));

    calc_ConfIsin<<<grid5,threads>>>(n,maxP_c,iP_c,maxPP_c,maxN_c,isIn_c,conf1_c,conf2_c);
    cudaThreadSynchronize();
	

    cutilSafeCall(cudaMemcpy( isin,isIn_c,3*n*sizeof(int),  cudaMemcpyDeviceToHost ));
    cutilSafeCall(cudaMemcpy( conf1,conf1_c,n*sizeof(float),  cudaMemcpyDeviceToHost ));
    cutilSafeCall(cudaMemcpy( conf2,conf2_c,n*sizeof(float),  cudaMemcpyDeviceToHost ));


    cudaFree(pex_c);
    cudaFree(nex_c);
    cudaFree(x_c);
   
    cudaFree(isIn_c);
    cudaFree(conf1_c);
    cudaFree(conf2_c);

    cudaFree(maxP_c);
    cudaFree(iP_c);
    cudaFree(maxPP_c);
    cudaFree(maxN_c);

    cudaFree(nccP_c);
    cudaFree(nccN_c);

}
int** CudaFunctions::calcntuples(int x1[],int x2[],int num_x1_col,int num_x2_col)
{
	int *cuda_x1;
    int *cuda_x2;
    int *cuda_x3;
	
	
	int* resp =(int*)malloc(num_x1_col * num_x2_col * 2*sizeof(int));
	
    
	cudaMalloc( (void**)&cuda_x1, num_x1_col*sizeof(int));
    cudaMemcpy( cuda_x1, x1,num_x1_col*sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc( (void**)&cuda_x2, num_x2_col*sizeof(int));
    cudaMemcpy( cuda_x2, x2,num_x2_col*sizeof(int), cudaMemcpyHostToDevice);
    
	cudaMalloc( (void**)&cuda_x3, num_x1_col * num_x2_col * 2*sizeof(int) );
   

    int block_size = 256 ;
    dim3 threads(block_size,1);//16*1 threads per block
    dim3 grid((num_x1_col*num_x2_col+block_size-1)/ threads.x,1/ threads.y);//no of thread blocks in x direction and y direction [x,y]=rows,cols
   
    calc_grid_kernel<<< grid, threads>>>(cuda_x1,cuda_x2,cuda_x3,num_x1_col,num_x2_col);
	cudaThreadSynchronize();
	  
    cudaMemcpy( resp, cuda_x3,num_x1_col * num_x2_col *2* sizeof(int), cudaMemcpyDeviceToHost); 
	
	int** out=(int**)malloc(2*sizeof(int*));
    out[0]=&resp[0];
    out[1]=&resp[num_x1_col * num_x2_col];
	
	 
	cudaFree(cuda_x1);
    cudaFree(cuda_x2);
	cudaFree(cuda_x3);
	cudaThreadSynchronize();
	return out;
}

