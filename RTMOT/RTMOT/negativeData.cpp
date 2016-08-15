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
#include "negativeData.h"
// Generates negative data for online lerning process.
// The grid box indexes with overlap value less than a certain limit are used to calculate nX values.

negativeData::negativeData(void)
{
	overlapLimit=0.2;
	num_patches=100;


}
negativeData::~negativeData(void)
{

}
negativeDataOut negativeData::generateNegativeData(floatArrayStruct overlap,float tld_var,IplImage* img_input,IplImage* img_blur,CudaFunctions *fernObj,float2DArrayStruct tld_grid)
{
	
	negativeDataOut output;

	intArrayStruct idxN =find(overlap); /* find grid box indexes where overlap ration < 0.2 */

	int2DArrayStruct nX1;
	nX1.cols = idxN.size;
	nX1.rows = 10;

	char *status = (char*)malloc(idxN.size*sizeof(char));
	nX1.ptr = (int*)malloc(nX1.cols*nX1.rows*sizeof(int));

	fernObj->fern5Negative(nX1,status,idxN.ptr,tld_var/2.0); 

	/* nX values( 13 bit feature patterns) are found for grid boxes < 0.2 */

	intArrayStruct idxN2;	
	idxN2=select1(idxN,status);	/* grid box indexes with status==1 is taken*/
	output.nX=select2(nX1,status); /* nX values with status = 1 is taken*/

	if(idxN2.size<num_patches)
	{
		num_patches=idxN2.size-1;
	}

	int *idx = (int*)malloc(num_patches*sizeof(int));

	randvalues(idxN2.size,1,num_patches,idx); /*indexes from 1 to idxN2.size are randomly arranged and put into idx.*/


	int nROWS=tld_grid.rows;
	float *bb_array=(float*)malloc(sizeof(float)*tld_grid.rows*num_patches);

	for(int i=0;i<num_patches;i++)
	{
		for(int j=i+1;j<num_patches;j++)
		{
			int temp;
			if(idx[i]<idx[j])
			{
				temp = idx[i];
				idx[i] = idx[j];
				idx[j] = temp;
			}
		}
	}
	
	for(int j=0;j<num_patches;j++)
	{
		for(int k=0;k<tld_grid.rows;k++)
		{
			bb_array[j*tld_grid.rows+k] = tld_grid.ptr[idxN2.ptr[idx[j]]*tld_grid.rows+k]; /*grid boxes are found*/			
		}
	}


	float2DArrayStruct nEx;
	nEx.ptr = (float*)malloc(2025*num_patches*sizeof(float));
	nEx.cols = num_patches;
	nEx.rows = 2025;
	fernObj->cudaGetPattern(num_patches,nEx.ptr,bb_array);
	/* nEx values (pixel value- mean of the patch) is found*/

	output.nEx=nEx;
	
	free(bb_array);

	return output;

}

void negativeData::randvalues(int upper,int lower,int k,int *out)
{

	int *range=(int*)malloc(sizeof(int)*((upper-lower)+1));
	int idx=0;
	for(int i=lower;i<=upper;i++)
	{
		range[idx]=i;
		idx++;
	}

	std::random_shuffle(range,range+idx);

	for(int j=0;j<k;j++)
	{
		out[j]=range[j];
	}
	free(range);

}


int2DArrayStruct negativeData::select2(int2DArrayStruct nX,char *status)
{
		
	int rsize=nX.rows;
	int csize=nX.cols;
	int* b=(int*)malloc(rsize*csize*sizeof(int));

	int statCount=0;
	for(int i=0;i<csize;i++)
	{
		if(status[i]==1)
		{
			for(int j=0;j<rsize;j++)
			{
				b[statCount]=nX.ptr[i*rsize+j];
				statCount++;
			}
		}
	}
	
	realloc(b,sizeof(int)*statCount);
	int2DArrayStruct output;
	output.ptr=b;
	output.cols=statCount/nX.rows;
	output.rows=nX.rows;

	return output;

}


intArrayStruct negativeData::select1(intArrayStruct idXn,char* status)
{
	int size=idXn.size;
	
	int *a=(int*)malloc(sizeof(int)*size);
	int statCount = 0;

	for(int i=0;i<size;i++)
	{
		if(status[i]==1)
		{
			a[statCount]= idXn.ptr[i];
			statCount++;
		}
	}
	
	realloc(a,sizeof(int)*statCount);

	intArrayStruct output;
	output.ptr=a;
	output.size=statCount;
	
	return output;

}


intArrayStruct negativeData::find(floatArrayStruct overlap)
{
	intArrayStruct out;
	
	int* temp=(int*)malloc(sizeof(int)*overlap.size);
	int k=0;

	for(int i=0;i<overlap.size;i++)
	{
		if(overlap.ptr[i]<overlapLimit)
		{
			temp[k]=i;
			k++;
			
		}

	}
	realloc(temp,sizeof(int)*k);
	out.ptr=temp;
	out.size=k;
	return out;

}