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
#ifndef GEN_POS_H
#define GEN_POS_H

#include "IncludeFiles.h"
#include "TLDstructs.h"
//#include "Fern.h"
//#include "getPattern.h"
#include "CudaFunctions.h"

#include <ctime>

#define FIND_MIN 0
#define FIND_MAX 1

class GenPositiveData
{
public:
	//getPattern gptn;
	void genPositiveData(tld_Structure *tld, floatArrayStruct *overlap, int2DArrayStruct *pX, float2DArrayStruct *pEx, floatArrayStruct *bbP_out,CudaFunctions* fernObj, bool first=0);
	int higestValueIndexes(float dataArray[],int indx[],int size,float thresh,int limit);
	float* bb_hull(float** bbP,int num_closest_Found);
	float findMinMax(float arr[],int end,int type);

	void getMatrix(float* bb,float* matrix,CvRandState RandState);
	void mulMat(float out[],float data1[],float data2[]);

};

#endif
