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
#ifndef NEGATIVEDATA_H
#define NEGATIVEDATA_H

#include "IncludeFiles.h"
#include "TLDstructs.h"
//#include "Fern.h"
//#include "img_patch.h"
#include "CudaFunctions.h"

struct negativeDataOut 
{
  int2DArrayStruct nX; 
  //float2DArrayStruct nX; 
  float2DArrayStruct nEx;

};

class negativeData
{

public:

	float overlapLimit;
	int num_patches;

	//img_patch imp;

	negativeData(void);
	~negativeData(void);
	negativeDataOut generateNegativeData(floatArrayStruct overlap,float tld_var,IplImage* img_input,IplImage* img_blur,CudaFunctions *fernObj,float2DArrayStruct tld_grid);
	intArrayStruct find(floatArrayStruct overlap);
	intArrayStruct select1(intArrayStruct idXn,char* status);
	int2DArrayStruct select2(int2DArrayStruct nX,char *status);
	void randvalues(int upper,int lower,int k,int *pIdx);	
};

#endif