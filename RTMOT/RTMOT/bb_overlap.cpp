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
#include "bb_overlap.h"

float bb_overlapClass::bb_overlappp(float *bb1, float *bb2) 
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

floatArrayStruct bb_overlapClass::bb_overlap(float *bb1,int M1,int N1,float *bb2,int M2,int N2)
{

	int n=N1*N2;      

	float *out=(float*) malloc(n*sizeof(float));
	float *overlp=out;
				
	for (int j = 0; j < N2; j++)	// N2 = # cols in the grid
	{
		for (int i = 0; i < N1; i++)  // N1 = 1
		{
			*overlp++ = bb_overlappp(bb1 + M1*i, bb2 + M2*j);
		}
	}


	floatArrayStruct output;
	output.ptr=out;
	output.size=n;
		
	return output;	
}
