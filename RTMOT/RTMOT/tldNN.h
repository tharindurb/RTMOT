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
#ifndef TLD_NN_H
#define TLD_NN_H

#include "IncludeFiles.h"
//#include "Distances.h"
#include "CudaFunctions.h"

class tldNN
{
public:
	long start, end;	

	//Distances dis;
	CudaFunctions cudaNN;

	void dotldNN(float* x,int n,int M,float* pex,int N1,float* nex,int N2,float* conf1,float* conf2,int* isin);
	float findMax(float arr[],int end);

};

#endif
