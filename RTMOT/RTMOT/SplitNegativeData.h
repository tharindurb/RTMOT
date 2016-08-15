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
#ifndef SPLIT_NEG_H
#define SPLIT_NEG_H

#include "IncludeFiles.h"

class SplitNegativeData
{
public:

	SplitNegativeData();
	~SplitNegativeData();
								
	void tldSplitNegativeData(	int* nX,int M1,int N1,float* nEx,int M2,int N2,int* nX1,int* nX2,float* nEx1,float* nEx2);
	void randperm2(int n,int* perm);



};


#endif
