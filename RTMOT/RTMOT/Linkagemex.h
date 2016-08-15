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
#ifndef LINKAGE_MEX
#define LINKAGE_MEX

#include "IncludeFiles.h"
#include "TLDstructs.h"

#include <limits>

#define ISNAN(a) (a != a)
#define MAX_NUM_OF_INPUT_ARG_FOR_PDIST 50


class Linkagemex
{
public:
	void mexLinkageTEMPLATE(float2DArrayStruct* out,float2DArrayStruct* bbd,char* method,bool call_pdist=0);
	floatArrayStruct pdist(float2DArrayStruct in);

};

#endif
