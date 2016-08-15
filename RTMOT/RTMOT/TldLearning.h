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
#ifndef TLD_LEARN
#define TLD_LEARN

#include "IncludeFiles.h"
//#include "getPattern.h"
#include "bb_overlap.h"
#include "GenPositiveData.h"
#include "tldNN.h"
#include "tldTrainNN.h"
#include "TLDstructs.h"
//#include "Fern.h"
#include "negativeData.h"
#include "CudaFunctions.h"

class TldLearning
{
public:

	tldNN tnn;
	bb_overlapClass bb_lap;
	GenPositiveData genPos;
	tldTrainNN train;

	void doLearning(tld_Structure* tld,CudaFunctions* fernObj);
	float varience(const float* dataArr,int size);


};

#endif
