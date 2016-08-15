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
#ifndef TLD_EXEC
#define TLD_EXEC

#include "IncludeFiles.h"
#include "TLDstructs.h"

#include <algorithm>
#include <math.h>

#include "CudaFunctions.h"

class TLDExecution
{

public:
	
	tld_Structure* my_tld;
	CvSize img_size;
	CvSize bb_size;
	
	int total_grid_cols;
	
	int *leftVals, *topVals;

	CudaFunctions ntuples_cuda;

	TLDExecution(CvSize fr_size,tld_Structure* my_tld);
	void bb_scan_method(IplImage *firstFrame,CvRect bb);
	int** ntuples(int* top, int *left, int top_elms, int left_elms);

};

#endif
