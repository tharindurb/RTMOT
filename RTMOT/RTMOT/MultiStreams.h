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
#ifndef STREAMS_H
#define STREAMS_H

#include "IncludeFiles.h"
#include "VideoHandler.h"
#include "ObjectClass.h"

#include <process.h>
#include <windows.h>
#include <omp.h>

class MultiStreams
{
public:

	VideoHandler vih[10];
	ObjectClass *obj[10];

	HANDLE  hObjectMutex; /* Mutex to control all video streams accessing object processing part..*/

	int num_of_cams;

	void initStreams(int n);
	void freeStreams(int n);
	void runStreams(int arg);


};


#endif
