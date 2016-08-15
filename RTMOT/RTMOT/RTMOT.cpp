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
#include "IncludeFiles.h"
#include "MultiStreams.h"
#include <omp.h>

MultiStreams ms;


void  videoStreams( void *arg )
{
	int count = (INT_PTR)arg;
	ms.runStreams(count);

}
int _tmain(int argc, _TCHAR* argv[])
{


	int num_devices=0;
    CUDA_SAFE_CALL( cudaGetDeviceCount(&num_devices) );
    if(0==num_devices)
    { 
		std::cout << "your system does not have a CUDA capable device.." << std::endl ;
        return 1;
	}

	int device;
	cudaGetDevice(&device);
	std::cout << "Cuda Device " << device << std::endl;

	struct cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,device);

	std::cout << "Deivce Name : " << prop.name << std::endl;
	std::cout << "Concurrent Kernels : " << prop.concurrentKernels << std::endl;
	std::cout << "Compute Mode : " << prop.computeMode <<std::endl;


	int numStreams = 1;	/* set the no of cameras at the begining*/

	ms.initStreams(numStreams);
	ms.hObjectMutex = CreateMutex( NULL, FALSE, NULL ); /* create the mutex */
	
	#pragma omp parallel for
	
	for(int i=0;i<numStreams;i++)
	{
		_beginthread(videoStreams, 0, (void*)i );	
	}
	
	ms.freeStreams(numStreams);
	_getch();
	return 0;
}



