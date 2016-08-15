#ifndef TIMINGMICRO
#define TIMINGMICRO

#include <windows.h>
class timingMicro{
	double PCFreq;
	__int64 CounterStart;

public:
	void StartCounter()
	{
		LARGE_INTEGER li;
		if(!QueryPerformanceFrequency(&li))
			printf("QueryPerformanceFrequency failed!\n");

		PCFreq = double(li.QuadPart)/1000000.0;

		QueryPerformanceCounter(&li);
		CounterStart = li.QuadPart;
	}
	double GetCounter()
	{
		LARGE_INTEGER li;
		QueryPerformanceCounter(&li);
		return double(li.QuadPart-CounterStart)/PCFreq;
	}
};

#endif