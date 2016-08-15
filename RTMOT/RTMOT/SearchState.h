#ifndef SEARCH_H
#define SEARCH_H

#include "IncludeFiles.h"

class SearchState
{

public:
	
	int search_attempts_per_cam;
	int search_count;
	int current_search_cam;
	int next_search_cam;

	SearchState(int attempts)
	{
		search_attempts_per_cam = attempts;
	}

};

#endif SEARCH_H
