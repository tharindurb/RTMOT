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
#ifndef DATASTRUCTS_H
#define DATASTRUCTS_H

#pragma once

struct floatArrayStruct 
{
	float* ptr;
	int size;
} ;
struct intArrayStruct 
{
	int* ptr;
	int size;
} ;
struct charArrayStruct 
{
	char* ptr;
	int size;
 

} ;
struct boolArrayStruct
{
	bool* ptr;
	int size;
};

struct int2DArrayStruct 
{
	int* ptr;
	int rows;
	int cols;
} ;

struct float2DArrayStruct 
{
	float* ptr;
	int rows;
	int cols;
} ;

#endif