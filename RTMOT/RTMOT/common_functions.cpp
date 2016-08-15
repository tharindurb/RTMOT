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
#include "StdAfx.h"
#include "common_functions.h"

common_functions::common_functions(void)
{
}

common_functions::~common_functions(void)
{
}
float common_functions::varience(float* dataArr,int size)
{
	float sum=0;
	float sqSum = 0;
	float mean;

	for(int i =0;i<size;i++)
		sum += dataArr[i];
	mean = sum/size;

	for(int i=0;i < size;i++)
		sqSum += (dataArr[i] - mean)*(dataArr[i] - mean);

	return sqSum/(size-1);

}

 void common_functions::max(floatArrayStruct* arr1,floatArrayStruct* arr2,floatArrayStruct* out)
 {
	 if(arr1->size==arr2->size){
		 out->size=arr1->size;
		 out->ptr=(float*)malloc(out->size*sizeof(float));

		 for(int i=0;i<out->size;i++){

			 if(arr1->ptr[i]<arr2->ptr[i]){
				 out->ptr[i]=arr2->ptr[i];
			 }
			 else{
				 out->ptr[i]=arr1->ptr[i];
			 }
		 }

	 }
	 else{
		 std::cout<<"Array sizes are not equal"<<std::endl;
		 return;

	 }



}
