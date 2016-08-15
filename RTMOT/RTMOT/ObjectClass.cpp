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
#include "ObjectClass.h"

CvScalar ObjectClass::color_tab[] = {0};

int ObjectClass::objectCount = 0;

void ObjectClass::setColors()
{
	ObjectClass::color_tab[0] = CV_RGB(255,0,0);
	ObjectClass::color_tab[1] = CV_RGB(0,255,0);
	ObjectClass::color_tab[2] = CV_RGB(0,0,255);
	ObjectClass::color_tab[2] = CV_RGB(100,100,255);
	ObjectClass::color_tab[2] = CV_RGB(255,100,100);
}

void ObjectClass::setObjectColor(int id)
{
	objColor = color_tab[id];
}
void ObjectClass::displayObject()
{
	std::cout << "Object ID: " << objectID << " - Capture ID: " << captureID << std::endl;
}
float2DArrayStruct ObjectClass::createOffsets(float2DArrayStruct scales, float *features)
{

	float2DArrayStruct offsets;
	offsets.rows  = scales.cols; //row1: f0x row2: f0y row3: f1x row4: f1y 
	offsets.cols = 130*4;
	
	offsets.ptr  = (float*)malloc(offsets.rows*offsets.cols*sizeof(float));
	
	float* off  = offsets.ptr;

	for(int scale = 0 ;scale < scales.cols ; scale++)
	{		
		for(int tree = 0; tree < 10; tree++)
		{
			for(int feature  = 0; feature < 13; feature++)
			{
				*off++ = (scales.ptr[scale*2]  -1.0)*features[52*tree + 4*feature+0]; // f0x
				*off++ = (scales.ptr[scale*2+1]-1.0)*features[52*tree + 4*feature+1]; // f0y
				*off++ = (scales.ptr[scale*2]  -1.0)*features[52*tree + 4*feature+2]; // f1x
				*off++ = (scales.ptr[scale*2+1]-1.0)*features[52*tree + 4*feature+3]; // f1y
				
			}
		}
	}
	return offsets;

}
