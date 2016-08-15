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
#ifndef VIDEO_HANDLE
#define VIDEO_HANDLE

#include "IncludeFiles.h"
#include <vector>

using namespace std;

class VideoHandler
{
	
public:


	int frameWidth;
	int frameHeight;

	CvRect selection;
	
	vector<CvRect> boxes;
	vector<bool> boxStatus;
	int numRectangles;

	CvCapture *capture;

	IplImage *frame;
	IplImage *image;
	IplImage *rgbFrame;
	IplImage *rgbImage;
	IplImage *blurImage;
	IplImage *blurFrame;

    bool initialized;
	bool doneDrawing;
    bool doneSetting;
	bool onProcess;

	void MouseHandle(int event, int x, int y, int flags, void* param);
	void loadVideo(int n);
	void displayVideo();
	void drawRectangle();
	void drawRectangleColor(CvScalar color,CvRect temDraw);
		
	VideoHandler()
	{
		onProcess = false;
		initialized = false;
		doneDrawing = false;
		doneSetting = false;
		
		selection = cvRect(0,0,0,0);
		numRectangles = 0;

		boxes.reserve(5);

		boxStatus.reserve(5);
	}

	~VideoHandler();
	void initVideoHandler();
	void releaseMemory();
	
};

#endif
