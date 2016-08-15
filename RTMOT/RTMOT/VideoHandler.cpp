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
#include "VideoHandler.h"

void VideoHandler::loadVideo(int n)
{
	capture = cvCaptureFromCAM(n);

	frameHeight = cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_HEIGHT);
	frameWidth = cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_WIDTH);

	rgbFrame = cvCreateImage(cvSize(frameWidth,frameHeight),IPL_DEPTH_8U,3);	
	rgbImage = cvCreateImage(cvSize(frameWidth,frameHeight),IPL_DEPTH_8U,3);
	blurFrame = cvCreateImage(cvSize(frameWidth,frameHeight),IPL_DEPTH_8U,1);	
	blurImage = cvCreateImage(cvSize(frameWidth,frameHeight),IPL_DEPTH_8U,1);
	frame = cvCreateImage(cvSize(frameWidth,frameHeight),IPL_DEPTH_8U,1);	
	image = cvCreateImage(cvSize(frameWidth,frameHeight),IPL_DEPTH_8U,1);
    IplImage* temp1 = cvCreateImage(cvSize(frameWidth,frameHeight),IPL_DEPTH_8U,1);

	numRectangles = 0;
	for(int i=0;i<5;i++)
		temp1=cvQueryFrame(capture);
}


void VideoHandler::initVideoHandler()
{
	onProcess = false;
	initialized = false;
	doneDrawing = false;
	doneSetting = false;
	
	selection = cvRect(0,0,0,0);
	numRectangles = 0;

	boxes.reserve(5);
}


VideoHandler::~VideoHandler()
{
	cvReleaseCapture(&capture);
	cvReleaseImage(&frame);
	cvReleaseImage(&image);
	cvReleaseImage(&rgbImage);
	cvReleaseImage(&rgbFrame);
	

}

void VideoHandler::releaseMemory()
{
	cvReleaseCapture(&capture);
	cvReleaseImage(&frame);
	cvReleaseImage(&image);
	cvReleaseImage(&rgbImage);
	cvReleaseImage(&rgbFrame);

}
void VideoHandler::drawRectangleColor(CvScalar color,CvRect temDraw)
{
	
	CvPoint pt1,pt2;
	pt1.x = temDraw.x;
	pt2.x = temDraw.x + temDraw.width;

	pt1.y = temDraw.y;
	pt2.y = temDraw.y + temDraw.height;

	cvRectangle(rgbFrame,pt1,pt2,color,2,CV_AA,0);

}
void VideoHandler::drawRectangle()
{
	CvRect temDraw; 
	for(int i=0;i<boxes.size() ; i++)
	{
		temDraw = boxes[i];	
	
		CvPoint pt1,pt2;
		pt1.x = temDraw.x;
		pt2.x = temDraw.x + temDraw.width;

		pt1.y = temDraw.y;
		pt2.y = temDraw.y + temDraw.height;

		cvRectangle(rgbFrame,pt1,pt2,CV_RGB(255,255,255),1,CV_AA,0);

		
	}
	
}


void VideoHandler::MouseHandle(int event, int x, int y, int flags, void *param)
{

	switch(event){

		case CV_EVENT_MOUSEMOVE:
			
			if(onProcess){
				selection.width = x - selection.x;
				selection.height = y - selection.y;
			}

			break;

		case CV_EVENT_LBUTTONDOWN:
            
			onProcess = true;
			selection = cvRect(x,y,0,0);
			break;

		case CV_EVENT_LBUTTONUP:
			
			onProcess = false;
			if(selection.width > 0 && selection.height > 0){
				doneSetting = true;
			}

			if(doneSetting){
				doneDrawing = true;
			}

			break;
	}
}

