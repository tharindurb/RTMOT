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
#include "TLDExecution.h"

TLDExecution::TLDExecution(CvSize fr_size,tld_Structure* ptld)
{
	my_tld=ptld;
	img_size = fr_size;
}

void TLDExecution::bb_scan_method(IplImage *firstFrame, CvRect bb)
{

	my_tld->currentFrames.iplimage_ptr.rgb_ipl=firstFrame;
	my_tld->currentFrames.iplimage_ptr.input_ipl = cvCreateImage(cvSize(img_size.width,img_size.height),8,1);
	my_tld->currentFrames.iplimage_ptr.blur_ipl = cvCreateImage(cvSize(img_size.width,img_size.height),8,1);
	cvCvtColor(firstFrame,my_tld->currentFrames.iplimage_ptr.input_ipl,CV_RGB2GRAY);
	cvSmooth(my_tld->currentFrames.iplimage_ptr.input_ipl,my_tld->currentFrames.iplimage_ptr.blur_ipl,CV_GAUSSIAN, 11, 0,2 ,2);
	
	bb_size.width = bb.width;
	bb_size.height = bb.height;

	int min_win = my_tld->model.min_win;

	const int n=21;

	float SHIFT = 0.1;
	float SCALE[n];
	float factor = 1.2;
	int k=0;

	for(int i=-10; i<=10;i++)	/* There are 21 scales..*/
	{
		SCALE[k] = std::pow(factor,i);
		k++;
	}


	int bbW[n], bbH[n];
	float bbSHH[n], bbSHW[n];
	

	for(int k=0; k<n;k++)
	{
		bbW[k] = cvRound(this->bb_size.width * SCALE[k]);
		bbH[k] = cvRound(this->bb_size.height * SCALE[k]);
	}	

	for(int k=0;k<n;k++)
	{
		bbSHH[k] = SHIFT * std::min(bbH[k],bbH[k]);
		bbSHW[k] = SHIFT * std::min(bbH[k],bbW[k]);
	}
	int bbF[] = {2,2,img_size.width, img_size.height};
	
	int idxm = 1;

	int left_limit;
	int top_limit;

	//float left_step;
	int left_step;
	int top_step;
	
	int **grid;
	int cols ;
	CvPoint sca1;
	total_grid_cols = 0;

	float2DArrayStruct tldGrid;
	float2DArrayStruct scales;
	
	tldGrid.rows = 6;
	tldGrid.cols = 0;
	
	scales.ptr  = (float*)malloc(21*2*sizeof(float));
	scales.cols = 0;
	scales.rows  = 2;

	int flag = 0;
	int idx=0;
	int leftElms=0;
	int topElms=0;
	int lCount = 0;

	for(int i=0;i<n;i++)
	{

		if(bbW[i] < min_win || bbH[i] < min_win)
			continue;
		
		left_limit = bbF[2] - bbW[i] - 1;
		
		top_limit = bbF[3] - bbH[i] - 1;
		
		if(left_limit <=0 || top_limit <=0)
			continue;

		left_step = bbSHW[i];
		top_step = bbSHH[i];

		leftElms = ((left_limit-bbF[0])/left_step)+1;
		topElms = ((top_limit-bbF[1])/top_step)+1;

		topVals = (int*)malloc(topElms*sizeof(int));
		leftVals = (int*)malloc(leftElms*sizeof(int));

		idx = bbF[0];
		leftVals[0] = idx;
	
		for(int h=1;h<leftElms;h++)
		{
			idx+=left_step;
			if(idx>left_limit)
				break;
			leftVals[h] = idx;
			
		}
			
		idx = bbF[1];
		topVals[0]=idx;
		for(int h=1;h<topElms;h++)
		{
			
			idx+=top_step;
			if(idx>top_limit)
				break;
			topVals[h] = idx;
		}

		grid = ntuples_cuda.calcntuples(topVals,leftVals,topElms,leftElms);
		/*
		output of ntuples
		--------------------
		(t1,l1)(t2,l1)(tn,l1).....(t1,l2)(t2,l2)....(t1,lm)(t2,lm)(t3,lm)()
		*/
		cols = topElms * leftElms;
		
		if(cols==0)
			break;

		total_grid_cols += cols;
		flag++;
		
		if(flag==1)
		{
			tldGrid.ptr = (float*)malloc(n*tldGrid.rows*topElms*leftElms*sizeof(float)); 
		}		

		/*
		In the grid first 4 elements in a column give the indexes to a box in the grid..5th element gives the scale of the box
		*/
		for(int j=0;j<topElms*leftElms;j++)
		{
			tldGrid.ptr[0+tldGrid.rows*tldGrid.cols]    = static_cast<float>(grid[1][j]);				//x1
			tldGrid.ptr[1+tldGrid.rows*tldGrid.cols]    = static_cast<float>(grid[0][j]);				//y1
			tldGrid.ptr[2+tldGrid.rows*tldGrid.cols]    = static_cast<float>(grid[1][j] + bbW[i] - 1);	//x2
			tldGrid.ptr[3+tldGrid.rows*tldGrid.cols]    = static_cast<float>(grid[0][j] + bbH[i] - 1);	//y2
			tldGrid.ptr[4+tldGrid.rows*tldGrid.cols]    = static_cast<float> (1*idxm);					//scale index
			tldGrid.ptr[5+tldGrid.rows*tldGrid.cols++]  = static_cast<float> (leftElms*1);				//no of boxes in one row
		}
		scales.ptr[2*scales.cols] = bbW[i];
		scales.ptr[2*scales.cols+1] = bbH[i];
		scales.cols++;

		// call for ntuples and obtain the grid..
		idxm++;


		free(leftVals);
		free(topVals);
	}// end of 21 itr for loop.
	tldGrid.ptr = (float*)realloc((float*)tldGrid.ptr,tldGrid.rows*tldGrid.cols*sizeof(float));
	scales.ptr = (float*)realloc((float*)scales.ptr,scales.cols*scales.rows*sizeof(float));

	my_tld->grid = tldGrid;
	my_tld->scales = scales;
	free(*grid);

	
}
int** TLDExecution::ntuples(int *top, int* left,int tElms, int lElms)
{
	int rows = 2;
	int cols = lElms*tElms;

	cv::Mat grid_mat(rows,cols,CV_8U);
	int **grid;
	grid = (int**)malloc(rows*sizeof(int *));

	for(int i=0;i<rows;i++){
		grid[i] = (int* )malloc(cols*sizeof(int *));
	}

	int k=0;
	int m=0;
	int count = 1;
	for(int i=0;i<cols;i++)
	{		
		grid[0][i] = top[k];	
		grid[1][i] = left[m];
		m++;
		if(m==lElms){
			k++;
			m=0;
		}
	}
	return grid;
}
