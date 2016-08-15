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
#ifndef TLDSTRUCTS_H
#define TLDSTRUCTS_H

#pragma once

#include <stdlib.h>
#include <cv.h>
#include "DataStructs.h"

struct image
{
	int* ptr;
	 
	int width;
	int height;
	int channels;


};

struct Detection
{
	int num_detections;  
	float* bb;
	float* patch;
	float* patt;
	float* conf1;
	float* conf2;
	int* isin;

};

struct double2DArrayCell
{
	float2DArrayStruct* ptr;
	int no_of_arrays;//no of cells


};
struct tld_tmp
{
	tld_tmp(){};

	floatArrayStruct conf;
	//float2DArrayStruct patt;
	int2DArrayStruct patt;

};
struct tld_features
{
	tld_features(){};

       /*x: [52x10 float]
    type: 'forest'*/

	float2DArrayStruct x;
	char* type;

};

struct tld_p_par_update
{
	tld_p_par_update(){};
	int num_closest;
	int num_warps;
    int noise;
    int angle;
    float shift;
    float scale;
};


struct tld_n_par
{
	tld_n_par(){};
	float overlap;
	int num_patches;
};



struct tld_p_par_init
{
	tld_p_par_init(){};
	int num_closest;
	int num_warps;
	int noise;
	int angle;
	float shift;
	float scale;
};

struct tld_model
{
	tld_model(){};
	int min_win;
	int patchsize[2];
	int fliplr;
	bool update_detector;
	float ncc_thesame;
	float  valid;
	int  num_trees;
	int	  num_features;
	float  thr_fern;
	float   thr_nn;
	float   thr_nn_valid;
	int num_init;



};

struct tld_im0
{
	tld_im0(){};
	image input;
	
	image blur;
	
	image rgb;
	

};

struct tld_im0_ipl
{
	tld_im0_ipl(){};
	IplImage* input_ipl;
	IplImage* blur_ipl;
	IplImage* rgb_ipl;
};

struct imgSet
{
	tld_im0 image_ptr;
	tld_im0_ipl iplimage_ptr;

};


struct tld_control
{
	tld_control(){};
	int maxbox;
	bool update_detector;
	bool drop_img;
	bool repeat;

};


struct tld_tracker
{

	tld_tracker(){};
	int occlusion;


};



struct tld_Structure 
{

	tld_Structure() {};
	
	char output[25];
	
	tld_model model;
	tld_p_par_init p_par_init;
	tld_p_par_update   p_par_update;
	tld_n_par  n_par;
	
	float2DArrayStruct grid;//ngrid=grid.cols
	
	float2DArrayStruct scales;
	
	tld_features features;

	
	float2DArrayStruct bb;
	floatArrayStruct conf;
	floatArrayStruct  valid;
	floatArrayStruct  size;
	floatArrayStruct  trackerfailure;
	float2DArrayStruct  draw;
	float2DArrayStruct  pts;
	int imgsize[2];

	float2DArrayStruct* X;
	float2DArrayStruct* Y; 
	float2DArrayStruct* pEx;
	float2DArrayStruct* nEx;
	float2DArrayStruct target;//: [157x122 uint8]
	float var;
	float2DArrayStruct pex;
	float2DArrayStruct  nex;

	Detection dt;

	tld_tmp tmp;
	imgSet currentFrames;
	imgSet previousFrames;

	tld_tracker tracker;
	tld_control control;


};


#endif

