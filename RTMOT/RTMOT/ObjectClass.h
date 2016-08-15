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
#ifndef OBJ_H
#define OBJ_H

#include "IncludeFiles.h"
#include "TLDstructs.h"
#include "TLDParams.h"
#include "TLDExecution.h"
#include "GenerateFeatures.h"
#include "bb_overlap.h"
#include "GenPositiveData.h"
#include "common_functions.h"
#include "negativeData.h"
#include "SplitNegativeData.h"
#include "tldTrainNN.h"
#include "tldTracking.h"
#include "tldDetection.h"
#include "ClusterConfidence.h"
#include "TldLearning.h"

#include "CudaFunctions.h"
#include "SearchState.h"



class ObjectClass
{
public:

	long start,  end;	/* for timing*/
	
	int captureID;	/* store the camera ID*/
	int objectID;
	int frameWidth;
	int frameHeight;
	CvSize frameSize;

	bool boxStatus;

	static int objectCount;
	
	static CvScalar color_tab[5];
	void setColors();
	void setObjectColor(int id);
	CvScalar objColor;
	void displayObject();
	CvRect bb;
	CudaFunctions *cu_func;

	tld_Structure *tld;
	TLDParams *tld_para;
	TLDExecution *tld_exec;
	GenerateFeatures genFeat;

	bb_overlapClass bb_lap;
	GenPositiveData genPos;
	common_functions* comfunc;
	negativeData *nd;
	SplitNegativeData *splitNeg;
	tldTrainNN *train_nn;
	tldTracking tld_track;
	tldDetection tldDetect;
	ClusterConfidence clusConf;
	TldLearning tldLearn;

	float2DArrayStruct offsets;
	floatArrayStruct overlap;

	SearchState *s_state;

	ObjectClass(int oId, int cid,CvRect bb,int w, int h,IplImage *rgbCurr,IplImage *grayFrame,IplImage *blurFrame)
	{
		frameWidth = w;
		frameHeight = h;
		frameSize = cvSize(w,h);
		objectID = oId;
		captureID = cid;
		boxStatus = true;

		tld = new tld_Structure();
		tld_para = new TLDParams(tld);
		tld_exec = new TLDExecution(frameSize,tld);	
		s_state = new SearchState(3);

		tld_exec->bb_scan_method(rgbCurr,bb);	/* Creating the GRID */
		
		genFeat.featurePatterns(tld);			/* Generate Feature Patterns*/
		/* TREES = 10, FEATURES = 13. A feature has 4 points (x1,y1) and (x2,y2)*/

		cu_func = new CudaFunctions(oId);
		offsets = createOffsets(tld->scales,tld->features.x.ptr);

		cu_func->initMemTrack(w,h);
		cu_func->loadImageTrack((unsigned char*)rgbCurr->imageData);

		cu_func->initializeFern(tld->grid,offsets,w,h);
		cu_func->loadImage((unsigned char*)rgbCurr->imageData); 

		tld_para->tldnGrid = tld->grid.cols;
		tld_para->create_temporal_structure();

		tld->conf.ptr=(float*) malloc(1*sizeof(float));
		tld->conf.ptr[0]=1;
		tld->conf.size=1;

		tld->valid.ptr=new float[1];
		tld->valid.ptr[0]=1;
		tld->valid.size=1;

		tld->size.ptr=new float[1];
		tld->size.ptr[0]=1;
		tld->size.size=1;

		float2DArrayStruct tldGridArray=tld->grid;

		float bb1[]= {bb.x,bb.y,bb.x+bb.width,bb.y+bb.height} ;//[xmin ymin width height]
		
		overlap.ptr = (float*) malloc(sizeof(float)*tldGridArray.cols);
		overlap.size = tldGridArray.cols;
		cu_func->calcBBOverLapWithGrid(bb1,1,overlap.ptr);

		int2DArrayStruct pX;
		float2DArrayStruct pEx; /* pixels in image patch - mean of the patch.*/
		floatArrayStruct bbP;	/* the grid box that has maximum overlap*/

		genPos.genPositiveData(tld,&overlap,&pX,&pEx,&bbP,cu_func,true);

		tld->bb.ptr=(float*)malloc(bbP.size*sizeof(float));
		tld->bb.cols=1;
		tld->bb.rows=4;
		memcpy(tld->bb.ptr,bbP.ptr,bbP.size*sizeof(float));	/*Copy the bounding box to tld*/
		
		comfunc=new common_functions();		
		tld->var=comfunc->varience(pEx.ptr,pEx.rows)/2; /* sum((x-mean)^2)/#elements */

		std::cout << "Variance : " << tld->var << std::endl;

		nd=new negativeData();
		negativeDataOut output=nd->generateNegativeData(overlap,tld->var,tld->currentFrames.iplimage_ptr.input_ipl,tld->currentFrames.iplimage_ptr.blur_ipl,cu_func,tld->grid);
		/*nX and nEx values are found*/

		int M1=output.nX.rows;
		int N1=output.nX.cols;

		int M2=output.nEx.rows;
		int N2=output.nEx.cols;

		int p=((int)N1/2)*M1*sizeof(int);
		int q=((int)N2/2)*M2*sizeof(float);

		int2DArrayStruct nX1;
		int2DArrayStruct nX2;
		nX1.cols = N1/2;
		nX1.rows = M1;		
		nX1.ptr = (int*)malloc(sizeof(int)*nX1.cols*nX1.rows);

		nX2.cols = N1/2;
		nX2.rows = M1;
		nX2.ptr = (int*)malloc(sizeof(int)*nX2.cols*nX2.rows);
		
		float* nEx1=(float*) malloc(q);
		float* nEx2=(float*) malloc(q);


		splitNeg = new SplitNegativeData(); /*nX and nEx are split into two sets*/
		splitNeg->tldSplitNegativeData(output.nX.ptr,M1,N1,output.nEx.ptr,M2,N2,nX1.ptr,nX2.ptr,nEx1,nEx2);

		//-----------------Introducing some sort of randomness before training------------------------//

		int* idx=(int*)malloc((pX.cols+N1/2)*sizeof(int)); // pX.Cols = ?
		for(int i=0;i<pX.cols+N1/2;i++)
			idx[i]=i;
		std::random_shuffle(idx,idx+pX.cols+N1/2); // random_shuffle(first,last)

		int2DArrayStruct tmp;

		tmp.rows=pX.rows;
		tmp.cols=pX.cols+N1/2;
		tmp.ptr=(int*)malloc(tmp.cols*tmp.rows*sizeof(int));
		int* tmp2=(int*)malloc(tmp.cols*tmp.rows*sizeof(int));

		memcpy(tmp2,pX.ptr,pX.rows*pX.cols*sizeof(int));	// first copy pX to tmp2
		memcpy(tmp2+pX.rows*pX.cols,nX1.ptr,output.nX.cols/2*output.nX.rows*sizeof(int)); // then copy nX1 next to pX..
		
		for(int i=0;i<tmp.cols;i++)
		{
			memcpy(tmp.ptr+i*tmp.rows,tmp2+idx[i]*tmp.rows,tmp.rows*sizeof(int));			
		}		
		free(tmp2);

		bool* pY = (bool*)calloc((pX.cols+N1/2),sizeof(bool));		// all pYs are set to zero
		bool* temp3 = (bool*)malloc((pX.cols+N1/2)*sizeof(bool));   
		memset(pY,1,pX.cols*sizeof(bool));	// in pY, set 1 upto pX.cols.
		memcpy(temp3,pY,(pX.cols+N1/2)*sizeof(bool)); // in temp3 first part has 1s and second part has zeros..

		for(int i=0;i<(pX.cols+N1/2);i++)
		{
			pY[i]=temp3[idx[i]];
		}

		// Train using training set ------------------------------------------------

		int bootstrap = 2;
		cu_func->fernLearning(pX,nX1,5.0,5.0,bootstrap); /*---fernLearning CUDA implementation---*/

		free(pY);
		free(pX.ptr);
		free(temp3);		
		free(idx);
	
		tld->pex.ptr=NULL;
		tld->pex.cols=0;
		tld->pex.rows=0;
		tld->nex.ptr=NULL;
		tld->nex.cols=0;
		tld->nex.rows=0;

		train_nn = new tldTrainNN();
		train_nn->doTrainNN(pEx.ptr,pEx.rows,pEx.cols,nEx1,N2/2,tld); /*tld->pex and tld->nex are updated */
		tld->model.num_init = tld->pex.cols;
		std::cout << "num init " << tld->model.num_init;

		float* conf_fern = (float*)malloc(sizeof(float)*nX2.cols);

		cu_func->fern3(nX2,conf_fern);	/*--- fern3 CUDA implementation---*/


		float *conf_fern_max=std::max_element(conf_fern,conf_fern+nX2.cols);
		float a=conf_fern_max[0]/tld->model.num_trees;
		float b=tld->model.thr_fern;
		tld->model.thr_fern=std::max(a,b);

		std::cout << "new thr fern " << tld->model.thr_fern;
		
		free(conf_fern);

		floatArrayStruct conf1_nn;
		conf1_nn.size=N2/2;//n
		conf1_nn.ptr=(float*) malloc(conf1_nn.size*sizeof(float));
	
		floatArrayStruct conf2_nn;
		conf2_nn.size=N2/2;//n
		conf2_nn.ptr=(float*) malloc(conf2_nn.size*sizeof(float));

		int2DArrayStruct isin_nn;
		isin_nn.rows=3;
		isin_nn.cols=N2/2;
		isin_nn.ptr=(int*) malloc(isin_nn.rows*isin_nn.cols*sizeof(int));

		train_nn->tnn.dotldNN(nEx2,N2/2,tld->pex.rows,tld->pex.ptr,tld->pex.cols,tld->nex.ptr,tld->nex.cols,conf1_nn.ptr,conf2_nn.ptr,isin_nn.ptr);
		/* Update tld->pex and tld->nex based on nEx2*/

		float *conf_nn_max=std::max_element(conf1_nn.ptr,conf1_nn.ptr+conf1_nn.size);
		tld->model.thr_nn=std::max(tld->model.thr_nn,conf_nn_max[0]);	/* Update nn threshold*/
	
		free(conf1_nn.ptr);
		free(conf2_nn.ptr);		
		free(isin_nn.ptr);

		tld->model.thr_nn_valid = std::max(tld->model.thr_nn_valid,tld->model.thr_nn);

		free(nX1.ptr);
		free(nX2.ptr);
		free(nEx1);
		free(nEx2);

		free(overlap.ptr);



	} //end of ObjectClass Constructor.

	float2DArrayStruct createOffsets(float2DArrayStruct scales,float* features);

	~ObjectClass()
	{
		delete tld;
		delete tld_para;
		delete tld_exec;
		delete comfunc;
		delete nd;
		delete train_nn;
		delete s_state;
	}
	

};

#endif OBJ_H
