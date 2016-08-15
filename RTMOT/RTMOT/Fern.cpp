#include "stdafx.h"
#include "Fern.h"

Fern::Fern()
{
	nBIT = 1;
	BBOX_STEP = 7;
	BBOX=0;
	OFF = 0;
	IIMG = 0;
	IIMG2 = 0;

	tldObj=new tld();	/* tld is for creating Integral Images*/

}
Fern::~Fern()
{
	free(tldObj);
	free(image);

	free(BBOX);
	free(OFF );
	free(IIMG);
	free(IIMG2);

	//free(sca);
	//free(tldFeatures);
	//free(gray_image);
	//free(matImg);
}
int* Fern::create_offsets_bbox(float *bb0) 
{

	int *offsets = (int*) malloc(BBOX_STEP * nBBOX * sizeof(int));
	int *off = offsets;
	int *print = offsets;

	int temp;
	for (int i = 0; i < nBBOX; i++) // nBBOX => a large value..Columns in the grid
	{
		float *bb = bb0+mBBOX*i; // bb[0] = x1, bb[1] = y1, bb[2] = x2, bb[3] = y2, bb[4] = scale, bb[5] = Num of boxes

		*off++ = sub2idx(bb[1]-1,bb[0]-1,iWIDTH);		// (y1+0.5)*iWIDTH + (x1+0.5) // X1,Y1 point in the image
		*off++ = sub2idx(bb[3]-1,bb[0]-1,iWIDTH);		// (y2+0.5)*iWIDTH + (x1+0.5) // X1,Y2 point in the image
		*off++ = sub2idx(bb[1]-1,bb[2]-1,iWIDTH);		// (y1+0.5)*iWIDTH + (x2+0.5) // X2,Y1 point in the image
		*off++ = sub2idx(bb[3]-1,bb[2]-1,iWIDTH);		// (y2+0.5)*iWIDTH + (x2+0.5) // X2,Y2 point in the image
		*off++ = (int) ((bb[2]-bb[0])*(bb[3]-bb[1]));	// (X2-X1)*(Y2-Y1) // BOX_width* BOX_height
		*off++ = (int) (bb[4]-1)* 2 * nFEAT * nTREES;	// pointer to features for this scale
		*off++ = bb[5];									// number of left-right bboxes, will be used for searching neighbours
	}

	return offsets;
}
int* Fern::create_offsets(float *scale0, float *x0) 
{

	int *offsets = (int*) malloc(nSCALE * nTREES * nFEAT*2*sizeof(int));
	int *off = offsets;
	
	int g=0;
	for (int k = 0; k < nSCALE; k++) // SCALES = 21, scale[0] = width scale, scale[1] - height scale
	{
		float *scale = scale0+2*k;
		
		for (int i = 0; i < nTREES; i++) // nTREES = 10
		{
			for (int j = 0; j < nFEAT; j++)  // nFEAT = 13. 
			{
				float *x  = x0 +(4*nFEAT)*i +4*j;
				*off++ = sub2idx((scale[1]-1)*x[1],(scale[0]-1)*x[0],iWIDTH); // Height SCALE *y1* iWIDTH + Width SCALE*x1
				*off++ = sub2idx((scale[1]-1)*x[3],(scale[0]-1)*x[2],iWIDTH); // Height SCALE *y2* iWIDTH + Width SCALE*x2		
			}
		}
	}

	return offsets;
}

// ---------------------------------FERN main Functions ----------------------------------//
void Fern::fern0()
{
		srand(0); // fix state of random generator

		thrN = 0; 
		nBBOX = 0;
		mBBOX = 0; 
		nTREES = 0; 
		nFEAT = 0; 
		nSCALE = 0; 
		iHEIGHT = 0; 
		iWIDTH = 0;

		free(BBOX); BBOX = 0;// free deallocates memory assigned to BBOX. But keeps the pointer
		free(OFF);  OFF = 0;
		free(IIMG); IIMG = 0;
		free(IIMG2);IIMG2 = 0;
		
		WEIGHT.clear();
		nP.clear();
		nN.clear();

}


void Fern::fern1(tld_Structure* tld)
{
	image =tld->currentFrames.iplimage_ptr.input_ipl; //tld_source_img0_input;
	iWIDTH  = image->width;
	iHEIGHT = image->height;	

	nTREES = tld->features.x.rows; // 10
	nFEAT = tld->features.x.cols/4; // 52/4 = 13
	thrN = 0.5 * nTREES;
	nSCALE = tld->scales.cols;

	IIMG       = (float*) malloc(iHEIGHT*iWIDTH*sizeof(float)); // for Integral Image
	IIMG2      = (float*) malloc(iHEIGHT*iWIDTH*sizeof(float));	// for square Integral Image

	mBBOX		=tld->grid.rows; // 6
	nBBOX       =tld->grid.cols; // large value

	BBOX= create_offsets_bbox(tld->grid.ptr);	
	// BBOX has (x1,y1), (x2,y1), (x1,y2), (x2,y2) on the Image for the prepared grid.

	float *x  = tld->features.x.ptr;	
	float *s  = tld->scales.ptr;

	OFF	= create_offsets(s,x);
	//OFF has p1,p2 points in the image in all scales for a box in the grid

	for (int i = 0; i<nTREES; i++) 
	{
		WEIGHT.push_back(std::vector<float>((int)std::pow(2.0,nBIT*nFEAT),0));			
		nP.push_back(std::vector<int>((int)std::pow(2.0,nBIT*nFEAT),0));
		nN.push_back(std::vector<int>((int)std::pow(2.0,nBIT*nFEAT),0));	
	}

}

fern5DataOut Fern::fern5(char *input,char *blur,int *idx,int Idx_rows,int Idx_cols,float minVar)
{
	// input = gray image, blur = smoothed image, idx = indexes of overlapped grid boxes, Idx_rows = # grid boxes(<=100)
	// Idx_cols = 1, 
	// initially minVar = 0	
	// in negative data call = minVar > 0


	int numIdx = Idx_rows * Idx_cols;	// initially numIdx = 100 x 1;

	if (minVar > 0)	 /*Initially minVar =0*/
	{
		tldObj->iimg(input,IIMG,iHEIGHT,iWIDTH);	// integral image creation
		tldObj->iimg2(input,IIMG2,iHEIGHT,iWIDTH);
	}

	float *patt = new float[nTREES*numIdx];		// 10 x numIdx (<=100)
	float *status = new float[numIdx];			// 1 x numIdx (<=100)

	for (int j = 0; j < numIdx; j++) // numIdx usually <=100
	{

		if (minVar > 0)	// Initially minVar =0
		{
			float bboxvar = tldObj->bbox_var_offset(IIMG,IIMG2,BBOX+j*BBOX_STEP);			
			if (bboxvar < minVar) {	continue; }
		}
		status[j] = 1;
		float *tPatt = patt + j*nTREES;
		
		for (int i = 0; i < nTREES; i++) // nTREES = 10 
		{
			tPatt[i] = (float) measure_tree_offset(blur, idx[j], i); // (blur image, grid box index, treeid)
			// a 13bit value is return corresponding to feature points in the grid box
		}
	}

	fern5DataOut a;
	a.nX.ptr=patt;
	a.nX.rows=nTREES;
	a.nX.cols=numIdx;

	a.status.ptr=status;
	a.status.size=numIdx;
	
	return a;

}



float* Fern::fern2(float* X1,int X1_cols,bool* Y1,float Margin,int bootstrap)
{ //output is not checked all the time

	// X1 == > [pX nX1], Y1 ==> [111..00000], initially bootstrap =2

	float *X = X1;			// tls [pX nX1] randomly selected..
	int numX = X1_cols;		// size of tld->X
	bool *Y = Y1;			// tld->Y [pY nY1] randomly selected..
	float thrP = Margin * nTREES;
//	int bootstrap = Bootstrap;

	int step = numX / 10;


	for (int j = 0; j < bootstrap; j++) 
	{
		for (int i = 0; i < step; i++) 
		{
			for (int k = 0; k < 10; k++) 
			{
				int I = k*step + i;
				float *x = X+nTREES*I;
				if (Y[I] == 1) /*if feature in *x is labled as positive in Y[] */
				{
					if (measure_forest(x) <= thrP)
						update(x,1,1);											
				} 
				else 
				{
					if (measure_forest(x) >= thrN)
						update(x,0,1);
				
				}
			}
		}
	}



	float *resp0= new float[numX];
	float* tempresp0 = resp0;
	

	for (int i = 0; i < numX; i++) {
		*tempresp0++ = measure_forest(X+nTREES*i);
	}


	return resp0;

}

void Fern::fern3(float* X1,int X1_cols,float2DArrayStruct* conf_fern)
{
	// X1 = nX2 array, X_cols = #cols in nX2

	float *X = X1;
	int numX = X1_cols;
	conf_fern->cols=numX;
	conf_fern->rows=1;
	conf_fern->ptr=(float*) malloc(conf_fern->cols*conf_fern->rows*sizeof(float));
	
	float *resp0= conf_fern->ptr;

	for (int i = 0; i < numX; i++) 
	{
		*resp0++ = measure_forest(X+nTREES*i);
	}
	

}

void Fern::fern4(char *input,char *blur,float *conf,int conf_cols,float *patt,int patt_cols ,float maxBBox,float minVar)
{

		if ( conf_cols!= nBBOX) { printf("Wrong input.\n"); return; }

		//float *patt = mxGetPr(prhs[5]); 
		if (patt_cols != nBBOX) { printf("Wrong input.\n"); return; }
		
		for (int i = 0; i < nBBOX; i++) 
		{ 
			conf[i] = -1; 
		}

		// Setup sampling of the BBox
		//float probability = *mxGetPr(prhs[2]);
		float probability=maxBBox;

		float nTest  = nBBOX * probability; if (nTest <= 0) return;
		if (nTest > nBBOX) nTest = nBBOX;
		float pStep  = (float) nBBOX / nTest;
		float pState = randdouble() * pStep;


		// Integral images
	
		tldObj->iimg(input,IIMG,iHEIGHT,iWIDTH);
		tldObj->iimg2(input,IIMG2,iHEIGHT,iWIDTH);

		// log: 0 - not visited, 1 - visited
		int *log = (int*) calloc(nBBOX,sizeof(int));

		// variance
		//float minVar = *mxGetPr(prhs[3]);

		// totalrecall
		//float totalrecall = *mxGetPr(prhs[6]);
		int I = 0;
		int K = 2;

		while (1) 
		{
			// Get index of bbox
			I = (int) floor(pState);
			pState += pStep;
			if (pState >= nBBOX) { break; }

			// measure bbox
			log[I] = 1;
			float *tPatt = patt + nTREES*I;
			conf[I] = measure_bbox_offset(blur,I,minVar,tPatt);
		}

		free(log);

}

float Fern::randdouble() 
{ 
	return rand()/(float(RAND_MAX)+1); 
}  

float Fern::measure_bbox_offset(char *blur, int idx_bbox, float minVar, float *tPatt) 
{

	float conf = 0.0;
	float bboxvar = tldObj->bbox_var_offset(IIMG,IIMG2,BBOX+idx_bbox*BBOX_STEP);
	if (bboxvar < minVar) {	return conf; }

	for (int i = 0; i < nTREES; i++) { 
		int idx = measure_tree_offset(blur,idx_bbox,i);
		tPatt[i] = idx;
		conf += WEIGHT[i][idx];
	}
	return conf;
}

void Fern::update(float *x, int C, int N) {
	
	for (int i = 0; i < nTREES; i++) {

		int idx = (int) x[i];

		(C==1) ? nP[i][idx] += N : nN[i][idx] += N;

		if (nP[i][idx]==0) {
			WEIGHT[i][idx] = 0;
		} else {
			WEIGHT[i][idx] = ((float) (nP[i][idx])) / (nP[i][idx] + nN[i][idx]);
		}
	}
}


int Fern::measure_tree_offset(char *img, int idx_bbox, int idx_tree) 
{
	// img = blured image, idx_bbox = grid box index, idx_tree = TREE number

	int index = 0;
	
	// BBOX = grid box points in image, BBOX_STEP = 7
	int *bbox = BBOX + idx_bbox*BBOX_STEP; 
	
	// OFF = feature points in image, bbox[5] = scale related to this grid box, 
	int *off = OFF + bbox[5] + idx_tree*2*nFEAT;
		
	for (int i=0; i<nFEAT; i++) // nFEAT
	{
		index<<=1;
		int position1 = off[1]+bbox[0];
		position1 = (position1>iHEIGHT*iWIDTH)?iHEIGHT*iWIDTH:position1;

		int position0 = off[0]+bbox[0];
		position0 = (position0>iHEIGHT*iWIDTH)?iHEIGHT*iWIDTH:position0;

		int fp0 = img[position0]; // pixel value in position0
		int fp1 = img[position1]; // pixel value in position1
		if (fp0>fp1) { index |= 1;}
		off += 2;
	}
	
	return index;	
}
float Fern::measure_forest(float *idx) 
{
	float votes = 0;
	for (int i = 0; i < nTREES; i++) 
	{ 
		//std::cout<<idx[i]<<"  ";
		votes += WEIGHT[i][idx[i]];
	}
	//std::cout<<std::endl;
	return votes;
}
