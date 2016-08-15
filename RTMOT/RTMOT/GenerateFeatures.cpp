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
#include "GenerateFeatures.h"

void GenerateFeatures::randperm(int n,int perm[])
{
	int i, j, t;

	for(i=0; i<n; i++)
		perm[i] = i;
	for(i=0; i<n; i++) {
		j = rand()%(n-i)+i;
		t = perm[j];
		perm[j] = perm[i];
		perm[i] = t;
	}
}


void GenerateFeatures::featurePatterns(tld_Structure* my_tld)
{
	int nTREES=my_tld->model.num_trees;			
	int nFEAT=my_tld->model.num_features;		
	
	float SHI=0.2;
	float SCA=1.0;
    int n=5;

	srand((unsigned)time(NULL));
	float* x= (float*)malloc(4*288*sizeof(float)); 

    int k=0;
	for(int i=0;i<=n;i++){ 
		for(int j=0;j<=n;j++){ 
			x[k]	= i*SHI;
			x[k+1]	= j*SHI;
			x[k+2]	= i*SHI + SCA*((float)rand()/(RAND_MAX+1))+SHI;
			x[k+3]	= j*SHI;

			x[288+k]		= i*SHI;
			x[288+k+1]	= j*SHI;
			x[288+k+2]	= i*SHI - SCA*((float) rand() / (RAND_MAX+1))+SHI;
			x[288+k+3]	= j*SHI;

			x[2*288+k]	= i*SHI;
			x[2*288+k+1]	= j*SHI;
			x[2*288+k+2]	= i*SHI;
			x[2*288+k+3]	= j*SHI - SCA*((float) rand() / (RAND_MAX+1))+SHI;
			
			x[3*288+k]	= i*SHI;
			x[3*288+k+1]	= j*SHI;
			x[3*288+k+2]	= i*SHI;
			x[3*288+k+3]	= j*SHI + SCA*((float) rand() / (RAND_MAX+1))+SHI;
			k=k+4;

		}
	}

	for(int i=0;i<=n;i++){
		for(int j=0;j<=n;j++){
			
			x[k]	= i*SHI + 0.1;
			x[k+1]	= j*SHI + 0.1;
			x[k+2]	= i*SHI + 0.1 + SCA*((float) rand() / (RAND_MAX+1))+SHI;
			x[k+3]	= j*SHI + 0.1;

			x[288+k]	= i*SHI + 0.1;
			x[288+k+1]	= j*SHI + 0.1;
			x[288+k+2]	= i*SHI + 0.1 - SCA*((float) rand() / (RAND_MAX+1))+SHI;
			x[288+k+3]	= j*SHI + 0.1;

			x[2*288+k]	= i*SHI + 0.1;
			x[2*288+k+1]	= j*SHI + 0.1;
			x[2*288+k+2]	= i*SHI + 0.1;
			x[2*288+k+3]	= j*SHI + 0.1- SCA*((float) rand() / (RAND_MAX+1))+SHI;
			
			x[3*288+k]		= i*SHI + 0.1;
			x[3*288+k+1]	= j*SHI + 0.1;
			x[3*288+k+2]	= i*SHI + 0.1;
			x[3*288+k+3]	= j*SHI + 0.1 + SCA*((float) rand() / (RAND_MAX+1))+SHI;
			k = k+4;

		}
	}

	k=0;
    float* y= (float*)malloc(4*288*sizeof(float)); 
	for(int i=0;i<288;i++){
		if(x[4*i] < 1 && x[4*i] > 0 && x[4*i+1] < 1 && x[4*i+1] > 0){
			y[4*k]=x[4*i]; 
			y[4*k+1] = x[4*i+1];

			if(x[4*i+2]>1){
				y[4*k+2]=1.0;}
			else if(x[4*i+2]<0){
				y[4*k+2]=0.0;}
			else{
				y[4*k+2]=x[4*i+2];}

			if(x[4*i+3]>1){
				y[4*k+3]=1.0;}
			else if(x[4*i+3]<0){
				y[4*k+3]=0.0;}
			else{
				y[4*k+3]=x[4*i+3];}

			k=k++;
		}
	}

	free(x);
	y=(float*)realloc(y,4*4*k);
    
 	int nPts = k;

	int* rp = (int*)malloc(nPts*sizeof(int));
	randperm(nPts,rp);

	float* z = (float*)malloc(4*nTREES*nFEAT*sizeof(float));
	float test;

	for(int i=0; i< nTREES*nFEAT;i++){
		z[4*i]   = y[4*(rp[i])];
		z[4*i+1] = y[4*(rp[i])+1];
		z[4*i+2] = y[4*(rp[i])+2];
		test=y[4*(rp[i])+3];
		z[4*i+3] = y[4*(rp[i])+3];
	}
	free(y);
	free(rp);

	for(int i =0;i< 4*nTREES*nFEAT;i++){
	   if(z[i] > 1 || z[i] < 0){ z[i] = 0.0;}
	}


	my_tld->features.type="forest";
	my_tld->features.x.ptr=z;
	my_tld->features.x.rows=nTREES;
	my_tld->features.x.cols=nFEAT*4;
	
}