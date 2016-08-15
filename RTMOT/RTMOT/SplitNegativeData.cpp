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
#include "SplitNegativeData.h"

SplitNegativeData::SplitNegativeData()
{

}
SplitNegativeData::~SplitNegativeData()
{

}
void SplitNegativeData::randperm2(int n,int* perm)
{
    int i, j, t;
	srand ((unsigned) time(0) );

	for(i=0; i<n; i++)
	{
		perm[i] = i;
	}
    for(i=0; i<n; i++) 
	{
        j = rand()%(n-i)+i;
        t = perm[j];
        perm[j] = perm[i];
        perm[i] = t;
    }
}

void SplitNegativeData::tldSplitNegativeData(int* nX,int M1,int N1,float* nEx,int M2,int N2,int* nX1,int* nX2,float* nEx1,float* nEx2)
{
	int* idx=(int*) malloc(N1*sizeof(int));
	randperm2(N1,idx);// get the random permutation of numbers from 1 to N1

	int* nXn=(int*) malloc(N1*M1*sizeof(int));
	
	for (int i=0; i<N1;i++) // no need..
	{
		int index=idx[i];
		int inx=index*M1;
		for (int j=0; j<M1;j++)
		{
			nXn[i*M1+j]=nX[inx+j];
		}
	}


	for (int i=0; i<((int)N1/2)*M1;i++)
	{
		nX1[i]=nXn[i];
	}

	for (int i=0; i<((int)N1/2)*M1;i++)
	{
		nX2[i]=nXn[((int)N1/2)*M1+i];
	}

	free(nXn);
	free(idx);
	

//nEx

	int* idx2=(int*) malloc(N2*sizeof(int));
	randperm2(N2,idx2);

	float* nExn=(float*) malloc(N2*M2*sizeof(float));

	for (int i=0; i<N2;i++)
	{
		int index=idx2[i];
		int inx=index*M2;
		for (int j=0; j<M2;j++)
		{
			nExn[i*M2+j]=nEx[inx+j];
		}
	}

	for (int i=0; i<((int)N2/2)*M2;i++)
	{
		nEx1[i]=nExn[i];
	}

	for (int i=0; i<((int)N2/2)*M2;i++)
	{
		nEx2[i]=nExn[((int)N2/2)*M2+i];
	}

	free(nExn);
	free(idx2);

}
