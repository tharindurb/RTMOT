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
#include "tldTrainNN.h"

//Training the nearest neighbor classifier using negative and positive data

void tldTrainNN::randperm_tldNN(int n,int* perm)
{
    int i, j, t;
	srand ((unsigned) time(0) );

    for(i=0; i<n; i++)
        perm[i] = i;
    for(i=0; i<n; i++) 
	{
        j = rand()%(n-i)+i;
        t = perm[j];
        perm[j] = perm[i]; 
        perm[i] = t;
    }
}


void tldTrainNN::doTrainNN(float *pEx, int M1, int N1, float *nEx1, int N2, tld_Structure *tld)
{
	

    int pN1=tld->pex.cols;        //no of coloumns of pex
    int nN1=tld->nex.cols;        //no of coloumns of nex

    int nP = N1;// get the number of positive examples
    int nN = N2; // get the number of negative examples

    int s=M1*N1+M1*N2;      
    int p=M1*N1;
    int q=M1*N2;

    float* x=(float*) malloc(s*sizeof(float));
    memcpy(x,pEx,M1*N1*sizeof(float));
    memcpy(x+M1*N1,nEx1,M1*N2*sizeof(float));

    int t=nP+nN;
    float* y=(float*) malloc(t*sizeof(float));

    for(int i=0;i<nP;i++)
       y[i]=1;
  

    for(int i=0;i<nN;i++)
       y[nP+i]=0;
  
    int* idx=(int*) malloc(t*sizeof(int));
    randperm_tldNN(t,idx);

    int s2=s+M1;
    float* x2=(float*) malloc(s2*sizeof(float));
    memcpy(x2,pEx,M1*sizeof(float));
  
    for (int i=0; i<t;i++)
    {
        int index=idx[i];
        int inx=index*M1;
        memcpy(x2+i*M1+M1,x+inx,M1*sizeof(float));
    }  
    free(x);

    float* y2=(float*) malloc((t+1)*sizeof(float));
    y2[0]=1;
    for (int i=1;i<(t+1);i++)
    {
       y2[i]=y[idx[i-1]];
    }

    free(y);
    free(idx);

    float* conf1=(float*) malloc((t+1)*sizeof(float));
    float* conf2=(float*) malloc((t+1)*sizeof(float));
    int* isin=(int*) malloc(3*(t+1)*sizeof(int));

    tnn.dotldNN(x2,(t+1),M1,tld->pex.ptr,tld->pex.cols,tld->nex.ptr,tld->nex.cols,conf1,conf2,isin); // measure Relative similarity


    for(int i=0;i<t+1;i++)
    {
    
    // Positive
       if (y2[i] == 1 && conf1[i] <= tld->model.thr_nn)
       {   
           if(isin[3*i+1]== -1)

           {    
               if(tld->pex.cols==0)
               {
                   tld->pex.ptr=(float*) malloc(M1*sizeof(float));
                   tld->pex.rows=M1;
                   
               }
               else
               {
                   tld->pex.ptr=(float*) realloc(tld->pex.ptr,M1*sizeof(float));
                    
               }
               
               memcpy(tld->pex.ptr,x2+i*M1,M1*sizeof(float));              
               tld->pex.cols=1;
               pN1=1;
                if(i<t)
                {
                    tnn.dotldNN(x2+(i+1)*M1,(t-i),M1,tld->pex.ptr,tld->pex.cols,tld->nex.ptr,tld->nex.cols,conf1+i+1,conf2+i+1,isin+3*(i+1));

                }
               continue;
          }
                           
          if(tld->pex.cols != 0)
          {  
               int num=isin[3*i+1]+1;
               float* pex2=(float*) malloc(M1*(pN1+1)*sizeof(float)); 
                           
               memcpy(pex2,tld->pex.ptr,M1*num*sizeof(float));      
               memcpy(pex2+num*M1,x2+i*M1,M1*sizeof(float));              
               memcpy(pex2+(num+1)*M1,tld->pex.ptr+num*M1,M1*(pN1-num)*sizeof(float));     
               free(tld->pex.ptr);
               tld->pex.ptr=pex2;
               pN1=pN1+1;
               tld->pex.cols=pN1;
              
                if(i<t)
                {
                    tnn.dotldNN(x2+(i+1)*M1,(t-i),M1,tld->pex.ptr,tld->pex.cols,tld->nex.ptr,tld->nex.cols,conf1+i+1,conf2+i+1,isin+3*(i+1));

                }
          }
          
        }
  
        // Negative
        if (y2[i] == 0 && conf1[i] > 0.5)
        {
            float* nex2=(float*) malloc(M1*(nN1+1)*sizeof(float));         
            if(tld->nex.cols != 0)
            {
                memcpy(nex2,tld->nex.ptr,M1*nN1*sizeof(float));                                   
                free(tld->nex.ptr);
            }
          
            memcpy(nex2+M1*nN1,x2+i*M1,M1*sizeof(float));
            tld->nex.ptr=nex2;
            nN1++;
            tld->nex.cols=nN1;

                if(i<t)
                {
                    tnn.dotldNN(x2+(i+1)*M1,(t-i),M1,tld->pex.ptr,tld->pex.cols,tld->nex.ptr,tld->nex.cols,conf1+i+1,conf2+i+1,isin+3*(i+1));

                }

        }
    }
      
    free(conf1);
    free(conf2);
    free(isin);

    free(x2);
    free(y2);
   
  

}
