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
#include "MultiStreams.h"
#include <time.h>
#include <omp.h>

void on_mouse(int event, int x, int y, int flags, void* param)
{
	VideoHandler *pvih = (VideoHandler*)param;	
	pvih->MouseHandle(event,x,y,flags,param);
}

void MultiStreams::initStreams(int n)
{
	num_of_cams = n;
	
	for(int i=0;i<num_of_cams;i++)
		vih[i].initVideoHandler();
	
}

void MultiStreams::freeStreams(int n)
{

}

void MultiStreams::runStreams(int arg)
{
	IplImage* timg[2];
	char* winnames[4]={"cam1","cam2","cam3","cam4"};

	int stream = arg;
	std::cout << stream <<" running now..." << std::endl;
	
	cvNamedWindow(winnames[stream],CV_WINDOW_AUTOSIZE);
	
	vih[stream].loadVideo(stream);
	cvSetMouseCallback(winnames[stream],on_mouse,(void*)&vih[stream]);
	
	int frameCount = 0;
	
	///time variables
	long start,  end; start = 0; end = 0; 

	
	for(;;)
	{
		frameCount++;
		vih[stream].rgbFrame =cvQueryFrame(vih[stream].capture);
		start = clock();/*---Timing Start---*/
		if(!timg[stream])
		{
			std::cout << "end of frames.reaches..\n";
			break;
		}

		cvCvtColor(vih[stream].rgbFrame,vih[stream].frame,CV_RGB2GRAY);
		
		if(vih[stream].onProcess && vih[stream].selection.height > 0 && vih[stream].selection.width > 0)
		{
			cvSetImageROI(vih[stream].rgbFrame,vih[stream].selection);
			cvXorS(vih[stream].rgbFrame,cvScalarAll(255),vih[stream].rgbFrame,0);
			cvResetImageROI(vih[stream].rgbFrame);
		}

		if(vih[stream].doneDrawing)
		{	
			
			vih[stream].initialized=true;
			vih[stream].numRectangles++;
			
			vih[stream].doneDrawing = false;
			vih[stream].doneSetting = false;
			vih[stream].onProcess = false;

			// create an object..
			obj[ObjectClass::objectCount] = new ObjectClass(ObjectClass::objectCount,stream,vih[stream].selection,vih[stream].frameWidth,vih[stream].frameHeight,vih[stream].rgbFrame,vih[stream].frame,vih[stream].blurFrame);
			
			
			vih[stream].boxes.push_back(vih[stream].selection);
			vih[stream].boxStatus.push_back(true);

			obj[ObjectClass::objectCount]->setColors();
			obj[ObjectClass::objectCount]->setObjectColor(ObjectClass::objectCount);
			
			obj[ObjectClass::objectCount]->cu_func->initPointSetMemoryTrack(vih[stream].numRectangles);

			obj[ObjectClass::objectCount]->s_state->current_search_cam = stream;
			obj[ObjectClass::objectCount]->s_state->search_count = 0;

			std::cout << "object " << ObjectClass::objectCount << " created" << std::endl;
			ObjectClass::objectCount++;

			cvCopyImage(vih[stream].rgbFrame,vih[stream].rgbImage);	

			continue;
			
		}	


		WaitForSingleObject( hObjectMutex, INFINITE );
		
		if(ObjectClass::objectCount>0)
		{
			vih[stream].boxes.clear();

			int streamBoxCount = 0;

			#pragma omp parallel for			/*--- Parallel processing of objects in OpenMP--- */ 
			// openMP didnt work for multiple video streams unless mutexes are used to control..

			for(int od=0;od<ObjectClass::objectCount;od++)
			{

				int oId = obj[od]->objectID;
				int streamId = obj[od]->captureID;
				
				if(streamId != stream)
					continue;

				obj[oId]->tld->currentFrames.iplimage_ptr.input_ipl=vih[stream].frame;
				obj[oId]->tld->currentFrames.iplimage_ptr.rgb_ipl=vih[stream].rgbFrame;

				obj[oId]->tld->previousFrames.iplimage_ptr.input_ipl=vih[stream].image;
				obj[oId]->tld->previousFrames.iplimage_ptr.rgb_ipl=vih[stream].rgbImage;
				
				obj[oId]->cu_func->loadImage((unsigned char*)vih[stream].rgbFrame->imageData);

				floatArrayStruct tBB;	/* Output bounding box from tracker.*/
				floatArrayStruct tConf;
				int tValid;
				
				
				tBB.size=0;
				tBB.ptr=(float*)calloc(4,sizeof(float));
				tConf.size=0;//patch width
				tConf.ptr=(float*) calloc(tConf.size,sizeof(float));
				tValid=0;

				intArrayStruct confIdx;
				confIdx.ptr = (int*)malloc(100*sizeof(int));
				confIdx.size = 100;
				
				if(obj[oId]->boxStatus) //if the there is a bounding box to track
					/*Tracking Function uses Median Flow tracking method refer to paper -> Forward-Backward Error: Automatic Detection of Tracking Failures*/				
					obj[oId]->tld_track.doTldTracking(obj[oId]->tld,&tBB,&tConf,&tValid,oId,obj[oId]->cu_func,&confIdx); 


				float2DArrayStruct dBB;
				floatArrayStruct dConf;

				/*Evaluate the Grid and try to find matching patches for the object*/
				obj[oId]->tldDetect.doTldDetection(obj[oId]->tld,obj[oId]->cu_func,&dBB,&dConf,&confIdx,obj[oId]->boxStatus); // ? what happens


				bool DT=((dBB.cols>0)?1:0);				
				bool TR=((tBB.ptr[2]<vih[stream].frameWidth)?1:0) && ((tBB.ptr[3]<vih[stream].frameHeight)?1:0) && ((tBB.ptr[1]>0)?1:0) && ((tBB.ptr[0]>0)?1:0);

				if(TR)
				{
					obj[oId]->tld->bb.rows=4;
					obj[oId]->tld->bb.cols=1;
					obj[oId]->tld->conf.ptr[0] = tConf.ptr[0];
					obj[oId]->tld->valid.ptr[0] = tValid;
					memcpy(obj[oId]->tld->bb.ptr,tBB.ptr,obj[oId]->tld->bb.rows*obj[oId]->tld->bb.cols*sizeof(float));

					if(DT)
					{
						
						float2DArrayStruct cBB;
						floatArrayStruct cConf;
						intArrayStruct cSize;


						obj[oId]->clusConf.bb_cluster_confidance(&dBB,&dConf,&cBB,&cConf,&cSize);
						floatArrayStruct out=obj[oId]->bb_lap.bb_overlap(obj[oId]->tld->bb.ptr,obj[oId]->tld->bb.rows,obj[oId]->tld->bb.cols,cBB.ptr,cBB.rows,cBB.cols);

						boolArrayStruct id;
						id.size=out.size;
						id.ptr=(bool*)malloc (id.size*sizeof(bool));
						int sum=0;
						for(int j=0;j<out.size;j++)
						{
							id.ptr[j]=((out.ptr[j]< 0.5) && (cConf.ptr[j] > obj[oId]->tld->conf.ptr[0])); // id = bb_overlap(my_tld.bb(:,I),cBB) < 0.5 & cConf > my_tld.conf(I);
							if(id.ptr[j]==1)
							{
								sum=sum+1;
							}
						}

						if(sum==1)
						{//one bounding box ,one cluster
						
							for(int k=0;k<id.size;k++)
							{
								if(id.ptr[k])
								{
									memcpy(obj[oId]->tld->bb.ptr,cBB.ptr+k*cBB.rows,cBB.rows*sizeof(float)) ;
									obj[oId]->tld->bb.cols=1;
									obj[oId]->tld->bb.rows=cBB.rows;


									memcpy(obj[oId]->tld->conf.ptr,(cConf.ptr+k),1*sizeof(float));
									obj[oId]->tld->conf.size=1;

									memcpy(obj[oId]->tld->size.ptr,(cSize.ptr+k),1*sizeof(float));
									obj[oId]->tld->size.size=1;

									obj[oId]->tld->valid.ptr[0] = 0;
									obj[oId]->tld->valid.size=1;
								}
							}
						}
						else
						{	// if sum!=1..

							floatArrayStruct idTr=obj[oId]->bb_lap.bb_overlap(tBB.ptr,tBB.size,1,obj[oId]->tld->dt.bb,4,obj[oId]->tld->dt.num_detections);
							float r0=0;
							float r1=0;
							float r2=0;
							float r3=0;
							int count=0;
							
							for(int k=0;k<idTr.size;k++)
							{
								if(idTr.ptr[k]>0.7)
								{
									r0 += obj[oId]->tld->dt.bb[4*k];
									r1 += obj[oId]->tld->dt.bb[4*k+1];
									r2 += obj[oId]->tld->dt.bb[4*k+2];
									r3 += obj[oId]->tld->dt.bb[4*k+3];
									count=count+1;
									
								}
							}
							r0=(r0+tBB.ptr[0]*10.0f)/(float)(10+count);
							r1=(r1+tBB.ptr[1]*10.0f)/(float)(10+count);
							r2=(r2+tBB.ptr[2]*10.0f)/(float)(10+count);
							r3=(r3+tBB.ptr[3]*10.0f)/(float)(10+count);


							obj[oId]->tld->bb.ptr[0]=r0;
							obj[oId]->tld->bb.ptr[1]=r1;
							obj[oId]->tld->bb.ptr[2]=r2;
							obj[oId]->tld->bb.ptr[3]=r3;				
							free(idTr.ptr);
						
						}//end else

						free(cBB.ptr);
						free(cConf.ptr);
						free(cSize.ptr);
						free(out.ptr);
						free(id.ptr);
					}// end of DT if

					else
					{
						//Tracker only
					}
					
				} // end of main TR if

				else
				{
					if(DT)
					{

						float2DArrayStruct cBB;
						floatArrayStruct cConf;
						intArrayStruct cSize;

						obj[oId]->clusConf.bb_cluster_confidance(&dBB,&dConf,&cBB,&cConf,&cSize );

						if(cConf.size==1)
						{

							obj[oId]->tld->bb.rows=4;
							obj[oId]->tld->bb.cols=1;
							memcpy(obj[oId]->tld->bb.ptr,cBB.ptr,obj[oId]->tld->bb.rows*obj[oId]->tld->bb.cols*sizeof(float));
							
							obj[oId]->tld->conf.size  = cConf.size;
							memcpy(obj[oId]->tld->conf.ptr,cConf.ptr,obj[oId]->tld->conf.size*sizeof(float));
							
							obj[oId]->tld->size.size=1;
							memcpy(obj[oId]->tld->size.ptr,cSize.ptr,obj[oId]->tld->size.size*sizeof(float));


							obj[oId]->tld->valid.ptr[0]=0 ;
							obj[oId]->tld->valid.size=1;
						}
						else
							std::cout << "more than one cConf" << std::endl;

						free(cBB.ptr);
						free(cConf.ptr);
						free(cSize.ptr);
					}
					else
					{
						obj[oId]->tld->bb.rows=0;

					}
				} //end of else in TR if
			   
				if(obj[oId]->tld->control.update_detector && (obj[oId]->tld->valid.ptr[0] == 1))
				{

					obj[oId]->tldLearn.doLearning(obj[oId]->tld,obj[oId]->cu_func); //Learninig process start if only the box is valid
				}

				if(obj[oId]->tld->dt.num_detections >0)
				{
					free(dBB.ptr);
					free(dConf.ptr);
					free(obj[oId]->tld->dt.bb);
					free(obj[oId]->tld->dt.conf1);
					free(obj[oId]->tld->dt.conf2);
					free(obj[oId]->tld->dt.isin);
					free(obj[oId]->tld->dt.patch);
					free(obj[oId]->tld->dt.patt);
				}
				
				free(tBB.ptr);
				free(tConf.ptr);

				if(obj[oId]->tld->bb.rows !=0)
				{
					obj[oId]->bb.x = obj[oId]->tld->bb.ptr[0];
					obj[oId]->bb.y = obj[oId]->tld->bb.ptr[1];
					obj[oId]->bb.width = obj[oId]->tld->bb.ptr[2]-obj[oId]->tld->bb.ptr[0];
					obj[oId]->bb.height = obj[oId]->tld->bb.ptr[3]-obj[oId]->tld->bb.ptr[1];

					vih[stream].boxes.push_back(obj[oId]->bb);
					vih[stream].drawRectangleColor(obj[oId]->objColor,obj[oId]->bb);
					streamBoxCount++;
					obj[oId]->boxStatus = true;
					obj[oId]->captureID = stream;
				}

				else
				{
					std::cout << "object lost from stream.. "<< stream << std::endl;
								
					obj[oId]->boxStatus=false;
								
					obj[oId]->s_state->search_count++;
					
					obj[oId]->captureID = obj[oId]->s_state->current_search_cam;

					if(obj[oId]->s_state->search_count>obj[oId]->s_state->search_attempts_per_cam)
					{
						obj[oId]->s_state->next_search_cam = obj[oId]->s_state->current_search_cam+1;
						obj[oId]->s_state->current_search_cam = obj[oId]->s_state->next_search_cam;
						obj[oId]->s_state->search_count=0;
					}

								
					if(obj[oId]->s_state->current_search_cam == num_of_cams)
						obj[oId]->s_state->current_search_cam = 0;

					std::cout << "object searched in stream " << obj[oId]->captureID << std::endl;					

				}			

			} 
			
			cvCopyImage(vih[stream].rgbFrame,vih[stream].rgbImage);
			

		}
		ReleaseMutex(hObjectMutex);

		end = clock();
		std::cout << " processing time: " << ((double)(end-start)/(double)(CLOCKS_PER_SEC)) << std::endl;
		cvShowImage(winnames[stream],vih[stream].rgbFrame);
		cvWaitKey(2);


	}// end of main for loop for image capturing..
	
	for(int i=0;i<ObjectClass::objectCount;i++)
	{
		delete obj[i];
	}

	vih[stream].releaseMemory();

}