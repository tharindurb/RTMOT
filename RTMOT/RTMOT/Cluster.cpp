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
#include "Cluster.h"

//This Constructs clusters from a hierarchical cluster tree.
// Uses distance as the criterion for forming clusters.  Each node's height 
// in the tree represents the distance between the two subnodes merged at that
// node.  All leaves below any node whose height is less than limit are
// grouped into a cluster (a singleton if the node itself is a leaf).This code
// is constructed with the help of cluster implementation of MATLAB

void Cluster::getCluster(float2DArrayStruct* Z,float limit,intArrayStruct* T)
{
	int depth = 2;
	char* criterion = "distance";
	float cutoff =limit; 
	int inconsistent_set=0;
	bool usecutoff = true;   
	char* pname = "cutoff";
	int m = Z->rows+1;
	boolArrayStruct conn;

	T->size=m;
	T->ptr=(int*) malloc(T->size*sizeof(float));	  
	checkcut(Z, cutoff,&conn);
	labeltree(Z, &conn,T);

	delete(conn.ptr);
}

void Cluster::checkcut(float2DArrayStruct* X,float cutoff,boolArrayStruct* conn)
{

	conn->size=X->rows;
	conn->ptr=(bool*) malloc(sizeof(bool)*conn->size);
	
	 for(int i=0;i< conn->size;i++){
		 if(( X->ptr)[2*X->rows+i]<=cutoff){			
		    (conn->ptr)[i]=1;//the elements in the 3rd column of X			
		 }
		 else{
			(conn->ptr)[i]=0;
		 }
	 }

	//See which nodes are below the cutoff, disconnect those that aren't
	 int n = X->rows;

	// We may still disconnect a node unless all non-leaf children are
	// below the cutoff, and grand-children, and so on
	boolArrayStruct todo;
	todo.size=n;
	todo.ptr=new bool[n];
	intArrayStruct rows;
	rows.size=0;
	rows.ptr=new int[n];


	for(int i=0;i<n;i++){
		todo.ptr[i] = conn->ptr[i] && (X->ptr[i,0]>n || X->ptr[i,1]>n);

		if( todo.ptr[i]==true){
			rows.ptr[rows.size]=i;
			rows.size=rows.size+1;
		 }
	 }

	 while(rows.size>0){  
		std::vector<std::vector<bool>> cdone;
		for(int i=0;i<rows.size;i++){
			std::vector<bool> cdone_temp;//initializing cdone as 2*m array of 1
			cdone_temp.push_back(1);
			cdone_temp.push_back(1);
			cdone.push_back(cdone_temp);
		}
		for( int j=0;j<2;j++){ //for4    // 1=left child, 2=right child
			std::vector<int> crows;
			std::vector<bool> t ;
			int count=0;
			for(int k=0;k<rows.size;k++){
				crows.push_back((int)X->ptr[rows.ptr[k]+j*X->rows]);
				t.push_back(crows.at(k)>n);
				if(t.at(k)==1){
					count++;
				}			
			}			
			if (count>0){
				std::vector<float> child;				
				for(int m=0;m<t.size();m++){
					if(t.at(m)==1){
						child.push_back ( crows.at(m)-n);
					}
				}				
				std::vector<bool> value;
					for(int p=0;p<child.size();p++){
							value.push_back(!todo.ptr[p]);
					}				
				int value_count=0;
				for(int m=0;m<t.size();m++){					
					if(t.at(m)==1){
						cdone.at(m).at(j) =value.at(value_count); 
						value_count++;
						}								
				}
				value_count=0;
				for(int m=0;m<t.size();m++){
					if(t.at(m)==1){
						conn->ptr[rows.ptr[m]] =conn->ptr[rows.ptr[m]] && conn->ptr[(int)child.at(value_count)]; 
						value_count++;
						}								
				}
			}
		}

			// Update todo list
			for(int m=0;m<cdone.size();m++){
				if(cdone.at(m).at(0) && cdone.at(m).at(1)){
					todo.ptr[rows.ptr[m]] = 0;
				}
			}
			//calculate no of non zero rows
			rows.size=0;
			for(int i=0;i<todo.size;i++){
				if( todo.ptr[i]==true){//if6		 
					rows.ptr[rows.size]=i;
					rows.size=rows.size+1;
				}
			}
	}
}
void Cluster::labeltree(float2DArrayStruct* X,boolArrayStruct* conn,intArrayStruct* T)
{
   
	int n = X->rows;
	int nleaves = n+1;
	T->size=n+1;
	T->ptr =new int[T->size]; 
	std::fill(T->ptr,T->ptr+T->size,1);

	//Each cut potentially yields an additional cluster
	std::vector<bool> todo(n,1);
	
	// Define cluster numbers for each side of each non-leaf node
	//clustlist = reshape(1:2*n,n,2);
	std::vector<std::vector<int>> clustlist;
	for(int i=1;i<=n;i++){
		std::vector<int> temp;
		temp.push_back(i);
		temp.push_back(i+n);
		clustlist.push_back(temp);
		
	}
	// Propagate cluster numbers down the tree
	bool check=true;

	while(check){
	   // Work on rows that are now split but not yet processed
		check=false;
		for(int i=0;i<todo.size();i++){
			if(todo.at(i)==1){
				check=true;
				break;
			}
		}

		//rows = find(todo & ~conn);
		std::vector<int> rows;
		for(int i=0;i<todo.size();i++){
			bool k=todo.at(i) & ~(conn->ptr[i]);
			if(k==1){
				rows.push_back(i);
			}
		}

	   
		if( rows.size()==0){ break;}
	   
		for (int j=0;j<2;j++){    

				// children = X(rows,j);
				std::vector<float> children;
				for(int x=0;x<rows.size();x++){
					float num=X->ptr[rows.at(x)+j*X->rows];
					children.push_back(num );
				}
		   
				// Assign cluster number to child leaf node
				 //leaf = (children <= nleaves);
				std::vector<bool> leaf;
				bool check2=false;
				for(int x=0;x<children.size();x++){
					if(children.at(x)<= nleaves){
						leaf.push_back(1);
						check2=true;
					}
					else{
							leaf.push_back(0);
					}
				}

				if (check2){
					//T(children(leaf)) = clustlist(rows(leaf),j);
					for(int x=0;x<leaf.size();x++){
						if(leaf.at(x)==1){
							
							T->ptr[(int)children.at(x)-1]=clustlist.at(rows.at(x)).at(j);
							int h=T->ptr[(int)children.at(x)-1];
							int y=(int)children.at(x)-1;
							int p=0;
						}					
					}

				}
		   
				  //Also assign it to both children of any joined child non-leaf nodes
				std::vector<bool> joint(leaf.size());
				for(int x=0;x<leaf.size();x++){
					joint.at(x) = ~leaf.at(x);
				}

				// joint(joint) = conn(children(joint)-nleaves);
				for(int x=0;x<joint.size();x++){
					if(joint.at(x)==1){
						joint.at(x)=conn->ptr[(int)(children.at(x)-nleaves)];
					}

				}

				std::vector<bool>::iterator it;
				it=std::find(joint.begin(),joint.end(),1);

				if (it != joint.end()){
					std::vector<int> clustnum;
					std::vector<int> childnum;
					for(int x=0;x<joint.size();x++){
						if(joint.at(x)==1){
							clustnum.push_back(clustlist.at(x).at(j));
							clustnum.push_back(children.at(x)-nleaves);
						}
					}

					for(int x=0;x<childnum.size();x++){
						clustlist.at(childnum.at(x)).at(0)=clustnum.at(x);
						clustlist.at(childnum.at(x)).at(1)=clustnum.at(x);
						conn->ptr[childnum.at(x)]=0;

					}					 					
				}				 
		}
			   
		//Mark these rows as done  
		for(int x=0;x<rows.size();x++){
			todo.at(rows.at(x)) = 0;
		}			
	}

	get_unique(T);
}
void Cluster::get_unique(intArrayStruct* pp)
{


	int* t=new int[pp->size];
	int* tt=new int[pp->size];
	std::vector<int> w(pp->ptr,pp->ptr+pp->size);


	std::vector<int>::iterator it,it2;
	std::sort(w.begin(),w.end());
	it=std::unique(w.begin(),w.end()); 

	  for(int i=0;i<w.size();i++){
		int k=0;
		for (it2=w.begin(); it2!=it; ++it2){
		
			if(pp->ptr[i]==*it2){
			  tt[i]=k;		 
			}
			  k=k+1;
		}
	 }

	  memmove( pp->ptr, tt, pp->size*sizeof(int) );
	  
free(t);
free(tt);
}

