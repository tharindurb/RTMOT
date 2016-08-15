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
#ifndef CLUSTER_CONF_H
#define CLUSTER_CONF_H

#include "IncludeFiles.h"
#include "TLDstructs.h"
#include "Linkagemex.h"
#include "Cluster.h"

class ClusterConfidence
{
public:

	Linkagemex link;
	Cluster clus;

	void bb_cluster_confidance(float2DArrayStruct* iBB,floatArrayStruct* iConf,float2DArrayStruct* oBB,floatArrayStruct* oConf,intArrayStruct* oSize );
	float2DArrayStruct bb_distance(float2DArrayStruct* bb1);
	float2DArrayStruct bb_overlap1(float2DArrayStruct* bb);
	float bb_overlap_calc(float *bb1, float *bb2);
	void get_unique_T(intArrayStruct* pp,intArrayStruct* idx);

};

#endif

