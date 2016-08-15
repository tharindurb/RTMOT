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
#include "TLDParams.h"

TLDParams::TLDParams(tld_Structure *ptld)
{

	my_tld = ptld;
	
	my_tld->model.min_win=24;			
	my_tld->model.patchsize[0]=45;	
	my_tld->model.patchsize[1]=45;
	my_tld->model.fliplr=0;
					
		
									
	my_tld->model.ncc_thesame=0.95;
	my_tld->model.valid=0.5;
	my_tld->model.num_trees=10;
	my_tld->model.num_features=13;
	my_tld->model.thr_fern=0.5;
	my_tld->model.thr_nn=0.65;
	my_tld->model.thr_nn_valid=0.7;

	my_tld->p_par_init.num_closest=10;
	my_tld->p_par_init.num_warps=20;
	my_tld->p_par_init.noise=5;
	my_tld->p_par_init.angle=20;
	my_tld->p_par_init.shift=0.02;
	my_tld->p_par_init.scale=0.02;

	my_tld->p_par_update.num_closest=10;
	my_tld->p_par_update.num_warps=10;
	my_tld->p_par_update.angle=10;
	my_tld->p_par_update.noise=5;
	my_tld->p_par_update.scale=0.02;
	my_tld->p_par_update.shift=0.02;
		
		
	my_tld->n_par.overlap=0.2;
	my_tld->n_par.num_patches=100;

	my_tld->tracker.occlusion=10;

	my_tld->control.drop_img=true;
	my_tld->control.update_detector=true;
	my_tld->control.repeat=true;
	my_tld->control.maxbox=1;
}

tld_Structure* TLDParams::get_tld()
{
	return my_tld;

}

void TLDParams::create_temporal_structure()
{
	
	my_tld->tmp.conf.size=my_tld->grid.cols;
	my_tld->tmp.conf.ptr=(float*)calloc(this->my_tld->tmp.conf.size,sizeof(float));

	my_tld->tmp.patt.cols=my_tld->grid.cols;
	my_tld->tmp.patt.rows=my_tld->model.num_trees;
	my_tld->tmp.patt.ptr=(int*)calloc(this->my_tld->tmp.patt.cols*this->my_tld->tmp.patt.rows,sizeof(int));

}
