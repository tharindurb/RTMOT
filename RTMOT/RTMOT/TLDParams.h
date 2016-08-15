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
#ifndef TLD_PARA_H
#define TLD_PARA_H

#include "IncludeFiles.h"
#include "TLDstructs.h"

class TLDParams
{
private:
	tld_Structure *my_tld;
	


public:
	TLDParams(tld_Structure *ptld);
	tld_Structure* get_tld();
	int tldnGrid;
	void create_temporal_structure();
	//VideoImageHandler* vidObj;

};

#endif;