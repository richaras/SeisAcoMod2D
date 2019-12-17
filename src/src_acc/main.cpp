/*
  Copyright (C) 2019 C-DAC All rights reserved

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  This program or modified versions of it may be distributed under
  the condition that this code and any modifications made to it in
  the same file remain under copyright of the original authors, both
  source and object code are made freely available without charge, and
  clear notice is given of the modifications. Distribution of this
  code as part of a commercial system is permissible only by direct
  arrangement with C-DAC.

  Contact:
  Seismic Data Processing (SDP),
  High Performance Computing - Scientific and Engeering Application (HPC-S&EA),
  Centre for Development of Advanced Computing (C-DAC), Pune, India
  <richar@cdac.in, abhisheks@cdac.in>

  Acknowledgement: The FD modules of this code are the GPU adaptation of FD functions
  in FWM2DA FORTRAN code developed by by J. Virieux & S. Operto under SEISCOPE project.
  -------------------------------------------------------------------------------------

  This is program SeisAcoMod2D.
    Parallel 2D Acoustic Finite Difference Seismic Modeling using the Staggered Grid
    OpenACC implementation
  -------------------------------------------------------------------------------------

*/


#include "mpi.h"
#include <iostream>
#include <cstring>

#include "modelling.h"

using namespace std;

mod_t *mod_sp = new mod_t[1];           //(mod_t*) calloc(1, sizeof(mod_t));
wave_t *wave_sp = new wave_t[1];        //(wave_t*) calloc(1, sizeof(wave_t));
geo2d_t *geo2d_sp = new geo2d_t[1];     //(geo2d_t*) calloc(1, sizeof(geo2d_t));
job_t *job_sp = new job_t[1];           //(job_t*) calloc(1, sizeof(job_t));

int main(int argc, char **argv)
{
    int rank, size, length;
    char proc_name[32], job[128];
	
    // Initialise MPI environment
    MPI::Init(argc, argv);
    
    if (argc < 2)
    {
        cerr <<"\n Error!!! Insufficient number of argument";
        cerr <<"\n Please provide input parameter file";
        MPI::COMM_WORLD.Abort(-1);
    }
    rank = MPI::COMM_WORLD.Get_rank();
    size = MPI::COMM_WORLD.Get_size();

    strcpy(job, argv[1]);
    
    MPI::Get_processor_name(proc_name, length);
    
    cout<<"\n Rank : "<<rank<<" on processor : "<<proc_name;
    cout<<"\n Rank : "<<rank<<" Job file : "<<job;    

    if(rank == 0)
    {
        Modelling_master(job);
    }
    else
    {
        Modelling_worker(job);
    }
    			    
    // Finalize MPI
    MPI_Finalize();                					
    
	return 0;	

}//End main

