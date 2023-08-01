#include "mpi.h"
#include <iostream>
#include <cmath>

#include "modelling.h"

using namespace std;

void Modelling_worker(char *job)
{
    int ix, iz, nobject, rank, size;
	geo2d_t *mygeo2d_sp;
	wrkld_t* wrkld_sp;
	int my_nsrc;
	int nxpad, nypad, nzpad;
	
	map<string, string> map_json;	
	map<string, string>::iterator it_json;	

	rank = MPI::COMM_WORLD.Get_rank();
	size = MPI::COMM_WORLD.Get_size();

    set_job_name(job);

	// Read input job card
	nobject = read_json_objects(job, map_json);

	// Set input parameters
	set_json_object( map_json);

	// Receive workload count	
	wrkld_sp = new wrkld_t[size];
	mygeo2d_sp = new geo2d_t[1];

	calculate_workload(geo2d_sp->nsrc, wrkld_sp);

	my_nsrc = wrkld_sp[rank].myNsrc;
	cout<<"\n Shot to be handled for rank "<<rank<<" is : "<<my_nsrc;

	mygeo2d_sp->nrec = new int[my_nsrc];
	mygeo2d_sp->src2d_sp = new crd2d_t[my_nsrc];
	mygeo2d_sp->rec2d_sp = new crd2d_t*[my_nsrc];

	receive_workload(wrkld_sp, mygeo2d_sp);
		


	

	// Read velocity and density model
	mod_sp->vel2d = new float*[mod_sp->nx];
    mod_sp->rho2d = new float*[mod_sp->nx];
	for(ix = 0; ix < mod_sp->nx; ix++)	
    {
		mod_sp->vel2d[ix] = new float[mod_sp->nz];
        mod_sp->rho2d[ix] = new float[mod_sp->nz];
    }
	
	read2d(mod_sp->velfile, mod_sp->vel2d, mod_sp->nx, mod_sp->nz);
    read2d(mod_sp->densfile, mod_sp->rho2d, mod_sp->nx, mod_sp->nz);
    
    minmax2d(mod_sp->vel2d, mod_sp->nx, mod_sp->nz, mod_sp->vmin, mod_sp->vmax);
    cout<<"\n Min. Velocity : "<<mod_sp->vmin<<"\t Max. Velocity : "<<mod_sp->vmax;


    // Check stability and dispersion
    check_stability(); 

    // Call wave propagation
    modelling_module(my_nsrc, mygeo2d_sp); 	
	
}//End of Modelling_worker

