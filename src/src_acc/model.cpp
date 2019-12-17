#include "mpi.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "modelling.h"

using namespace std;

void read2d(char *file, float **arr, int nx, int nz)
{
	int i;
	FILE *fp_mod;
	
	fp_mod = fopen(file, "r");
	if(fp_mod == NULL)
	{
		cerr<<"\n Error!!! Unable to open input mode : "<<file;
        MPI::COMM_WORLD.Abort(-2);
	}	
    
    fseek(fp_mod, 0, SEEK_END);
    float file_size = (float) ftell(fp_mod);
    rewind(fp_mod); 

    // Checking whether file is having nx*nz elements or not
    float rem = file_size/(float)(nx*nz);
    if(rem != 4.0f)     // 4.0f = size of float
    {
        cerr<<"\n Error!!! Either input file: "<<file<<"   is empty or didn't contain ";
        cerr<<"NX*NZ element.....\n";
        MPI::COMM_WORLD.Abort(-2);
    }
	
	for(i = 0; i < nx; i++)
	{
		fread(arr[i], sizeof(float), nz, fp_mod);
	}	
	
	fclose(fp_mod);

}//End of read2d function

//get the min and max value ....
void minmax2d(float **arr, int nx, int nz, float &vmin, float &vmax)
{   
    int ix, iz;
    
    vmax = arr[0][0];      vmin = arr[0][0];        //we can assing int val or float direct rather than copying value...

    for(ix = 0; ix < nx; ix++)   
    {        
        for(iz = 0; iz < nz; iz++)
        {
            if(arr[ix][iz] < vmin)      
                vmin = arr[ix][iz];
            if(arr[ix][iz] > vmax)
                vmax = arr[ix][iz];
        }
    }

}//End of minmax2d function

