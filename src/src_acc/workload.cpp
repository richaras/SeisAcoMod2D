#include "mpi.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "modelling.h"

using namespace std;

void calculate_workload(int nsrc, wrkld_t* wrkld_sp)
{
	int rank, size;
	
	rank = MPI::COMM_WORLD.Get_rank();
	size = MPI::COMM_WORLD.Get_size();

	if(rank == 0)			// Master will calculate worload and send to worker
	{
        if(nsrc < size)
        {
            cout<<"\n********************************************************************";
            cout<<"\n*                                                                  *";
            cout<<"\n*  Number of sources should be greater than or equal to MPI ranks  *";
            cout<<"\n*  for proper workload distribution. Please re-submit the job      *"; 
            cout<<"\n*  with suggested change                                           *";
            cout<<"\n*                                                                  *";
            cout<<"\n********************************************************************"; 
            MPI::COMM_WORLD.Abort(-8);
        }        

		int div, mod;
		div = nsrc / size;
		mod = nsrc % size;

		int i;
		wrkld_sp[0].start = 0;
		for(i = 0; i < size; i++)
		{
			if(i!=0)	
				wrkld_sp[i].start = wrkld_sp[i-1].end+1;
					
			if(i < mod)
				wrkld_sp[i].myNsrc = div+1; 
			else
				wrkld_sp[i].myNsrc = div; 
			
			wrkld_sp[i].end = wrkld_sp[i].start + wrkld_sp[i].myNsrc - 1;			
		}

		for(i = 1; i < size; i++)
		{			
			MPI::COMM_WORLD.Send(wrkld_sp, size*sizeof(wrkld_t), MPI_BYTE, i, i);
		}
	}
	else					//Worker will receiver workload count
	{
		MPI::COMM_WORLD.Recv(wrkld_sp, size*sizeof(wrkld_t), MPI_BYTE, 0, rank);		
	}	
	
}//End of calculate_workload function

void send_workload(wrkld_t* wrkld_sp)
{
	int i, j, k, rank, size;
	
	rank = MPI::COMM_WORLD.Get_rank();
	size = MPI::COMM_WORLD.Get_size();
	
	geo2d_t *mygeo2d_sp = new geo2d_t[1];
	mygeo2d_sp->nrec = new int[1];
	mygeo2d_sp->src2d_sp = new crd2d_t[1];
	mygeo2d_sp->rec2d_sp = new crd2d_t*[1];	

	// Send workload to worker
	for(i = 1; i < size; i++)
	{
		// For each source
		for(j = wrkld_sp[i].start; j <= wrkld_sp[i].end; j++)
		{
			mygeo2d_sp->src2d_sp[0].x = geo2d_sp->src2d_sp[j].x;		
            mygeo2d_sp->src2d_sp[0].z = geo2d_sp->src2d_sp[j].z;
			mygeo2d_sp->nrec[0] = geo2d_sp->nrec[j];
			mygeo2d_sp->rec2d_sp[0] = new crd2d_t[mygeo2d_sp->nrec[0]];

			for(k = 0; k < mygeo2d_sp->nrec[0]; k++)
			{
				mygeo2d_sp->rec2d_sp[0][k].x = geo2d_sp->rec2d_sp[j][k].x;		
                mygeo2d_sp->rec2d_sp[0][k].z = geo2d_sp->rec2d_sp[j][k].z;	
			}

												
		
			MPI::COMM_WORLD.Send(&mygeo2d_sp->src2d_sp[0], 1*sizeof(crd2d_t), MPI_BYTE, i, i); 			
			MPI::COMM_WORLD.Send(&mygeo2d_sp->nrec[0], 1, MPI_INT, i, i);
			MPI::COMM_WORLD.Send(mygeo2d_sp->rec2d_sp[0], mygeo2d_sp->nrec[0]*sizeof(crd2d_t), MPI_BYTE, i, i);   		
			
			delete[] mygeo2d_sp->rec2d_sp[0];	
		}
	}
	delete[] mygeo2d_sp->nrec;
	delete[] mygeo2d_sp->src2d_sp;
	delete[] mygeo2d_sp->rec2d_sp;	
	delete[] mygeo2d_sp;

}// End of send_workload function

void receive_workload(wrkld_t* wrkld_sp, geo2d_t *mygeo2d_sp)
{
	int i, j, nrec, rank, size;
	
	rank = MPI::COMM_WORLD.Get_rank();
	size = MPI::COMM_WORLD.Get_size();
	
	// Receive workload from master
	for(i = 0; i < wrkld_sp[rank].myNsrc; i++)
	{
		// Receive no. of receiver of single shot
		MPI::COMM_WORLD.Recv(&mygeo2d_sp->src2d_sp[i], 1*sizeof(crd2d_t), MPI_BYTE, 0, rank); 			
		MPI::COMM_WORLD.Recv(&nrec, 1, MPI_INT, 0, rank);
    	mygeo2d_sp->nrec[i] = nrec;          

		mygeo2d_sp->rec2d_sp[i] = new crd2d_t[nrec];
		MPI::COMM_WORLD.Recv(mygeo2d_sp->rec2d_sp[i], nrec*sizeof(crd2d_t), MPI_BYTE, 0, rank);  
	}

}// End of receive_workload function

void master_workload(wrkld_t* wrkld_sp, geo2d_t *mygeo2d_sp)
{
	int j, k, rank;

	rank = MPI::COMM_WORLD.Get_rank();		

	// Master workload
	int count=0;
	for(j = wrkld_sp[rank].start; j <= wrkld_sp[rank].end; j++)
	{	
		mygeo2d_sp->src2d_sp[count].x = geo2d_sp->src2d_sp[j].x;		
        mygeo2d_sp->src2d_sp[count].z = geo2d_sp->src2d_sp[j].z;
		mygeo2d_sp->nrec[count] = geo2d_sp->nrec[j];
		mygeo2d_sp->rec2d_sp[count] = new crd2d_t[mygeo2d_sp->nrec[count]];

		for(k = 0; k < mygeo2d_sp->nrec[0]; k++)
		{
			mygeo2d_sp->rec2d_sp[count][k].x = geo2d_sp->rec2d_sp[j][k].x;		
            mygeo2d_sp->rec2d_sp[count][k].z = geo2d_sp->rec2d_sp[j][k].z;	
		}
		count++;
	}

}// End of master_workload function

void print_workload(wrkld_t* wrkld_sp, geo2d_t *mygeo2d_sp)
{
	int i, j, rank;
    char file[512], rnk[5];
	rank = MPI::COMM_WORLD.Get_rank();
        
    strcpy(file, "source_receiver_geom_rank_");
    sprintf(rnk, "%d\0", rank);
    strcat(file, rnk);      strcat(file, ".txt\0");    

    FILE *fp = fopen(file, "w");

	for(i = 0; i < wrkld_sp[rank].myNsrc; i++)
	{

        
        fprintf(fp, "\n Source[%d](x,z): ( %d, %d )", i+1, mygeo2d_sp->src2d_sp[i].x, mygeo2d_sp->src2d_sp[i].z);


		fprintf(fp, "\n Number of Receviers for source[%d] are: %d", i+1, mygeo2d_sp->nrec[i]); 
		for(j = 0; j < mygeo2d_sp->nrec[i]; j++)
		{

         
            fprintf(fp, "\n Receiver[%d](x,z): ( %d, %d )", j+1, mygeo2d_sp->rec2d_sp[i][j].x, mygeo2d_sp->rec2d_sp[i][j].z);
		}		
        fflush(fp);
	}
		
    fclose(fp);

}// End of print_workload function

