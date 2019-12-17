#include <mpi.h>
#include <iostream>
#include <cmath>
#include <cstring>
#include <omp.h>

#include "modelling.h"
#include "cpu_modelling_kernels.h"
#define PI 3.141592654
#define Block_size1 16
#define Block_size2 16

using namespace std;

const int npml=32;      /* thickness of PML boundary */

static int      nz1, nx1, nz, nx, nnz, nnx, N, NJ, ns, ng, fd_nt, real_nt, nts;
static float    fm, fd_dt, real_dt, dz, dx, _dz, _dx, rec_len/*, vmute, tdmute*/;

float   *v0, *h_vel, *h_rho;
/* variables on device */
int     *sx_pos, *sz_pos;
int     *gx_pos, *gz_pos;
int     *h_Sxz, *h_Gxz;     /* set source and geophone position */

double dtdx, dtdz;

float *bu1, *bu2, *bue;
float *spg, *spg1, *damp1a1, *damp2a1, *damp1b1, *damp2b1;
float *damp1a, *damp2a, *damp1b, *damp2b;

double *h_wlt;  //source wavelet
float *h_dobs, *h_temp;  // seismogram

double *p, *px, *pz, *vx, *vz, *der1, *der2;
double *kappa, *kappaw1, *kappaw2;
double *buw1, *buw2;

void expand(float *b, float *a, int npml, int nnz, int nnx, int nz1, int nx1)
/*< expand domain of 'a' to 'b':  a, size=nz1*nx1; b, size=nnz*nnx;  >*/
{
    int iz,ix;
    for(ix = 0; ix < nx1; ix++)
        for (iz = 0; iz < nz1; iz++)
        {
            b[(npml+ix)*nnz+(npml+iz)] = a[ix*nz1+iz];
        }

    for(ix = 0; ix < nnx; ix++)
    {
        for (iz = 0; iz < npml; iz++)       b[ix*nnz+iz] = b[ix*nnz+npml];//top
        for (iz = nz1+npml; iz < nnz; iz++) b[ix*nnz+iz] = b[ix*nnz+npml+nz1-1];//bottom
    }

    for(iz = 0; iz < nnz; iz++)
    {
        for(ix = 0; ix < npml; ix++)        b[ix*nnz+iz] = b[npml*nnz+iz];//left
        for(ix = npml+nx1; ix < nnx; ix++)  b[ix*nnz+iz] = b[(npml+nx1-1)*nnz+iz];//right
    }
}// End of expand function


void host_alloc()
{
    // source wavelet
    h_wlt = new double[nts];    

    h_vel = new float[N];
    h_rho = new float[N];

    // staggered grid		
    bue = new float[N];
    bu1 = new float[N];
    bu2 = new float[N];
    kappa = new double[N];
   		
    // sponge
    spg = new float[npml+1];
    spg1 = new float[npml+1];

    damp2a = new float[nnx];
    damp2a1 = new float[nnx];
    damp2b = new float[nnx];
    damp2b1 = new float[nnx];

    damp1a = new float[nnz];
    damp1a1 = new float[nnz];
    damp1b = new float[nnz];
    damp1b1 = new float[nnz];	

    // wavefields
    p = new double[N];
    px = new double[N];
    pz = new double[N];
    vx = new double[N];
    vz = new double[N];
    
    // diffrential operators
    der1 = new double[N];
    der2 = new double[N];

    kappaw2 = new double[N];
    kappaw1 = new double[N];
    buw2 = new double[N];
    buw1 = new double[N];

    double float_mem, double_mem;
    float_mem  =  4.0*nnz*sizeof(float) + 4.0*nnx*sizeof(float) + 5.0*N*sizeof(float) + 2.0*(npml+1)*sizeof(float);
    double_mem =  12.0*N*sizeof(double) + nts*sizeof(double);
	
    cout<<"\n Rough memory estimate is:"<<(float_mem+double_mem)/(1024.f*1024.f*1024.f)<<"  GB.";

}// End of host_alloc function

void FDOperator_4(int it, int isource)
{
    int ompi;
    if(it%1000 == 0)
        cout<<"\n it: "<<it;

    if(NJ == 4)
    {  
        #pragma omp parallel for schedule(static)
        for(ompi = 0; ompi < N; ompi++)
        {
            der1[ompi] = 0.0;           der2[ompi] = 0.0;
        }


        fdoperator420(vx, der2, nnz, nnx);
        fdoperator410(vz, der1, nnz, nnx);
        
        cpu_pml_p(px, pz, kappaw2, kappaw1, der2, der1, damp2a, damp1a, nnz, nnx);    
     
        // increment source
        if(it <= nts)
        {
            px[isource] = px[isource] + fd_dt*h_wlt[it]*kappa[isource] / (dx*dz);

        }
    
        compute_p(p, px, pz, nnz, nnx);
   
        #pragma omp parallel for schedule(static)
        for(ompi = 0; ompi < N; ompi++)
        {
            der1[ompi] = 0.0;           der2[ompi] = 0.0;
        }
 
        fdoperator421(p, der2, nnz, nnx);
        fdoperator411(p, der1, nnz, nnx);
    
        cpu_pml_v(vx, vz, buw2, buw1, der2, der1, damp2b, damp1b, nnz, nnx);
    }// NJ=4

}//End of FDOperator function

void modelling_module(int my_nsrc, geo2d_t *mygeo2d_sp)
{
    int rank = MPI::COMM_WORLD.Get_rank();    	
    char log_file[1024], seis_file[1024], rnk[5];
    FILE *fp_log;
    sprintf(rnk, "%d\0", rank);

    int ix, iz, id, is, ig, kt, istart, dt_factor, indx, ompi;

    // variable init
    NJ = job_sp->fdop;              ns = my_nsrc;
    nx1 = mod_sp->nx;               nz1 = mod_sp->nz;

    dx  = mod_sp->dx;               dz   = mod_sp->dz;
    _dx = 1.0f/dx;                    _dz = 1.0f/dz;
    fm = wave_sp->dom_freq;             rec_len = wave_sp->rec_len;
    fd_dt = wave_sp->fd_dt_sec;         fd_nt = (int)ceil(rec_len/fd_dt);
    real_dt = wave_sp->real_dt_sec;     real_nt = (int)ceil(rec_len/real_dt);

    dtdx = (double) (fd_dt * _dx);      dtdz = (double) (fd_dt * _dz);

    dt_factor = (int) (real_dt/fd_dt);
    cout<<"\n Rank: "<<rank<<"   fd_nt: "<<fd_nt
        <<"   real_nt: "<<real_nt<<"   DT factor: "<<dt_factor; 

    double td = sqrt(6.0f) / (PI*fm);
    nts = (int) (5.0f * td/fd_dt) + 1;
    if (nts > fd_nt)
    {
        nts = fd_nt;
    }
    
    nnz = 2*npml+nz1;
    nnx = 2*npml+nx1;

        N = nnz*nnx;
    
    cout<<"\n Rank: "<<rank<<"\t nnx: "<<nnx<<"   nnz: "<<nnz<<"    nts: "<<nts;

    #pragma omp parallel
    {
        #pragma omp master
            cout<<"\n Rank: "<<rank<<"\t number of OMP threads: "<<omp_get_num_threads();
    }


    // allocate host memory
    host_alloc();
 
    v0 = new float[nx1*nz1];
    // expand velocity model
    memset(v0, 0, nz1*nx1*sizeof(float));       memset(h_vel, 0, N*sizeof(float));
    for(ix = 0; ix < nx1; ix++)
        for(iz = 0; iz < nz1; iz++)
            v0[ix*nz1+iz] = mod_sp->vel2d[ix][iz];
    
    expand(h_vel, v0, npml, nnz, nnx, nz1, nx1);

    // expand density model
    memset(v0, 0, nz1*nx1*sizeof(float));       memset(h_rho, 0, N*sizeof(float));
    for(ix = 0; ix < nx1; ix++)
        for(iz = 0; iz < nz1; iz++)
            v0[ix*nz1+iz] = mod_sp->rho2d[ix][iz];

    expand(h_rho, v0, npml, nnz, nnx, nz1, nx1);
    
    #pragma omp parallel for schedule(static)
        for(ompi = 0; ompi < N; ompi++)
        {
            bue[ompi] = 0.0f;        kappa[ompi] = 0.0;
            bu2[ompi] = 0.0f;        bu1[ompi] = 0.0f;
        }
    cpu_stagger(h_vel, h_rho, bue, bu2, bu1, kappa, nnz, nnx); 

    // compute source wavelet
    memset(h_wlt, 0, nts*sizeof(double));
    cpu_ricker(fm, fd_dt, h_wlt, nts); 
    
    // build sponges
    memset(spg, 0, (npml+1)*sizeof(float));
    memset(spg1, 0, (npml+1)*sizeof(float));
    
     #pragma omp parallel for schedule(static)
        for(ompi = 0; ompi < nnx; ompi++)
        {
            damp2a[ompi] = 0.0f;        damp2a1[ompi] = 0.0f;
            damp2b[ompi] = 0.0f;        damp2b1[ompi] = 0.0f;
        }

     #pragma omp parallel for schedule(static)
        for(ompi = 0; ompi < nnz; ompi++)
        {
            damp1a[ompi] = 0.0f;        damp1a1[ompi] = 0.0f;
            damp1b[ompi] = 0.0f;        damp1b1[ompi] = 0.0f;
        }
     
    float xxfac = 0.05f;
    cpu_pml_coefficient_a(xxfac, damp2a, damp2a1, spg, spg1, npml, nx1);
    cpu_pml_coefficient_b(xxfac, damp2b, damp2b1, spg, spg1, npml, nx1);

    cpu_pml_coefficient_a(xxfac, damp1a, damp1a1, spg, spg1, npml, nz1);
    cpu_pml_coefficient_b(xxfac, damp1b, damp1b1, spg, spg1, npml, nz1); 

    #pragma omp parallel for schedule(static)
        for(ompi = 0; ompi < nnz; ompi++)
        {
            kappaw2[ompi] = 0.0;        kappaw1[ompi] = 0.0;            
            buw2[ompi] = 0.0;           buw1[ompi] = 0.0;
        }

    cout<<"\n File: "<<__FILE__<<"  Line: "<<__LINE__;    

    for(ix = 0; ix < nnx; ix++)
    {
       for(iz = 0; iz < nnz; iz++)
       {
            id = iz + ix*nnz;

            kappaw2[id] = dtdx*kappa[id]*damp2a1[ix];
            kappaw1[id] = dtdz*kappa[id]*damp1a1[iz];
            buw2[id] = dtdx*bu2[id]*damp2b1[ix];
            buw1[id] = dtdz*bu1[id]*damp1b1[iz];
        }
    }
    
    // set source positions
    sz_pos = new int[ns];       sx_pos = new int[ns];
    for(is = 0; is < ns; is++)
    {
        cout<<"\n sx: "<<mygeo2d_sp->src2d_sp[is].x<<"  sz: "<<mygeo2d_sp->src2d_sp[is].z;
        sz_pos[is] = (int)(mygeo2d_sp->src2d_sp[is].z*_dz);
        sx_pos[is] = (int)(mygeo2d_sp->src2d_sp[is].x*_dx);
    }
    h_Sxz = new int[ns];
    cpu_set_sg(h_Sxz, sx_pos, sz_pos, ns, npml, nnz);

    // checking job status - New/Restart
    strcpy(log_file, job_sp->tmppath);  strcat(log_file, job_sp->jobname);
    strcat(log_file, "checkpoint");
    strcat(log_file, "_rank");      strcat(log_file, rnk);  strcat(log_file, ".txt\0");

    if(strcmp(job_sp->jbtype, "New") == 0)
    {
        istart = 0;
    }
    else if(strcmp(job_sp->jbtype, "Restart") == 0)
    {
        fp_log = fopen(log_file, "r");
        if(fp_log == NULL)
        {
            cerr<<"\n Rank: "<<rank<<"   Error !!! in openining log file: "<<log_file;
            cerr<<"\n Please make sure that same job is executed before with same resource \
                      configuration";
            MPI::COMM_WORLD.Abort(-10);
        }

        fscanf(fp_log, "%d", &istart);
        fclose(fp_log);
    }
    else
    {
        cerr<<"\n Error !!! Rank: "<<rank<<"   Unknow job status, Please make respective changes in job card";
        MPI::COMM_WORLD.Abort(-11);
    }


    cout<<"\n Rank: "<<rank<<"   Started Modelling.......";
    // shot loop
    for(is = istart; is < ns; is++)
    {
        // set geaophone positions
        ng = mygeo2d_sp->nrec[is];
        gz_pos = new int[ng];               gx_pos = new int[ng];
        for(ig = 0; ig < ng; ig++)
        {
            gz_pos[ig] = (int)(mygeo2d_sp->rec2d_sp[is][ig].z*_dz);
            gx_pos[ig] = (int)(mygeo2d_sp->rec2d_sp[is][ig].x*_dx);
        }

        h_Gxz = new int[ng];
        cpu_set_sg(h_Gxz, gx_pos, gz_pos, ng, npml, nnz);
    
        h_dobs = new float[ng*real_nt];         memset(h_dobs, 0, ng*real_nt*sizeof(float));
        h_temp = new float[ng*real_nt];         memset(h_temp, 0, ng*real_nt*sizeof(float));

        #pragma omp parallel for schedule(static)
        for(ompi = 0; ompi < nnz; ompi++)
        {
            p[ompi] = 0.0;          px[ompi] = 0.0;         pz[ompi] = 0.0;
            vx[ompi] = 0.0;         vz[ompi] = 0.0;
        }
        
        for(kt = 0; kt < fd_nt; kt++)
        {
           
            FDOperator_4(kt, h_Sxz[is]);
               
            // storing pressure seismograms
            if( (kt+1)%dt_factor == 0 )
            {
                indx = ((kt+1)/dt_factor) - 1;
                cpu_record(p, &h_temp[indx*ng], h_Gxz, ng);
            }

        } // End of NT loop

        delete[] gz_pos;    delete[] gx_pos;
        delete[] h_Gxz;

        cpu_transpose(h_temp, h_dobs, ng, real_nt);

        // write seismogram
        strcpy(seis_file, job_sp->tmppath);  strcat(seis_file, job_sp->jobname);
        strcat(seis_file, "seismogram.bin\0");
        FILE *fp_seis = fopen(seis_file, "w"); 
            fwrite(h_dobs, sizeof(float), ng*real_nt, fp_seis);
        fclose(fp_seis);

        delete[] h_dobs;    delete[] h_temp;

        cout<<"\n Rank: "<<rank<<"   Shot remaining: "<<ns - (is+1);
    
    }// End of shot loop


}//End of modelling_module function



