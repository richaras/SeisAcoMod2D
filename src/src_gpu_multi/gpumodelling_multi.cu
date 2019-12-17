#include <mpi.h>
#include <iostream>
#include <cmath>
#include <cstring>
#include <omp.h>

#include "modelling.h"
#include "gpu_modelling_kernels.h"
#define PI 3.141592654
#define Block_size1 16
#define Block_size2 16

using namespace std;

const int npml=32;      /* thickness of PML boundary */

static int      nz1, nx1, nz, nx, nnz, nnx, N, NJ, ns, ng, fd_nt, real_nt, nts;
static float    fm, fd_dt, real_dt, dz, dx, _dz, _dx, rec_len/*, vmute, tdmute*/;
static dim3     dimg0, dimb0;


float   *v0, *h_vel, *h_rho;
/* variables on device */
int     *sx_pos, *sz_pos;
int     *gx_pos[2], *gz_pos[2];
int     *h_Sxz, *h_Gxz[2];     /* set source and geophone position */
int     *d_Gxz[2];

double dtdx, dtdz;

float *bu1, *bu2, *bue;
float *spg, *spg1, *damp1a1, *damp2a1, *damp1b1, *damp2b1;
float *damp1a, *damp2a, *damp1b, *damp2b;
float *d_damp1a[2], *d_damp2a[2], *d_damp1b[2], *d_damp2b[2];

double *h_wlt;  //source wavelet
double *d_wlt[2];
float *h_dobs[2], *h_temp[2];  // seismogram
float *d_dobs[2], *d_temp[2];  // seismogram

double *d_p[2], *d_px[2], *d_pz[2], *d_vx[2], *d_vz[2], *d_der2[2], *d_der1[2];

double *kappa, *kappaw2, *kappaw1;
double *d_kappa[2], *d_kappaw2[2], *d_kappaw1[2];
double *buw2, *buw1;
double *d_buw2[2], *d_buw1[2];

cudaStream_t stream1[2], stream2[2];

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

    kappaw2 = new double[N];
    kappaw1 = new double[N];
    buw2 = new double[N];
    buw1 = new double[N];
	
}// End of host_alloc function

void host_free()
{
    // source wavelet
    delete[] h_wlt;
    delete[] h_vel;     delete[] h_rho;
    
    // staggered grid
    delete[] bue;       
    delete[] bu2;       delete[] bu1;
    delete[] kappa;

    // sponge
    delete[] spg;       delete[] spg1;

    delete[] damp2a;    delete[] damp2b;
    delete[] damp1a;    delete[] damp1b;    

    delete[] damp2a1;   delete[] damp2b1;
    delete[] damp1a1;   delete[] damp1b1;

    delete[] kappaw2;   delete[] kappaw1;
    delete[] buw2;      delete[] buw1;

}// End of host_free function

void check_gpu_error(const char *msg)
{
    cudaError_t gpu_error;
    gpu_error = cudaGetLastError();
    if(gpu_error != cudaSuccess)
    {
        cout<<"\n Device error: "<<msg<<"   error string: "<<cudaGetErrorString(gpu_error)<<"\n";
        MPI::COMM_WORLD.Abort(-20);
    }

}// End of check_gpu_error function

void device_alloc(int device_id)
{
    // source wavelet
    cudaMalloc((void**)&d_wlt[device_id], nts*sizeof(double));    

    // sponge
    cudaMalloc((void**)&d_damp2a[device_id], nnx*sizeof(float));
    cudaMalloc((void**)&d_damp2b[device_id], nnx*sizeof(float));
    cudaMalloc((void**)&d_damp1a[device_id], nnz*sizeof(float));
    cudaMalloc((void**)&d_damp1b[device_id], nnz*sizeof(float));

    // wavefields
    cudaMalloc((void**)&d_p[device_id], N*sizeof(double));
    cudaMalloc((void**)&d_px[device_id], N*sizeof(double));
    cudaMalloc((void**)&d_pz[device_id], N*sizeof(double));
    cudaMalloc((void**)&d_vx[device_id], N*sizeof(double));
    cudaMalloc((void**)&d_vz[device_id], N*sizeof(double));

    // diffrential operators
    cudaMalloc((void**)&d_der2[device_id], N*sizeof(double));
    cudaMalloc((void**)&d_der1[device_id], N*sizeof(double));

    cudaMalloc((void**)&d_kappa[device_id], N*sizeof(double));

    cudaMalloc((void**)&d_kappaw2[device_id], N*sizeof(double));
    cudaMalloc((void**)&d_kappaw1[device_id], N*sizeof(double));
    cudaMalloc((void**)&d_buw2[device_id], N*sizeof(double));
    cudaMalloc((void**)&d_buw1[device_id], N*sizeof(double));

}// End of device_alloc function

void device_free(int device_id)
{   
    // source wavelet
    cudaFree(d_wlt[device_id]);

    // sponge
    cudaFree(d_damp2a[device_id]);     cudaFree(d_damp2b[device_id]);
    cudaFree(d_damp1a[device_id]);     cudaFree(d_damp1b[device_id]);
    
    // wavefields
    cudaFree(d_p[device_id]);         
    cudaFree(d_px[device_id]);         cudaFree(d_pz[device_id]);
    cudaFree(d_vx[device_id]);         cudaFree(d_vz[device_id]);
    
    // diffrential operators
    cudaFree(d_der2[device_id]);       cudaFree(d_der1[device_id]);

    cudaFree(d_kappa[device_id]);
    cudaFree(d_kappaw2[device_id]);    cudaFree(d_kappaw1[device_id]);
    cudaFree(d_buw2[device_id]);       cudaFree(d_buw1[device_id]);

}// End of device_free function

void FDOperator_4(int it, int isource, int device_id)
{	    
    if(it%1000 == 0)
        cout<<"\n it: "<<it;

    if(NJ == 4)
    {
        cudaMemset(d_der1[device_id], 0, N*sizeof(double));        
        cudaMemset(d_der2[device_id], 0, N*sizeof(double)); 
    
        cuda_fdoperator420<<<dimg0, dimb0>>>(d_vx[device_id], d_der2[device_id], nnz, nnx);
        cuda_fdoperator410<<<dimg0, dimb0>>>(d_vz[device_id], d_der1[device_id], nnz, nnx);

        cuda_pml_p<<<dimg0, dimb0>>>(d_px[device_id], d_pz[device_id], d_kappaw2[device_id], 
                                     d_kappaw1[device_id], d_der2[device_id], d_der1[device_id], 
                                     d_damp2a[device_id], d_damp1a[device_id], nnz, nnx);

        // increment source
        if(it <= nts)
        {
            cuda_incr_p<<<1,1>>>(d_px[device_id], d_wlt[device_id], d_kappa[device_id], 
                                 fd_dt, dx, dz, isource, it);
        }
        
        cuda_compute_p<<<dimg0, dimb0>>>(d_p[device_id], d_px[device_id], d_pz[device_id], nnz, nnx);

        cudaMemset(d_der1[device_id], 0, N*sizeof(double));        
        cudaMemset(d_der2[device_id], 0, N*sizeof(double));
 
        cuda_fdoperator421<<<dimg0, dimb0>>>(d_p[device_id], d_der2[device_id], nnz, nnx);
        cuda_fdoperator411<<<dimg0, dimb0>>>(d_p[device_id], d_der1[device_id], nnz, nnx);
    
        cuda_pml_v<<<dimg0, dimb0>>>(d_vx[device_id], d_vz[device_id], d_buw2[device_id], 
                                     d_buw1[device_id], d_der2[device_id], d_der1[device_id], 
                                     d_damp2b[device_id], d_damp1b[device_id], nnz, nnx);
    }// NJ=4
    
}//End of FDOperator function


void modelling_module(int my_nsrc, geo2d_t *mygeo2d_sp)
{
    int rank = MPI::COMM_WORLD.Get_rank();    	
    char log_file[2][1024], seis_file[2][1024], rnk[5], sx_ch[5];
    FILE *fp_log[2], *fp_seis[2];
    sprintf(rnk, "%d\0", rank);

    int ix, iz, id, is, ig, kt, istart[2], iend[2], dt_factor, indx;
    int device_id = 1, d_count;
    
    // variable init
    NJ = job_sp->fdop;              ns = my_nsrc;
    nx1 = mod_sp->nx;               nz1 = mod_sp->nz;

    dx  = mod_sp->dx;               dz   = mod_sp->dz;
    _dx = 1.0f/dx;                  _dz = 1.0f/dz;
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
    
    nx = (int)((nx1+Block_size2-1)/Block_size2) * Block_size2;
    nz = (int)((nz1+Block_size1-1)/Block_size1) * Block_size1;
        nnz = 2*npml+nz;
        nnx = 2*npml+nx;

        N = nnz*nnx;

    dimb0=dim3(Block_size1, Block_size2);       dimg0=dim3(nnz/Block_size1, nnx/Block_size2);    
    
    cout<<"\n Rank: "<<rank<<"\t nnx: "<<nnx<<"   nnz: "<<nnz<<"    nts: "<<nts;

    // allocate host memory
    host_alloc();

    // Get Device count
    cudaGetDeviceCount(&d_count);
    cout<<"\n Total number of devices attached are: "<<d_count;

    omp_set_num_threads(d_count);
    #pragma omp parallel private(device_id)
    {   
        device_id = omp_get_thread_num();

        // Set Device and allocate memory
        cudaSetDevice(device_id);
        check_gpu_error("Failed to initialise device..");
        
        device_alloc(device_id);
        check_gpu_error("Failed to allocate memory on device");
        
        // create two streams for memset
        cudaStreamCreate(&stream1[device_id]);          cudaStreamCreate(&stream2[device_id]);
    }

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
    
    // build staggered grids
    memset(bue, 0, N*sizeof(float));
    memset(bu2, 0, N*sizeof(float));
    memset(bu1, 0, N*sizeof(float));
    memset(kappa, 0, N*sizeof(double));
    cpu_stagger(h_vel, h_rho, bue, bu2, bu1, kappa, nnz, nnx); 

    // compute source wavelet
    memset(h_wlt, 0, nts*sizeof(double));
    cpu_ricker(fm, fd_dt, h_wlt, nts); 

    // build sponges
    memset(spg, 0, (npml+1)*sizeof(float));
    memset(spg1, 0, (npml+1)*sizeof(float));
    memset(damp2a, 0, nnx*sizeof(float));
    memset(damp2a1, 0, nnx*sizeof(float));
    memset(damp2b, 0, nnx*sizeof(float));
    memset(damp2b1, 0, nnx*sizeof(float));
    memset(damp1a, 0, nnz*sizeof(float));
    memset(damp1a1, 0, nnz*sizeof(float));
    memset(damp1b, 0, nnz*sizeof(float));
    memset(damp1b1, 0, nnz*sizeof(float));
     
    float xxfac = 0.05f;
    cpu_pml_coefficient_a(xxfac, damp2a, damp2a1, spg, spg1, npml, nx1);
    cpu_pml_coefficient_b(xxfac, damp2b, damp2b1, spg, spg1, npml, nx1);

    cpu_pml_coefficient_a(xxfac, damp1a, damp1a1, spg, spg1, npml, nz1);
    cpu_pml_coefficient_b(xxfac, damp1b, damp1b1, spg, spg1, npml, nz1); 

    memset(kappaw2, 0, N*sizeof(double));
    memset(kappaw1, 0, N*sizeof(double));
    memset(buw2, 0, N*sizeof(double));
    memset(buw1, 0, N*sizeof(double));

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

    #pragma omp parallel private(device_id)
    {
        device_id = omp_get_thread_num();
    
        cudaMemcpy(d_wlt[device_id], h_wlt, nts*sizeof(double), cudaMemcpyHostToDevice);    
        cudaMemcpy(d_damp2a[device_id], damp2a, nnx*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_damp2b[device_id], damp2b, nnx*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_damp1a[device_id], damp1a, nnz*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_damp1b[device_id], damp1b, nnz*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kappa[device_id], kappa, N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kappaw2[device_id], kappaw2, N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kappaw1[device_id], kappaw1, N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_buw2[device_id], buw2, N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_buw1[device_id], buw1, N*sizeof(double), cudaMemcpyHostToDevice);
    }

    // free host memory
    host_free();

    // set source positions
    sz_pos = new int[ns];      sx_pos = new int[ns];
    for(is = 0; is < ns; is++)
    {
        sz_pos[is] = (int)(mygeo2d_sp->src2d_sp[is].z*_dz);
        sx_pos[is] = (int)(mygeo2d_sp->src2d_sp[is].x*_dx);
    }
    h_Sxz = new int[ns];
    cpu_set_sg(h_Sxz, sx_pos, sz_pos, ns, npml, nnz);

    // checking job status - New/Restart
    #pragma omp parallel private(device_id)
    {
        device_id = omp_get_thread_num();
        strcpy(log_file[device_id], job_sp->tmppath);   strcat(log_file[device_id], job_sp->jobname);
        strcat(log_file[device_id], "checkpoint");
        strcat(log_file[device_id], "_rank");           strcat(log_file[device_id], rnk);  
        strcat(log_file[device_id], ".txt\0");

        if(device_id == 0)
            iend[0] = ns/2;
        else if(device_id == 1)
            iend[1] = ns;

        if(strcmp(job_sp->jbtype, "New") == 0)
        {
            if(device_id == 0) 
                istart[0] = 0;
            else if(device_id == 1)
                istart[1] = ns/2;
        }
        else if(strcmp(job_sp->jbtype, "Restart") == 0)
        {
            fp_log[device_id] = fopen(log_file[device_id], "r");
            if(fp_log[device_id] == NULL)
            {
                cerr<<"\n Rank: "<<rank<<"   Error !!! in openining log file: "<<log_file[device_id];
                cerr<<"\n Please make sure that same job is executed before with same resource \
                          configuration";
                MPI::COMM_WORLD.Abort(-10);
            }

            fscanf(fp_log[device_id], "%d", &istart[device_id]);
            fclose(fp_log[device_id]);
        }
        else
        {
            cerr<<"\n Error !!! Rank: "<<rank<<"   Unknow job status, Please make respective changes in job card";
            MPI::COMM_WORLD.Abort(-11);
        }
    }

    cout<<"\n Rank: "<<rank<<"   Started Modelling.......";
    // shot loop

    #pragma omp parallel private(is, device_id, ng, ig, kt, indx, sx_ch)
    {   
            device_id = omp_get_thread_num();
            cout<<"\n thread id: "<<device_id<<"  start: "<<istart[device_id];
            cudaSetDevice(device_id);
            check_gpu_error("Failed to initialise device..");
                
            for(is = istart[device_id]; is < iend[device_id]; is++)
            {    
                // set geaophone positions
                ng = mygeo2d_sp->nrec[is];
                gz_pos[device_id] = new int[ng];               gx_pos[device_id] = new int[ng];
                for(ig = 0; ig < ng; ig++)
                {
                    gz_pos[device_id][ig] = (int)(mygeo2d_sp->rec2d_sp[is][ig].z*_dz);
                    gx_pos[device_id][ig] = (int)(mygeo2d_sp->rec2d_sp[is][ig].x*_dx);
                }

                h_Gxz[device_id] = new int[ng];
                cpu_set_sg(h_Gxz[device_id], gx_pos[device_id], gz_pos[device_id], ng, npml, nnz);
                
                cudaMalloc((void**)&d_Gxz[device_id], ng*sizeof(int));
                    cudaMemcpy(d_Gxz[device_id], h_Gxz[device_id], ng*sizeof(int), cudaMemcpyHostToDevice);

                h_dobs[device_id] = new float[ng*real_nt];         
                    memset(h_dobs[device_id], 0, ng*real_nt*sizeof(float));
                h_temp[device_id] = new float[ng*real_nt];         
                    memset(h_temp[device_id], 0, ng*real_nt*sizeof(float));

                cudaMalloc((void**)&d_temp[device_id], ng*real_nt*sizeof(float));
                    cudaMemset(d_temp[device_id], 0, ng*real_nt*sizeof(float));

                cudaMalloc((void**)&d_dobs[device_id], ng*real_nt*sizeof(float));
                    cudaMemset(d_dobs[device_id], 0, ng*real_nt*sizeof(float));

                cudaMemset(d_p[device_id], 0, N*sizeof(double));
                cudaMemset(d_px[device_id], 0, N*sizeof(double));
                cudaMemset(d_pz[device_id], 0, N*sizeof(double));
                cudaMemset(d_vx[device_id], 0, N*sizeof(double));
                cudaMemset(d_vz[device_id], 0, N*sizeof(double));
                
                for(kt = 0; kt < fd_nt; kt++)
                {

                    FDOperator_4(kt, h_Sxz[is],device_id);

                    // storing pressure seismograms
                    if( (kt+1)%dt_factor == 0 )
                    {
                        indx = ((kt+1)/dt_factor) - 1;
                        cuda_record<<<(ng+255)/256, 256>>>(d_p[device_id], &d_temp[device_id][indx*ng], d_Gxz[device_id], ng);
                    }
                } // End of NT loop

                delete[] gz_pos[device_id];         gz_pos[device_id] = NULL;
                delete[] gx_pos[device_id];         gx_pos[device_id] = NULL;
                delete[] h_Gxz[device_id];          h_Gxz[device_id]  = NULL;
                cudaFree(d_Gxz[device_id]);

                cuda_transpose<<<dim3((ng+15)/16,(real_nt+15)/16), dim3(16,16)>>>(d_temp[device_id], d_dobs[device_id], ng, real_nt);
                cudaMemcpy(h_dobs[device_id], d_dobs[device_id], ng*real_nt*sizeof(float), cudaMemcpyDeviceToHost);
                // write seismogram
                strcpy(seis_file[device_id], job_sp->tmppath);  
                strcat(seis_file[device_id], job_sp->jobname);
                strcpy(sx_ch, "\0");                    sprintf(sx_ch, "%.2f\0", sx_pos[is]*dx);     
                strcat(seis_file[device_id], "sx");     strcat(seis_file[device_id], sx_ch);
                strcat(seis_file[device_id], "_seismogram.bin\0");

                fp_seis[device_id] = fopen(seis_file[device_id], "w");
                    fwrite(h_dobs[device_id], sizeof(float), ng*real_nt, fp_seis[device_id]);
                fclose(fp_seis[device_id]);

                delete[] h_dobs[device_id];         h_dobs[device_id] = NULL;
                delete[] h_temp[device_id];         h_temp[device_id] = NULL;
                cudaFree(d_dobs[device_id]);        cudaFree(d_temp[device_id]);

                cout<<"\n Rank: "<<rank<<"   thread id: "<<device_id<<"   Shot remaining: "<<iend[device_id]- (is+1);
            }// End of shot loop
            
            device_free(device_id);
    }// End of parallel sections

}// End of modelling_module function



