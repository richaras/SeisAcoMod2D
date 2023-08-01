#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
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
static float    fm, fd_dt, real_dt, dz, dx, _dz, _dx, rec_len;
/* DPCT_ORIG static dim3     dimg0, dimb0;*/
static sycl::range<3> dimg0(1, 1, 1), dimb0(1, 1, 1);

float   *v0, *h_vel, *h_rho;
/* variables on device */
int     *sx_pos, *sz_pos;
int     *gx_pos, *gz_pos;
int     *h_Sxz, *h_Gxz;     /* set source and geophone position */
int     *d_Gxz;

double dtdx, dtdz;

float *bu1, *bu2, *bue;
float *spg, *spg1, *damp1a1, *damp2a1, *damp1b1, *damp2b1;
float *damp1a, *damp2a, *damp1b, *damp2b;
float *d_damp1a, *d_damp2a, *d_damp1b, *d_damp2b;

double *h_wlt;  //source wavelet
double *d_wlt;
float *h_dobs, *h_temp;  // seismogram
float *d_dobs, *d_temp;  // seismogram

double *d_p, *d_px, *d_pz, *d_vx, *d_vz, *d_der2, *d_der1;

double *kappa, *kappaw2, *kappaw1;
double *d_kappa, *d_kappaw2, *d_kappaw1;
double *buw2, *buw1;
double *d_buw2, *d_buw1;

/* DPCT_ORIG cudaStream_t stream1, stream2;*/
dpct::queue_ptr stream1, stream2;

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
/* DPCT_ORIG     cudaError_t gpu_error;*/
    int gpu_error;
/* DPCT_ORIG     gpu_error = cudaGetLastError();*/
    /*
    DPCT1010:9: SYCL uses exceptions to report errors and does not use the error
    codes. The call was replaced with 0. You need to rewrite this code.
    */
    gpu_error = 0;
/* DPCT_ORIG     if(gpu_error != cudaSuccess)*/
    /*
    DPCT1000:8: Error handling if-stmt was detected but could not be rewritten.
    */
    if (gpu_error != 0)
    {
/* DPCT_ORIG         cout<<"\n Device error: "<<msg<<"   error string:
 * "<<cudaGetErrorString(gpu_error)<<"\n";*/
        /*
        DPCT1009:10: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        /*
        DPCT1001:7: The statement could not be removed.
        */
        cout
            << "\n Device error: " << msg << "   error string: "
            << "cudaGetErrorString is not supported" /*cudaGetErrorString(gpu_error)*/
            << "\n";
        MPI::COMM_WORLD.Abort(-20);
    }

}// End of check_gpu_error function

void device_alloc()
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    // source wavelet
/* DPCT_ORIG     cudaMalloc((void**)&d_wlt, nts*sizeof(double));    */
    d_wlt = sycl::malloc_device<double>(nts, dpct::get_default_queue());

    // sponge
/* DPCT_ORIG     cudaMalloc((void**)&d_damp2a, nnx*sizeof(float));*/
    d_damp2a = sycl::malloc_device<float>(nnx, dpct::get_default_queue());
/* DPCT_ORIG     cudaMalloc((void**)&d_damp2b, nnx*sizeof(float));*/
    d_damp2b = sycl::malloc_device<float>(nnx, dpct::get_default_queue());
/* DPCT_ORIG     cudaMalloc((void**)&d_damp1a, nnz*sizeof(float));*/
    d_damp1a = sycl::malloc_device<float>(nnz, dpct::get_default_queue());
/* DPCT_ORIG     cudaMalloc((void**)&d_damp1b, nnz*sizeof(float));*/
    d_damp1b = sycl::malloc_device<float>(nnz, dpct::get_default_queue());

    // wavefields
/* DPCT_ORIG     cudaMalloc((void**)&d_p, N*sizeof(double));*/
    d_p = sycl::malloc_device<double>(N, dpct::get_default_queue());
/* DPCT_ORIG     cudaMalloc((void**)&d_px, N*sizeof(double));*/
    d_px = sycl::malloc_device<double>(N, dpct::get_default_queue());
/* DPCT_ORIG     cudaMalloc((void**)&d_pz, N*sizeof(double));*/
    d_pz = sycl::malloc_device<double>(N, dpct::get_default_queue());
/* DPCT_ORIG     cudaMalloc((void**)&d_vx, N*sizeof(double));*/
    d_vx = sycl::malloc_device<double>(N, dpct::get_default_queue());
/* DPCT_ORIG     cudaMalloc((void**)&d_vz, N*sizeof(double));*/
    d_vz = sycl::malloc_device<double>(N, dpct::get_default_queue());

    // diffrential operators
/* DPCT_ORIG     cudaMalloc((void**)&d_der2, N*sizeof(double));*/
    d_der2 = sycl::malloc_device<double>(N, dpct::get_default_queue());
/* DPCT_ORIG     cudaMalloc((void**)&d_der1, N*sizeof(double));*/
    d_der1 = sycl::malloc_device<double>(N, dpct::get_default_queue());

/* DPCT_ORIG     cudaMalloc((void**)&d_kappa, N*sizeof(double));*/
    d_kappa = sycl::malloc_device<double>(N, dpct::get_default_queue());

/* DPCT_ORIG     cudaMalloc((void**)&d_kappaw2, N*sizeof(double));*/
    d_kappaw2 = sycl::malloc_device<double>(N, dpct::get_default_queue());
/* DPCT_ORIG     cudaMalloc((void**)&d_kappaw1, N*sizeof(double));*/
    d_kappaw1 = sycl::malloc_device<double>(N, dpct::get_default_queue());
/* DPCT_ORIG     cudaMalloc((void**)&d_buw2, N*sizeof(double));*/
    d_buw2 = sycl::malloc_device<double>(N, dpct::get_default_queue());
/* DPCT_ORIG     cudaMalloc((void**)&d_buw1, N*sizeof(double));*/
    d_buw1 = sycl::malloc_device<double>(N, dpct::get_default_queue());

}// End of device_alloc function

void device_free()
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    // source wavelet
/* DPCT_ORIG     cudaFree(d_wlt);*/
    sycl::free(d_wlt, dpct::get_default_queue());

    // sponge
/* DPCT_ORIG     cudaFree(d_damp2a);     cudaFree(d_damp2b);*/
    sycl::free(d_damp2a, dpct::get_default_queue());
        sycl::free(d_damp2b, dpct::get_default_queue());
/* DPCT_ORIG     cudaFree(d_damp1a);     cudaFree(d_damp1b);*/
    sycl::free(d_damp1a, dpct::get_default_queue());
        sycl::free(d_damp1b, dpct::get_default_queue());

    // wavefields
/* DPCT_ORIG     cudaFree(d_p);         */
    sycl::free(d_p, dpct::get_default_queue());
/* DPCT_ORIG     cudaFree(d_px);         cudaFree(d_pz);*/
    sycl::free(d_px, dpct::get_default_queue());
        sycl::free(d_pz, dpct::get_default_queue());
/* DPCT_ORIG     cudaFree(d_vx);         cudaFree(d_vz);*/
    sycl::free(d_vx, dpct::get_default_queue());
        sycl::free(d_vz, dpct::get_default_queue());

    // diffrential operators
/* DPCT_ORIG     cudaFree(d_der2);       cudaFree(d_der1);*/
    sycl::free(d_der2, dpct::get_default_queue());
        sycl::free(d_der1, dpct::get_default_queue());

/* DPCT_ORIG     cudaFree(d_kappa);*/
    sycl::free(d_kappa, dpct::get_default_queue());
/* DPCT_ORIG     cudaFree(d_kappaw2);    cudaFree(d_kappaw1);*/
    sycl::free(d_kappaw2, dpct::get_default_queue());
        sycl::free(d_kappaw1, dpct::get_default_queue());
/* DPCT_ORIG     cudaFree(d_buw2);       cudaFree(d_buw1);*/
    sycl::free(d_buw2, dpct::get_default_queue());
        sycl::free(d_buw1, dpct::get_default_queue());

}// End of device_free function

void FDOperator_4(int it, int isource)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
    if(it%1000 == 0)
        cout<<"\n it: "<<it;

    if(NJ == 4)
    {
        //cudaMemset(d_der1, 0, N*sizeof(double));        cudaMemset(d_der2, 0, N*sizeof(double));
/* DPCT_ORIG         cudaMemsetAsync(d_der1, 0, N*sizeof(double), stream1); */
        //stream1->memset(d_der1, 0, N * sizeof(double));
/* DPCT_ORIG         cudaMemsetAsync(d_der2, 0, N*sizeof(double), stream2);*/
        //stream2->memset(d_der2, 0, N * sizeof(double));
	stream1->fill(d_der1,(double) 0, N);
	stream2->fill(d_der2,(double) 0, N);
/* DPCT_ORIG         cuda_fdoperator420<<<dimg0, dimb0>>>(d_vx, d_der2, nnz,
 * nnx);*/
        /*
        DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            sycl::local_accessor<double, 2> s_vx_acc_ct1(
                sycl::range<2>(16 /*16*/, 19 /*Block_size2+3*/), cgh);

            auto d_vx_ct0 = d_vx;
            auto d_der2_ct1 = d_der2;
            auto nnz_ct2 = nnz;
            auto nnx_ct3 = nnx;

            cgh.parallel_for(sycl::nd_range<3>(dimg0 * dimb0, dimb0),
                             [=](sycl::nd_item<3> item_ct1) {
                                 cuda_fdoperator420(d_vx_ct0, d_der2_ct1,
                                                    nnz_ct2, nnx_ct3, item_ct1,
                                                    s_vx_acc_ct1);
                             });
        });
/* DPCT_ORIG         cuda_fdoperator410<<<dimg0, dimb0>>>(d_vz, d_der1, nnz,
 * nnx);*/
        /*
        DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            sycl::local_accessor<double, 2> s_vz_acc_ct1(
                sycl::range<2>(19 /*Block_size1+3*/, 16 /*16*/), cgh);

            auto d_vz_ct0 = d_vz;
            auto d_der1_ct1 = d_der1;
            auto nnz_ct2 = nnz;
            auto nnx_ct3 = nnx;

            cgh.parallel_for(sycl::nd_range<3>(dimg0 * dimb0, dimb0),
                             [=](sycl::nd_item<3> item_ct1) {
                                 cuda_fdoperator410(d_vz_ct0, d_der1_ct1,
                                                    nnz_ct2, nnx_ct3, item_ct1,
                                                    s_vz_acc_ct1);
                             });
        });

/* DPCT_ORIG         cuda_pml_p<<<dimg0, dimb0>>>(d_px, d_pz, d_kappaw2,
   d_kappaw1, d_der2, d_der1, d_damp2a, d_damp1a, nnz, nnx);*/
        /*
        DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            auto d_px_ct0 = d_px;
            auto d_pz_ct1 = d_pz;
            auto d_kappaw2_ct2 = d_kappaw2;
            auto d_kappaw1_ct3 = d_kappaw1;
            auto d_der2_ct4 = d_der2;
            auto d_der1_ct5 = d_der1;
            auto d_damp2a_ct6 = d_damp2a;
            auto d_damp1a_ct7 = d_damp1a;
            auto nnz_ct8 = nnz;
            auto nnx_ct9 = nnx;

            cgh.parallel_for(
                sycl::nd_range<3>(dimg0 * dimb0, dimb0),
                [=](sycl::nd_item<3> item_ct1) {
                    cuda_pml_p(d_px_ct0, d_pz_ct1, d_kappaw2_ct2, d_kappaw1_ct3,
                               d_der2_ct4, d_der1_ct5, d_damp2a_ct6,
                               d_damp1a_ct7, nnz_ct8, nnx_ct9, item_ct1);
                });
        });
        dpct::get_default_queue().wait(); // added by om
        // increment source
        if(it <= nts)
        {
/* DPCT_ORIG             cuda_incr_p<<<1,1>>>(d_px, d_wlt, d_kappa, fd_dt, dx,
 * dz, isource, it);*/
            dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                auto d_px_ct0 = d_px;
                auto d_wlt_ct1 = d_wlt;
                auto d_kappa_ct2 = d_kappa;
                auto fd_dt_ct3 = fd_dt;
                auto dx_ct4 = dx;
                auto dz_ct5 = dz;

                cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1),
                                                   sycl::range<3>(1, 1, 1)),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     cuda_incr_p(d_px_ct0, d_wlt_ct1,
                                                 d_kappa_ct2, fd_dt_ct3, dx_ct4,
                                                 dz_ct5, isource, it);
                                 });
            });
        }

/* DPCT_ORIG         cuda_compute_p<<<dimg0, dimb0>>>(d_p, d_px, d_pz, nnz,
 * nnx);*/
        /*
        DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            auto d_p_ct0 = d_p;
            auto d_px_ct1 = d_px;
            auto d_pz_ct2 = d_pz;
            auto nnz_ct3 = nnz;
            auto nnx_ct4 = nnx;

            cgh.parallel_for(sycl::nd_range<3>(dimg0 * dimb0, dimb0),
                             [=](sycl::nd_item<3> item_ct1) {
                                 cuda_compute_p(d_p_ct0, d_px_ct1, d_pz_ct2,
                                                nnz_ct3, nnx_ct4, item_ct1);
                             });
        });

/* DPCT_ORIG         cudaMemsetAsync(d_der1, 0, N*sizeof(double), stream1);*/
       // stream1->memset(d_der1, 0, N * sizeof(double));
/* DPCT_ORIG         cudaMemsetAsync(d_der2, 0, N*sizeof(double), stream2);*/
      //  stream2->memset(d_der2, 0, N * sizeof(double));
	stream1->fill(d_der1,(double) 0, N);
	stream2->fill(d_der2,(double) 0, N);

/* DPCT_ORIG         cuda_fdoperator421<<<dimg0, dimb0>>>(d_p, d_der2, nnz,
 * nnx);*/
        /*
        DPCT1049:4: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            sycl::local_accessor<double, 2> s_p_acc_ct1(
                sycl::range<2>(16 /*16*/, 19 /*Block_size2+3*/), cgh);

            auto d_p_ct0 = d_p;
            auto d_der2_ct1 = d_der2;
            auto nnz_ct2 = nnz;
            auto nnx_ct3 = nnx;

            cgh.parallel_for(sycl::nd_range<3>(dimg0 * dimb0, dimb0),
                             [=](sycl::nd_item<3> item_ct1) {
                                 cuda_fdoperator421(d_p_ct0, d_der2_ct1,
                                                    nnz_ct2, nnx_ct3, item_ct1,
                                                    s_p_acc_ct1);
                             });
        });
/* DPCT_ORIG         cuda_fdoperator411<<<dimg0, dimb0>>>(d_p, d_der1, nnz,
 * nnx);*/
        /*
        DPCT1049:5: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            sycl::local_accessor<double, 2> s_p_acc_ct1(
                sycl::range<2>(19 /*Block_size1+3*/, 16 /*16*/), cgh);

            auto d_p_ct0 = d_p;
            auto d_der1_ct1 = d_der1;
            auto nnz_ct2 = nnz;
            auto nnx_ct3 = nnx;

            cgh.parallel_for(sycl::nd_range<3>(dimg0 * dimb0, dimb0),
                             [=](sycl::nd_item<3> item_ct1) {
                                 cuda_fdoperator411(d_p_ct0, d_der1_ct1,
                                                    nnz_ct2, nnx_ct3, item_ct1,
                                                    s_p_acc_ct1);
                             });
        });

/* DPCT_ORIG         cuda_pml_v<<<dimg0, dimb0>>>(d_vx, d_vz, d_buw2, d_buw1,
   d_der2, d_der1, d_damp2b, d_damp1b, nnz, nnx);*/
        /*
        DPCT1049:6: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            auto d_vx_ct0 = d_vx;
            auto d_vz_ct1 = d_vz;
            auto d_buw2_ct2 = d_buw2;
            auto d_buw1_ct3 = d_buw1;
            auto d_der2_ct4 = d_der2;
            auto d_der1_ct5 = d_der1;
            auto d_damp2b_ct6 = d_damp2b;
            auto d_damp1b_ct7 = d_damp1b;
            auto nnz_ct8 = nnz;
            auto nnx_ct9 = nnx;

            cgh.parallel_for(sycl::nd_range<3>(dimg0 * dimb0, dimb0),
                             [=](sycl::nd_item<3> item_ct1) {
                                 cuda_pml_v(d_vx_ct0, d_vz_ct1, d_buw2_ct2,
                                            d_buw1_ct3, d_der2_ct4, d_der1_ct5,
                                            d_damp2b_ct6, d_damp1b_ct7, nnz_ct8,
                                            nnx_ct9, item_ct1);
                             });
        });
	dpct::get_default_queue().wait(); //  added by om
    }// NJ=4
    
}//End of FDOperator function


void modelling_module(int my_nsrc, geo2d_t *mygeo2d_sp)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
    int rank = MPI::COMM_WORLD.Get_rank();    	
    char log_file[1024], seis_file[1024], rnk[5], sx_ch[5];
    FILE *fp_log, *fp_seis;
    sprintf(rnk, "%d\0", rank);

    int ix, iz, id, is, ig, kt, istart, dt_factor, indx;

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

/* DPCT_ORIG     dimb0=dim3(Block_size1, Block_size2);
 * dimg0=dim3(nnz/Block_size1, nnx/Block_size2);    */
    dimb0 = sycl::range<3>(1, Block_size2, Block_size1);
        dimg0 = sycl::range<3>(1, nnx / Block_size2, nnz / Block_size1);

    cout<<"\n Rank: "<<rank<<"\t nnx: "<<nnx<<"   nnz: "<<nnz<<"    nts: "<<nts;

    // allocate host memory
    host_alloc();

    // Set Device and allocate memory
/* DPCT_ORIG     cudaSetDevice(0);*/
    /*
    DPCT1093:11: The "0" may not be the best XPU device. Adjust the selected
    device if needed.
    */
    dpct::select_device(0);
    check_gpu_error("Failed to initialise device..");

    device_alloc();
    check_gpu_error("Failed to allocate memory on device");
    
    // create two streams for memset
/* DPCT_ORIG     cudaStreamCreate(&stream1); cudaStreamCreate(&stream2);*/
    stream1 = dpct::get_current_device().create_queue();
        stream2 = dpct::get_current_device().create_queue();
    //stream1 = &(dpct::get_default_queue());
    //stream2 = &(dpct::get_default_queue());      // edited for result correction - om 

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
/* DPCT_ORIG     cudaMemcpy(d_wlt, h_wlt, nts*sizeof(double),
 * cudaMemcpyHostToDevice);    */
    dpct::get_default_queue().memcpy(d_wlt, h_wlt, nts * sizeof(double)).wait();

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
/* DPCT_ORIG     cudaMemcpy(d_damp2a, damp2a, nnx*sizeof(float),
 * cudaMemcpyHostToDevice);*/
    dpct::get_default_queue()
        .memcpy(d_damp2a, damp2a, nnx * sizeof(float))
        .wait();
    cpu_pml_coefficient_b(xxfac, damp2b, damp2b1, spg, spg1, npml, nx1);
/* DPCT_ORIG     cudaMemcpy(d_damp2b, damp2b, nnx*sizeof(float),
 * cudaMemcpyHostToDevice);*/
    dpct::get_default_queue()
        .memcpy(d_damp2b, damp2b, nnx * sizeof(float))
        .wait();

    cpu_pml_coefficient_a(xxfac, damp1a, damp1a1, spg, spg1, npml, nz1);
/* DPCT_ORIG     cudaMemcpy(d_damp1a, damp1a, nnz*sizeof(float),
 * cudaMemcpyHostToDevice);*/
    dpct::get_default_queue()
        .memcpy(d_damp1a, damp1a, nnz * sizeof(float))
        .wait();
    cpu_pml_coefficient_b(xxfac, damp1b, damp1b1, spg, spg1, npml, nz1);
/* DPCT_ORIG     cudaMemcpy(d_damp1b, damp1b, nnz*sizeof(float),
 * cudaMemcpyHostToDevice);*/
    dpct::get_default_queue()
        .memcpy(d_damp1b, damp1b, nnz * sizeof(float))
        .wait();

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

/* DPCT_ORIG     cudaMemcpy(d_kappa, kappa, N*sizeof(double),
 * cudaMemcpyHostToDevice);*/
    dpct::get_default_queue().memcpy(d_kappa, kappa, N * sizeof(double));
/* DPCT_ORIG     cudaMemcpy(d_kappaw2, kappaw2, N*sizeof(double),
 * cudaMemcpyHostToDevice);*/
    dpct::get_default_queue().memcpy(d_kappaw2, kappaw2, N * sizeof(double));
/* DPCT_ORIG     cudaMemcpy(d_kappaw1, kappaw1, N*sizeof(double),
 * cudaMemcpyHostToDevice);*/
    dpct::get_default_queue().memcpy(d_kappaw1, kappaw1, N * sizeof(double));
/* DPCT_ORIG     cudaMemcpy(d_buw2, buw2, N*sizeof(double),
 * cudaMemcpyHostToDevice);*/
    dpct::get_default_queue().memcpy(d_buw2, buw2, N * sizeof(double));
/* DPCT_ORIG     cudaMemcpy(d_buw1, buw1, N*sizeof(double),
 * cudaMemcpyHostToDevice);*/
    dpct::get_default_queue().memcpy(d_buw1, buw1, N * sizeof(double)).wait();

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

/* DPCT_ORIG         cudaMalloc((void**)&d_Gxz, ng*sizeof(int));  */
        d_Gxz = sycl::malloc_device<int>(ng, dpct::get_default_queue());
/* DPCT_ORIG         cudaMemcpy(d_Gxz, h_Gxz, ng*sizeof(int),
 * cudaMemcpyHostToDevice);*/
        dpct::get_default_queue().memcpy(d_Gxz, h_Gxz, ng * sizeof(int)).wait();

        h_dobs = new float[ng*real_nt];         memset(h_dobs, 0, ng*real_nt*sizeof(float));
        h_temp = new float[ng*real_nt];         memset(h_temp, 0, ng*real_nt*sizeof(float));

/* DPCT_ORIG         cudaMalloc((void**)&d_temp, ng*real_nt*sizeof(float));*/
        d_temp =
            sycl::malloc_device<float>(ng * real_nt, dpct::get_default_queue());
/* DPCT_ORIG         cudaMemset(d_temp, 0, ng*real_nt*sizeof(float));*/
        dpct::get_default_queue()
            .memset(d_temp, 0, ng * real_nt * sizeof(float))
            .wait();

/* DPCT_ORIG         cudaMalloc((void**)&d_dobs, ng*real_nt*sizeof(float));*/
        d_dobs =
            sycl::malloc_device<float>(ng * real_nt, dpct::get_default_queue());
/* DPCT_ORIG         cudaMemset(d_dobs, 0, ng*real_nt*sizeof(float));*/
        dpct::get_default_queue()
            .memset(d_dobs, 0, ng * real_nt * sizeof(float))
            .wait();

/* DPCT_ORIG         cudaMemset(d_p, 0, N*sizeof(double));*/
        dpct::get_default_queue().memset(d_p, 0, N * sizeof(double)).wait();
/* DPCT_ORIG         cudaMemset(d_px, 0, N*sizeof(double));*/
        dpct::get_default_queue().memset(d_px, 0, N * sizeof(double)).wait();
/* DPCT_ORIG         cudaMemset(d_pz, 0, N*sizeof(double));*/
        dpct::get_default_queue().memset(d_pz, 0, N * sizeof(double)).wait();
/* DPCT_ORIG         cudaMemset(d_vx, 0, N*sizeof(double));*/
        dpct::get_default_queue().memset(d_vx, 0, N * sizeof(double)).wait();
/* DPCT_ORIG         cudaMemset(d_vz, 0, N*sizeof(double));*/
        dpct::get_default_queue().memset(d_vz, 0, N * sizeof(double)).wait();

        for(kt = 0; kt < fd_nt; kt++)
        {
           
            FDOperator_4(kt, h_Sxz[is]);
               
            // storing pressure seismograms
            if( (kt+1)%dt_factor == 0 )
            {
                indx = ((kt+1)/dt_factor) - 1;

/* DPCT_ORIG               cuda_record<<<(ng+255)/256, 256>>>(d_p,
 * &d_temp[indx*ng], d_Gxz, ng);*/
                dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                    auto d_p_ct0 = d_p;
                    auto d_temp_indx_ng_ct1 = &d_temp[indx * ng];
                    auto d_Gxz_ct2 = d_Gxz;
                    auto ng_ct3 = ng;

                    cgh.parallel_for(
                        sycl::nd_range<3>(
                            sycl::range<3>(1, 1, (ng + 255) / 256) *
                                sycl::range<3>(1, 1, 256),
                            sycl::range<3>(1, 1, 256)),
                        [=](sycl::nd_item<3> item_ct1) {
                            cuda_record(d_p_ct0, d_temp_indx_ng_ct1, d_Gxz_ct2,
                                        ng_ct3, item_ct1);
                        });
                });
            }

        } // End of NT loop

        delete[] gz_pos;    delete[] gx_pos;
        delete[] h_Gxz;
/* DPCT_ORIG         cudaFree(d_Gxz);*/
        sycl::free(d_Gxz, dpct::get_default_queue());

/* DPCT_ORIG         cuda_transpose<<<dim3((ng+15)/16,(real_nt+15)/16),
 * dim3(16,16)>>>(d_temp, d_dobs, ng, real_nt);*/
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            auto d_temp_ct0 = d_temp;
            auto d_dobs_ct1 = d_dobs;
            auto ng_ct2 = ng;
            auto real_nt_ct3 = real_nt;

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, (real_nt + 15) / 16, (ng + 15) / 16) *
                        sycl::range<3>(1, 16, 16),
                    sycl::range<3>(1, 16, 16)),
                [=](sycl::nd_item<3> item_ct1) {
                    cuda_transpose(d_temp_ct0, d_dobs_ct1, ng_ct2, real_nt_ct3,
                                   item_ct1);
                });
        });
/* DPCT_ORIG         cudaMemcpy(h_dobs, d_dobs, ng*real_nt*sizeof(float),
 * cudaMemcpyDeviceToHost);*/
        dpct::get_default_queue()
            .memcpy(h_dobs, d_dobs, ng * real_nt * sizeof(float))
            .wait();

        // write seismogram
        strcpy(seis_file, job_sp->tmppath);  strcat(seis_file, job_sp->jobname);       
        strcpy(sx_ch, "\0");        sprintf(sx_ch, "%.2f\0", sx_pos[is]*dx);
        strcat(seis_file, "sx");    strcat(seis_file, sx_ch);
        strcat(seis_file, "_seismogram.bin\0");
        fp_seis = fopen(seis_file, "w");
            fwrite(h_dobs, sizeof(float), ng*real_nt, fp_seis);
        fclose(fp_seis);

        delete[] h_dobs;    delete[] h_temp;
/* DPCT_ORIG         cudaFree(d_dobs);   cudaFree(d_temp);   */
        sycl::free(d_dobs, dpct::get_default_queue());
            sycl::free(d_temp, dpct::get_default_queue());

        cout<<"\n Rank: "<<rank<<"   Shot remaining: "<<ns - (is+1);   
 
    }// End of shot loop

   // device_free();
/* DPCT_ORIG     cudaStreamDestroy(stream1);     cudaStreamDestroy(stream2);*/
    dpct::get_current_device().destroy_queue(stream1);
        dpct::get_current_device().destroy_queue(stream2);

    device_free();    // edited - om
}//End of modelling_module function



