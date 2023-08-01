#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#ifndef CPU_MODELLING_KERNELS_H
#define CPU_MODELLING_KERNELS_H

void cpu_set_sg(int *sxz, int *sx_pos, int *sz_pos, int ns, int npml, int nnz);

void cpu_record(double *h_p, float *seis_kt, int *h_Gxz, int ng);
/* DPCT_ORIG __global__ void cuda_record(double *d_p, float *seis_kt, int
 * *d_Gxz, int ng);*/
SYCL_EXTERNAL void cuda_record(double *d_p, float *seis_kt, int *d_Gxz, int ng,
                               sycl::nd_item<3> item_ct1);

void cpu_ricker(float fm, float dt, double *wlt, int nts);

void cpu_stagger(float *vpe, float *rhoe, float *bue, float *bu2, float *bu1, double *kappa, int nnz, int nnx);

void cpu_pml_coefficient_a(float fac, float *damp, float *damp1, float *spg, float *spg1, int npml, int n);

void cpu_pml_coefficient_b(float fac,float *damp,float *damp1, float *spg, float *spg1, int npml, int n);

void FDOperator_4(int it, int isource);

void fdoperator420(double *vx, double *der2, int nnz, int nnx);
/* DPCT_ORIG __global__ void cuda_fdoperator420(double *vx, double *der2, int
 * nnz, int nnx);*/
SYCL_EXTERNAL void cuda_fdoperator420(double *vx, double *der2, int nnz,
                                      int nnx, sycl::nd_item<3> item_ct1,
                                      sycl::local_accessor<double, 2> s_vx);
void fdoperator410(double *vz, double *der1, int nnz, int nnx);
/* DPCT_ORIG __global__ void cuda_fdoperator410(double *vz, double *der1, int
 * nnz, int nnx);*/
SYCL_EXTERNAL void cuda_fdoperator410(double *vz, double *der1, int nnz,
                                      int nnx, sycl::nd_item<3> item_ct1,
                                      sycl::local_accessor<double, 2> s_vz);

void fdoperator421(double *p, double *der2, int nnz, int nnx);
/* DPCT_ORIG __global__ void cuda_fdoperator421(double *p, double *der2, int
 * nnz, int nnx);*/
SYCL_EXTERNAL void cuda_fdoperator421(double *p, double *der2, int nnz, int nnx,
                                      sycl::nd_item<3> item_ct1,
                                      sycl::local_accessor<double, 2> s_p);
void fdoperator411(double *p, double *der1, int nnz, int nnx);
/* DPCT_ORIG __global__ void cuda_fdoperator411(double *p, double *der1, int
 * nnz, int nnx);*/
SYCL_EXTERNAL void cuda_fdoperator411(double *p, double *der1, int nnz, int nnx,
                                      sycl::nd_item<3> item_ct1,
                                      sycl::local_accessor<double, 2> s_p);

/* DPCT_ORIG __global__ void cuda_incr_p(double *px, double *wlt, double *kappa,
 * float fd_dt, float dx, float dz, int isource, int it);*/
SYCL_EXTERNAL void cuda_incr_p(double *px, double *wlt, double *kappa,
                               float fd_dt, float dx, float dz, int isource,
                               int it);

void compute_p(double *p, double *px, double *pz, int nnz, int nnx);
/* DPCT_ORIG __global__ void cuda_compute_p(double *p, double *px, double *pz,
 * int nnz, int nnx);*/
SYCL_EXTERNAL void cuda_compute_p(double *p, double *px, double *pz, int nnz,
                                  int nnx, sycl::nd_item<3> item_ct1);

void cpu_pml_p(double *px, double *pz, double *kappaw2, double *kappaw1, double *der2, double *der1, float *damp2a, float *damp1a, int nnz, int nnx);
/* DPCT_ORIG __global__ void cuda_pml_p(double *px, double *pz, double *kappaw2,
 * double *kappaw1, double *der2, double *der1, float *damp2a, float *damp1a,
 * int nnz, int nnx);*/
SYCL_EXTERNAL void cuda_pml_p(double *px, double *pz, double *kappaw2,
                              double *kappaw1, double *der2, double *der1,
                              float *damp2a, float *damp1a, int nnz, int nnx,
                              sycl::nd_item<3> item_ct1);
void cpu_pml_v(double *vx, double *vz, double *buw2, double *buw1, double *der2, double *der1, float *damp2b, float *damp1b, int nnz, int nnx);
/* DPCT_ORIG __global__ void cuda_pml_v(double *vx, double *vz, double *buw2,
 * double *buw1, double *der2, double *der1, float *damp2b, float *damp1b, int
 * nnz, int nnx);*/
SYCL_EXTERNAL void cuda_pml_v(double *vx, double *vz, double *buw2,
                              double *buw1, double *der2, double *der1,
                              float *damp2b, float *damp1b, int nnz, int nnx,
                              sycl::nd_item<3> item_ct1);

void cpu_transpose(float *inp, float *out, int n1, int n2);
/* DPCT_ORIG __global__ void cuda_transpose(float *inp, float *out, int n1, int
 * n2);*/
SYCL_EXTERNAL void cuda_transpose(float *inp, float *out, int n1, int n2,
                                  sycl::nd_item<3> item_ct1);

#endif
