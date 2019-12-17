#include <iostream>
#include <cmath>
#include <cstring>
#include <omp.h>
#include "cpu_modelling_kernels.h"

#define PI 3.141592654

#define a0 1.125f
#define a1 -1.0f/24.0f

using namespace std;

// set the positions of sources and geophones in whole domain
void cpu_set_sg(int *sxz, int *sx_pos, int *sz_pos, int ns, int npml, int nnz)
{
    int id;

    for(id = 0; id < ns; id++)
        sxz[id] = nnz*(sx_pos[id]+npml) + (sz_pos[id]+npml);

}// End of cpu_set_sg function

void cpu_record(double *p, float *seis_kt, int *h_Gxz, int ng)
{
    int id;
    
    #pragma acc parallel loop independent present(p, h_Gxz)
    for(id = 0; id < ng; id++)
        seis_kt[id] = (float) p[h_Gxz[id]];

}

void cpu_ricker(float fm, float dt, double *wlt, int nts)
{
    int iss = 2;
    double t0 = 1.5*sqrt(6.0)/(PI*(double)fm);
    double da = PI*(double)fm;
    int i;
    double t, a, a2;
   
    for(i = 0;i < nts; i++)
    {
        t = (double)(i*dt);
        a = PI*(double)fm*(t-t0);
        a2= a*a;

        if(iss == 3)
        {
            wlt[i] = (a/da)*exp(-a2);
        }
        else if(iss == 2)
        {
            wlt[i] = (1.0f-2.0f*a2)*exp(-a2);
        }
        else
        {
            wlt[i] = -4*a*da*exp(-a2)-2.0f*a*da*(1.0f-2.0f*a2)*exp(-a2);
        }

    }

}

void cpu_stagger(float *vpe, float *rhoe, float *bue, float *bu2, float *bu1, double *kappa, int nnz, int nnx)
{
    int i1, i2, id;
    double vp2;

    for(i2 = 0; i2 < nnx; i2++)
    {
        for(i1 = 0; i1 < nnz; i1++)
        {
            id = i1 + i2*nnz;
            vp2 = (double)vpe[id] * (double)vpe[id];
            kappa[id] = vp2*(double)rhoe[id];
            if(rhoe[id] > 0.0f)
            {
                bue[id] = 1.0f/rhoe[id];
            }
            else
            {
                bue[id] = 10000000000.0f;
            }
        }
    }

    //bu2 (grid coincident with Tauzy)
    for(i2 = 0; i2 < nnx; i2++)
    {
        for(i1 = 0; i1 < nnz-1; i1++)
        {
            id = i1 + i2*nnz;
            bu1[id] = 0.5f*(bue[id+1] + bue[id]);
        }   
        
        bu1[i2*nnz + nnz-1]=bu2[i2*nnz + nnz-2];
    }

    //bu1 (grid coincident with Tauxy)
    for(i2 = 0; i2 < nnx-1; i2++)
    {
       
        for(i1 = 0; i1 < nnz; i1++)
        {
            id = i1 + i2*nnz;
            bu2[id] = 0.5f*(bue[id+nnz] + bue[id]);
        }
    }

    
    for(i1 = 0; i1 < nnz; i1++)
    {
        bu2[(nnx-1)*nnz + i1] = bu1[(nnx-2)*nnz + i1];
    }

}

void cpu_pml_coefficient_a(float fac, float *damp, float *damp1, float *spg, float *spg1, int npml, int n)
{
    int i;
    float x, temp;
    
    #pragma acc parallel loop present (spg, spg1, damp, damp1)
    {
        for(i = 0; i < npml; i++)
        {
            x = (float)(npml-(i+1));
            temp = (fac*x);
            spg[i] = exp(-(temp*temp));
            spg1[i] = exp(-0.5*(temp*temp));
        }

        for(i = 0; i < npml; i++)
        {
            damp[i] = spg[i];
            damp1[i] = spg1[i];
            damp[npml+n+i] = spg[npml-i-1];
            damp1[npml+n+i] = spg1[npml-i-1];
        }

        for(i = 0; i < n; i++)
        {
            damp[i+npml] = 1.0f;
            damp1[i+npml] = 1.0f;
        }
    }

}

void cpu_pml_coefficient_b(float fac,float *damp,float *damp1, float *spg, float *spg1, int npml, int n)
{
    int i;
    float x, temp;

    #pragma acc parallel loop present (spg, spg1, damp, damp1)
    {
        for(i = 0; i < npml-1; i++)
        {
            x = (float)(npml-1-(i+1)) + 0.5f;
            temp = (fac*x);
            spg[i] = exp(-(temp*temp));
            spg1[i] = exp(-0.5*(temp*temp));
        }

        for(i = 0; i < npml-1; i++)
        {
            damp[i] = spg[i];
            damp1[i] = spg1[i];
        }

        for(i = 0; i < n+1; i++)
        {
            damp[i+npml-1] = 1.0f;
            damp1[i+npml-1] = 1.0f;
        }

        for(i = 0; i < npml; i++)
        {
            x = (float)((i+1)-1)+0.5f;
            temp = (fac*x);
            spg[i] = exp(-(temp*temp));
            spg1[i] = exp(-0.5*(temp*temp));
        }
    
        for(i = 0; i < npml; i++)
        {
            damp[npml+n+i] = spg[i];
            damp1[npml+n+i] = spg1[i];
        }
    }

}

void fdoperator420(double *vx, double *der2, int nnz, int nnx)
{
    int i1, i2, id;
    int i21, i22;    

    #pragma acc parallel loop independent present (vx, der2) 
	for(i2 = 2; i2 < nnx-1; i2++)
	{
        #pragma acc loop independent 
        for(i1 = 0; i1 < nnz; i1++)
        {
            id = i1 + i2*nnz;
            der2[id] = a0*(vx[id]-vx[id-nnz]) + a1*(vx[id+nnz]-vx[id-2*nnz]);
        }
	}
    
    i21 = 1;        i22 = nnx-1;
    
    #pragma acc parallel loop independent present(vx, der2) 
    for(i1 = 0; i1 < nnz; i1++)
    {
        id = i1 + i21*nnz;
        der2[id] = vx[id] - vx[id-nnz];

        id = i1 + i22*nnz;
        der2[id] = vx[id] - vx[id-nnz];
    }

}

void fdoperator410(double *vz, double *der1, int nnz, int nnx)
{
    int i1, i2, id;
    int i11, i12;

    #pragma acc parallel loop independent gang present(vz, der1)
    for(i2 = 0; i2 < nnx; i2++)
    {
	    #pragma acc loop independent vector
        for(i1 = 2; i1 < nnz-1; i1++)
        {
            id = i1 + i2*nnz;
            der1[id] = a0*(vz[id]-vz[id-1]) + a1*(vz[id+1]-vz[id-2]);
        }
	}
	
    i11 = nnz-1;        i12 = 1;
    #pragma acc parallel loop independent present(vz, der1)
    for(i2 = 0; i2 < nnx; i2++)
    {
        id = i11 + i2*nnz;
        der1[id] = vz[id] - vz[id-1];

        id = i12 + i2*nnz;
        der1[id] = vz[id] - vz[id-1];
    }
}

void fdoperator421(double *p, double *der2, int nnz, int nnx)
{
    int i1, i2, id;
    int i21, i22;

    #pragma acc parallel loop independent gang present(der2, p)
    for(i2 = 1; i2 < nnx-2; i2++)
    {
	    #pragma acc loop independent vector
        for(i1 = 0; i1 < nnz; i1++)
        {
            id = i1 + i2*nnz;
            der2[id] = a0*(p[id+nnz]-p[id]) + a1*(p[id+2*nnz]-p[id-nnz]);
        }
	}

    i21 = 0;        i22 = nnx-2;
    #pragma acc parallel loop independent present(p, der2)
    for(i1 = 0; i1 < nnz; i1++)
    {
        id = i1 + i21*nnz;
        der2[id] = p[id+nnz] - p[id];

        id = i1 + i22*nnz;
        der2[id] = p[id+nnz] - p[id];
    }
}

void fdoperator411(double *p, double *der1, int nnz, int nnx)
{
    int i1, i2, id;
    int i11, i12;

    #pragma acc parallel loop independent gang present(der1, p)
    for(i2 = 0; i2 < nnx; i2++)
    {
        #pragma acc loop independent vector
        for(i1 = 1; i1 < nnz-2; i1++)
        {
            id = i1 + i2*nnz;
            der1[id] = a0*(p[id+1]-p[id]) + a1*(p[id+2]-p[id-1]);
        }
    }

    i11 = nnz-2;        i12 = 0;
    #pragma acc parallel loop independent present(der1, p)
    for(i2 = 0; i2 < nnx; i2++)
    {
        id = i11 + i2*nnz;
        der1[id] = p[id+1] - p[id];

        id = i12 + i2*nnz;
        der1[id] = p[id+1] - p[id];
    }
}

void compute_p(double *p, double *px, double *pz, int nnz, int nnx)
{
    int i2, i1, id;
        
    #pragma acc parallel loop independent gang present(p, px, pz)
    for(i2 = 0; i2 < nnx; i2++)
    {
	    #pragma acc loop independent vector
        for(i1 = 0; i1 < nnz; i1++)
        {
            id = i1 + i2*nnz;
            p[id] = px[id] + pz[id];
        }
	}
}

void cpu_pml_p(double *px, double *pz, double *kappaw2, double *kappaw1, double *der2, double *der1, float *damp2a, float *damp1a, int nnz, int nnx)
{
    int i2, i1, id;
    int temp = nnx;
    #pragma acc parallel loop independent gang present(px, pz, damp2a, damp1a, kappaw1, kappaw2, der1, der2)
    for(i2 = 0; i2 < temp; i2++)
    {
	    #pragma acc loop independent vector
        for(i1 = 0; i1 < nnz; i1++)
        {
            id = i1 + i2*nnz;
            px[id] = damp2a[i2]*px[id] + kappaw2[id]*der2[id];
            pz[id] = damp1a[i1]*pz[id] + kappaw1[id]*der1[id];
        }
	}
}

void cpu_pml_v(double *vx, double *vz, double *buw2, double *buw1, double *der2, double *der1, float *damp2b, float *damp1b, int nnz, int nnx)
{
    int i2, i1, id;

    #pragma acc parallel loop independent gang present(vx, vz, damp2b, damp1b, buw1, buw2, der1, der2)
    for(i2 = 0; i2 < nnx; i2++)
    {
	    #pragma acc loop independent vector
        for(i1 = 0; i1 < nnz; i1++)
        {
            id = i1 + i2*nnz;
            vx[id] = damp2b[i2]*vx[id] + buw2[id]*der2[id];
            vz[id] = damp1b[i1]*vz[id] + buw1[id]*der1[id];
        }
	}
}


void cpu_transpose(float *inp, float *out, int n1, int n2)
{
    int i1, i2, id1, id2;

    for(i2 = 0; i2< n2; i2++)
        for(i1 = 0; i1 < n1; i1++)
        {
            id1 = i1+i2*n1;         id2 = i2+i1*n2;
            out[id2] = inp[id1];
        }

}
