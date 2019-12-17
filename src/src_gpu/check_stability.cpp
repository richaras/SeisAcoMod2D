#include <mpi.h>
#include <iostream>
#include <cmath>
#include "modelling.h"

using namespace std;

void check_stability()
{

    int i, j, stability;
    float criteria, limit;

    int order;    
    float vel_min, vel_max, fm, dx, dz, dt;
   

    vel_min  = mod_sp->vmin;
    vel_max  = mod_sp->vmax;
    order = job_sp->fdop;
    fm = 2.0f * wave_sp->dom_freq;
    dx = mod_sp->dx;      dz = mod_sp->dz;
    dt = wave_sp->fd_dt_sec;
    
    float hmin, hmax;
    hmin = min2(dx, dz);        hmax = max2(dx, dz);

    //Stability check
    // dt < (0.606*hmin/vmax)  - 0.606 for 4th order, 1/sqrt(2) for 2nd order
    criteria = (0.606f * hmin)/vel_max;

    if(dt > criteria)
    {
        cout<<"\n****************************************";
        cout<<"\n*                                      *";
        cout<<"\n*     STABILITY CONDITION VIOLATED     *";
        cout<<"\n*                                      *";
        cout<<"\n****************************************";

        cout<<"\n Current Time Step(dt)     : "<<dt<<" sec ";
        cout<<"\n Required Time Step(dt)    : "<<criteria<<" sec ";
        MPI::COMM_WORLD.Abort(-5);        
    }
    else
    {
        cout<<"\n Current Time Step(dt)     : "<<dt<<" sec ";
        cout<<"\n Required Time Step(dt)    : "<<criteria<<" sec ";
        cout<<"\n     STABILITY CONDITION FULFILLED      \n";
    }

    // Dispersion check  -  maxfreq < vmin/(hmax*5)
    criteria = vel_min / (hmax*5.0f);

    if(fm > criteria)
    {
        cout<<"\n*************************************************";
        cout<<"\n*                                               *";
        cout<<"\n*   Non-dispersion relation not satisfied!      *";
        cout<<"\n*                                               *";
        cout<<"\n*************************************************";
        cout<<"\n Current peak frequency     : "<<wave_sp->dom_freq;
        cout<<"\n Required peak frequency    : "<<criteria/2.0f;
        MPI::COMM_WORLD.Abort(-5);
    }
    else
    {
        cout<<"\n Current peak frequency     : "<<wave_sp->dom_freq;
        cout<<"\n Required peak frequency    : "<<criteria/2.0f;
        cout<<"\n     GRID SPACING CRITERIA (DISPERSION) FULFILLED \n";
    }
    
}// End of stability_check function
