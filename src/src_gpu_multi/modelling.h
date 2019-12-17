#ifndef MODELLING_H_
#define MODELLING_H_

#include <iostream>
#include <map>
#include <string>

#define min2(x,z) ( (x)<(z) ? (x) : (z))
#define max2(x,z) ( (x)>(z) ? (x) : (z))

using namespace std;

typedef struct ModelInfo
{
    int nx, nz;                         // Grid dimension in x,and z direction
    float dx, dz;                       // Grid size in x,and z direction
    char velfile[224];                  // Name of velocity file
    char densfile[224];                 // Name of density file
 
    float **vel2d, **rho2d;             // 2D array to store velocity and density model 
    float vmax, vmin;

}mod_t;

extern mod_t *mod_sp;

typedef struct Wavelet
{
    float rec_len, fd_dt_sec, real_dt_sec;      // Time and sampling interval of data
    float dom_freq;                 // Dominant Frequency of source wavelet
}wave_t;

extern wave_t *wave_sp;

typedef struct Coord2d
{
    float x, z;
}crd2d_t;

typedef struct Geo2d
{
    int nsrc, *nrec;                // Number of source and number of receiver for each source
    crd2d_t *src2d_sp;              // Source coordinate
    crd2d_t **rec2d_sp;             // nrec recievers for each source
}geo2d_t;

extern geo2d_t *geo2d_sp;

typedef struct Job
{
    int adj, geomflag, fdop, frsf;
    char jbtype[8], shotfile[224], geomfile[224], logfile[224];    
    char jobname[256], tmppath[512];
}job_t;

typedef struct Workload
{
	int start, end, myNsrc;
}wrkld_t;

extern job_t *job_sp;

void Modelling_master(char *);

void Modelling_worker(char *);

void set_job_name(char *job);

// Read variable name and value from json file and insert in map<string, string>
int read_json_objects(char *job, map<string, string> &map_json);

// Setting value of parameter as per description in json from map<string, string>
void set_json_object(map<string, string> map_json);

// Geometry
void read_recv2d(char *recv_file);
void read_source2d(char *shot_file);
void set_int(string token, int  &var, int src_indx, char* err_msg);
void set_float(string token, float  &var, int src_indx, char* err_msg);

int count_src(char *recv_file, map<int, int> &map_srccount);
void check_geom_para(float sx, float sz, float ngeoph);
void check_geom_coord(float sx, float sz, float gx, float gz);
void set_receiver2d(float sx, float sz, int nrec, float nearoff, int geotype, float ngeoph, int srcIndx);
void set_geom2d_split(float sx, float sz, int nrec, float nearoff, float ngeoph, int srcIndx);
void set_geom2d_endonleft(float sx, float sz, int nrec, float nearoff, float ngeoph, int srcIndx);
void set_geom2d_endonright(float sx, float sz, int nrec, float nearoff, float ngeoph, int srcIndx);

// Workload
void calculate_workload(int nsrc, wrkld_t* wrkld_sp);
void send_workload(wrkld_t* wrkld_sp);
void receive_workload(wrkld_t* wrkld_sp, geo2d_t *mygeo2d_sp);
void master_workload(wrkld_t* wrkld_sp, geo2d_t *mygeo2d_sp);
void print_workload(wrkld_t* wrkld_sp, geo2d_t *mygeo2d_sp);

// Model
void read2d(char *file, float **arr, int nx, int nz);
void minmax2d(float **arr, int nx, int nz, float &vmin, float &vmax);

// Stability and Dispersion check
void check_stability();
void check_stability_2();

void modelling_module(int my_nsrc, geo2d_t *mygeo2d_sp);

#endif
