#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "include.h"

#define GCGS 6.674e-8
#define KPCTOCM 3.086e21
#define CODETOGRAMS 2e43

#define ALLTOGETHER 1.407e-7 // GCGS*CODETOGRAMS/(KPCTOCM*KPCTOCM)

#define GGAL 4.302e-6 // kpc (km/s)^2 / Msun

float multiCalculateGravityZ(float * point, float * masses, float * xs, float * ys, float * zs,int Narr,int nthreads){
    float total_grav = 0;
    int i = 0;
    #pragma omp parallel for num_threads(nthreads) shared(xs,ys,zs,point,masses) reduction(+:total_grav) private(i)
    for (i=0;i<Narr; i++){
        float dx = xs[i]-point[0];
        float dy = ys[i]-point[1];
        float dz = zs[i]-point[2];

        float dr = sqrt(dx*dx + dy*dy + dz*dz); // RIP speed
        total_grav += masses[i]/(dr*dr*dr)*dz; 
    }
    return ALLTOGETHER*total_grav;//cgs
}

int multiCalculateZGravityAtLocations(
    int nthreads, 
    int Narr,
    float * xs, float * ys, float * zs,
    float * masses,
    int Ntest,
    float * test_xs, float * test_ys, float * test_zs,
    float * H_OUT ){

    float point[3];
    int i = 0;
    //#pragma omp parallel private(i) num_threads(2)
    for (int i=0; i<Ntest; i++){

        point[0]=test_xs[i]; point[1]=test_ys[i]; point[2]=test_zs[i];

        H_OUT[i]=multiCalculateGravityZ(point,masses,xs,ys,zs,Narr,nthreads);

    }

    return 1;
}

float multiCalculateGravityR(float * point, float * masses, float * xs, float * ys, float * zs,int Narr,int nthreads){
    float total_grav = 0;
    int i = 0;
    #pragma omp parallel for num_threads(nthreads) shared(xs,ys,zs,point,masses) reduction(+:total_grav) private(i)
    for (i=0;i<Narr; i++){
        float dx = xs[i]-point[0];
        float dy = ys[i]-point[1];
        float dz = zs[i]-point[2];

        float dr2 = dx*dx + dy*dy + dz*dz;
        total_grav += masses[i]/dr2; 
    }
    return ALLTOGETHER*total_grav;//cgs
}

float multiCalculatePotential(float * point, float * masses, float * xs, float * ys, float * zs,int Narr,int nthreads){
    float total_pot = 0;
    int i = 0;
    #pragma omp parallel for num_threads(nthreads) shared(xs,ys,zs,point,masses) reduction(+:total_pot) private(i)
    for (i=0;i<Narr; i++){
        float dx = xs[i]-point[0];
        float dy = ys[i]-point[1];
        float dz = zs[i]-point[2];

        float dr2 = dx*dx + dy*dy + dz*dz;
        float dr = sqrt(dr2);

        if(dr>1e-14) {  //skip the particle itself

            total_pot += masses[i]/dr; 

        } 
         
    }
    return -1.0*GGAL*total_pot; //in (km/s)^2
}

int multiCalculateRGravityAtLocations(
    int nthreads, 
    int Narr,
    float * xs, float * ys, float * zs,
    float * masses,
    int Ntest,
    float * test_xs, float * test_ys, float * test_zs,
    float * H_OUT ){

    float point[3];
    int i = 0;
    //#pragma omp parallel private(i) num_threads(2)
    for (int i=0; i<Ntest; i++){

        point[0]=test_xs[i]; point[1]=test_ys[i]; point[2]=test_zs[i];

        H_OUT[i]=multiCalculateGravityR(point,masses,xs,ys,zs,Narr,nthreads);

    }

    return 1;
}

int multiCalculatePotentialAtLocations(
    int nthreads, 
    int Narr,
    float * xs, float * ys, float * zs,
    float * masses,
    int Ntest,
    float * test_xs, float * test_ys, float * test_zs,
    float * H_OUT ){

    float point[3];
    int i = 0;
    //#pragma omp parallel private(i) num_threads(2)
    for (int i=0; i<Ntest; i++){

        point[0]=test_xs[i]; point[1]=test_ys[i]; point[2]=test_zs[i];

        H_OUT[i]=multiCalculatePotential(point,masses,xs,ys,zs,Narr,nthreads);

    }

    return 1;
}

