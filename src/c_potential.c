#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "include.h"

#define GCGS 6.674e-8
#define KPCTOCM 3.08e21
#define CODETOGRAMS 2e43

#define ALLTOGETHER 1.407e-7 // GCGS*CODETOGRAMS/(KPCTOCM*KPCTOCM)


#define GGAL 4.302e-6 // kpc (km/s)^2 / Msun

void printArray(float * arr,int Narr){
    for (int i=0; i< Narr; i++){
        printf("%.2f\t",arr[i]);
    }
    printf("\n");
}

float calculateGravityZ(float * point, float * masses, float * xs, float * ys, float * zs,int Narr){
    float summ = 0;
    for (int i =0; i<Narr; i++){
        float dx = xs[i]-point[0];
        float dy = ys[i]-point[1];
        float dz = zs[i]-point[2];

        float dr = sqrt(dx*dx + dy*dy + dz*dz); // RIP speed
        summ += masses[i]/(dr*dr*dr)*dz; 
    }
    return ALLTOGETHER*summ;//cgs
}

int calculateZGravityAtLocations(
    int Narr,
    float * xs, float * ys, float * zs,
    float * masses,
    int Ntest,
    float * test_xs, float * test_ys, float * test_zs,
    float * H_OUT ){

    float point[3];
    for (int i=0; i<Ntest; i++){

        point[0]=test_xs[i]; point[1]=test_ys[i]; point[2]=test_zs[i];

        H_OUT[i]=calculateGravityZ(point,masses,xs,ys,zs,Narr);

    }

    return 1;
}

float calculateGravityR(float * point, float * masses, float * xs, float * ys, float * zs,int Narr){
    float summ = 0;
    for (int i =0; i<Narr; i++){
        float dx = xs[i]-point[0];
        float dy = ys[i]-point[1];
        float dz = zs[i]-point[2];

        float dr2 = dx*dx + dy*dy + dz*dz;
        summ += masses[i]/dr2; 
    }
    return ALLTOGETHER*summ;//cgs
}

int calculateRGravityAtLocations(
    int Narr,
    float * xs, float * ys, float * zs,
    float * masses,
    int Ntest,
    float * test_xs, float * test_ys, float * test_zs,
    float * H_OUT ){

    float point[3];
    for (int i=0; i<Ntest; i++){

        point[0]=test_xs[i]; point[1]=test_ys[i]; point[2]=test_zs[i];

        H_OUT[i]=calculateGravityR(point,masses,xs,ys,zs,Narr);

    }

    return 1;
}

float calculatePotential(float * point, float * masses, float * xs, float * ys, float * zs,int Narr){
    float summ = 0;
    //int nskip = 0;
    for (int i =0; i<Narr; i++){
        float dx = xs[i]-point[0];
        float dy = ys[i]-point[1];
        float dz = zs[i]-point[2];

        float dr2 = dx*dx + dy*dy + dz*dz;
        float dr = sqrt(dr2);

        if(dr>1e-14) {  //skip the particle itself

            summ += masses[i]/dr; 

        } 
        //else {
        //    nskip+=1;
        //    fprintf(stderr, "skipped dr = %.3g\n", dr);
        //}


    }
    //printf("skipped %d particle(s)\n", nskip);
    //printf("%.3g\n", summ);
    return -1.0*GGAL*summ;// in (km/s)^2
}

int calculatePotentialAtLocations(
    int Narr,
    float * xs, float * ys, float * zs,
    float * masses,
    int Ntest,
    float * test_xs, float * test_ys, float * test_zs,
    float * H_OUT ){

    float point[3];
    for (int i=0; i<Ntest; i++){

        point[0]=test_xs[i]; point[1]=test_ys[i]; point[2]=test_zs[i];

        //printArray(point, 3);

        H_OUT[i]=calculatePotential(point,masses,xs,ys,zs,Narr);

    }

    return 1;
}
