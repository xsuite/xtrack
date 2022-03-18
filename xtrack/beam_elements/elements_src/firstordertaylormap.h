#ifndef XTRACK_FIRSTORDERTAYLORMAP_H
#define XTRACK_FIRSTORDERTAYLORMAP_H

#include <stdio.h>

/*gpufun*/
void FirstOrderTaylorMap_track_local_particle(FirstOrderTaylorMapData el, LocalParticle* part0){

    int64_t radiation_flag = FirstOrderTaylorMapData_get_radiation_flag(el);

    for(unsigned int i = 0; i<6;++i){
        printf("m0,%i,%E\n",i,FirstOrderTaylorMapData_get_m0(el,i));
    }
    for(unsigned int i = 0; i<6;++i){
        for(unsigned int j = 0;j<6;++j){
            printf("m1,%i/%i,%E\n",i,j,FirstOrderTaylorMapData_get_m1(el));
        }
    }

    //start_per_particle_block (part0->part)

        double x0 = LocalParticle_get_x(part);
        double px0 = LocalParticle_get_px(part);
        double y0 = LocalParticle_get_y(part);
        double py0 = LocalParticle_get_py(part);
        double tau0 = LocalParticle_get_zeta(part)/LocalParticle_get_beta(part);
        double psigma0 = LocalParticle_get_psigma(part);
        double beta0 = LocalParticle_get_beta0(part);
        double p0c = LocalParticle_get_p0c(part);
        double ptau0 = psigma * beta0;


        LocalParticle_set_x(FirstOrderTaylorMapData_get_m0(el,0) +
                            FirstOrderTaylorMapData_get_m1(el,0,0)*x0 +
                            FirstOrderTaylorMapData_get_m1(el,0,1)*px0 +
                            FirstOrderTaylorMapData_get_m1(el,0,2)*y0 +
                            FirstOrderTaylorMapData_get_m1(el,0,3)*py0 +
                            FirstOrderTaylorMapData_get_m1(el,0,4)*tau0 +
                            FirstOrderTaylorMapData_get_m1(el,0,5)*ptau0);
        LocalParticle_set_px(FirstOrderTaylorMapData_get_m0(el,1) +
                            FirstOrderTaylorMapData_get_m1(el,1,0)*x0 +
                            FirstOrderTaylorMapData_get_m1(el,1,1)*px0 +
                            FirstOrderTaylorMapData_get_m1(el,1,2)*y0 +
                            FirstOrderTaylorMapData_get_m1(el,1,3)*py0 +
                            FirstOrderTaylorMapData_get_m1(el,1,4)*tau0 +
                            FirstOrderTaylorMapData_get_m1(el,1,5)*ptau0);
        LocalParticle_set_y(FirstOrderTaylorMapData_get_m0(el,2) +
                            FirstOrderTaylorMapData_get_m1(el,2,0)*x0 +
                            FirstOrderTaylorMapData_get_m1(el,2,1)*px0 +
                            FirstOrderTaylorMapData_get_m1(el,2,2)*y0 +
                            FirstOrderTaylorMapData_get_m1(el,2,3)*py0 +
                            FirstOrderTaylorMapData_get_m1(el,2,4)*tau0 +
                            FirstOrderTaylorMapData_get_m1(el,2,5)*ptau0);
        LocalParticle_set_py(FirstOrderTaylorMapData_get_m0(el,3) +
                            FirstOrderTaylorMapData_get_m1(el,3,0)*x0 +
                            FirstOrderTaylorMapData_get_m1(el,3,1)*px0 +
                            FirstOrderTaylorMapData_get_m1(el,3,2)*y0 +
                            FirstOrderTaylorMapData_get_m1(el,3,3)*py0 +
                            FirstOrderTaylorMapData_get_m1(el,3,4)*tau0 +
                            FirstOrderTaylorMapData_get_m1(el,3,5)*ptau0);
        double tau = FirstOrderTaylorMapData_get_m0(el,4) +
                            FirstOrderTaylorMapData_get_m1(el,4,0)*x0 +
                            FirstOrderTaylorMapData_get_m1(el,4,1)*px0 +
                            FirstOrderTaylorMapData_get_m1(el,4,2)*y0 +
                            FirstOrderTaylorMapData_get_m1(el,4,3)*py0 +
                            FirstOrderTaylorMapData_get_m1(el,4,4)*tau0 +
                            FirstOrderTaylorMapData_get_m1(el,4,5)*ptau0);
        LocalParticle_set_zeta(tau*LocalParticle_get_beta(part));
        double ptau = FirstOrderTaylorMapData_get_m0(el,5) +
                            FirstOrderTaylorMapData_get_m1(el,5,0)*x0 +
                            FirstOrderTaylorMapData_get_m1(el,5,1)*px0 +
                            FirstOrderTaylorMapData_get_m1(el,5,2)*y0 +
                            FirstOrderTaylorMapData_get_m1(el,5,3)*py0 +
                            FirstOrderTaylorMapData_get_m1(el,5,4)*tau0 +
                            FirstOrderTaylorMapData_get_m1(el,5,5)*ptau0);
        LocalParticle_update_delta(part,sqrt(ptau*ptau + 2.0*ptau/beta0+1.0)-1.0);

    //end_per_particle_block
}

#endif
