#ifndef XTRACK_FIRSTORDERTAYLORMAP_H
#define XTRACK_FIRSTORDERTAYLORMAP_H

#include <stdio.h>

/*gpufun*/
void FirstOrderTaylorMap_track_local_particle(FirstOrderTaylorMapData el, LocalParticle* part0){

    int64_t const radiation_flag = FirstOrderTaylorMapData_get_radiation_flag(el);
    double const length = FirstOrderTaylorMapData_get_length(el); // m

    //for(unsigned int i=0;i<6;i++){
    //     printf("%i:%E\n",i,FirstOrderTaylorMapData_get_m0(el,i));
    //}

    //for(unsigned int i=0;i<6;i++){
    //    for(unsigned int j=0;j<6;j++){
    //        printf("%i,%i:%E\n",i,j,FirstOrderTaylorMapData_get_m1(el,i*6+j));
    //    }
    //}

    //start_per_particle_block (part0->part)

        double x0 = LocalParticle_get_x(part);
        double px0 = LocalParticle_get_px(part);
        double y0 = LocalParticle_get_y(part);
        double py0 = LocalParticle_get_py(part);
        double beta0 = LocalParticle_get_beta0(part);
        double beta = LocalParticle_get_rvv(part)*beta0;
        double tau0 = LocalParticle_get_zeta(part)/beta;
        double psigma0 = LocalParticle_get_psigma(part);
        double ptau0 = psigma0 * beta0;
        
        LocalParticle_set_x(part,FirstOrderTaylorMapData_get_m0(el,0) +
                            FirstOrderTaylorMapData_get_m1(el,0)*x0 +
                            FirstOrderTaylorMapData_get_m1(el,1)*px0 +
                            FirstOrderTaylorMapData_get_m1(el,2)*y0 +
                            FirstOrderTaylorMapData_get_m1(el,3)*py0 +
                            FirstOrderTaylorMapData_get_m1(el,4)*tau0 +
                            FirstOrderTaylorMapData_get_m1(el,5)*ptau0);
        LocalParticle_set_px(part,FirstOrderTaylorMapData_get_m0(el,1) +
                            FirstOrderTaylorMapData_get_m1(el,6)*x0 +
                            FirstOrderTaylorMapData_get_m1(el,7)*px0 +
                            FirstOrderTaylorMapData_get_m1(el,8)*y0 +
                            FirstOrderTaylorMapData_get_m1(el,9)*py0 +
                            FirstOrderTaylorMapData_get_m1(el,10)*tau0 +
                            FirstOrderTaylorMapData_get_m1(el,11)*ptau0);
        LocalParticle_set_y(part,FirstOrderTaylorMapData_get_m0(el,2) +
                            FirstOrderTaylorMapData_get_m1(el,12)*x0 +
                            FirstOrderTaylorMapData_get_m1(el,13)*px0 +
                            FirstOrderTaylorMapData_get_m1(el,14)*y0 +
                            FirstOrderTaylorMapData_get_m1(el,15)*py0 +
                            FirstOrderTaylorMapData_get_m1(el,16)*tau0 +
                            FirstOrderTaylorMapData_get_m1(el,17)*ptau0);
        LocalParticle_set_py(part,FirstOrderTaylorMapData_get_m0(el,3) +
                            FirstOrderTaylorMapData_get_m1(el,18)*x0 +
                            FirstOrderTaylorMapData_get_m1(el,19)*px0 +
                            FirstOrderTaylorMapData_get_m1(el,20)*y0 +
                            FirstOrderTaylorMapData_get_m1(el,21)*py0 +
                            FirstOrderTaylorMapData_get_m1(el,22)*tau0 +
                            FirstOrderTaylorMapData_get_m1(el,23)*ptau0);
        double tau = FirstOrderTaylorMapData_get_m0(el,4) +
                            FirstOrderTaylorMapData_get_m1(el,24)*x0 +
                            FirstOrderTaylorMapData_get_m1(el,25)*px0 +
                            FirstOrderTaylorMapData_get_m1(el,26)*y0 +
                            FirstOrderTaylorMapData_get_m1(el,27)*py0 +
                            FirstOrderTaylorMapData_get_m1(el,28)*tau0 +
                            FirstOrderTaylorMapData_get_m1(el,29)*ptau0;
        double ptau = FirstOrderTaylorMapData_get_m0(el,5) +
                            FirstOrderTaylorMapData_get_m1(el,30)*x0 +
                            FirstOrderTaylorMapData_get_m1(el,31)*px0 +
                            FirstOrderTaylorMapData_get_m1(el,32)*y0 +
                            FirstOrderTaylorMapData_get_m1(el,33)*py0 +
                            FirstOrderTaylorMapData_get_m1(el,34)*tau0 +
                            FirstOrderTaylorMapData_get_m1(el,35)*ptau0;

        LocalParticle_update_delta(part,sqrt(ptau*ptau + 2.0*ptau/beta0+1.0)-1.0);
        beta = LocalParticle_get_rvv(part)*beta0;
        LocalParticle_set_zeta(part,tau*beta);

        // Radiation
        if (radiation_flag > 0 && length > 0){
            double dpx = LocalParticle_get_px(part)-px0;
            double dpy = LocalParticle_get_py(part)-py0;
            double curv = sqrt(dpx*dpx+dpy*dpy);
            if (radiation_flag == 1){
                synrad_average_kick(part, curv, length);
            }
            else if (radiation_flag == 2){
                synrad_emit_photons(part, curv, length);
            }
        }

    //end_per_particle_block
}

#endif
