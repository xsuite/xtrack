// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_FIRSTORDERTAYLORMAP_H
#define XTRACK_FIRSTORDERTAYLORMAP_H

#include <headers/track.h>


GPUFUN
void FirstOrderTaylorMap_track_local_particle(FirstOrderTaylorMapData el, LocalParticle* part0){

    double const length = FirstOrderTaylorMapData_get_length(el);

    START_PER_PARTICLE_BLOCK(part0, part);
        double x0 = LocalParticle_get_x(part);
        double px0 = LocalParticle_get_px(part);
        double y0 = LocalParticle_get_y(part);
        double py0 = LocalParticle_get_py(part);
        double beta0 = LocalParticle_get_beta0(part);
        double tau0 = LocalParticle_get_zeta(part)/beta0;
        double ptau0 = LocalParticle_get_ptau(part);

        LocalParticle_set_x(part,FirstOrderTaylorMapData_get_m0(el,0) +
                            FirstOrderTaylorMapData_get_m1(el,0,0)*x0 +
                            FirstOrderTaylorMapData_get_m1(el,0,1)*px0 +
                            FirstOrderTaylorMapData_get_m1(el,0,2)*y0 +
                            FirstOrderTaylorMapData_get_m1(el,0,3)*py0 +
                            FirstOrderTaylorMapData_get_m1(el,0,4)*tau0 +
                            FirstOrderTaylorMapData_get_m1(el,0,5)*ptau0);
        LocalParticle_set_px(part,FirstOrderTaylorMapData_get_m0(el,1) +
                            FirstOrderTaylorMapData_get_m1(el,1,0)*x0 +
                            FirstOrderTaylorMapData_get_m1(el,1,1)*px0 +
                            FirstOrderTaylorMapData_get_m1(el,1,2)*y0 +
                            FirstOrderTaylorMapData_get_m1(el,1,3)*py0 +
                            FirstOrderTaylorMapData_get_m1(el,1,4)*tau0 +
                            FirstOrderTaylorMapData_get_m1(el,1,5)*ptau0);
        LocalParticle_set_y(part,FirstOrderTaylorMapData_get_m0(el,2) +
                            FirstOrderTaylorMapData_get_m1(el,2,0)*x0 +
                            FirstOrderTaylorMapData_get_m1(el,2,1)*px0 +
                            FirstOrderTaylorMapData_get_m1(el,2,2)*y0 +
                            FirstOrderTaylorMapData_get_m1(el,2,3)*py0 +
                            FirstOrderTaylorMapData_get_m1(el,2,4)*tau0 +
                            FirstOrderTaylorMapData_get_m1(el,2,5)*ptau0);
        LocalParticle_set_py(part,FirstOrderTaylorMapData_get_m0(el,3) +
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
                            FirstOrderTaylorMapData_get_m1(el,4,5)*ptau0;
        double ptau = FirstOrderTaylorMapData_get_m0(el,5) +
                            FirstOrderTaylorMapData_get_m1(el,5,0)*x0 +
                            FirstOrderTaylorMapData_get_m1(el,5,1)*px0 +
                            FirstOrderTaylorMapData_get_m1(el,5,2)*y0 +
                            FirstOrderTaylorMapData_get_m1(el,5,3)*py0 +
                            FirstOrderTaylorMapData_get_m1(el,5,4)*tau0 +
                            FirstOrderTaylorMapData_get_m1(el,5,5)*ptau0;

        LocalParticle_update_ptau(part, ptau);
        LocalParticle_set_zeta(part,tau*beta0);
        LocalParticle_add_to_s(part, length);
    END_PER_PARTICLE_BLOCK;
}

#endif
