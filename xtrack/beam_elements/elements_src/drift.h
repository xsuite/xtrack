// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_DRIFT_H
#define XTRACK_DRIFT_H

/*gpufun*/
void Drift_track_local_particle(DriftData el, LocalParticle* part0){

    double const length = DriftData_get_length(el);

    //start_per_particle_block (part0->part)

        double const rpp    = LocalParticle_get_rpp(part);
        double const rv0v    = 1./LocalParticle_get_rvv(part);
        double const xp     = LocalParticle_get_px(part) * rpp;
        double const yp     = LocalParticle_get_py(part) * rpp;
        double const dzeta  = 1 - rv0v * ( 1. + ( xp*xp + yp*yp ) / 2. );

        LocalParticle_add_to_x(part, xp * length );
        LocalParticle_add_to_y(part, yp * length );
        LocalParticle_add_to_s(part, length);
        LocalParticle_add_to_zeta(part, length * dzeta );
    //end_per_particle_block

}

#endif
