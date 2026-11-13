// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_TRACK_DRIFT_H
#define XTRACK_TRACK_DRIFT_H


GPUFUN
void Drift_single_particle_expanded(LocalParticle* part, double length){
    xt_float_or_tpsa const rpp    = LocalParticle_get_rpp(part);
    xt_float_or_tpsa const rv0v    = 1./LocalParticle_get_rvv(part);
    xt_float_or_tpsa const xp     = LocalParticle_get_px(part) * rpp;
    xt_float_or_tpsa const yp     = LocalParticle_get_py(part) * rpp;
    xt_float_or_tpsa const dzeta  = 1 - rv0v * ( 1. + ( xp*xp + yp*yp ) / 2. );

    LocalParticle_add_to_x(part, xp * length );
    LocalParticle_add_to_y(part, yp * length );
    LocalParticle_add_to_s(part, length);
    LocalParticle_add_to_zeta(part, length * dzeta );
}


GPUFUN
void Drift_single_particle_exact(LocalParticle* part, double length){
    xt_float_or_tpsa const px = LocalParticle_get_px(part);
    xt_float_or_tpsa const py = LocalParticle_get_py(part);
    xt_float_or_tpsa const rv0v    = 1./LocalParticle_get_rvv(part);
    xt_float_or_tpsa const one_plus_delta = 1. + LocalParticle_get_delta(part);

    xt_float_or_tpsa const one_over_pz = 1./sqrt(one_plus_delta*one_plus_delta
                                       - px * px - py * py);
    xt_float_or_tpsa const dzeta = 1 - rv0v * one_plus_delta * one_over_pz;

    LocalParticle_add_to_x(part, px * one_over_pz * length);
    LocalParticle_add_to_y(part, py * one_over_pz * length);
    LocalParticle_add_to_zeta(part, dzeta * length);
    LocalParticle_add_to_s(part, length);
}


GPUFUN
void Drift_single_particle(LocalParticle* part, double length){
    #ifndef XTRACK_USE_EXACT_DRIFTS
        Drift_single_particle_expanded(part, length);
    #else
        Drift_single_particle_exact(part, length);
    #endif
}


#endif /* XTRACK_TRACK_DRIFT_H */
