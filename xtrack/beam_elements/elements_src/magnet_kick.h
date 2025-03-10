// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_MAGNET_KICK_H
#define XTRACK_MAGNET_KICK_H

/*gpufun*/
void MagnetKick_track_local_particle(
    MagnetKickData el,
    LocalParticle* part0
) {
    const double length = MagnetKickData_get_length(el);
    const int64_t order = MagnetKickData_get_order(el);
    const double inv_factorial_order = MagnetKickData_get_inv_factorial_order(el);
    const double* knl = MagnetKickData_getp1_knl(el, 0);
    const double* ksl = MagnetKickData_getp1_ksl(el, 0);
    const double factor_knl_ksl = MagnetKickData_get_factor_knl_ksl(el);
    const double kick_weight = MagnetKickData_get_kick_weight(el);
    const double k0 = MagnetKickData_get_k0(el);
    const double k1 = MagnetKickData_get_k1(el);
    const double k2 = MagnetKickData_get_k2(el);
    const double k3 = MagnetKickData_get_k3(el);
    const double k0s = MagnetKickData_get_k0s(el);
    const double k1s = MagnetKickData_get_k1s(el);
    const double k2s = MagnetKickData_get_k2s(el);
    const double k3s = MagnetKickData_get_k3s(el);
    const double h = MagnetKickData_get_h(el);

    //start_per_particle_block (part0->part)
        track_magnet_kick_single_particle(
            part, length, order, inv_factorial_order, knl, ksl, factor_knl_ksl,
            kick_weight, k0, k1, k2, k3, k0s, k1s, k2s, k3s, h
        );
    //end_per_particle_block
}

#endif