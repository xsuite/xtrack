// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_MAGNET_H
#define XTRACK_MAGNET_H

/*gpufun*/
void Magnet_track_local_particle(
    MagnetData el,
    LocalParticle* part0
) {
    const double length = MagnetData_get_length(el);
    const int64_t order = MagnetData_get_order(el);
    const double inv_factorial_order = MagnetData_get_inv_factorial_order(el);
    /*gpuglmem*/ const double* knl = MagnetData_getp1_knl(el, 0);
    /*gpuglmem*/ const double* ksl = MagnetData_getp1_ksl(el, 0);
    const int64_t num_multipole_kicks = MagnetData_get_num_multipole_kicks(el);
    const double h = MagnetData_get_h(el);
    const double k0 = MagnetData_get_k0(el);
    const double k1 = MagnetData_get_k1(el);
    const double k2 = MagnetData_get_k2(el);
    const double k3 = MagnetData_get_k3(el);
    const double k0s = MagnetData_get_k0s(el);
    const double k1s = MagnetData_get_k1s(el);
    const double k2s = MagnetData_get_k2s(el);
    const double k3s = MagnetData_get_k3s(el);
    const int64_t model = MagnetData_get_model(el);

    const double factor_knl_ksl = 1.0;

    track_magnet_body_particles(
        part0,
        length,
        order,
        inv_factorial_order,
        knl,
        ksl,
        factor_knl_ksl,
        num_multipole_kicks,
        model,
        h,
        k0,
        k1,
        k2,
        k3,
        k0s,
        k1s,
        k2s,
        k3s
    );
}

#endif