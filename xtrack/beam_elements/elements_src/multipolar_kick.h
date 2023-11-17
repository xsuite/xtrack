// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_MULTIPOLAR_KICK_H
#define XTRACK_MULTIPOLAR_KICK_H

/*gpufun*/
void multipolar_kick(
    LocalParticle* part,
    const int order,
    const double inv_factorial_order,
    /*gpuglmem*/ const double *knl,
    /*gpuglmem*/ const double *ksl,
    const double weight
) {

    #ifdef XSUITE_BACKTRACK
        LocalParticle_kill_particle(part, -31);
        return;
    #else

    int64_t index = order;
    double inv_factorial = inv_factorial_order;

    double dpx = knl[index] * inv_factorial;
    double dpy = ksl[index] * inv_factorial;

    double const x   = LocalParticle_get_x(part);
    double const y   = LocalParticle_get_y(part);
    double const chi = LocalParticle_get_chi(part);

    while( index > 0 )
    {
        double const zre = dpx * x - dpy * y;
        double const zim = dpx * y + dpy * x;

        inv_factorial *= index;
        index -= 1;

        dpx = knl[index] * inv_factorial + zre;
        dpy = ksl[index] * inv_factorial + zim;
    }

    dpx = -chi * dpx; // rad
    dpy =  chi * dpy; // rad

    LocalParticle_add_to_px(part, weight * dpx);
    LocalParticle_add_to_py(part, weight * dpy);

    #endif // not XSUITE_BACKTRACK
}

#endif // XTRACK_MULTIPOLAR_KICK_H
