// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_SEXTUPOLE_H
#define XTRACK_SEXTUPOLE_H

/*gpufun*/
void Sextupole_track_local_particle(
        SextupoleData el,
        LocalParticle* part0
) {
    double length = SextupoleData_get_length(el);

    #ifdef XSUITE_BACKTRACK
        length = -length;
    #endif

    double const k2 = SextupoleData_get_k2(el);
    double const k2s = SextupoleData_get_k2s(el);

    double const knl2 = k2 * length;
    double const ksl2 = k2s * length;

    //start_per_particle_block (part0->part)

        // Drift
        Drift_single_particle(part, length / 2.);

        // Sextupolar kick
        int64_t index = 2;
        double const inv_factorial = 0.5; // 1 / factorial(2)
        double dpx = knl2 * inv_factorial;
        double dpy = ksl2 * inv_factorial;

        double const x   = LocalParticle_get_x(part);
        double const y   = LocalParticle_get_y(part);
        double const chi = LocalParticle_get_chi(part);

        while( index > 0 )
        {
            double const zre = dpx * x - dpy * y;
            double const zim = dpx * y + dpy * x;

            index -= 1;

            dpx = zre;
            dpy = zim;
        }

        dpx = -chi * dpx; // rad
        dpy =  chi * dpy; // rad

        LocalParticle_add_to_px(part, dpx);
        LocalParticle_add_to_py(part, dpy);

        // Drift
        Drift_single_particle(part, length / 2.);


    //end_per_particle_block


}

#endif // XTRACK_SEXTUPOLE_H