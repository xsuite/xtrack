// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_OCTUPOLE_H
#define XTRACK_OCTUPOLE_H

/*gpufun*/
void Octupole_track_local_particle(
        OctupoleData el,
        LocalParticle* part0
) {
    double length = OctupoleData_get_length(el);

    #ifdef XSUITE_BACKTRACK
        length = -length;
    #endif

    double const k3 = OctupoleData_get_k3(el);
    double const k3s = OctupoleData_get_k3s(el);

    double const knl3 = k3 * length;
    double const ksl3 = k3s * length;

    //start_per_particle_block (part0->part)

        // Drift
        Drift_single_particle(part, length / 2.);

        int64_t index = 3;
        double const inv_factorial = 1/6.; // 1 / factorial(3)
        double dpx = knl3 * inv_factorial;
        double dpy = ksl3 * inv_factorial;

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

#endif // XTRACK_OCTUPOLE_H