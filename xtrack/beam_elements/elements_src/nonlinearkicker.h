// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //
#ifndef XTRACK_NONLINEARKICKER_H
#define XTRACK_NONLINEARKICKER_H

#include <headers/track.h>


GPUFUN
void NonlinearKicker_track_local_particle(NonlinearKickerData el, LocalParticle* part0){


    // Data from nonlinear kicker
    double const L_phy = NonlinearKickerData_get_L_phy(el);
    double const L_int = NonlinearKickerData_get_L_int(el); 

    // Get the number of wires defined in the arrays
    int64_t const n_wires = NonlinearKickerData_len_xma(el);

    START_PER_PARTICLE_BLOCK(part0, part);

        // Retrieve particle coordinates
        double x      = LocalParticle_get_x(part);
        double y      = LocalParticle_get_y(part);
        
        // chi = q/q0 * m0/m
        // p0c : reference particle momentum
        // q0  : reference particle charge
        //double const chi    = LocalParticle_get_chi(part);
        double const p0c    = LocalParticle_get_p0c(part);
        double const q0     = LocalParticle_get_q0(part);

        double dpx_total = 0.0;
        double dpy_total = 0.0;

        // Loop over each wire
        for (int64_t i = 0; i < n_wires; i++){
            
            // Retrieve parameters for the i-th wire from the data arrays
            double const cx = NonlinearKickerData_get_xma(el, i);
            double const cy = NonlinearKickerData_get_yma(el, i);
            double const curr = NonlinearKickerData_get_current(el, i);

            // Calculate distance relative to the wire center
            double D_x    = x-cx;
            double D_y    = y-cy;
            double R2     = D_x*D_x + D_y*D_y;
            
            // Skip the calculation if the distance is too small to avoid division by zero
            if (R2 < 1e-20) continue;

            // Computing the kick
            double const L1   = L_int + L_phy;
            double const L2   = L_int - L_phy;
            double const N    = MU_0*curr*q0/(4*PI*p0c/C_LIGHT);

            double dpx_i  =  -N*D_x*(sqrt(L1*L1 + 4.0*R2) - sqrt(L2*L2 + 4.0*R2))/R2;
            double dpy_i  =  -N*D_y*(sqrt(L1*L1 + 4.0*R2) - sqrt(L2*L2 + 4.0*R2))/R2;

            dpx_total += dpx_i;
            dpy_total += dpy_i;
        }

        // Update the particle properties
        LocalParticle_add_to_px(part, dpx_total);
        LocalParticle_add_to_py(part, dpy_total);
    END_PER_PARTICLE_BLOCK;
}

#endif
