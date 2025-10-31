// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //

#ifndef XTRACK_ACDIPOLE_HORIZONTAL_H
#define XTRACK_ACDIPOLE_HORIZONTAL_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_acdipole.h>


GPUFUN void ACDipole_track_local_particle(
    ACDipoleData el,
    LocalParticle* part0
) {
    const uint8_t plane = ACDipoleData_get_plane(el);
    if (plane == 0) {
        // No kick applied
        return;
    }

    const uint64_t twiss_mode = ACDipoleData_get_twiss_mode(el);
    if (twiss_mode == 0) {
        const double vrf = ACDipoleData_get_volt(el) * 300e-3;
        const double omega = ACDipoleData_get_freq(el) * 2 * PI;
        const double phirf = ACDipoleData_get_lag(el) * 2 * PI;
        const uint32_t ramp1 = ACDipoleData_get_ramp(el, 0);
        const uint32_t ramp2 = ACDipoleData_get_ramp(el, 1);
        const uint32_t ramp3 = ACDipoleData_get_ramp(el, 2);
        const uint32_t ramp4 = ACDipoleData_get_ramp(el, 3);


        START_PER_PARTICLE_BLOCK(part0, part);
            track_ramped_ac_dipole_single_particle(
                part,
                vrf,
                omega,
                phirf,
                ramp1,
                ramp2,
                ramp3,
                ramp4,
                plane
            );
        END_PER_PARTICLE_BLOCK;
    }
    else {
        const double eff_grad = ACDipoleData_get_eff_grad(el);
        START_PER_PARTICLE_BLOCK(part0, part);
            track_effgrad_ac_dipole_single_particle(
                part,
                eff_grad,
                plane
            );
        END_PER_PARTICLE_BLOCK;

    }
}

#endif
