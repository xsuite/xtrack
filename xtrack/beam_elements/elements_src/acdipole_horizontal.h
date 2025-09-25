// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //

#ifndef XTRACK_ACDIPOLE_HORIZONTAL_H
#define XTRACK_ACDIPOLE_HORIZONTAL_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_acdipole.h>


GPUFUN void ACDipoleThickHorizontal_track_local_particle(
    ACDipoleThickHorizontalData el,
    LocalParticle* part0
) {
    const double vrf = ACDipoleThickHorizontalData_get_volt(el) * 300e-3;
    const double omega = ACDipoleThickHorizontalData_get_freq(el) * 2 * PI;
    const double phirf = ACDipoleThickHorizontalData_get_lag(el) * 2 * PI;
    const uint16_t ramp1 = ACDipoleThickHorizontalData_get_ramp(el, 0);
    const uint16_t ramp2 = ACDipoleThickHorizontalData_get_ramp(el, 1);
    const uint16_t ramp3 = ACDipoleThickHorizontalData_get_ramp(el, 2);
    const uint16_t ramp4 = ACDipoleThickHorizontalData_get_ramp(el, 3);


    START_PER_PARTICLE_BLOCK(part0, part);
        track_ac_dipole_horizontal_single_particle(
            part,
            vrf,
            omega,
            phirf,
            ramp1,
            ramp2,
            ramp3,
            ramp4
        );
    END_PER_PARTICLE_BLOCK;
}

#endif
