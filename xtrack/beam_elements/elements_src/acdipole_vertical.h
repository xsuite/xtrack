// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //

#ifndef XTRACK_ACDIPOLE_VERTICAL_H
#define XTRACK_ACDIPOLE_VERTICAL_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_acdipole.h>


GPUFUN void ACDipoleThickVertical_track_local_particle(
    ACDipoleThickVerticalData el,
    LocalParticle* part0
) {
    const double vrf = ACDipoleThickVerticalData_get_volt(el) * 300e-3;
    const double omega = ACDipoleThickVerticalData_get_freq(el) * 2 * PI;
    const double phirf = ACDipoleThickVerticalData_get_lag(el) * 2 * PI;
    const int16_t ramp1 = ACDipoleThickVerticalData_get_ramp(el, 0);
    const int16_t ramp2 = ACDipoleThickVerticalData_get_ramp(el, 1);
    const int16_t ramp3 = ACDipoleThickVerticalData_get_ramp(el, 2);
    const int16_t ramp4 = ACDipoleThickVerticalData_get_ramp(el, 3);
    

    START_PER_PARTICLE_BLOCK(part0, part);
        track_ac_dipole_vertical_single_particle(
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
