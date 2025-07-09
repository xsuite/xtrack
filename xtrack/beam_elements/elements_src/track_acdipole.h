// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //
#ifndef XTRACK_TRACK_ACDIPOLE_H
#define XTRACK_TRACK_ACDIPOLE_H

#include <headers/track.h>

GPUFUN
void track_ac_dipole_vertical_single_particle(
    LocalParticle* part,
    double vrf,
    double omega,
    double phirf,
    uint16_t ramp1,
    uint16_t ramp2,
    uint16_t ramp3,
    uint16_t ramp4)
{
    double const at_turn = LocalParticle_get_at_turn(part);
    double const p0c = LocalParticle_get_p0c(part) / 1e9; // Convert to GeV/c

    double vrf_scaled;
    if (at_turn < ramp1)
    { // voltage stable at zero
        vrf_scaled = 0.0;
    }
    else if (at_turn < ramp2)
    { // ramping up the voltage
        vrf_scaled = (at_turn - ramp1) * vrf / (ramp2 - ramp1);
    }
    else if (at_turn < ramp3)
    { // voltage stable at maximum
        vrf_scaled = vrf;
    }
    else if (at_turn < ramp4)
    { // ramping down the voltage
        vrf_scaled = (ramp4 - at_turn) * vrf / (ramp4 - ramp3);
    }
    else
    { // stable again at zero
        vrf_scaled = 0.0;
    }
    double const kick_y = vrf_scaled / p0c * sin(phirf + omega * at_turn);
    LocalParticle_add_to_py(part, kick_y);
}

GPUFUN
void track_ac_dipole_horizontal_single_particle(
    LocalParticle* part,
    double vrf,
    double omega,
    double phirf,
    uint16_t ramp1,
    uint16_t ramp2,
    uint16_t ramp3,
    uint16_t ramp4)
{
    double const at_turn = LocalParticle_get_at_turn(part);
    double const p0c = LocalParticle_get_p0c(part) / 1e9; // Convert to GeV/c

    double vrf_scaled;
    if (at_turn < ramp1)
    { // voltage stable at zero
        vrf_scaled = 0.0;
    }
    else if (at_turn < ramp2)
    { // ramping up the voltage
        vrf_scaled = (at_turn - ramp1) * vrf / (ramp2 - ramp1);
    }
    else if (at_turn < ramp3)
    { // voltage stable at maximum
        vrf_scaled = vrf;
    }
    else if (at_turn < ramp4)
    { // ramping down the voltage
        vrf_scaled = (ramp4 - at_turn) * vrf / (ramp4 - ramp3);
    }
    else
    { // stable again at zero
        vrf_scaled = 0.0;
    }

    double const kick_x = vrf_scaled / p0c * sin(phirf + omega * at_turn);
    LocalParticle_add_to_px(part, kick_x);
}

GPUFUN
void track_thin_ac_dipole_vertical_single_particle(
    LocalParticle* part,
    double eff_grad)
{
    double const y = LocalParticle_get_y(part);
    LocalParticle_add_to_py(part, eff_grad * y);
}

GPUFUN
void track_thin_ac_dipole_horizontal_single_particle(
    LocalParticle* part,
    double eff_grad)
{
    double const x = LocalParticle_get_x(part);
    LocalParticle_add_to_px(part, eff_grad * x);
}

#endif
