// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_CAVITY_H
#define XTRACK_CAVITY_H

#include <beam_elements/elements_src/track_cavity.h>

GPUFUN
void Cavity_track_local_particle(CavityData el, LocalParticle* part0)
{
    double const volt = CavityData_get_voltage(el);
    double const freq = CavityData_get_frequency(el);
    double const lag = CavityData_get_lag(el);
    double const lag_taper = CavityData_get_lag_taper(el);
    int64_t const absolute_time = CavityData_get_absolute_time(el);

    track_cavity_particles(part0, volt, freq, lag, lag_taper, absolute_time);
}

#endif  // XTRACK_CAVITY_H
