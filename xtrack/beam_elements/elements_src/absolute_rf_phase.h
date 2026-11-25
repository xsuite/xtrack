// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //
#ifndef XTRACK_ABSOLUTE_RF_PHASE_H
#define XTRACK_ABSOLUTE_RF_PHASE_H

#include "xtrack/headers/track.h"

GPUFUN
double absolute_rf_phase(double frequency, double time_s, double local_time){
    // Subtract an integer number of cycles using a fused operation, so the
    // rounding of frequency * time_s does not erase the fractional phase.
    double const cycles = fma(frequency, time_s, -rint(frequency * time_s));
    double const local_cycles = frequency * local_time;
    return 2. * PI * ((cycles - rint(cycles))
                    + (local_cycles - rint(local_cycles)));
}

#endif
