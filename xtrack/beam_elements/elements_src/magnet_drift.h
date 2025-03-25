// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_MAGNETDRIFT_H
#define XTRACK_MAGNETDRIFT_H

/*gpufun*/
void MagnetDrift_track_local_particle(
    MagnetDriftData el,
    LocalParticle* part0
) {
    const double length = MagnetDriftData_get_length(el);
    const double k0 = MagnetDriftData_get_k0(el);
    const double k1 = MagnetDriftData_get_k1(el);
    const double h = MagnetDriftData_get_h(el);
    const int64_t drift_model = MagnetDriftData_get_drift_model(el);

    //start_per_particle_block (part0->part)
        track_magnet_drift_single_particle(
            part, length, k0, k1, h, drift_model
        );
    //end_per_particle_block
}

#endif