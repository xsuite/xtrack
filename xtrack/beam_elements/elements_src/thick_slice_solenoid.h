// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THICK_SLICE_SOLENOID_H
#define XTRACK_THICK_SLICE_SOLENOID_H

/*gpufun*/
void ThickSliceSolenoid_track_local_particle(
        ThickSliceSolenoidData el,
        LocalParticle* part0
) {

    double weight = ThickSliceSolenoidData_get_weight(el);
    double const ks = ThickSliceSolenoidData_get__parent_ks(el);

    #ifndef XSUITE_BACKTRACK
        double const length = weight * ThickSliceSolenoidData_get__parent_length(el); // m
    #else
        double const length = -weight * ThickSliceSolenoidData_get__parent_length(el); // m
    #endif



    //start_per_particle_block (part0->part)
        Solenoid_thick_with_radiation_track_single_particle(
                part, length, ks,
                0, // radiation flag, not supported for now
                NULL, NULL, NULL);
    //end_per_particle_block

}

#endif