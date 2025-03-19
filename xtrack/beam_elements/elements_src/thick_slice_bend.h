// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THICK_SLICE_BEND_H
#define XTRACK_THICK_SLICE_BEND_H

/*gpufun*/
void ThickSliceBend_track_local_particle(
        ThickSliceBendData el,
        LocalParticle* part0
) {

    double weight = ThickSliceBendData_get_weight(el);
    int64_t radiation_flag = 0;
    double delta_taper = 0.0;
    #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        int64_t radiation_flag = ThickSliceBendData_get_radiation_flag(el);
        if (radiation_flag == 10){ // from parent
            radiation_flag = ThickSliceBendData_get__parent_radiation_flag(el);
        }
    #endif


}

#endif