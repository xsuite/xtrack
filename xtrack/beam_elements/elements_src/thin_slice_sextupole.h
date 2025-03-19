// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THIN_SLICE_SEXTUPOLE_H
#define XTRACK_THIN_SLICE_SEXTUPOLE_H

/*gpufun*/
void ThinSliceSextupole_track_local_particle(
        ThinSliceSextupoleData el,
        LocalParticle* part0
) {


    double weight = ThickSliceSextupoleData_get_weight(el);

    int64_t num_multipole_kicks_parent = ThickSliceSextupoleData_get__parent_num_multipole_kicks(el);
    int64_t model = ThickSliceSextupoleData_get__parent_model(el);
    int64_t integrator = ThickSliceSextupoleData_get__parent_integrator(el);

    int64_t num_multipole_kicks = (int64_t) ceil(num_multipole_kicks_parent * weight);

    if (model == 0) {  // adaptive
        model = 4; // mat-kick-mat
    }
    if (integrator == 0) {  // adaptive
        integrator = 3; // uniform
    }
    if (num_multipole_kicks == 0) {
        num_multipole_kicks = 1;
    }

    int64_t radiation_flag = 0;
    double delta_taper = 0.0;
    #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        radiation_flag = ThickSliceSextupoleData_get_radiation_flag(el);
        if (radiation_flag == 10){ // from parent
            radiation_flag = ThickSliceSextupoleData_get__parent_radiation_flag(el);
        }
        delta_taper = ThickSliceSextupoleData_get_delta_taper(el);
    #endif



}

#endif
