// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THIN_SLICE_BEND_H
#define XTRACK_THIN_SLICE_BEND_H

/*gpufun*/
void ThinSliceBend_track_local_particle(
        ThinSliceBendData el,
        LocalParticle* part0
) {

    double weight = ThinSliceBendData_get_weight(el);

    const double k0 = ThinSliceBendData_get__parent_k0(el);
    const double k1 = ThinSliceBendData_get__parent_k1(el);
    const double h = ThinSliceBendData_get__parent_h(el);
    const double order = ThinSliceBendData_get__parent_order(el);
    const double inv_factorial_order = ThinSliceBendData_get__parent_inv_factorial_order(el);
    const double* knl = ThinSliceBendData_getp1__parent_knl(el, 0);
    const double* ksl = ThinSliceBendData_getp1__parent_ksl(el, 0);

    SynchrotronRadiationRecordData record = NULL;
    RecordIndex record_index = NULL;

    #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
    int64_t radiation_flag = ThinSliceBendData_get_radiation_flag(el);

    // Extract record and record_index
    if (radiation_flag==2){
        record = (SynchrotronRadiationRecordData) ThinSliceBendData_getp_internal_record(el, part0);
        if (record){
            record_index = SynchrotronRadiationRecordData_getp__index(record);
        }
    }

    #else
    int64_t radiation_flag = 0;
    #endif

    double dp_record_entry = 0.;
    double dpx_record_entry = 0.;
    double dpy_record_entry = 0.;
    double dp_record_exit = 0.;
    double dpx_record_exit = 0.;
    double dpy_record_exit = 0.;

    #ifdef XTRACK_MULTIPOLE_NO_SYNRAD
    #define delta_taper (0)
    #else
        double delta_taper = ThinSliceBendData_get_delta_taper(el);
    #endif


    #ifndef XSUITE_BACKTRACK
        double const length = weight * ThinSliceBendData_get__parent_length(el); // m
        double const backtrack_sign = 1;
    #else
        double const length = -weight * ThinSliceBendData_get__parent_length(el); // m
        double const backtrack_sign = -1;
    #endif

    double const knl_bend[2] = {backtrack_sign * k0 * length / weight,
                                backtrack_sign * k1 * length / weight}; // the length is supposed to be already scaled by the weight
    double const ksl_bend[2] = {0., 0.};
    double const hxl_bend = h * length;

    //start_per_particle_block (part0->part)

        #ifdef XTRACK_MULTIPOLE_TAPER
            delta_taper = LocalParticle_get_delta(part);
        #endif

        Multipole_track_single_particle(part,
            hxl_bend, length, weight, // weight 1
            knl, ksl, order, inv_factorial_order, // first tap unused
            knl_bend, ksl_bend, 1, 1,
            backtrack_sign,
            delta_taper, radiation_flag,
            &dp_record_entry, &dpx_record_entry, &dpy_record_entry,
            &dp_record_exit, &dpx_record_exit, &dpy_record_exit,
            record, record_index);

    //end_per_particle_block

}


#endif