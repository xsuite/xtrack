// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_MULTIPOLE_H
#define XTRACK_MULTIPOLE_H


/*gpufun*/
void Multipole_track_local_particle(MultipoleData el, LocalParticle* part0){

    SynchrotronRadiationRecordData record = NULL;
    RecordIndex record_index = NULL;

    #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
    int64_t radiation_flag = MultipoleData_get_radiation_flag(el);

    // Extract record and record_index
    if (radiation_flag==2){
        record = (SynchrotronRadiationRecordData) MultipoleData_getp_internal_record(el, part0);
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
        double delta_taper = MultipoleData_get_delta_taper(el);
    #endif

    int64_t const order = MultipoleData_get_order(el);
    double const inv_factorial_order_0 = MultipoleData_get_inv_factorial_order(el);

    #ifndef XSUITE_BACKTRACK
        double const hxl = MultipoleData_get_hxl(el);
    #else
        double const hxl = -MultipoleData_get_hxl(el);
    #endif

    /*gpuglmem*/ double const* knl = MultipoleData_getp1_knl(el, 0);
    /*gpuglmem*/ double const* ksl = MultipoleData_getp1_ksl(el, 0);

    #ifndef XSUITE_BACKTRACK
        double const length = MultipoleData_get_length(el); // m
        double const backtrack_sign = 1;
    #else
        double const length = -MultipoleData_get_length(el); // m
        double const backtrack_sign = -1;
    #endif

    //start_per_particle_block (part0->part)

        #ifdef XTRACK_MULTIPOLE_TAPER
            delta_taper = LocalParticle_get_delta(part);
        #endif

        Multipole_track_single_particle(part,
            hxl, length, 1, //weight 1
            knl, ksl, order, inv_factorial_order_0,
            NULL, NULL, -1, -1., // second tap unused
            backtrack_sign,
            delta_taper, radiation_flag,
            &dp_record_entry, &dpx_record_entry, &dpy_record_entry,
            &dp_record_exit, &dpx_record_exit, &dpy_record_exit,
            record, record_index);

    //end_per_particle_block
}

#endif
