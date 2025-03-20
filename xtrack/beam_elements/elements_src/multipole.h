// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_MULTIPOLE_H
#define XTRACK_MULTIPOLE_H


/*gpufun*/
void Multipole_track_local_particle(MultipoleData el, LocalParticle* part0){

    SynchrotronRadiationRecordData record = NULL;
    int64_t radiation_flag = 0;
    #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        radiation_flag = MultipoleData_get_radiation_flag(el);
        if (radiation_flag==2){
            record = (SynchrotronRadiationRecordData) MultipoleData_getp_internal_record(el, part0);
        }
    #endif

    track_magnet_particles(
        /*part0*/                 part0,
        /*length*/                MultipoleData_get_length(el),
        /*order*/                 MultipoleData_get_order(el),
        /*inv_factorial_order*/   MultipoleData_get_inv_factorial_order(el),
        /*knl*/                   MultipoleData_getp1_knl(el, 0),
        /*ksl*/                   MultipoleData_getp1_ksl(el, 0),
        /*factor_knl_ksl*/        1.,
        /*num_multipole_kicks*/   1,
        /*model*/                 -1, // kick only
        /*integrator*/            3, // uniform
        /*radiation_flag*/        radiation_flag,
        /*radiation_record*/      record,
        /*delta_taper*/           MultipoleData_get_delta_taper(el),
        /*h*/                     0.,
        /*hxl*/                   MultipoleData_get_hxl(el),
        /*k0*/                    0.,
        /*k1*/                    0.,
        /*k2*/                    0.,
        /*k3*/                    0.,
        /*k0s*/                   0.,
        /*k1s*/                   0.,
        /*k2s*/                   0.,
        /*k3s*/                   0.,
        /*edge_entry_active*/     0,
        /*edge_exit_active*/      0,
        /*edge_entry_model*/      0,
        /*edge_exit_model*/       0,
        /*edge_entry_angle*/      0.,
        /*edge_exit_angle*/       0.,
        /*edge_entry_angle_fdown*/0.,
        /*edge_exit_angle_fdown*/ 0.,
        /*edge_entry_fint*/       0.,
        /*edge_exit_fint*/        0.,
        /*edge_entry_hgap*/       0.,
        /*edge_exit_hgap*/        0.
    );

}

#endif
