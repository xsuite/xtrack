// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THIN_SLICE_OCTUPOLE_H
#define XTRACK_THIN_SLICE_OCTUPOLE_H

/*gpufun*/
void ThinSliceOctupole_track_local_particle(
        ThinSliceOctupoleData el,
        LocalParticle* part0
) {

    double weight = ThinSliceOctupoleData_get_weight(el);

    int64_t radiation_flag = 0;
    double delta_taper = 0.0;
    SynchrotronRadiationRecordData record = NULL;
    #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        radiation_flag = ThinSliceOctupoleData_get_radiation_flag(el);
        if (radiation_flag == 10){ // from parent
            radiation_flag = ThinSliceOctupoleData_get__parent_radiation_flag(el);
        }
        delta_taper = ThinSliceOctupoleData_get_delta_taper(el);
        if (radiation_flag==2){
            record = (SynchrotronRadiationRecordData) ThinSliceOctupoleData_getp_internal_record(el, part0);
        }
    #endif

    track_magnet_particles(
        /*part0*/                 part0,
        /*length*/                ThinSliceOctupoleData_get__parent_length(el) * weight,
        /*order*/                 ThinSliceOctupoleData_get__parent_order(el),
        /*inv_factorial_order*/   ThinSliceOctupoleData_get__parent_inv_factorial_order(el),
        /*knl*/                   ThinSliceOctupoleData_getp1__parent_knl(el, 0),
        /*ksl*/                   ThinSliceOctupoleData_getp1__parent_ksl(el, 0),
        /*factor_knl_ksl*/        weight,
        /*num_multipole_kicks*/   1,
        /*model*/                 -1, // kick only
        /*integrator*/            3, // uniform
        /*radiation_flag*/        radiation_flag,
        /*radiation_record*/      record,
        /*delta_taper*/           delta_taper,
        /*h*/                     0.,
        /*hxl*/                   0.,
        /*k0*/                    0.,
        /*k1*/                    0.,
        /*k2*/                    0.,
        /*k3*/                    ThinSliceOctupoleData_get__parent_k3(el),
        /*k0s*/                   0.,
        /*k1s*/                   0.,
        /*k2s*/                   0.,
        /*k3s*/                   ThinSliceOctupoleData_get__parent_k3s(el),
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
