// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THIN_SLICE_SEXTUPOLE_H
#define XTRACK_THIN_SLICE_SEXTUPOLE_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_magnet.h>


GPUFUN
void ThinSliceSextupole_track_local_particle(
        ThinSliceSextupoleData el,
        LocalParticle* part0
) {

    double weight = ThinSliceSextupoleData_get_weight(el);

    track_magnet_particles(
        /*part0*/                 part0,
        /*length*/                ThinSliceSextupoleData_get__parent_length(el) * weight,
        /*order*/                 ThinSliceSextupoleData_get__parent_order(el),
        /*inv_factorial_order*/   ThinSliceSextupoleData_get__parent_inv_factorial_order(el),
        /*knl*/                   ThinSliceSextupoleData_getp1__parent_knl(el, 0),
        /*ksl*/                   ThinSliceSextupoleData_getp1__parent_ksl(el, 0),
        /*factor_knl_ksl*/        weight,
        /*num_multipole_kicks*/   1,
        /*model*/                 -1, // kick only
        /*default_model*/         0, // unused
        /*integrator*/            3, // uniform
        /*default_integrator*/    0, // unused
        /*radiation_flag*/        ThinSliceSextupoleData_get_radiation_flag(el),
        /*radiation_flag_parent*/ ThinSliceSextupoleData_get__parent_radiation_flag(el),
        /*radiation_record*/      (SynchrotronRadiationRecordData) ThinSliceSextupoleData_getp_internal_record(el, part0),
        /*delta_taper*/           ThinSliceSextupoleData_get_delta_taper(el),
        /*h*/                     0.,
        /*hxl*/                   0.,
        /*k0*/                    0.,
        /*k1*/                    0.,
        /*k2*/                    ThinSliceSextupoleData_get__parent_k2(el),
        /*k3*/                    0.,
        /*k0s*/                   0.,
        /*k1s*/                   0.,
        /*k2s*/                   ThinSliceSextupoleData_get__parent_k2s(el),
        /*k3s*/                   0.,
        /*ks*/                    0.,
        /*dks_ds*/                0.,
        /*body_active*/           1,
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
