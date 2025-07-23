// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //
#ifndef XTRACK_THIN_SLICE_QUADRUPOLE_H
#define XTRACK_THIN_SLICE_QUADRUPOLE_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_magnet.h>


GPUFUN
void ThinSliceQuadrupole_track_local_particle(
        ThinSliceQuadrupoleData el,
        LocalParticle* part0
) {
    double weight = ThinSliceQuadrupoleData_get_weight(el);

    track_magnet_particles(
        /*part0*/                 part0,
        /*length*/                ThinSliceQuadrupoleData_get__parent_length(el) * weight,
        /*order*/                 ThinSliceQuadrupoleData_get__parent_order(el),
        /*inv_factorial_order*/   ThinSliceQuadrupoleData_get__parent_inv_factorial_order(el),
        /*knl*/                   ThinSliceQuadrupoleData_getp1__parent_knl(el, 0),
        /*ksl*/                   ThinSliceQuadrupoleData_getp1__parent_ksl(el, 0),
        /*factor_knl_ksl*/        weight,
        /*num_multipole_kicks*/   1,
        /*model*/                 -1, // kick only
        /*default_model*/         0, // unused
        /*integrator*/            3, // uniform
        /*default_integrator*/    0, // unused
        /*radiation_flag*/        ThinSliceQuadrupoleData_get_radiation_flag(el),
        /*radiation_flag_parent*/ ThinSliceQuadrupoleData_get__parent_radiation_flag(el),
        /*radiation_record*/      (SynchrotronRadiationRecordData) ThinSliceQuadrupoleData_getp_internal_record(el, part0),
        /*delta_taper*/           ThinSliceQuadrupoleData_get_delta_taper(el),
        /*h*/                     0.,
        /*hxl*/                   0.,
        /*k0*/                    0.,
        /*k1*/                    ThinSliceQuadrupoleData_get__parent_k1(el),
        /*k2*/                    0.,
        /*k3*/                    0.,
        /*k0s*/                   0.,
        /*k1s*/                   ThinSliceQuadrupoleData_get__parent_k1s(el),
        /*k2s*/                   0.,
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
