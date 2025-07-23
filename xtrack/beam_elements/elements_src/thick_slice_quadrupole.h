// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THICK_SLICE_QUADRUPOLE_H
#define XTRACK_THICK_SLICE_QUADRUPOLE_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_magnet.h>
#include <beam_elements/elements_src/default_magnet_config.h>

GPUFUN
void ThickSliceQuadrupole_track_local_particle(
        ThickSliceQuadrupoleData el,
        LocalParticle* part0
) {
    double weight = ThickSliceQuadrupoleData_get_weight(el);

    int64_t num_multipole_kicks_parent = ThickSliceQuadrupoleData_get__parent_num_multipole_kicks(el);
    int64_t model = ThickSliceQuadrupoleData_get__parent_model(el);
    int64_t integrator = ThickSliceQuadrupoleData_get__parent_integrator(el);

    int64_t num_multipole_kicks = (int64_t) ceil(num_multipole_kicks_parent * weight);

    track_magnet_particles(
        /*part0*/                 part0,
        /*length*/                ThickSliceQuadrupoleData_get__parent_length(el) * weight,
        /*order*/                 ThickSliceQuadrupoleData_get__parent_order(el),
        /*inv_factorial_order*/   ThickSliceQuadrupoleData_get__parent_inv_factorial_order(el),
        /*knl*/                   ThickSliceQuadrupoleData_getp1__parent_knl(el, 0),
        /*ksl*/                   ThickSliceQuadrupoleData_getp1__parent_ksl(el, 0),
        /*factor_knl_ksl*/        weight,
        /*num_multipole_kicks*/   num_multipole_kicks,
        /*model*/                 model,
        /*default_model*/         QUADRUPOLE_DEFAULT_MODEL,
        /*integrator*/            integrator,
        /*default_integrator*/    QUADRUPOLE_DEFAULT_INTEGRATOR,
        /*radiation_flag*/        ThickSliceQuadrupoleData_get_radiation_flag(el),
        /*radiation_flag_parent*/ ThickSliceQuadrupoleData_get__parent_radiation_flag(el),
        /*radiation_record*/      NULL,
        /*delta_taper*/           ThickSliceQuadrupoleData_get_delta_taper(el),
        /*h*/                     0.,
        /*hxl*/                   0.,
        /*k0*/                    0.,
        /*k1*/                    ThickSliceQuadrupoleData_get__parent_k1(el),
        /*k2*/                    0.,
        /*k3*/                    0.,
        /*k0s*/                   0.,
        /*k1s*/                   ThickSliceQuadrupoleData_get__parent_k1s(el),
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
