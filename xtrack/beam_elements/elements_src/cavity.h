// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_CAVITY_H
#define XTRACK_CAVITY_H

#include <beam_elements/elements_src/track_rf.h>

GPUFUN
void Cavity_track_local_particle(CavityData el, LocalParticle* part0)
{
    track_rf_particles(
        /*weight*/                1.,
        /*part0*/                 part0,
        /*length*/                CavityData_get_length(el),
        /*voltage*/               CavityData_get_voltage(el),
        /*frequency*/             CavityData_get_frequency(el),
        /*lag*/                   CavityData_get_lag(el),
        /*transverse_voltage*/    0.,
        /*transverse_lag*/        0.,
        /*absolute_time*/         CavityData_get_absolute_time(el),
        /*order*/                 -1, // not used here
        /*knl*/                   NULL,
        /*ksl*/                   NULL,
        /*pn*/                    NULL,
        /*ps*/                    NULL,
        /*num_kicks*/             CavityData_get_num_kicks(el),
        /*model*/                 CavityData_get_model(el),
        /*default_model*/         6, // drift-kick-drift-expanded
        /*integrator*/            CavityData_get_integrator(el),
        /*default_integrator*/    3, // Uniform
        /*radiation_flag*/        0, // not used here
        /*radiation_flag_parent*/ 0, // not used here
        /*lag_taper*/             CavityData_get_lag_taper(el),
        /*body_active*/           1,
        /*edge_entry_active*/     0, // not used here
        /*edge_exit_active*/      0  // not used here
    );

}

#endif  // XTRACK_CAVITY_H
