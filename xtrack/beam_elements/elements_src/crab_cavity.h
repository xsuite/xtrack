// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_CRABCAVITY_H
#define XTRACK_CRABCAVITY_H

#include <beam_elements/elements_src/track_rf.h>

GPUFUN
void CrabCavity_track_local_particle(CrabCavityData el, LocalParticle* part0)
{
    track_rf_particles(
        /*weight*/                1.,
        /*part0*/                 part0,
        /*length*/                CrabCavityData_get_length(el),
        /*voltage*/               0,
        /*frequency*/             CrabCavityData_get_frequency(el),
        /*lag*/                   0.,
        /*transverse_voltage*/    CrabCavityData_get_crab_voltage(el),
        /*transverse_lag*/        CrabCavityData_get_lag(el),
        /*absolute_time*/         CrabCavityData_get_absolute_time(el),
        /*order*/                 -1, // not used here
        /*knl*/                   NULL,
        /*ksl*/                   NULL,
        /*pn*/                    NULL,
        /*ps*/                    NULL,
        /*num_kicks*/             CrabCavityData_get_num_kicks(el),
        /*model*/                 CrabCavityData_get_model(el),
        /*default_model*/         6, // drift-kick-drift-expanded
        /*integrator*/            CrabCavityData_get_integrator(el),
        /*default_integrator*/    3, // Uniform
        /*radiation_flag*/        0, // not used here
        /*radiation_flag_parent*/ 0, // not used here
        /*lag_taper*/             CrabCavityData_get_lag_taper(el),
        /*body_active*/           1,
        /*edge_entry_active*/     0, // not used here
        /*edge_exit_active*/      0  // not used here
    );

}

#endif  // XTRACK_CRABCAVITY_H