// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_TEMPRF_H
#define XTRACK_TEMPRF_H

#include <beam_elements/elements_src/track_rf.h>

GPUFUN
void TempRF_track_local_particle(TempRFData el, LocalParticle* part0)
{

    track_rf_particles(
        /*weight*/                1.,
        /*part0*/                 part0,
        /*length*/                TempRFData_get_length(el),
        /*voltage*/               TempRFData_get_voltage(el),
        /*frequency*/             TempRFData_get_frequency(el),
        /*lag*/                   TempRFData_get_lag(el),
        /*absolute_time*/         0, // not used here
        /*order*/                 TempRFData_get_order(el),
        /*knl*/                   TempRFData_getp1_knl(el, 0),
        /*ksl*/                   TempRFData_getp1_ksl(el, 0),
        /*pn*/                    TempRFData_getp1_pn(el, 0),
        /*ps*/                    TempRFData_getp1_ps(el, 0),
        /*num_kicks*/             TempRFData_get_num_kicks(el),
        /*model*/                 TempRFData_get_model(el),
        /*default_model*/         6, // drift-kick-drift-expanded
        /*integrator*/            TempRFData_get_integrator(el),
        /*default_integrator*/    3, // Uniform
        /*radiation_flag*/        0, // not used here
        /*radiation_flag_parent*/ 0, // not used here
        /*lag_taper*/             0., // not used here
        /*body_active*/           1,
        /*edge_entry_active*/     0, // not used here
        /*edge_exit_active*/      0  // not used here
    );

}

#endif  // XTRACK_TEMPRF_H
