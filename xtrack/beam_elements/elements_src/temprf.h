// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_TEMPRF_H
#define XTRACK_TEMPRF_H

#include <beam_elements/elements_src/track_cavity.h>

GPUFUN
void TempRF_track_local_particle(TempRFData el, LocalParticle* part0)
{

    rack_rf_particles(
        /*weight*/                1.,
        /*part0*/                 part0,
        /*length*/                TempRFData_get_length(el);
        /*voltage*/               TempRFData_get_voltage(el),
        /*frequency*/             TempRFData_get_frequency(el),
        /*lag*/                   TempRFData_get_lag(el),
        /*absolute_time*/         0, // not used hereTempRFData_get_order(el),
        /*order*/                 -1, // not used here
        /*knl*/                   NULL, // not used here
        /*ksl*/                   NULL, // not used here
        /*pn*/                    NULL, // not used here
        /*ps*/                    NULL, // not used here
        /*num_kicks*/             0, // auto
        /*model*/                 6, // drift-kick-drift-expanded
        /*default_model*/         0, // unused
        /*integrator*/            1, // teapot
        /*default_integrator*/    0, // unused
        /*radiation_flag*/        0, // not used here
        /*radiation_flag_parent*/ 0, // not used here
        /*radiation_record*/      NULL, // not used here
        /*delta_taper*/           0., // not used here
        /*body_active*/           1,
        /*edge_entry_active*/     0, // not used here
        /*edge_exit_active*/      0, // not used here
    );

}

#endif  // XTRACK_CAVITY_H
