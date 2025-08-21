// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_RFMULTIPOLE_H
#define XTRACK_RFMULTIPOLE_H

#include <beam_elements/elements_src/track_rf.h>

GPUFUN
void RFMultipole_track_local_particle(RFMultipoleData el, LocalParticle* part0){

    track_rf_particles(
        /*weight*/                1.,
        /*part0*/                 part0,
        /*length*/                0,
        /*voltage*/               RFMultipoleData_get_voltage(el),
        /*frequency*/             RFMultipoleData_get_frequency(el),
        /*lag*/                   RFMultipoleData_get_lag(el),
        /*transverse_voltage*/    0.,
        /*transverse_lag*/        0.,
        /*absolute_time*/         0, // not used here
        /*order*/                 RFMultipoleData_get_order(el),
        /*knl*/                   RFMultipoleData_getp1_knl(el, 0),
        /*ksl*/                   RFMultipoleData_getp1_ksl(el, 0),
        /*pn*/                    RFMultipoleData_getp1_pn(el, 0),
        /*ps*/                    RFMultipoleData_getp1_ps(el, 0),
        /*num_kicks*/             1, // not used here
        /*model*/                 -1, // kick only
        /*default_model*/         0, // not used here
        /*integrator*/            0, // not used here
        /*default_integrator*/    0, // not used here
        /*radiation_flag*/        0, // not used here
        /*radiation_flag_parent*/ 0, // not used here
        /*lag_taper*/             0., // not used here
        /*body_active*/           1,
        /*edge_entry_active*/     0, // not used here
        /*edge_exit_active*/      0  // not used here
    );

}

#endif
