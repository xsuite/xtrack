// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_MISALIGNMENT_H
#define XTRACK_MISALIGNMENT_H

#include <beam_elements/elements_src/track_misalignments.h>


GPUFUN
void Misalignment_track_local_particle(MisalignmentData el, LocalParticle* part0)
{
    const double dx = MisalignmentData_get_dx(el);
    const double dy = MisalignmentData_get_dy(el);
    const double ds = MisalignmentData_get_ds(el);
    const double theta = MisalignmentData_get_theta(el);
    const double phi = MisalignmentData_get_phi(el);
    const double psi = MisalignmentData_get_psi(el);
    const double anchor = MisalignmentData_get_anchor(el);
    const double length = MisalignmentData_get_length(el);
    const double angle = MisalignmentData_get_angle(el);
    const double tilt = MisalignmentData_get_tilt(el);
    const double is_exit = MisalignmentData_get_is_exit(el);
    const int8_t backtrack = 0;

    if (!is_exit) {
        if (NONZERO(angle)) {
            track_misalignment_entry_curved(part0, dx, dy, ds, theta, phi, psi, anchor, length, angle, tilt, backtrack);
        } else {
            track_misalignment_entry_straight(part0, dx, dy, ds, theta, phi, psi, anchor, length, tilt, backtrack);
        }
    } else {
        if (NONZERO(angle)) {
            track_misalignment_exit_curved(part0, dx, dy, ds, theta, phi, psi, anchor, length, angle, tilt, backtrack);
        } else {
            track_misalignment_exit_straight(part0, dx, dy, ds, theta, phi, psi, anchor, length, tilt, backtrack);
        }
    }
}

#endif  // XTRACK_MISALIGNMENT_H
