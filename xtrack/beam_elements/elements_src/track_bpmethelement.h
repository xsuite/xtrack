// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //
#ifndef XTRACK_TRACK_BPMETHELEMENT_H
#define XTRACK_TRACK_BPMETHELEMENT_H

#include <headers/track.h>
// Here, include:
// 1. Header that contains the magnetic field callable.
// 2. Header that contains the Boris integrator?


GPUFUN
void BPMethElement_single_particle(LocalParticle* part, double Bs, double length)
{
    double const x  = LocalParticle_get_x(part);
    double const y  = LocalParticle_get_y(part);
    double const px = LocalParticle_get_px(part);
    double const py = LocalParticle_get_py(part);

    double const q0 = LocalParticle_get_q0(part);
    double const mass = LocalParticle_get_mass0(part);
    double const beta = LocalParticle_get_beta0(part);
    double const gamma = LocalParticle_get_gamma0(part);
    double const c = C_LIGHT;
    double const q_elem = QELEM;

    double const q0_coulomb = q0 * q_elem;
    double const mass_kg = mass * q_elem / c / c;

    // Simple Euler integration for now, it's just a test.
    double const x_hat  = x + px * length / beta / c / gamma / mass_kg;
    double const y_hat  = y + py * length / beta / c / gamma / mass_kg;
    double const px_hat = px + q0_coulomb / beta / c / gamma / mass_kg * Bs * py * length;
    double const py_hat = py - q0_coulomb / beta / c / gamma / mass_kg * Bs * px * length;

    LocalParticle_set_x(part, x_hat);
    LocalParticle_set_y(part, y_hat);
    LocalParticle_set_px(part, px_hat);
    LocalParticle_set_py(part, py_hat);
}

#endif