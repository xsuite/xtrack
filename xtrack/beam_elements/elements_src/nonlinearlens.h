// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_NONLINEARLENS_H
#define XTRACK_NONLINEARLENS_H

#include <headers/track.h>

// Implementation of a non-linear lens with elliptic potential
// (based on the corresponding element in MAD-X, reference: Danilov and Nagaitsev,
// https://journals.aps.org/prab/abstract/10.1103/PhysRevSTAB.13.084002)

GPUFUN
void NonLinearLens_track_local_particle(
        NonLinearLensData el,
        LocalParticle* part0
) {

    double const cnll = NonLinearLensData_get_cnll(el);
    double const knll = NonLinearLensData_get_knll(el) / cnll;

    START_PER_PARTICLE_BLOCK(part0, part);

        double const x = LocalParticle_get_x(part) / cnll;
        double const y = LocalParticle_get_y(part) / cnll;

        double const u = 0.5 * sqrt(POW2(x - 1.) + POW2(y)) + 0.5 * sqrt(POW2(x + 1.) + POW2(y));
        double const v = 0.5 * sqrt(POW2(x + 1.) + POW2(y)) - 0.5 * sqrt(POW2(x - 1.) + POW2(y));

        double dd;
        if (u == 1.){
            dd = 0.;
        }
        else {
            dd = POW2(u) * log(u + sqrt(POW2(u) - 1.0)) / sqrt(POW2(u) - 1.0);
        }

        double const dUu =
            (u + log(u + sqrt(u*u - 1.0)) * sqrt(POW2(u) - 1.) + dd)/(POW2(u) - POW2(v))
            - 2. * u * (u * log(u + sqrt(u*u - 1.)) * sqrt(POW2(u) - 1.)
            + v * (acos(v) - PI / 2.) * sqrt(1. - POW2(v))) /POW2(POW2(u) - POW2(v));
        double const dUv =
            2. * v * (u * log(u + sqrt(u*u - 1.)) * sqrt(POW2(u) - 1.)
            + v * (acos(v) - PI/2.) * sqrt(1. - POW2(v))) / POW2(POW2(u) - POW2(v))
            - (v - (acos(v)- PI/2.) * sqrt(1. - POW2(v)) + POW2(v) * (acos(v) - PI / 2.) / sqrt(1. - POW2(v)))
            / (POW2(u) - POW2(v));

        double const dux = 0.5 * (x - 1.) / sqrt(POW2(x - 1.) + POW2(y)) + 0.5 * (x + 1.) / sqrt(POW2(x + 1.) + POW2(y));
        double const duy = 0.5 * y / sqrt(POW2(x - 1.) + POW2(y)) + 0.5 * y / sqrt(POW2(x + 1.) + POW2(y));
        double const dvx = 0.5 * (x + 1.) / sqrt(POW2(x + 1.) + POW2(y)) - 0.5 * (x - 1.) / sqrt(POW2(x - 1.) + POW2(y));
        double const dvy = 0.5 * y / sqrt(POW2(x + 1.) + POW2(y)) - 0.5 * y / sqrt(POW2(x - 1.) + POW2(y));

        LocalParticle_add_to_px(part, knll * (dUu * dux + dUv * dvx));
        LocalParticle_add_to_py(part, knll * (dUu * duy + dUv * dvy));

    END_PER_PARTICLE_BLOCK;

}


#endif // XTRACK_NONLINEARLENS_H