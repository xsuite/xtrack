// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //
#ifndef XTRACK_EDGE_LOAD_H
#define XTRACK_EDGE_LOAD_H

#include "xtrack/beam_elements/elements_src/edge_flags.h"

// Fill the settings of one face (a MagnetFace) from the flat edge fields
// of an element. GET is the getter prefix (e.g. BendData_get_, or
// ThinSliceBendData_get__parent_ for a slice), side is entry or exit.
// Wanted, not implemented: an exit face following the entry one, which would
// resolve the exit mask and the NaN exit parameters here (see edge_flags.h).

#define EDGE_FLAGS(GET, el, side)        GET##_edge_##side##_flags(el)
#define EDGE_PARAM(GET, el, side, param) GET##edge_##side##_##param(el)

// Straight magnets: mask, soft dipole fringe (fringe integral, half gap) and
// the integrals of the soft quadrupole fringe (Lee-Whiting), f1 and f2
// (one statement each: they can follow an if)
#define EDGE_LOAD_STRAIGHT(face, GET, el, side) do {    \
    (face).flags = EDGE_FLAGS(GET, el, side);           \
    (face).fint = EDGE_PARAM(GET, el, side, fint);      \
    (face).hgap = EDGE_PARAM(GET, el, side, hgap);      \
    (face).f1 = EDGE_PARAM(GET, el, side, qint1);       \
    (face).f2 = EDGE_PARAM(GET, el, side, qint2);       \
} while (0)

// Bends and magnets: as straight magnets, plus face angle and linear feed-down
#define EDGE_LOAD_BEND(face, GET, el, side) do {                 \
    EDGE_LOAD_STRAIGHT(face, GET, el, side);                     \
    (face).angle = EDGE_PARAM(GET, el, side, angle);             \
    (face).angle_fdown = EDGE_PARAM(GET, el, side, angle_fdown); \
} while (0)

#endif  // XTRACK_EDGE_LOAD_H
