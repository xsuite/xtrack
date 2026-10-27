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

// Straight magnets: mask and the SAD soft quadrupole fringe integrals (f1, f2)
#define EDGE_LOAD_STRAIGHT(face, GET, el, side)   \
    (face).flags = EDGE_FLAGS(GET, el, side);     \
    (face).f1 = EDGE_PARAM(GET, el, side, qint1); \
    (face).f2 = EDGE_PARAM(GET, el, side, qint2)

// Bends and magnets: as straight magnets, plus face angle, linear feed-down,
// fringe integral, half gap
#define EDGE_LOAD_BEND(face, GET, el, side)                      \
    EDGE_LOAD_STRAIGHT(face, GET, el, side);                     \
    (face).angle = EDGE_PARAM(GET, el, side, angle);             \
    (face).angle_fdown = EDGE_PARAM(GET, el, side, angle_fdown); \
    (face).fint = EDGE_PARAM(GET, el, side, fint);               \
    (face).hgap = EDGE_PARAM(GET, el, side, hgap)

#endif  // XTRACK_EDGE_LOAD_H
