// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //
#ifndef XTRACK_EDGE_FLAGS_H
#define XTRACK_EDGE_FLAGS_H

#include "xtrack/headers/track.h"

// Bits of the per-face fringe mask: the switch, then the model (linear edge
// or a set of fringe maps). The only definition: beam_elements/_edge.py reads
// them from here (one `#define EDGE_<NAME> <integer>` per line).
#define EDGE_ACTIVE  1
#define EDGE_LINEAR  2   // linear edge model, instead of the fringe maps
#define EDGE_BEND    4   // dipole fringe
#define EDGE_MULT    8   // hard-edge multipole fringe
#define EDGE_QUAD_LW 16  // soft quadrupole fringe, Lee-Whiting (qint1, qint2)

// Below these the face angle (rotation, wedge, patch) and the dipole strength
// of the fringe maps count as zero
#define EDGE_ANGLE_EPS 1e-10
#define EDGE_K0_EPS    1e-9

// Highest order of the k0...k3 (k0s...k3s) strengths of a magnet
#define MAGNET_K_ORDER 3

#endif  // XTRACK_EDGE_FLAGS_H
