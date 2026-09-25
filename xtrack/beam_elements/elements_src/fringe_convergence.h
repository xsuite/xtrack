// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //
#ifndef XTRACK_FRINGE_CONVERGENCE_H
#define XTRACK_FRINGE_CONVERGENCE_H

// Shared by the dipole and multipole fringe backtracking iterations
// (track_dipole_fringe.h, track_mult_fringe.h).
#define XT_FRINGE_MAX_ITER 10
#define XT_FRINGE_TOL_FLOOR 1e-20 // only matters for backtracking through zeros

#endif // XTRACK_FRINGE_CONVERGENCE_H
