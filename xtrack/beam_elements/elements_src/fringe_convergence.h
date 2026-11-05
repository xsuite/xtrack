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

// Converged when the step from prev to next (constant parts) is below the
// relative tolerance
#define XT_FRINGE_STEP_CONVERGED(prev, next)            \
    (fabs(xt_float_or_tpsa_const_part((next) - (prev))) \
     <= 1e-13 * fabs(xt_float_or_tpsa_const_part(next)) + XT_FRINGE_TOL_FLOOR)

#endif // XTRACK_FRINGE_CONVERGENCE_H
