// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //
#ifndef XTRACK_TRACK_SPLINEBORIS_H
#define XTRACK_TRACK_SPLINEBORIS_H

#include <headers/track.h>
#include "_spline_B_field_eval_scalar.h" // evaluate_B for Bx, By, Bs (scalar version)

// Spatial Boris integrator for BPMethElement, using a fitted field map.
//
// Arguments:
//   part           : LocalParticle pointer (single particle tracking kernel)
//   params    : pointer to the flat parameter array for THIS ELEMENT
//                    (start of the block corresponding to this element)
//   multipole_order: multipole order used in evaluate_B
//   s_start        : starting s-position in field map [m] (absolute, can be negative)
//   s_end          : ending s-position in field map [m] (absolute)
//   n_steps        : number of Boris substeps along the element
//
// Internally:
//   - Uses physical momenta px, py, ps, P in SI units (kg m / s)
//   - P0 is derived from p0c (eV) via P0 = p0c * QELEM / C_LIGHT
//   - px, py in the particle object are dimensionless (px_phys / P0)
//   - Bs from evaluate_B is used as Bs
//   - Uses local s coordinate (0 to L) for stepping, converts to absolute s_field
//     for field evaluation: s_field = s_start + s_local
//   - Particle's s coordinate is updated by adding element length L
//   - zeta is updated as in the Python version: sum over steps of
//       zeta += (ds - dt * c * beta0)
//     -> equivalent to zeta += (L - total_dt * c * beta0)
//        where L = s_end - s_start
GPUFUN
void SplineBoris_single_particle(
    LocalParticle* part,
    const double* const* params,
    const int      multipole_order,
    const double   s_start,
    const double   s_end,
    const int      n_steps
){
    // Skip dead particles (state <= 0)
    if (LocalParticle_get_state(part) <= 0){
        return;
    }

    // ----------------------------------------------------------------------
    //  Extract particle/reference parameters
    // ----------------------------------------------------------------------
    const double c      = C_LIGHT;   // [m/s]
    const double qe     = QELEM;     // [C], elementary charge (absolute value)

    const double q0     = LocalParticle_get_q0(part);      // charge in units of e
    const double mass0  = LocalParticle_get_mass0(part);   // [eV]
    const double delta  = LocalParticle_get_delta(part);   // relative momentum deviation
    const double p0c_ev = LocalParticle_get_p0c(part);     // reference p0 c [eV]
    const double beta0  = LocalParticle_get_beta0(part);   // reference beta

    // Positions and momenta (dimensionless px, py)
    double x    = LocalParticle_get_x(part);   // [m]
    double y    = LocalParticle_get_y(part);   // [m]
    double px_r = LocalParticle_get_px(part);  // dimensionless px / p0
    double py_r = LocalParticle_get_py(part);  // dimensionless py / p0
    double zeta = LocalParticle_get_zeta(part);

    // ----------------------------------------------------------------------
    //  Convert to physical units (kg, m, s)
    // ----------------------------------------------------------------------
    // mass_kg = mass[eV] * qe [J/eV] / c^2
    const double mass_kg = mass0 * qe / (c * c);  // [kg]

    // Reference momentum P0 in SI units:
    //   p0c [eV] * qe [J/eV] / c [m/s] = kg m / s
    const double P0 = p0c_ev * qe / c;  // [kg m / s]

    // Total momentum magnitude for this particle:
    const double P = P0 * (1.0 + delta);  // [kg m / s]

    // gamma = sqrt(1 + (P / (m c))^2)  (relativistic)
    const double P_over_mc = P / (mass_kg * c);
    const double gamma     = sqrt(1.0 + P_over_mc * P_over_mc);

    // Physical transverse momenta
    double px = px_r * P0;  // [kg m / s]
    double py = py_r * P0;  // [kg m / s]

    const double q_coulomb = q0 * qe;  // [C]

    // ----------------------------------------------------------------------
    //  Set up longitudinal stepping
    // ----------------------------------------------------------------------
    const double L    = s_end - s_start;
    const double ds   = L / (double) n_steps;
	const double half_ds = 0.5 * ds;
    
    // Local s coordinate (0 to L) for stepping through the element
    double s_local = 0.0;

    double total_dt = 0.0;  // accumulated time [s] over all substeps

    // ----------------------------------------------------------------------
    //  Loop over Boris substeps
    // ----------------------------------------------------------------------
    for (int istep = 0; istep < n_steps; ++istep, ++params) {

        // --------------------------------------------------------------
        //  (0) Longitudinal momentum from constant |p| = P
        // --------------------------------------------------------------
        double tmp  = P * P - px * px - py * py;
        if (tmp < 0.0) tmp = 0.0;
        double ps   = sqrt(tmp);     // [kg m / s]
        if (ps == 0.0) {
            // If ps is zero, drift/rotation is ill-defined; we just skip.
            // Should not occur in practice.
            break;
        }

        // --------------------------------------------------------------
        //  (1) FIRST HALF-DRIFT in x, y, and time
        // --------------------------------------------------------------
        const double inv_ps  = 1.0 / ps;

        const double xh = x + (px * inv_ps) * half_ds;
        const double yh = y + (py * inv_ps) * half_ds;
        const double s_local_h = s_local + half_ds;

        double dt = half_ds * inv_ps * gamma * mass_kg; // [s]

        // --------------------------------------------------------------
        //  Evaluate B-field at mid-step (xh, yh, s_field)
        //  Convert local s to absolute s in field map for field evaluation
        //  Using evaluate_B from _bpmeth_B_field_eval.h
        // --------------------------------------------------------------
        double Bx;
        double By;
        double Bs;
        
        // Convert local s coordinate to absolute s in field map
        const double s_field = s_start + s_local_h;

        evaluate_B_scalar(
            xh, yh, s_field,
            *params,
            multipole_order,
            &Bx, &By, &Bs
        );

        // --------------------------------------------------------------
        //  (2) FIRST HALF-KICK from (Bx, By)
        // --------------------------------------------------------------
        const double half_qds = q_coulomb * half_ds;

        double pxm = px - half_qds * By;
        double pym = py + half_qds * Bx;

        // Enforce |p| = P
        tmp = P * P - pxm * pxm - pym * pym;
        if (tmp < 0.0) tmp = 0.0;
        double ps_mid = sqrt(tmp);
        if (ps_mid == 0.0) {
            // Same caveat: if ps vanishes, rotation is undefined.
            break;
        }

        // --------------------------------------------------------------
        //  (3) ROTATION around Bs
        // --------------------------------------------------------------
        double t  = q_coulomb * Bs * half_ds / ps_mid;
        double t2 = t * t;
        double inv_den = 1.0 / (1.0 + t2);

        double sR = 2.0 * t * inv_den;
        double c0 = (1.0 - t2) * inv_den;

        double pxp = c0 * pxm + sR * pym;
        double pyp = -sR * pxm + c0 * pym;

        // --------------------------------------------------------------
        //  (4) SECOND HALF-KICK from (Bx, By)
        // --------------------------------------------------------------
        double px1 = pxp - half_qds * By;
        double py1 = pyp + half_qds * Bx;

        tmp = P * P - px1 * px1 - py1 * py1;
        if (tmp < 0.0) tmp = 0.0;
        double ps1 = sqrt(tmp);
        if (ps1 == 0.0) {
            break;
        }
        double inv_ps1 = 1.0 / ps1;

        // --------------------------------------------------------------
        //  (5) SECOND HALF-DRIFT in x, y and time
        // --------------------------------------------------------------
        x = xh + (px1 * inv_ps1) * half_ds;
        y = yh + (py1 * inv_ps1) * half_ds;
		s_local += ds;  // Advance local s coordinate

        dt += half_ds * inv_ps1 * gamma * mass_kg;  // [s]

        // Store updated momenta for next step
        px = px1;
        py = py1;

        // Accumulate time
        total_dt += dt;
    }

    // ------------------------------------------------------------------
    //  Write back to LocalParticle
    // ------------------------------------------------------------------
    // Positions:
    LocalParticle_set_x(part, x);
    LocalParticle_set_y(part, y);

    // s: Add element length to particle's s coordinate (like Drift element)
    // The particle's s coordinate advances by L from its incoming position
    LocalParticle_add_to_s(part, L);

    // Convert physical momenta back to dimensionless px, py (relative to p0)
    LocalParticle_set_px(part, px / P0);
    LocalParticle_set_py(part, py / P0);

    // Update longitudinal coordinate zeta:
    // Python: per step zeta += (ds - dt * c * beta0)
    // -> here we do it once, using total_dt = sum(dt) over all substeps.
    const double delta_zeta = L - total_dt * c * beta0;
    LocalParticle_set_zeta(part, zeta + delta_zeta);
}

#endif // XTRACK_TRACK_SPLINEBORIS_H
