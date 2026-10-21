// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //
#ifndef XTRACK_TRACK_MAGNET_EDGE_H
#define XTRACK_TRACK_MAGNET_EDGE_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/elements_src/edge_flags.h"
#include "xtrack/beam_elements/elements_src/track_dipole_edge_linear.h"
#include "xtrack/beam_elements/elements_src/track_yrotation.h"
#include "xtrack/beam_elements/elements_src/track_wedge.h"
#include "xtrack/beam_elements/elements_src/track_mult_fringe.h"
#include "xtrack/beam_elements/elements_src/track_dipole_fringe.h"
#include "xtrack/beam_elements/elements_src/fringe_convergence.h"


// A magnet face is tracked as a fixed sequence of independent sub-maps, each
// selected by the fringe mask of the face (edge_flags.h), following MAD-NG
// (strex_fringe). Going from the drift into the field the order is
//
//     patch -> vector potential -> y-rotation -> bend fringe -> mult fringe
//         -> qsad fringe -> quad wedge -> wedge
//
// and going from the field into the drift it is the reverse. Every sub-map
// is written as the forward map of its face, in the element frame, and
// applies its exact inverse when backtracking, so that backtracking crosses
// the faces in the reverse order and undoes them one by one.
//
// EDGE_LINEAR replaces the magnet fringe (y-rotation to wedge) with the
// r21/r43 linear edge. A face without EDGE_ACTIVE is skipped altogether.
// MagnetFace_prepare turns the mask and the parameters into one do_* switch
// per sub-map (off for zero parameters), and every sub-map returns at once
// when its switch is off.


// Vector potential: picked up when entering the field region, dropped when
// leaving it (we assume a drift outside the magnet).
GPUFUN
void MagnetEdge_vector_potential_single_particle(
    LocalParticle* part,
    const uint8_t is_leaving_magnet,
    xt_float_or_tpsa_arg ksol,
    const double x0_solenoid,
    const double y0_solenoid
) {
    if (is_leaving_magnet) {
        LocalParticle_set_ax(part, 0.);
        LocalParticle_set_ay(part, 0.);
    }
    else {
        LocalParticle_set_ax(
            part, -0.5 * ksol * (LocalParticle_get_y(part) - y0_solenoid)
        );
        LocalParticle_set_ay(
            part, 0.5 * ksol * (LocalParticle_get_x(part) - x0_solenoid)
        );
    }
}


// cosh(f) and sinh(f)/f as functions of m = f^2, regular at m = 0 (series
// below m = 1 on the constant part, so that they also hold for TPSA)
GPUFUN
void MagnetEdge_cosh_sinhc(
    xt_float_or_tpsa_arg m,
    xt_float_or_tpsa* cosh_f,
    xt_float_or_tpsa* sinhc_f
) {
    if (xt_float_or_tpsa_const_part(m) < 1.) {
        xt_float_or_tpsa c = 1., s = 1.;
        for (int n = 12; n >= 1; n--) {
            c = 1. + m * c / ((2. * n - 1.) * (2. * n));
            s = 1. + m * s / ((2. * n) * (2. * n + 1.));
        }
        *cosh_f = c;
        *sinhc_f = s;
    }
    else {
        const xt_float_or_tpsa f = sqrt(m);
        *cosh_f = cosh(f);
        *sinhc_f = sinh(f) / f;
    }
}


// Quadrupole fringe of SAD (MAD-NG qsad_fringe): Lee-Whiting formula,
// E. Forest ch 13.2.3, eq 13.33. MAD-NG evaluates it in the frame rotated by
// -atan2(k1s, k1)/2, which is singular at k1 = k1s = 0. Here the rotation is
// absorbed into the matrix K = [[k1, -k1s], [-k1s, -k1]] (= b2 times the
// reflection in that frame, K^2 = b2^2), so that the exponentials become
// cosh(f) + sinh(f)/f f K/b2, with f = c1 b2, functions of f^2 = c1^2 (k1^2 +
// k1s^2). The map is then regular at zero strength, with TPSA derivatives.
// g1 carries MAD-NG's lw (+1 on the face crossed first, -1 on the other) and g2
// the direction sdir, so that each backtracking call inverts the forward one
// (as in MAD-NG's strex_fringe).
GPUFUN
void MagnetEdge_qsad_fringe_single_particle(
    LocalParticle* part,
    xt_float_or_tpsa_arg k1,   // total quadrupole strength
    xt_float_or_tpsa_arg k1s,  // total skew quadrupole strength
    const double g1,
    const double g2
) {
    const double chi = LocalParticle_get_chi(part);
    const double beta0 = LocalParticle_get_beta0(part);

    const xt_float_or_tpsa x = LocalParticle_get_x(part);
    const xt_float_or_tpsa px = LocalParticle_get_px(part);
    const xt_float_or_tpsa y = LocalParticle_get_y(part);
    const xt_float_or_tpsa py = LocalParticle_get_py(part);
    xt_float_or_tpsa t = LocalParticle_get_zeta(part) / beta0;
    const xt_float_or_tpsa pt = LocalParticle_get_ptau(part);

    const xt_float_or_tpsa _pz = 1 / sqrt(1 + 2 / beta0 * pt + POW2(pt));
    const xt_float_or_tpsa dt = (1 / beta0 + pt) * _pz;

    const xt_float_or_tpsa c1 = chi * g1 * _pz;
    const xt_float_or_tpsa c2 = chi * g2 * _pz;

    const xt_float_or_tpsa kk = POW2(k1) + POW2(k1s);
    xt_float_or_tpsa cosh_f = 1., sinhc_f = 1.;
    MagnetEdge_cosh_sinhc(POW2(c1) * kk, &cosh_f, &sinhc_f);

    // K q and K p
    const xt_float_or_tpsa kq_x = k1 * x - k1s * y;
    const xt_float_or_tpsa kq_y = -k1s * x - k1 * y;
    const xt_float_or_tpsa kp_x = k1 * px - k1s * py;
    const xt_float_or_tpsa kp_y = -k1s * px - k1 * py;

    // Time of flight: q.K p, p.K p, p.p
    const xt_float_or_tpsa a_s = cosh_f - POW2(c1) * kk / 2 * sinhc_f;
    const xt_float_or_tpsa sum =
        c1 * (x * kp_x + y * kp_y) + a_s * _pz * c2 * (px * kp_x + py * kp_y)
        + _pz * c1 * c2 * kk * (cosh_f / 2 - sinhc_f) * (POW2(px) + POW2(py));
    t -= dt * sum * _pz;

    LocalParticle_set_x(part, cosh_f * x + sinhc_f * c1 * kq_x + c2 * _pz * kp_x);
    LocalParticle_set_y(part, cosh_f * y + sinhc_f * c1 * kq_y + c2 * _pz * kp_y);
    LocalParticle_set_px(part, cosh_f * px - sinhc_f * c1 * kp_x);
    LocalParticle_set_py(part, cosh_f * py - sinhc_f * c1 * kp_y);
    LocalParticle_set_zeta(part, t * beta0);
}


// Field of a magnet, shared by its body and its faces: the strengths
// k = knorm / kskew (orders 0 ... k_order), plus knl_scale * knl / length
// (knl includes the relative multipoles); the solenoid strength for the
// vector potential.
// The strengths are pointers to the element's, not copies (copying a TPSA
// allocates).
typedef struct MagnetField_s {
    const xt_float_or_tpsa* knorm[MAGNET_K_ORDER + 1];
    const xt_float_or_tpsa* kskew[MAGNET_K_ORDER + 1];
    int64_t k_order;
    const xt_float_or_tpsa* knl;
    const xt_float_or_tpsa* ksl;
    int64_t kl_order;
    double knl_scale;
    double length;
    const xt_float_or_tpsa* ks;
    double x0_solenoid, y0_solenoid;

    // Derived by MagnetField_prepare and MagnetFace_prepare, only when a
    // face is active
    xt_float_or_tpsa k0, k1, k1s;  // total (skew) dipole and quadrupole
    // Linear edge, [is_exit_face]. Kept here and not in MagnetFace,
    // which holds only doubles: a TPSA is built only for an active face.
    xt_float_or_tpsa r21[2], r43[2];
    uint8_t has_mult;  // a multipole of order >= 1 (the multipole fringe)
#ifdef XTRACK_TPSA_TRACK
    // The TPSA type has no default constructor
    MagnetField_s() : k0(0.), k1(0.), k1s(0.), r21{0., 0.}, r43{0., 0.} {}
#endif
} MagnetField;


// Total dipole and quadrupole strengths: k + knl / length
GPUFUN
void MagnetField_prepare(MagnetField* field) {
    const double length = field->length;
    xt_float_or_tpsa k0 = 0., k1 = 0., k1s = 0.;
    if (field->k_order > -1) k0 += *field->knorm[0];
    if (field->k_order > 0) {
        k1 += *field->knorm[1];
        k1s += *field->kskew[1];
    }
    if (fabs(length) > 1e-10 && field->kl_order > -1) {
        k0 += field->knl_scale * field->knl[0] / length;
        if (field->kl_order > 0) {
            k1 += field->knl_scale * field->knl[1] / length;
            k1s += field->knl_scale * field->ksl[1] / length;
        }
    }
    field->k0 = k0;
    field->k1 = k1;
    field->k1s = k1s;

    uint8_t has_mult = 0;
    for (int64_t ii = 1; ii <= field->k_order && !has_mult; ii++) {
        has_mult = !xt_float_or_tpsa_is_zero(*field->knorm[ii])
                   || !xt_float_or_tpsa_is_zero(*field->kskew[ii]);
    }
    for (int64_t ii = 1; ii <= field->kl_order && !has_mult; ii++) {
        has_mult = !xt_float_or_tpsa_is_zero(field->knl[ii])
                   || !xt_float_or_tpsa_is_zero(field->ksl[ii]);
    }
    field->has_mult = has_mult;
}


// One face of a magnet. The settings are given in the element frame by the
// element; MagnetFace_prepare derives what the face does for one direction of
// travel, so that the per-particle code only branches on the do_* switches.
typedef struct MagnetFace_s {
    // Settings
    int64_t flags;         // fringe mask, see edge_flags.h
    uint8_t is_exit_face;  // 0: entry face, 1: exit face of the element
    double angle;          // face angle
    double angle_fdown;    // feed-down angle of the linear model
    double fint;           // fringe field integral
    double hgap;           // half gap
    double f1, f2;         // quadrupole fringe of SAD
    // Patch between the drift and the frame of the magnet (straight-body
    // RBend): a y-rotation by patch_angle and a shift by patch_dx
    uint8_t has_patch;
    double patch_angle, patch_dx;

    // Derived by MagnetFace_prepare; angles and kicks signed for the
    // direction of travel
    const MagnetField* field;
    uint8_t backtrack;
    uint8_t do_patch, do_vector_potential, do_linear;
    uint8_t do_rotate, do_bend, do_mult, do_qsad, do_quad_wedge, do_wedge;
    double patch_sin, patch_cos, patch_tan;
    double rot_sin, rot_cos, rot_tan;  // y-rotation by the face angle
    double wedge_angle;
    double qsad_g1, qsad_g2;
} MagnetFace;


// Face with all fringes switched off and no patch
GPUFUN
MagnetFace MagnetFace_empty(const uint8_t is_exit_face) {
    MagnetFace face;
    face.flags = 0;
    face.is_exit_face = is_exit_face;
    face.angle = 0;
    face.angle_fdown = 0;
    face.fint = 0;
    face.hgap = 0;
    face.f1 = 0;
    face.f2 = 0;
    face.has_patch = 0;
    face.patch_angle = 0;
    face.patch_dx = 0;
    face.field = NULL;
    return face;
}


// Derive the sub-maps of a face for the direction of travel. The field must
// have been prepared (MagnetField_prepare) if the face is active.
GPUFUN
void MagnetFace_prepare(MagnetFace* face, MagnetField* field, const uint8_t backtrack) {
    const int64_t flags = face->flags;
    // Rotations and wedges are inverted by flipping the angle sign
    const double rot_sign = backtrack ? 1. : -1.;

    face->field = field;
    face->backtrack = backtrack;
    if (!(flags & EDGE_ACTIVE)) {
        face->do_patch = 0;
        face->do_vector_potential = 0;
        face->do_linear = 0;
        face->do_rotate = 0;
        face->do_bend = 0;
        face->do_mult = 0;
        face->do_qsad = 0;
        face->do_quad_wedge = 0;
        face->do_wedge = 0;
        return;
    }

    // Patch
    face->do_patch = face->has_patch;
    double patch_sin = 0, patch_cos = 1;
    if (face->has_patch && fabs(face->patch_angle) > EDGE_ANGLE_EPS) {
        patch_sin = sin(face->patch_angle);
        patch_cos = cos(face->patch_angle);
    }
    face->patch_sin = rot_sign * patch_sin;
    face->patch_cos = patch_cos;
    face->patch_tan = rot_sign * patch_sin / patch_cos;

    face->do_vector_potential = 1;

    // Linear edge: a kick, inverted by flipping it
    face->do_linear = (flags & EDGE_LINEAR) != 0;
    if (face->do_linear) {
        xt_float_or_tpsa* r21 = &field->r21[face->is_exit_face];
        xt_float_or_tpsa* r43 = &field->r43[face->is_exit_face];
        compute_dipole_edge_linear_coefficients(
            field->k0, face->angle, face->angle_fdown, face->hgap, face->fint, r21, r43
        );
        if (backtrack) {
            *r21 = -*r21;
            *r43 = -*r43;
        }
    }

    // Magnet fringes; no geometric sub-map without a fringe map
    const uint8_t fringe =
        !(flags & EDGE_LINEAR) && (flags & (EDGE_BEND | EDGE_MULT | EDGE_QSAD));
    // A fringe of a vanishing field is the identity: skipped (the dipole
    // fringe uses the threshold of DipoleFringe_track_single_particle)
    face->do_bend = fringe && (flags & EDGE_BEND)
                    && !xt_float_or_tpsa_is_negligible(field->k0, EDGE_K0_EPS);
    face->do_mult = fringe && (flags & EDGE_MULT) && field->has_mult;
    face->do_rotate = fringe && fabs(face->angle) > EDGE_ANGLE_EPS;
    face->do_wedge = face->do_rotate && (field->k_order >= 0);
    face->do_quad_wedge = face->do_mult && face->do_rotate && (field->k_order >= 1);
    // Soft quadrupole fringe of SAD, with or without the hard-edge one
    face->do_qsad =
        fringe && (flags & EDGE_QSAD) && (face->f1 != 0 || face->f2 != 0)
        && !(xt_float_or_tpsa_is_zero(field->k1) && xt_float_or_tpsa_is_zero(field->k1s));

    double sin_ = 0, cos_ = 1, tan_ = 0;
    if (face->do_rotate) {
        sin_ = sin(face->angle);
        cos_ = cos(face->angle);
        tan_ = tan(face->angle);
    }
    face->rot_sin = rot_sign * sin_;
    face->rot_cos = cos_;
    face->rot_tan = rot_sign * tan_;
    face->wedge_angle = rot_sign * face->angle;

    // +1 on the face crossed first, -1 on the other (MAD-NG lw)
    const uint8_t entering = (face->is_exit_face == backtrack);
    face->qsad_g1 = (entering ? 1. : -1.) * (fabs(face->f1) * face->f1 / -24);
    face->qsad_g2 = face->f2 * (backtrack ? -1. : 1.);
}


// The sub-maps of a face, each the forward map of the face or, when
// backtracking, its inverse (the signs are set in MagnetFace_prepare)

GPUFUN
void MagnetFace_patch_single_particle(
    LocalParticle* part,
    const MagnetFace* f,
    const uint8_t entering
) {
    if (!f->do_patch) return;
    if (entering) {
        YRotation_single_particle(part, f->patch_sin, f->patch_cos, f->patch_tan);
        LocalParticle_add_to_x(part, f->patch_dx);
    }
    else {
        LocalParticle_add_to_x(part, -f->patch_dx);
        YRotation_single_particle(part, f->patch_sin, f->patch_cos, f->patch_tan);
    }
}

GPUFUN
void MagnetFace_vector_potential_single_particle(
    LocalParticle* part,
    const MagnetFace* f,
    const uint8_t entering
) {
    if (!f->do_vector_potential) return;
    MagnetEdge_vector_potential_single_particle(
        part,
        /*is_leaving_magnet*/ !entering,
        *f->field->ks,
        f->field->x0_solenoid,
        f->field->y0_solenoid
    );
}

GPUFUN
void MagnetFace_linear_single_particle(LocalParticle* part, const MagnetFace* f) {
    if (!f->do_linear) return;
    DipoleEdgeLinear_single_particle(
        part, f->field->r21[f->is_exit_face], f->field->r43[f->is_exit_face]
    );
}

GPUFUN
void MagnetFace_rotation_single_particle(LocalParticle* part, const MagnetFace* f) {
    if (!f->do_rotate) return;
    YRotation_single_particle(part, f->rot_sin, f->rot_cos, f->rot_tan);
}

GPUFUN
void MagnetFace_bend_fringe_single_particle(LocalParticle* part, const MagnetFace* f) {
    if (!f->do_bend) return;
    // The fringe kick changes sign between the two faces, a property of the
    // face itself
    if (f->is_exit_face) {
        DipoleFringe_track_single_particle(
            part, f->fint, f->hgap, -f->field->k0, f->backtrack
        );
    }
    else {
        DipoleFringe_track_single_particle(
            part, f->fint, f->hgap, f->field->k0, f->backtrack
        );
    }
}

GPUFUN
void MagnetFace_mult_fringe_single_particle(LocalParticle* part, const MagnetFace* f) {
    if (!f->do_mult) return;
    const MagnetField* field = f->field;
    MultFringe_track_single_particle(
        part,
        field->knorm,
        field->kskew,
        field->k_order,
        field->knl,
        field->ksl,
        field->kl_order,
        field->knl_scale,
        field->length,
        f->is_exit_face,
        /* min_order */ 1
    );
}

GPUFUN
void MagnetFace_qsad_fringe_single_particle(LocalParticle* part, const MagnetFace* f) {
    if (!f->do_qsad) return;
    MagnetEdge_qsad_fringe_single_particle(
        part, f->field->k1, f->field->k1s, f->qsad_g1, f->qsad_g2
    );
}

GPUFUN
void MagnetFace_quad_wedge_single_particle(LocalParticle* part, const MagnetFace* f) {
    if (!f->do_quad_wedge) return;
    Quad_wedge_single_particle(part, f->wedge_angle, *f->field->knorm[1]);
}

GPUFUN
void MagnetFace_wedge_single_particle(LocalParticle* part, const MagnetFace* f) {
    if (!f->do_wedge) return;
    // sin, cos and tan of wedge_angle are those of the face rotation
    Wedge_trig_single_particle(
        part, f->rot_sin, f->rot_cos, f->rot_tan, *f->field->knorm[0]
    );
}


// Cross a face from the drift into the field (the order at the top of this
// file)
GPUFUN
void MagnetFace_enter_single_particle(LocalParticle* part, const MagnetFace* f) {
    MagnetFace_patch_single_particle(part, f, 1);
    MagnetFace_vector_potential_single_particle(part, f, 1);
    MagnetFace_linear_single_particle(part, f);
    MagnetFace_rotation_single_particle(part, f);
    MagnetFace_bend_fringe_single_particle(part, f);
    MagnetFace_mult_fringe_single_particle(part, f);
    MagnetFace_qsad_fringe_single_particle(part, f);
    MagnetFace_quad_wedge_single_particle(part, f);
    MagnetFace_wedge_single_particle(part, f);
}


// Cross a face from the field into the drift: the same sub-maps in reverse
GPUFUN
void MagnetFace_leave_single_particle(LocalParticle* part, const MagnetFace* f) {
    MagnetFace_wedge_single_particle(part, f);
    MagnetFace_quad_wedge_single_particle(part, f);
    MagnetFace_qsad_fringe_single_particle(part, f);
    MagnetFace_mult_fringe_single_particle(part, f);
    MagnetFace_bend_fringe_single_particle(part, f);
    MagnetFace_rotation_single_particle(part, f);
    MagnetFace_linear_single_particle(part, f);
    MagnetFace_vector_potential_single_particle(part, f, 0);
    MagnetFace_patch_single_particle(part, f, 0);
}


// Cross a face, entering or leaving the field as follows from the face and the
// direction of travel: forward tracking enters at the entry face,
// backtracking at the exit one
GPUFUN
void MagnetFace_track_single_particle(LocalParticle* part, const MagnetFace* f) {
    if (f->is_exit_face == f->backtrack)
        MagnetFace_enter_single_particle(part, f);
    else
        MagnetFace_leave_single_particle(part, f);
}

#endif // XTRACK_TRACK_MAGNET_EDGE_H
