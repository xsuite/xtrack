// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //
#ifndef XTRACK_TRACK_MAGNET_KICK_H
#define XTRACK_TRACK_MAGNET_KICK_H

#include "xtrack/headers/track.h"

#if defined(XT_KNOBS) || defined(XT_TPSA_SLOTS)
#include <new>
#include <type_traits>

// Knob build only: lift a double multipole array to constant tpsas so it can
// go through the XT_STRENGTH* kick.  mad::tpsa is non-movable, so it cannot be
// stored portably in std::vector; construct the short-lived contiguous buffer
// directly instead.
class _xt_lifted_arr {
public:
    _xt_lifted_arr(const double* a, int64_t n, LocalParticle* part)
        : storage_(NULL), data_(NULL), size_(0) {
        if (a == NULL || n <= 0) return;

        storage_ = new storage_t[n];
        data_ = reinterpret_cast<mad::tpsa*>(storage_);

        XT_NUM proto = LocalParticle_get_x(part);
        for (; size_ < n; size_++) {
            new (&data_[size_]) mad::tpsa(0.0 * proto + a[size_]);
        }
    }

    ~_xt_lifted_arr() {
        for (int64_t i = 0; i < size_; i++) {
            data_[i].~tpsa();
        }
        delete[] storage_;
    }

    const mad::tpsa* ptr() const {
        return data_;
    }

private:
    typedef typename std::aligned_storage<
        sizeof(mad::tpsa), alignof(mad::tpsa)>::type storage_t;

    _xt_lifted_arr(const _xt_lifted_arr&);
    _xt_lifted_arr& operator=(const _xt_lifted_arr&);

    storage_t* storage_;
    mad::tpsa* data_;
    int64_t size_;
};

#define XT_KICK_SIMPLE(pt, ord, invf, KN, KS, fac, kw) do { \
        _xt_lifted_arr _kn((KN), (ord)+1, (pt)); \
        _xt_lifted_arr _ks((KS), (ord)+1, (pt)); \
        kick_simple_single_particle((pt),(ord),(invf),_kn.ptr(),_ks.ptr(),(fac),(kw)); \
    } while(0)
#else
#define XT_KICK_SIMPLE(pt, ord, invf, KN, KS, fac, kw) \
        kick_simple_single_particle((pt),(ord),(invf),(KN),(KS),(fac),(kw))
#endif


GPUFUN
void kick_simple_single_particle(
    LocalParticle* part,
    int64_t order,
    double inv_factorial,
    const XT_STRENGTH* knl,
    const XT_STRENGTH* ksl,
    XT_STRENGTH_CONST_ARG factor,
    double kick_weight
);


GPUFUN
void track_magnet_kick_single_particle(
    LocalParticle* part,
    double length,
    int64_t order,
    double inv_factorial_order,
    GPUGLMEM const double* knl,
    GPUGLMEM const double* ksl,
    int64_t order_rel,
    double inv_factorial_order_rel,
    GPUGLMEM const double* knl_rel,
    GPUGLMEM const double* ksl_rel,
    XT_STRENGTH_CONST_ARG rel_ref_strength,
    double const factor_knl_ksl,
    double kick_weight,
    XT_STRENGTH_CONST_ARG k0,
    XT_STRENGTH_CONST_ARG k1,
    XT_STRENGTH_CONST_ARG k2,
    XT_STRENGTH_CONST_ARG k3,
    XT_STRENGTH_CONST_ARG k0s,
    XT_STRENGTH_CONST_ARG k1s,
    XT_STRENGTH_CONST_ARG k2s,
    XT_STRENGTH_CONST_ARG k3s,
    double h,
    double hxl,
    XT_STRENGTH_CONST_ARG k0_h_correction,
    XT_STRENGTH_CONST_ARG k1_h_correction,
    uint8_t rot_frame
){

    double const chi = LocalParticle_get_chi(part);
    XT_NUM const x = LocalParticle_get_x(part);
    XT_NUM const y = LocalParticle_get_y(part);

    // Staging arrays scaled by length; brace-init avoids default-constructing
    // the non-default-constructible XT_STRENGTH (tpsa) elements.
    XT_STRENGTH knl_main[4] = {k0 * length, k1 * length, k2 * length, k3 * length};
    XT_STRENGTH ksl_main[4] = {k0s * length, k1s * length, k2s * length, k3s * length};

    // multipolar kick (element knl/ksl are doubles; lifted to const tpsa under XT_KNOBS)
    XT_KICK_SIMPLE(
        part,
        order,
        inv_factorial_order,
        knl,
        ksl,
        XT_STRENGTH_LIFT(factor_knl_ksl),
        kick_weight
    );

    // multipolar kick
    XT_KICK_SIMPLE(
        part,
        order_rel,
        inv_factorial_order_rel,
        knl_rel,
        ksl_rel,
        factor_knl_ksl * rel_ref_strength,
        kick_weight
    );

    // main kick: knl_main/ksl_main are already XT_STRENGTH (tpsa under XT_KNOBS)
    kick_simple_single_particle(
        part,
        /* order */ 3,
        /* inv_factorial_order */ 1. / (3 * 2),
        knl_main,
        ksl_main,
        /* factor_knl_ksl */ XT_STRENGTH_LIFT(1.0),
        kick_weight
    );

    // Correct for the curvature
    XT_NUM dpx = 0.0*x;
    XT_NUM dpy = 0.0*x;
    XT_NUM dzeta = 0.0*x;

    if (rot_frame) {
        double const hl = h * length * kick_weight + hxl * kick_weight;
        dpx += hl * (1. + LocalParticle_get_delta(part));
        XT_NUM const rv0v = 1./LocalParticle_get_rvv(part);
        dzeta += -rv0v * hl * x;
    }

    double htot = h;
    if (length != 0) {
        htot += hxl / length;
    }

    // Correct for the curvature
    // k0h correction can be computed from this term in the hamiltonian
    // H = 1/2 h k0 x^2
    // (see MAD 8 physics manual, eq. 5.15, and apply Hamilton's eq. dp/ds = -dH/dx)
    XT_STRENGTH k0l_mult = XT_STRENGTH_LIFT(0.0);
    if (order >= 0) {
        k0l_mult = knl[0] * factor_knl_ksl;
    }
    if (order_rel >= 0 && knl_rel != NULL) {
        k0l_mult += knl_rel[0] * factor_knl_ksl * rel_ref_strength;
    }
    dpx += -chi * (k0_h_correction  *length + k0l_mult) * kick_weight * htot * x;

    // k1h correction can be computed from this term in the hamiltonian
    // H = 1/3 hk1 x^3 - 1/2 hk1 xy^2
    // (see MAD 8 physics manual, eq. 5.15, and apply Hamilton's eq. dp/ds = -dH/dx)
    XT_STRENGTH k1l_mult = XT_STRENGTH_LIFT(0.0);
    if (order >= 1) {
        k1l_mult = knl[1] * factor_knl_ksl;
    }
    if (order_rel >= 1 && knl_rel != NULL) {
        k1l_mult += knl_rel[1] * factor_knl_ksl * rel_ref_strength;
    }
    dpx += htot * chi * (k1_h_correction * length + k1l_mult) * kick_weight * (-x * x + 0.5 * y * y);
    dpy += htot * chi * (k1_h_correction * length  + k1l_mult) * kick_weight * x * y;

    LocalParticle_add_to_px(part, dpx);
    LocalParticle_add_to_py(part, dpy);
    LocalParticle_add_to_zeta(part, dzeta);

}



GPUFUN
uint8_t kick_is_inactive(
    int64_t order,
    GPUGLMEM const double* knl,
    GPUGLMEM const double* ksl,
    XT_STRENGTH_CONST_ARG k0,
    XT_STRENGTH_CONST_ARG k1,
    XT_STRENGTH_CONST_ARG k2,
    XT_STRENGTH_CONST_ARG k3,
    XT_STRENGTH_CONST_ARG k0s,
    XT_STRENGTH_CONST_ARG k1s,
    XT_STRENGTH_CONST_ARG k2s,
    XT_STRENGTH_CONST_ARG k3s,
    double h
){
    if (h != 0) return 0;
    if (k0 != 0) return 0;
    if (k1 != 0) return 0;
    if (k2 != 0) return 0;
    if (k3 != 0) return 0;
    if (k0s != 0) return 0;
    if (k1s != 0) return 0;
    if (k2s != 0) return 0;
    if (k3s != 0) return 0;

    for (int index = order; index >= 0; index--) {
        if (knl[index] != 0) return 0;
        if (ksl[index] != 0) return 0;
    }

    return 1;

}

GPUFUN
void kick_simple_single_coordinates(
    XT_NUM_CONST_ARG x,   // by const-ref for non-scalar XT_NUM (tpsa copy-constructor doesn't copy coefficients);
    XT_NUM_CONST_ARG y,   // by value for native C doubles. See XT_NUM_CONST_ARG in track.h.
    double const chi,
    int64_t order,
    double inv_factorial,
    const XT_STRENGTH* knl,
    const XT_STRENGTH* ksl,
    XT_STRENGTH_CONST_ARG factor,
    double kick_weight,
    XT_NUM *dpx,
    XT_NUM *dpy
) {

    // Return if null knl/ksl pointers
    if (knl == NULL || ksl == NULL) {
        *dpx = 0.;
        *dpy = 0.;
        return;
    }

    int64_t index = order;

    XT_NUM dpx_mul = 0.0*x + chi * knl[index] * factor * inv_factorial;
    XT_NUM dpy_mul = 0.0*x + chi * ksl[index] * factor * inv_factorial;

    while( index > 0 )
    {
        XT_NUM const zre = dpx_mul * x - dpy_mul * y;
        XT_NUM const zim = dpx_mul * y + dpy_mul * x;

        inv_factorial *= index;
        index -= 1;

        XT_STRENGTH this_knl = chi * knl[index] * factor;
        XT_STRENGTH this_ksl = chi * ksl[index] * factor;

        dpx_mul = this_knl * inv_factorial + zre;
        dpy_mul = this_ksl * inv_factorial + zim;
    }

    dpx_mul = -dpx_mul; // rad

    *dpx = kick_weight * dpx_mul;
    *dpy = kick_weight * dpy_mul;
}


GPUFUN
void kick_simple_single_particle(
    LocalParticle* part,
    int64_t order,
    double inv_factorial,
    const XT_STRENGTH* knl,
    const XT_STRENGTH* ksl,
    XT_STRENGTH_CONST_ARG factor,
    double kick_weight
) {
    double const chi = LocalParticle_get_chi(part);
    XT_NUM const x = LocalParticle_get_x(part);
    XT_NUM const y = LocalParticle_get_y(part);

    XT_NUM dpx = 0.0*x, dpy = 0.0*x;

    kick_simple_single_coordinates(
        x,
        y,
        chi,
        order,
        inv_factorial,
        knl,
        ksl,
        factor,
        kick_weight,
        &dpx,
        &dpy);

    LocalParticle_add_to_px(part, dpx);
    LocalParticle_add_to_py(part, dpy);
}

#ifndef XT_FLAVOR_TPSA
// Radiation/field helper: emits physical fields to double* outputs, so it cannot be a
// Taylor map. This is unreachable in the bridge (only called from WITH_RADIATION, which is
// excluded through a macro by XTRACK_MULTIPOLE_NO_SYNRAD); excluded from the TPSA flavor so the
// coordinate args don't need to be XT_NUM.
GPUFUN
void evaluate_field_from_strengths(
    double const p0c,
    double const q0,
    double const x,
    double const y,
    double length,
    int64_t order,
    double inv_factorial_order,
    GPUGLMEM const double* knl,
    GPUGLMEM const double* ksl,
    int64_t order_rel,
    double inv_factorial_order_rel,
    GPUGLMEM const double* knl_rel,
    GPUGLMEM const double* ksl_rel,
    XT_STRENGTH_CONST_ARG rel_ref_strength,
    double const factor_knl_ksl,
    double k0,
    double k1,
    double k2,
    double k3,
    double k0s,
    double k1s,
    double k2s,
    double k3s,
    double ks,
    double dks_ds,
    double x0_solenoid,
    double y0_solenoid,
    double *Bx_T,
    double *By_T,
    double *Bz_T
){
    if (length == 0.0) {
        *Bx_T = 0.0;
        *By_T = 0.0;
        *Bz_T = 0.0;
        return;
    }

    double knl_main[4] = {k0, k1, k2, k3};
    double ksl_main[4] = {k0s, k1s, k2s, k3s};

    for (int index = 0; index < 4; index++) {
        knl_main[index] = knl_main[index] * length;
        ksl_main[index] = ksl_main[index] * length;
    }

    // multipolar kick
    double dpx_mul = 0.;
    double dpy_mul = 0.;
    kick_simple_single_coordinates(
        x,
        y,
        1., // chi
        order,
        inv_factorial_order,
        knl,
        ksl,
        factor_knl_ksl,
        1., // kick_weight
        &dpx_mul,
        &dpy_mul);

    // multipolar kick relevant for the field evaluation
    double dpx_mul_rel = 0.;
    double dpy_mul_rel = 0.;
    kick_simple_single_coordinates(
        x,
        y,
        1., // chi
        order_rel,
        inv_factorial_order_rel,
        knl_rel,
        ksl_rel,
        factor_knl_ksl * rel_ref_strength,
        1., // kick_weight
        &dpx_mul_rel,
        &dpy_mul_rel);


    // main kick
    double dpx_main=0.;
    double dpy_main=0.;
    kick_simple_single_coordinates(
        x,
        y,
        1., // chi
        3, // order
        1. / (3 * 2), //inv_factorial_order
        knl_main,
        ksl_main,
        1, // factor_knl_ksl,
        1., // kick_weight
        &dpx_main,
        &dpy_main);

    double const dpx = dpx_mul + dpx_main + dpx_mul_rel;
    double const dpy = dpy_mul + dpy_main + dpy_mul_rel;

    double const brho_0 = p0c / C_LIGHT / q0; // [T m]

    *Bx_T = dpy * brho_0 / length - 0.5 * dks_ds * brho_0 * (x - x0_solenoid); // [T]
    *By_T = -dpx * brho_0 / length - 0.5 * dks_ds * brho_0 * (y - y0_solenoid); // [T]
    *Bz_T = ks * brho_0; // [T]

}
#endif // XT_FLAVOR_TPSA (evaluate_field_from_strengths)

#endif
