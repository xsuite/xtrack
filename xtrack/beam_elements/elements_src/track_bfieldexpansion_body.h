#include "xtrack/beam_elements/elements_src/track_magnet_spin_helpers.h"

#ifndef XTRACK_BFIELDEXPANSION_SPIN
#define XTRACK_BFIELDEXPANSION_SPIN
GPUFUN
void bfieldexpansion_spin_rhs(
    const double h_ref,
    const double beta0,
    const double gamma0,
    const double q0,
    const double mass0,
    const double G_spin,
    const double chi,
    const double z[6],
    const HamiltonianFlow* flow,
    const double spin[3],
    double kspin[3]
){
    // Returns dS/ds based on T-BMT differential equation for magnetic fields 

    const double ptau = z[5];
    const double delta = flow->delta;
    const double gamma = gamma0 * (1.0 + beta0 * ptau);
    const double beta = sqrt(1.0 - 1.0 / (gamma * gamma));
    const double mass0_kg = mass0 * QELEM / (C_LIGHT * C_LIGHT);
    const double P_J = mass0_kg * beta * gamma * C_LIGHT;
    const double brho_part = P_J / (q0 * QELEM);
    const double brho_ref = mass0_kg * beta0 * gamma0 * C_LIGHT / (q0 * QELEM);

    const double kin_px = z[1] - chi * flow->pot.Ax;
    const double kin_py = z[3] - chi * flow->pot.Ay;
    const double Bx_T = flow->pot.Bx * brho_ref;
    const double By_T = flow->pot.By * brho_ref;
    const double Bz_T = flow->pot.Bs * brho_ref;

    double iv_x, iv_y, iv_z;
    direction_of_motion(kin_px, kin_py, delta,
                        &iv_x, &iv_y, &iv_z);

    double B_par_x, B_par_y, B_par_z;
    double B_perp_x, B_perp_y, B_perp_z;

    separate_par_perp_components(
        Bx_T,
        By_T,
        Bz_T,
        iv_x,
        iv_y,
        iv_z,
        &B_par_x,
        &B_par_y,
        &B_par_z,
        &B_perp_x,
        &B_perp_y,
        &B_perp_z);

    const double Omega_BMT_x = -1.0 / brho_part * (
        (1.0 + G_spin * gamma) * B_perp_x + (1.0 + G_spin) * B_par_x);
    const double Omega_BMT_y = -1.0 / brho_part * (
        (1.0 + G_spin * gamma) * B_perp_y + (1.0 + G_spin) * B_par_y);
    const double Omega_BMT_z = -1.0 / brho_part * (
        (1.0 + G_spin * gamma) * B_perp_z + (1.0 + G_spin) * B_par_z);

    const double dl_ds = (1.0 + h_ref * z[0]) * flow->one_plus_delta / flow->root;

    const double Omega_x = Omega_BMT_x * dl_ds;
    const double Omega_y = Omega_BMT_y * dl_ds + h_ref;
    const double Omega_z = Omega_BMT_z * dl_ds;

    kspin[0] = Omega_y * spin[2] - Omega_z * spin[1];
    kspin[1] = Omega_z * spin[0] - Omega_x * spin[2];
    kspin[2] = Omega_x * spin[1] - Omega_y * spin[0];
}

#endif // XTRACK_BFIELDEXPANSION_SPIN


GPUFUN
int HAMILTONIAN_FLOW(Expansion *f, const double beta0, const double chi,
                     double s, const double z[6], HamiltonianFlow *flow) {
    double delta1, delta, ddelta1;
    double q, pix, piy, rad, root;

    const HamiltonianFlow zero = {0};
    *flow = zero;

    if (EVALUATE_EXPANSION(f, z[0], z[2], s, &flow->pot) != 0) return -1;
    delta_from_ptau(beta0, z[5], &delta, &delta1, &ddelta1);
    q = 1.0 + f->h * z[0];
    // Expansion potentials are normalized to the reference rigidity.
    pix = z[1] - chi * flow->pot.Ax;
    piy = z[3] - chi * flow->pot.Ay;  /* A_y is zero in this gauge. */
    rad = delta1 * delta1 - pix * pix - piy * piy;
    root = sqrt(rad);

    flow->delta = delta;
    flow->one_plus_delta = delta1;
    flow->radicand = rad;
    flow->root = root;
    flow->H = z[5] / beta0 - q * (root + chi * flow->pot.As);

    flow->rhs[0] = q * pix / root;  // dx/ds = dH/dpx
    flow->rhs[2] = q * piy / root;  // dy/ds = dH/dpy
    flow->rhs[4] = 1.0 / beta0 - q * delta1 * ddelta1 / root;  // dtau/ds = dH/dptau

    flow->rhs[1] = f->h * (root + chi * flow->pot.As)
        + q * chi * (pix * flow->pot.dAx_dx / root + flow->pot.dAs_dx);  // dpx/ds = -dH/dx
    flow->rhs[3] = q * chi * (pix * flow->pot.dAx_dy / root + flow->pot.dAs_dy);  // dpy/ds = -dH/dy
    flow->rhs[5] = 0.0;  // dptau/ds = -dH/dtau, H has no tau-dependence for these static fields.

    flow->grad[0] = -flow->rhs[1];  // dH/dx
    flow->grad[1] =  flow->rhs[0];  // dH/dpx
    flow->grad[2] = -flow->rhs[3];  // dH/dy
    flow->grad[3] =  flow->rhs[2];  // dH/dpy
    flow->grad[4] = -flow->rhs[5];  // dH/dtau
    flow->grad[5] =  flow->rhs[4];  // dH/dptau
    flow->dH_ds = -q * chi * (pix * flow->pot.dAx_ds / root + flow->pot.dAs_ds);
    return 0;
}

GPUFUN
void TRACK_EXPANSION(
    BFieldExpansionData el,
    LocalParticle* part0,
    const double length,
    double s_start,
    const int64_t nstep)
{
    double ds = length / nstep;

    Expansion f;
    BFieldExpansionData_init_expansion(el, &f);

    int64_t const backtrack = LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK);
    if (backtrack) {s_start += length; ds = -ds;}

    int pkin_const = BFieldExpansionData_get_pkin_const(el);

    START_PER_PARTICLE_BLOCK(part0, part);
        HamiltonianFlow flow;
        FieldValue v;
        int valid = 1;
        const double beta0  = LocalParticle_get_beta0(part);
        const double chi    = LocalParticle_get_chi(part);
        const double gamma0 = LocalParticle_get_gamma0(part);
        const double q0     = LocalParticle_get_q0(part);
        const double mass0  = LocalParticle_get_mass0(part);
        const double G_spin = LocalParticle_get_anomalous_magnetic_moment(part);

        const double x      = LocalParticle_get_x(part);
        const double px     = LocalParticle_get_px(part);
        const double y      = LocalParticle_get_y(part);
        const double py     = LocalParticle_get_py(part);
        const double tau    = LocalParticle_get_zeta(part) / beta0;
        const double ptau   = LocalParticle_get_ptau(part);
        const double ax     = LocalParticle_get_ax(part);
        const double ay     = LocalParticle_get_ay(part);
        double z[6] = {x, px, y, py, tau, ptau};

        double spin[3] = {
            LocalParticle_get_spin_x(part),
            LocalParticle_get_spin_y(part),
            LocalParticle_get_spin_z(part)};
        const int track_spin = (spin[0] != 0. || spin[1] != 0. || spin[2] != 0.);

        // Momentum has to be continuous, vector potential discontinuous, update canonical momentum
        if (pkin_const) {
            valid = EVALUATE_EXPANSION(&f, z[0], z[2], s_start, &v) == 0;
            if (valid) {
                // Stored particle ax/ay already include its charge-to-mass ratio.
                z[1] += chi * v.Ax - ax;
                z[3] += chi * v.Ay - ay;
            }
        }

        double s = s_start;
        double ztmp[6];
        double spintmp[3] = {0.0, 0.0, 0.0};
        for (int step = 0; valid && step < nstep; ++step) {
            double k1[6], k2[6], k3[6], k4[6];
            double ks1[3] = {0.0, 0.0, 0.0};
            double ks2[3] = {0.0, 0.0, 0.0};
            double ks3[3] = {0.0, 0.0, 0.0};
            double ks4[3] = {0.0, 0.0, 0.0};

            if (HAMILTONIAN_FLOW(&f, beta0, chi, s, z, &flow) != 0) {
                valid = 0;
                break;
            }
            for (int i = 0; i < 6; ++i) k1[i] = flow.rhs[i];
            if (track_spin) bfieldexpansion_spin_rhs(f.h, beta0, gamma0, q0, mass0, G_spin, chi, z, &flow, spin, ks1);
            for (int i = 0; i < 6; ++i) ztmp[i] = z[i] + 0.5 * ds * k1[i];
            if (track_spin) for (int i = 0; i < 3; ++i) spintmp[i] = spin[i] + 0.5 * ds * ks1[i];

            if (HAMILTONIAN_FLOW(&f, beta0, chi, s + 0.5 * ds, ztmp, &flow) != 0) {
                valid = 0;
                break;
            }
            for (int i = 0; i < 6; ++i) k2[i] = flow.rhs[i];
            if (track_spin) bfieldexpansion_spin_rhs(f.h, beta0, gamma0, q0, mass0, G_spin, chi, ztmp, &flow, spintmp, ks2);
            for (int i = 0; i < 6; ++i) ztmp[i] = z[i] + 0.5 * ds * k2[i];
            if (track_spin) for (int i = 0; i < 3; ++i) spintmp[i] = spin[i] + 0.5 * ds * ks2[i];

            if (HAMILTONIAN_FLOW(&f, beta0, chi, s + 0.5 * ds, ztmp, &flow) != 0) {
                valid = 0;
                break;
            }
            for (int i = 0; i < 6; ++i) k3[i] = flow.rhs[i];
            if (track_spin) bfieldexpansion_spin_rhs(f.h, beta0, gamma0, q0, mass0, G_spin, chi, ztmp, &flow, spintmp, ks3);
            for (int i = 0; i < 6; ++i) ztmp[i] = z[i] + ds * k3[i];
            if (track_spin) for (int i = 0; i < 3; ++i) spintmp[i] = spin[i] + ds * ks3[i];

            if (HAMILTONIAN_FLOW(&f, beta0, chi, s + ds, ztmp, &flow) != 0) {
                valid = 0;
                break;
            }
            for (int i = 0; i < 6; ++i) k4[i] = flow.rhs[i];
            if (track_spin) bfieldexpansion_spin_rhs(f.h, beta0, gamma0, q0, mass0, G_spin, chi, ztmp, &flow, spintmp, ks4);

            for (int i = 0; i < 6; ++i) z[i] += ds * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]) / 6.0;
            if (track_spin) for (int i = 0; i < 3; ++i) spin[i] += ds * (ks1[i] + 2.0*ks2[i] + 2.0*ks3[i] + ks4[i]) / 6.0;

            s += ds;
        }

        // Back to zero vector potential for next element
        if (valid) valid = EVALUATE_EXPANSION(&f, z[0], z[2], s, &v) == 0;
        if (!valid) {
            // The curved coordinate chart is singular at 1 + h*x == 0.
            // Leave the last committed particle coordinates intact.
            LocalParticle_set_state(part, XT_INVALID_BFIELD_EXPANSION);
        }
        else {
            if (pkin_const) {
                z[1] -= chi * v.Ax;
                z[3] -= chi * v.Ay;
                LocalParticle_set_ax(part, 0);
                LocalParticle_set_ay(part, 0);
            }
            else {
                LocalParticle_set_ax(part, chi * v.Ax);
                LocalParticle_set_ay(part, chi * v.Ay);
            }

            LocalParticle_set_x(part, z[0]);
            LocalParticle_set_px(part, z[1]);
            LocalParticle_set_y(part, z[2]);
            LocalParticle_set_py(part, z[3]);
            LocalParticle_set_zeta(part, z[4]*beta0);
            LocalParticle_update_ptau(part, z[5]);
            LocalParticle_add_to_s(part, ds*nstep);

            if (track_spin){
                LocalParticle_set_spin_x(part, spin[0]);
                LocalParticle_set_spin_y(part, spin[1]);
                LocalParticle_set_spin_z(part, spin[2]);
            }
        }
    END_PER_PARTICLE_BLOCK
}
