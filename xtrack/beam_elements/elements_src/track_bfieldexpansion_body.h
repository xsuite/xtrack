/* Intentionally included twice: specialize the integration loop for each
 * geometry, with no geometry dispatch inside the Runge-Kutta stages. */

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

        const double x      = LocalParticle_get_x(part);
        const double px     = LocalParticle_get_px(part);
        const double y      = LocalParticle_get_y(part);
        const double py     = LocalParticle_get_py(part);
        const double tau    = LocalParticle_get_zeta(part) / beta0;
        const double ptau   = LocalParticle_get_ptau(part);
        const double ax     = LocalParticle_get_ax(part);
        const double ay     = LocalParticle_get_ay(part);
        double z[6] = {x, px, y, py, tau, ptau};

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
        for (int step = 0; valid && step < nstep; ++step) {
            double k1[6], k2[6], k3[6], k4[6];

            if (HAMILTONIAN_FLOW(&f, beta0, chi, s, z, &flow) != 0) {
                valid = 0;
                break;
            }
            for (int i = 0; i < 6; ++i) k1[i] = flow.rhs[i];
            for (int i = 0; i < 6; ++i) ztmp[i] = z[i] + 0.5 * ds * k1[i];

            if (HAMILTONIAN_FLOW(&f, beta0, chi, s + 0.5 * ds, ztmp, &flow) != 0) {
                valid = 0;
                break;
            }
            for (int i = 0; i < 6; ++i) k2[i] = flow.rhs[i];
            for (int i = 0; i < 6; ++i) ztmp[i] = z[i] + 0.5 * ds * k2[i];

            if (HAMILTONIAN_FLOW(&f, beta0, chi, s + 0.5 * ds, ztmp, &flow) != 0) {
                valid = 0;
                break;
            }
            for (int i = 0; i < 6; ++i) k3[i] = flow.rhs[i];
            for (int i = 0; i < 6; ++i) ztmp[i] = z[i] + ds * k3[i];

            if (HAMILTONIAN_FLOW(&f, beta0, chi, s + ds, ztmp, &flow) != 0) {
                valid = 0;
                break;
            }
            for (int i = 0; i < 6; ++i) k4[i] = flow.rhs[i];
            for (int i = 0; i < 6; ++i) z[i] += ds * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]) / 6.0;

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
        }
    END_PER_PARTICLE_BLOCK
}
