void HAMILTONIAN_FLOW(Expansion *f, const double beta0,
                      double s, const double z[6], HamiltonianFlow *flow) {
    double delta1, delta, ddelta1;
    double q, pix, piy, rad, root;

    memset(flow, 0, sizeof(*flow));

    EVALUATE_EXPANSION(f, z[0], z[2], s, &flow->pot);
    delta_from_ptau(beta0, z[5], &delta, &delta1, &ddelta1);
    q = 1.0 + f->h * z[0];
    pix = z[1] - flow->pot.Ax;
    piy = z[3] - flow->pot.Ay;  /* A_y is zero in this gauge. */
    rad = delta1 * delta1 - pix * pix - piy * piy;
    root = sqrt(rad);

    flow->delta = delta;
    flow->one_plus_delta = delta1;
    flow->radicand = rad;
    flow->root = root;
    flow->H = z[5] / beta0 - q * (root + flow->pot.As);

    flow->rhs[0] = q * pix / root;  // dx/ds = dH/dpx
    flow->rhs[2] = q * piy / root;  // dy/ds = dH/dpy
    flow->rhs[4] = 1.0 / beta0 - q * delta1 * ddelta1 / root;  // dtau/ds = dH/dptau

    flow->rhs[1] = f->h * (root + flow->pot.As)
        + q * (pix * flow->pot.dAx_dx / root + flow->pot.dAs_dx);  // dpx/ds = -dH/dx
    flow->rhs[3] = q * (pix * flow->pot.dAx_dy / root + flow->pot.dAs_dy);  // dpy/ds = -dH/dy
    flow->rhs[5] = 0.0;  // tptau/ds = -dH/dtau, H has no tau-dependence for these static fields.

    flow->grad[0] = -flow->rhs[1];  // dH/dx
    flow->grad[1] =  flow->rhs[0];  // dH/dpx
    flow->grad[2] = -flow->rhs[3];  // dH/dy
    flow->grad[3] =  flow->rhs[2];  // dH/dpy
    flow->grad[4] = -flow->rhs[5];  // dH/dtau
    flow->grad[5] =  flow->rhs[4];  // dH/dptau
    flow->dH_ds = -q * (pix * flow->pot.dAx_ds / root + flow->pot.dAs_ds);   
}

#ifndef CONCATDATA2
#define CONCATDATA2(a,b) a##b
#endif
#ifndef CONCATDATA
#define CONCATDATA(a,b) CONCATDATA2(a,b)
#endif

void TRACK_EXPANSION(
    FIELDEXPANSIONDATA el,
    LocalParticle* part0)
{

    const double nstep  = CONCATDATA(DATA, _get_nstep(el));
    const double ds     = CONCATDATA(DATA, _get_ds(el));

    Expansion f;
    f.ny         = CONCATDATA(DATA, _get_ny)(el);
    f.ncoef      = CONCATDATA(DATA, _get__ncoef)(el);
    f.na         = CONCATDATA(DATA, _get_na)(el);
    f.nb         = CONCATDATA(DATA, _get_nb)(el);
    f.deg        = CONCATDATA(DATA, _get_deg)(el);
    f.mmin       = CONCATDATA(DATA, _get__mmin)(el);
    f.mmax       = CONCATDATA(DATA, _get__mmax)(el);
    f.moff       = CONCATDATA(DATA, _get__moff)(el);
    f.nm         = CONCATDATA(DATA, _get__nm)(el);
    f.qemin      = CONCATDATA(DATA, _get__qemin)(el);
    f.nq         = CONCATDATA(DATA, _get__nq)(el);
    f.h          = CONCATDATA(DATA, _get_h)(el);
    f.straight   = CONCATDATA(DATA, _get_straight)(el);
    f.c          = (double *)CONCATDATA(DATA, _getp__c)(el);
    f.V          = (double *)CONCATDATA(DATA, _getp__V)(el);
    f.D1         = (double *)CONCATDATA(DATA, _getp__D1)(el);
    f.D2         = (double *)CONCATDATA(DATA, _getp__D2)(el);
    f.Q          = (double *)CONCATDATA(DATA, _getp__Q)(el);

    HamiltonianFlow flow;
    FieldValue v;

    START_PER_PARTICLE_BLOCK(part0, part);
        const double beta0  = LocalParticle_get_beta0(part);

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
        EVALUATE_EXPANSION(&f, z[0], z[2], 0, &v);
        z[1] += v.Ax - ax;
        z[3] += v.Ay - ay;

        double s = 0;
        double ztmp[6];
        for (int step = 0; step < nstep; ++step) {
            double k1[6], k2[6], k3[6], k4[6];

            HAMILTONIAN_FLOW(&f, beta0, s, z, &flow);
            memcpy(k1, flow.rhs, sizeof(k1));
            for (int i = 0; i < 6; ++i) ztmp[i] = z[i] + 0.5 * ds * k1[i];

            HAMILTONIAN_FLOW(&f, beta0, s + 0.5 * ds, ztmp, &flow);
            memcpy(k2, flow.rhs, sizeof(k2));
            for (int i = 0; i < 6; ++i) ztmp[i] = z[i] + 0.5 * ds * k2[i];

            HAMILTONIAN_FLOW(&f, beta0, s + 0.5 * ds, ztmp, &flow);
            memcpy(k3, flow.rhs, sizeof(k3));
            for (int i = 0; i < 6; ++i) ztmp[i] = z[i] + ds * k3[i];

            HAMILTONIAN_FLOW(&f, beta0, s + ds, ztmp, &flow);
            memcpy(k4, flow.rhs, sizeof(k4));
            for (int i = 0; i < 6; ++i) z[i] += ds * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]) / 6.0;

            s += ds;
        }

        // Back to zero vector potential for next element
        EVALUATE_EXPANSION(&f, z[0], z[2], s, &v);
        z[1] -= v.Ax;
        z[3] -= v.Ay;

        LocalParticle_set_x(part, z[0]);
        LocalParticle_set_px(part, z[1]);
        LocalParticle_set_y(part, z[2]);
        LocalParticle_set_py(part, z[3]);
        LocalParticle_set_zeta(part, z[4]*beta0);
        LocalParticle_set_ptau(part, z[5]);
        LocalParticle_set_ax(part, 0);
        LocalParticle_set_ay(part, 0);
        LocalParticle_add_to_s(part, ds*nstep);
    END_PER_PARTICLE_BLOCK
}
