#ifndef XTRACK_TRACK_FIELDEXPANSION_H
#define XTRACK_TRACK_FIELDEXPANSION_H

typedef struct {
    int ny;      /* requested output order in y */
    int ncoef;   /* stored phi_i coefficients: 0..ny+1 */
    int na, nb, deg;
    int mmin, mmax, moff, nm;
    int qemin, nq;
    double h;
    double *c;   /* c[i,m,k], polynomial coeff of s^k in q^m term */
    double *V;   /* scratch: c[i,m](s)   */
    double *D1;   /* scratch: d_s c[i,m]  */
    double *D2;   /* scratch: d2_s c[i,m]  */
    double *Q;   /* scratch: q^e, e=qemin.. */
} Expansion;

typedef struct {
    double phi;
    double Bx, By, Bs;
    double Ax, Ay, As;
    double dAx_dx, dAx_dy, dAx_ds;
    double dAs_dx, dAs_dy, dAs_ds;
} FieldValue;

typedef struct {
    double H;
    double delta;
    double one_plus_delta;
    double radicand;
    double root;
    double grad[6];  /* dH/d{x,px,y,py,tau,ptau} */
    double rhs[6];   /* canonical flow dz/ds */
    double dH_ds;    /* explicit derivative at fixed canonical variables */
    FieldValue pot;
} HamiltonianFlow;

static inline const double *ccptr(const Expansion *f, int i, int m) {
    return f->c + (((size_t)i * (size_t)f->nm + (size_t)m) * (size_t)(f->deg + 1));
}

static inline void poly_eval_d2(const double *p, int deg, double s, double *v, double *d1, double *d2) {
    double a = p[deg], b = 0.0, c=0.0;
    for (int k = deg - 1; k >= 0; --k) {
        c = c * s + 2.0 * b;
        b = b * s + a;
        a = a * s + p[k];
    }
    *v = a;
    *d1 = b;
    *d2 = c;
}

static void fs_prepare_s(Expansion *f, double s) {
    for (int i = 0; i < f->ncoef; ++i) {
        for (int m = 0; m < f->nm; ++m) {
            poly_eval_d2(ccptr(f, i, m), f->deg, s,
                         &f->V[i * f->nm + m],
                         &f->D1[i * f->nm + m],
                         &f->D2[i * f->nm + m]);
        }
    }
}

void delta_from_ptau(const double beta0, double ptau,
                     double *delta, double *delta1, double *ddelta1) {
    {
        const double r = 1.0 + 2.0 * ptau / beta0 + ptau * ptau;
        *delta1 = sqrt(r);
        *delta = *delta1 - 1.0;
        *ddelta1 = (1.0 / beta0 + ptau) / (*delta1);
    }
}



int evaluate_expansion(Expansion *f, double x, double y, double s, FieldValue *out) {
    const double q = 1.0 + f->h * x;
    if (q == 0.0) return -1; /* singular chart */

    memset(out, 0, sizeof(*out));

    double *V = f->V;
    double *D1 = f->D1;
    double *D2 = f->D2;
    double *Q = f->Q;

    fs_prepare_s(f, s);

    /* q powers from e = mmin-1 .. mmax+2 */
    Q[0] = pow(q, (double)f->qemin);
    for (int t = 1; t < f->nq; ++t) Q[t] = Q[t - 1] * q;
    #define QPOW(E) Q[(E) - f->qemin]

    /* As(x,0,s) 
    = 1/(1+hx) int_0^x dx' *(1+hx) By(x',0,s) 
    = 1/qh int_1^q dq' q' phi_1(q',s)
    = 1/qh sum_m c[1,m] q^(m+2)/(m+2) - 1/q sum_m c[1,m] 1/(m+2) */
    if (f->ncoef > 1) {
        for (int m = 0; m <= f->mmax; ++m) {
            int j = m + f->moff;
            const double c1m = V[1 * f->nm + j];  /* c[1,m] */
            const double dc1m = f->D1[1 * f->nm + j];  /* c[1,m]'*/
            const double g = QPOW(m + 1) - QPOW(-1);
            const double den = f->h * (double)(m + 2);
            if (c1m != 0.0) {
                out->As += c1m * g / den;
                out->dAs_dx += c1m * (((double)(m + 1)) * QPOW(m) + QPOW(-2)) / (double)(m + 2);
            }
            if (dc1m != 0.0) out->dAs_ds += dc1m * g / den;            
        }
    }

    double yi = 1.0; /* y^i / i! */
    for (int i = 0; i <= f->ny; ++i) {
        double sphi = 0.0, gx = 0.0, gs = 0.0, gy = 0.0;
        double dgx_dx = 0.0, dgx_ds = 0.0;
        double dgs_dx = 0.0, dgs_ds = 0.0;

        for (int m = f->mmin; m <= f->mmax; ++m) {
            const int j = m + f->moff;
            const double cim = V[i * f->nm + j];         /* c[i,m] */
            const double ci1m = V[(i + 1) * f->nm + j];  /* c[i+1,m] */
            const double dcim = D1[i * f->nm + j];       /* c[i,m]' */
            const double ddcim = D2[i * f->nm + j];      /* c[i,m]'' */
            const double qm = QPOW(m);                   /* q^m */
            const double qm1 = QPOW(m - 1);              /* q^(m-1) */
            const double qm2 = QPOW(m - 2);              /* q^(m-2) */

            sphi += cim * qm;                            /* c[i,m] q^m */
            gx   += f->h * (double)m * cim * qm1;        /* h m c[i,m] q^(m-1) */
            gy   += ci1m * qm;         /* c[i+1,m] q^m */
            gs   += dcim * qm1;                          /* c[i,m]' q^(m-1) */

            dgx_dx += f->h * f->h * (double)m * (double)(m-1) * cim * qm2;
            dgx_ds += f->h * (double)m * dcim * qm1;
            dgs_dx += f->h * (double)(m-1) * dcim * qm2;
            dgs_ds += ddcim * qm1;
        }

        out->phi += sphi * yi;  /* c[i,m] q^m y^i/i!*/
        out->Bx  -= gx   * yi;  /* -h m c[i,m] q^(m-1) y^i/i! */
        out->By  -= gy   * yi;  /* -c[i+1,m] q^m y^i/i! */
        out->Bs  -= gs   * yi;  /* -c[i,m]' q^(m-1) y^i/i! */

        /* A_x, A_s through order ny in y: need i = 0..ny-1 */
        if (i < f->ny) {
            double yi1 = yi * y / (double)(i + 1);  /* y^(i+1)/(i+1)! */
            out->Ax += gs * yi1;  /* -c[i,m]' q^(m-1) y^(i+1)/(i+1)! */
            out->As -= gx * yi1;  /* -h m c[i,m] q^(m-1) y^i/i! *//* -h m c[i,m] q^(m-1) y^i/i! */
            out->dAx_dx += dgs_dx  * yi1;
            out->dAx_ds += dgs_ds  * yi1;
            out->dAs_dx += -dgx_dx * yi1;
            out->dAs_ds += -dgx_ds * yi1;
        }

        yi *= y / (double)(i + 1);
    }

    out->dAx_dy = -out->Bs;
    out->dAs_dy =  out->Bx;

    return 0;
    #undef QPOW
}

void hamiltonian_flow(Expansion *f, const double beta0,
                      double s, const double z[6], HamiltonianFlow *flow) {
    double delta1, delta, ddelta1;
    double q, pix, piy, rad, root;

    memset(flow, 0, sizeof(*flow));

    evaluate_expansion(f, z[0], z[2], s, &flow->pot);
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

void FieldExpansion_track_local_particle(
    FieldExpansionData el,
    LocalParticle* part0)
{

    const double nstep  = FieldExpansionData_get_nstep(el);
    const double ds     = FieldExpansionData_get_ds(el);

    Expansion f;
    f.ny    = FieldExpansionData_get_ny(el);
    f.ncoef = FieldExpansionData_get__ncoef(el);
    f.na    = FieldExpansionData_get_na(el);
    f.nb    = FieldExpansionData_get_nb(el);
    f.deg   = FieldExpansionData_get_deg(el);
    f.mmin  = FieldExpansionData_get__mmin(el);
    f.mmax  = FieldExpansionData_get__mmax(el);
    f.moff  = FieldExpansionData_get__moff(el);
    f.nm    = FieldExpansionData_get__nm(el);
    f.qemin = FieldExpansionData_get__qemin(el);
    f.nq    = FieldExpansionData_get__nq(el);
    f.h     = FieldExpansionData_get_h(el);
    f.c     = (double *)FieldExpansionData_getp__c(el);
    f.V     = (double *)FieldExpansionData_getp__V(el);
    f.D1    = (double *)FieldExpansionData_getp__D1(el);
    f.D2    = (double *)FieldExpansionData_getp__D2(el);
    f.Q     = (double *)FieldExpansionData_getp__Q(el);

    HamiltonianFlow flow;

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
        FieldValue v;
        evaluate_expansion(&f, z[0], z[2], 0, &v);
        z[1] += v.Ax - ax;
        z[3] += v.Ay - ay;

        double s = 0;
        double ztmp[6];
        for (int step = 0; step < nstep; ++step) {
            double k1[6], k2[6], k3[6], k4[6];

            hamiltonian_flow(&f, beta0, s, z, &flow);
            memcpy(k1, flow.rhs, sizeof(k1));
            for (int i = 0; i < 6; ++i) ztmp[i] = z[i] + 0.5 * ds * k1[i];

            hamiltonian_flow(&f, beta0, s + 0.5 * ds, ztmp, &flow);
            memcpy(k2, flow.rhs, sizeof(k2));
            for (int i = 0; i < 6; ++i) ztmp[i] = z[i] + 0.5 * ds * k2[i];

            hamiltonian_flow(&f, beta0, s + 0.5 * ds, ztmp, &flow);
            memcpy(k3, flow.rhs, sizeof(k3));
            for (int i = 0; i < 6; ++i) ztmp[i] = z[i] + ds * k3[i];

            hamiltonian_flow(&f, beta0, s + ds, ztmp, &flow);
            memcpy(k4, flow.rhs, sizeof(k4));
            for (int i = 0; i < 6; ++i) z[i] += ds * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]) / 6.0;

            s += ds;
        }

        // Back to zero vector potential for next element
        evaluate_expansion(&f, z[0], z[2], 0, &v);
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

#endif