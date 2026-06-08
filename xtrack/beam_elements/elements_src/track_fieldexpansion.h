#ifndef XTRACK_TRACK_FIELDEXPANSION_H
#define XTRACK_TRACK_FIELDEXPANSION_H

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


void evaluate_expansion(Expansion *f, double x, double y, double s, FieldValue *out) {
}

void hamiltonian_flow(Expansion *f, const double beta0,
                      double s, const double z[6], HamiltonianFlow *flow) {
    memset(flow, 0, sizeof(*flow));    
}

void FieldExpansion_track_local_particle(
    FieldExpansionData el,
    LocalParticle* part)
{

    // HOW CAN I INITIALIZE AND KEEP THIS?
    Expansion f;

    HamiltonianFlow flow;

    const double nstep = FieldExpansionData_get_nstep(el);
    const double ds = FieldExpansionData_get_ds(el);
    printf("nstep = %e\n", nstep);
    const double beta0 = LocalParticle_get_beta0(part);

    const double x = LocalParticle_get_x(part);
    const double px = LocalParticle_get_px(part);
    const double y = LocalParticle_get_y(part);
    const double py = LocalParticle_get_py(part);
    const double tau = LocalParticle_get_zeta(part) / beta0;
    const double ptau = LocalParticle_get_ptau(part);

    // TO BE DONE 
    // Momentum has to be continuous, vector potential discontinuous, update canonical momentum
    // FieldValue v;
    // evaluate_expansion(f, z[0], z[2], s, &v);
    // z[1] += v.Ax;
    // z[3] += v.Ay;

    double s = 0;
    double z[6] = {x, px, y, py, tau, ptau};
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

    LocalParticle_set_x(part, z[0]);
    LocalParticle_set_px(part, z[1]);
    LocalParticle_set_y(part, z[2]);
    LocalParticle_set_py(part, z[3]);
    LocalParticle_set_zeta(part, z[4]*beta0);
    LocalParticle_set_ptau(part, z[5]);
}

#endif