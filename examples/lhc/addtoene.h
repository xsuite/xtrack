double LocalParticle_get_energy0(LocalParticle* part){

    double const p0c = LocalParticle_get_p0c(part);
    double const m0  = LocalParticle_get_mass0(part);

    return sqrt( p0c * p0c + m0 * m0 );
}

void LocalParticle_add_to_energy(LocalParticle* part, double delta_energy){

    double const beta0 = LocalParticle_get_beta0(part);
    double const delta_beta0 = LocalParticle_get_delta(part) * beta0;

    double const ptau_beta0 =
        delta_energy / LocalParticle_get_energy0(part) +
        sqrt( delta_beta0 * delta_beta0 + 2.0 * delta_beta0 * beta0
                + 1. ) - 1.;

    double const ptau   = ptau_beta0 / beta0;
    double const psigma = ptau / beta0;
    double const delta = sqrt( ptau * ptau + 2. * psigma + 1 ) - 1;

    double const one_plus_delta = delta + 1.;
    double const rvv = one_plus_delta / ( 1. + ptau_beta0 );

    LocalParticle_set_delta(part, delta );
    LocalParticle_set_psigma(part, psigma );
    LocalParticle_scale_zeta(part,
        rvv / LocalParticle_get_rvv(part));

    LocalParticle_set_rvv(part, rvv );
    LocalParticle_set_rpp(part, 1. / one_plus_delta );
}
