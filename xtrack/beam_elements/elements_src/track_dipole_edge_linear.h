#ifndef XTRACK_TRACK_DIPOLEEDGE_LINEAR_H
#define XTRACK_TRACK_DIPOLEEDGE_LINEAR_H


/*gpufun*/
void compute_dipole_edge_linear_coefficients(double const k, double const e1,
                    double const e1_fd, double const hgap, double const fint,
                    double* r21, double* r43){

    double const corr = 2.0 * k * hgap * fint;
    double const r21_out = k * tan(e1);

    printf("e1 = %e\n", e1);
    printf("e1_fd = %e\n", e1_fd);
    printf("r21_out = %e\n", r21_out);

    double const e1_v = e1 + e1_fd;
    double const sin_e1_v = sin(e1_v);
    double const temp = corr / cos(e1_v) * (1.0 + sin_e1_v * sin_e1_v);
    double const r43_out = -k * tan(e1_v - temp);
    printf("r43_out = %e\n", r43_out);

    *r21 = r21_out;
    *r43 = r43_out;

}

/*gpufun*/
void DipoleEdgeLinear_single_particle(LocalParticle* part, double r21, double r43){

    double const x = LocalParticle_get_x(part);
    double const y = LocalParticle_get_y(part);
    double const chi = LocalParticle_get_chi(part);

    LocalParticle_add_to_px(part, chi * r21*x);
    LocalParticle_add_to_py(part, chi * r43*y);

}

#endif