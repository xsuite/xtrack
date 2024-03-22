#ifndef XTRACK_TRACK_DIPOLEEDGE_LINEAR_H
#define XTRACK_TRACK_DIPOLEEDGE_LINEAR_H

/*gpufun*/
void DipoleEdgeLinear_single_particle(LocalParticle* part0, double r21, double r43){

    double const x = LocalParticle_get_x(part0);
    double const y = LocalParticle_get_y(part0);
    double const chi = LocalParticle_get_chi(part0);

    LocalParticle_add_to_px(part0, chi * r21*x);
    LocalParticle_add_to_py(part0, chi * r43*y);

}

#endif