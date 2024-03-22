#ifndef XTRACK_TRACK_DIPOLEEDGE_LINEAR_H
#define XTRACK_TRACK_DIPOLEEDGE_LINEAR_H

/*gpufun*/
void DipoleEdgeLinear_single_particle(LocalParticle* part, double r21, double r43){

    double const x = LocalParticle_get_x(part);
    double const y = LocalParticle_get_y(part);
    double const chi = LocalParticle_get_chi(part);

    LocalParticle_add_to_px(part, chi * r21*x);
    LocalParticle_add_to_py(part, chi * r43*y);

}

#endif