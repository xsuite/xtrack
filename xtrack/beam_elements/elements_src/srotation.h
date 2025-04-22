// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_SROTATION_H
#define XTRACK_SROTATION_H

/*gpufun*/
void SRotation_track_local_particle(SRotationData el, LocalParticle* part0){

    double sin_z = SRotationData_get_sin_z(el);
    double cos_z = SRotationData_get_cos_z(el);

    #ifdef XSUITE_BACKTRACK
        sin_z = -sin_z;
    #endif

    //start_per_particle_block (part0->part)
        SRotation_single_particle(part, sin_z, cos_z);
    //end_per_particle_block

    /* Spin tracking is disabled by the synrad compile flag */
    #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        // Rotate spin
        //start_per_particle_block (part0->part)
            double const spin_x_0 = LocalParticle_get_spin_x(part);
            double const spin_y_0 = LocalParticle_get_spin_y(part);
            if ((spin_x_0 != 0) || (spin_y_0 != 0)){
                double const spin_x_1 = cos_z*spin_x_0 + sin_z*spin_y_0;
                double const spin_y_1 = -sin_z*spin_x_0 + cos_z*spin_y_0;
                LocalParticle_set_spin_x(part, spin_x_1);
                LocalParticle_set_spin_y(part, spin_y_1);
            }
        //end_per_particle_block
    #endif

}

#endif /* XTRACK_SROTATION_H */
