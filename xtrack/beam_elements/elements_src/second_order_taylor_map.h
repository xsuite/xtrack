// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_SECONDORDERTAYLORMAP_H
#define XTRACK_SECONDORDERTAYLORMAP_H

/*gpufun*/
void SecondOrderTaylorMap_track_local_particle(SecondOrderTaylorMapData el,
                                               LocalParticle* part0){

    double const length = SecondOrderTaylorMapData_get_length(el);

    //start_per_particle_block (part0->part)

        double z_in[6];
        double z_out[6];

        z_in[0] = LocalParticle_get_x(part);
        z_in[1] = LocalParticle_get_px(part);
        z_in[2] = LocalParticle_get_y(part);
        z_in[3] = LocalParticle_get_py(part);
        z_in[4] = LocalParticle_get_zeta(part);
        z_in[5] = LocalParticle_get_ptau(part) / LocalParticle_get_beta0(part);

        for (int ii = 0; ii < 6; ii++){
            z_out[ii] = SecondOrderTaylorMapData_get_k(el, ii);
        }

        for (int ii = 0; ii < 6; ii++){
            for (int jj = 0; jj < 6; jj++){
                // printf("ii = %d, jj = %d, R = %e\n", ii, jj, SecondOrderTaylorMapData_get_R(el, ii, jj));
                z_out[ii] += SecondOrderTaylorMapData_get_R(el, ii, jj) * z_in[jj];
            }
        }

        for (int ii = 0; ii < 6; ii++){
            for (int jj = 0; jj < 6; jj++){
                for (int kk = 0; kk < 6; kk++){
                    z_out[ii] += SecondOrderTaylorMapData_get_T(el, ii, jj, kk) * z_in[jj] * z_in[kk];
                }
            }
        }

        LocalParticle_set_x(part, z_out[0]);
        LocalParticle_set_px(part, z_out[1]);
        LocalParticle_set_y(part, z_out[2]);
        LocalParticle_set_py(part, z_out[3]);
        LocalParticle_set_zeta(part, z_out[4]);
        LocalParticle_update_ptau(part, z_out[5] * LocalParticle_get_beta0(part));

        LocalParticle_add_to_s(part, length);


    //end_per_particle_block


    }

#endif