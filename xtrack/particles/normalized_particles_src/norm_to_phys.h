#ifndef XNLBD_NORM_TO_PHYS_H
#define XNLBD_NORM_TO_PHYS_H

/*gpukern*/
void norm_to_phys(ParticlesData part, NormalizedParticlesData norm_part, const int64_t nelem)
{
    for (int ii = 0; ii < nelem; ii++){ // vectorize_over ii nelem

        const int at_element = NormalizedParticlesData_get_force_at_element(norm_part) != -1 ? 0 : ParticlesData_get_at_element(part, ii);

        if (at_element >= 0 && at_element < NormalizedParticlesData_get_num_elements(norm_part))
        {
            const double part_beta0 = ParticlesData_get_beta0(part, ii);
            const double part_gamma0 = ParticlesData_get_gamma0(part, ii);
            const double *closed_orbit_data = NormalizedParticlesData_getp2_closed_orbit_data(norm_part, at_element, 0);
            const double *w = NormalizedParticlesData_getp3_w(norm_part, at_element, 0, 0);

            const double gemitt_x = isnan(NormalizedParticlesData_get_nemitt_x(norm_part)) ? 1.0 : NormalizedParticlesData_get_nemitt_x(norm_part) / part_beta0 / part_gamma0;
            const double gemitt_y = isnan(NormalizedParticlesData_get_nemitt_y(norm_part)) ? 1.0 : NormalizedParticlesData_get_nemitt_y(norm_part) / part_beta0 / part_gamma0;
            const double gemitt_zeta = isnan(NormalizedParticlesData_get_nemitt_zeta(norm_part)) ? 1.0 : NormalizedParticlesData_get_nemitt_zeta(norm_part) / part_beta0 / part_gamma0;

            const double x_norm = NormalizedParticlesData_get_x_norm(norm_part, ii) * sqrt(gemitt_x);
            const double px_norm = NormalizedParticlesData_get_px_norm(norm_part, ii) * sqrt(gemitt_x);
            const double y_norm = NormalizedParticlesData_get_y_norm(norm_part, ii) * sqrt(gemitt_y);
            const double py_norm = NormalizedParticlesData_get_py_norm(norm_part, ii) * sqrt(gemitt_y);
            const double zeta_norm = NormalizedParticlesData_get_zeta_norm(norm_part, ii) * sqrt(gemitt_zeta);
            const double pzeta_norm = NormalizedParticlesData_get_pzeta_norm(norm_part, ii) * sqrt(gemitt_zeta);

            ParticlesData_set_x(part, ii,
                                (w[0] * x_norm + w[1] * px_norm + w[2] * y_norm + w[3] * py_norm + w[4] * zeta_norm + w[5] * pzeta_norm) + closed_orbit_data[0]);
            ParticlesData_set_px(part, ii,
                                 (w[6] * x_norm + w[7] * px_norm + w[8] * y_norm + w[9] * py_norm + w[10] * zeta_norm + w[11] * pzeta_norm) + closed_orbit_data[1]);
            ParticlesData_set_y(part, ii,
                                (w[12] * x_norm + w[13] * px_norm + w[14] * y_norm + w[15] * py_norm + w[16] * zeta_norm + w[17] * pzeta_norm) + closed_orbit_data[2]);
            ParticlesData_set_py(part, ii,
                                 (w[18] * x_norm + w[19] * px_norm + w[20] * y_norm + w[21] * py_norm + w[22] * zeta_norm + w[23] * pzeta_norm) + closed_orbit_data[3]);
            ParticlesData_set_zeta(part, ii,
                                   (w[24] * x_norm + w[25] * px_norm + w[26] * y_norm + w[27] * py_norm + w[28] * zeta_norm + w[29] * pzeta_norm) + closed_orbit_data[4]);

            // since we are doing things "raw" without the standard "track" function,
            // we need to set all energy deviation variables ourselves

            const double ptau = (w[30] * x_norm + w[31] * px_norm + w[32] * y_norm + w[33] * py_norm + w[34] * zeta_norm + w[35] * pzeta_norm) * part_beta0 + closed_orbit_data[5];
            const double delta = sqrt(ptau * ptau + 2 * ptau / (part_beta0 * part_beta0) + 1) - 1;
            const double delta_beta0 = delta * part_beta0;
            const double ptau_beta0 = sqrt(delta_beta0 * delta_beta0 + 2 * delta_beta0 * part_beta0 + 1) - 1;

            const double new_rvv = (1 + delta) / (1 + ptau_beta0);
            const double new_rpp = 1 / (1 + delta);

            ParticlesData_set_ptau(part, ii, ptau);
            ParticlesData_set_delta(part, ii, delta);
            ParticlesData_set_rvv(part, ii, new_rvv);
            ParticlesData_set_rpp(part, ii, new_rpp);
        }
        else
        {
            // set everything to xt.particles.LAST_INVALID_STATE = -999999999
            ParticlesData_set_x(part, ii, -999999999.);
            ParticlesData_set_px(part, ii, -999999999.);
            ParticlesData_set_y(part, ii, -999999999.);
            ParticlesData_set_py(part, ii, -999999999.);
            ParticlesData_set_zeta(part, ii, -999999999.);
            ParticlesData_set_ptau(part, ii, -999999999.);
            ParticlesData_set_delta(part, ii, -999999999.);
        }
        // also copy the particle_id, state and at_turn just to be sure
        ParticlesData_set_particle_id(
            part, ii, NormalizedParticlesData_get_particle_id(norm_part, ii));
        ParticlesData_set_state(
            part, ii, NormalizedParticlesData_get_state(norm_part, ii));
        ParticlesData_set_at_turn(
            part, ii, NormalizedParticlesData_get_at_turn(norm_part, ii));
        ParticlesData_set_at_element(
            part, ii, NormalizedParticlesData_get_at_element(norm_part, ii));
    } // end_vectorize
}

#endif /* XNLBD_NORM_TO_PHYS_H */