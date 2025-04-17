#ifndef XNLBD_PHYS_TO_NORM_H
#define XNLBD_PHYS_TO_NORM_H

/*gpukern*/
void phys_to_norm(ParticlesData part, NormalizedParticlesData norm_part, const int64_t nelem)
{
    for (int ii = 0; ii < nelem; ii++){ // vectorize_over ii nelem

        const int at_element = NormalizedParticlesData_get_force_at_element(norm_part) != -1 ? 0 : ParticlesData_get_at_element(part, ii);

        if (at_element >= 0 && at_element < NormalizedParticlesData_get_num_elements(norm_part))
        {
            const double particle_gamma0 = ParticlesData_get_gamma0(part, ii);
            const double particle_beta0 = ParticlesData_get_beta0(part, ii);
            const double *closed_orbit_data = NormalizedParticlesData_getp2_closed_orbit_data(norm_part, at_element, 0);
            const double *w_inv = NormalizedParticlesData_getp3_w_inv(norm_part, at_element, 0, 0);

            const double gemitt_x = isnan(NormalizedParticlesData_get_nemitt_x(norm_part)) ? 1.0 : NormalizedParticlesData_get_nemitt_x(norm_part) / particle_beta0 / particle_gamma0;
            const double gemitt_y = isnan(NormalizedParticlesData_get_nemitt_y(norm_part)) ? 1.0 : NormalizedParticlesData_get_nemitt_y(norm_part) / particle_beta0 / particle_gamma0;
            const double gemitt_zeta = isnan(NormalizedParticlesData_get_nemitt_zeta(norm_part)) ? 1.0 : NormalizedParticlesData_get_nemitt_zeta(norm_part) / particle_beta0 / particle_gamma0;

            const double x_norm = ParticlesData_get_x(part, ii) - closed_orbit_data[0];
            const double px_norm = ParticlesData_get_px(part, ii) - closed_orbit_data[1];
            const double y_norm = ParticlesData_get_y(part, ii) - closed_orbit_data[2];
            const double py_norm = ParticlesData_get_py(part, ii) - closed_orbit_data[3];
            const double zeta_norm = ParticlesData_get_zeta(part, ii) - closed_orbit_data[4];
            const double pzeta_norm = (ParticlesData_get_ptau(part, ii) - closed_orbit_data[5]) / particle_beta0;

            NormalizedParticlesData_set_x_norm(norm_part, ii,
                (w_inv[0] * x_norm + w_inv[1] * px_norm + w_inv[2] * y_norm +
                w_inv[3] * py_norm + w_inv[4] * zeta_norm + w_inv[5] * pzeta_norm) / sqrt(gemitt_x));
            NormalizedParticlesData_set_px_norm(norm_part, ii,
                (w_inv[6] * x_norm + w_inv[7] * px_norm + w_inv[8] * y_norm +
                w_inv[9] * py_norm + w_inv[10] * zeta_norm + w_inv[11] * pzeta_norm) / sqrt(gemitt_x));
            NormalizedParticlesData_set_y_norm(norm_part, ii,
                (w_inv[12] * x_norm + w_inv[13] * px_norm + w_inv[14] * y_norm +
                w_inv[15] * py_norm + w_inv[16] * zeta_norm + w_inv[17] * pzeta_norm) / sqrt(gemitt_y));
            NormalizedParticlesData_set_py_norm(norm_part, ii,
                (w_inv[18] * x_norm + w_inv[19] * px_norm + w_inv[20] * y_norm +
                w_inv[21] * py_norm + w_inv[22] * zeta_norm + w_inv[23] * pzeta_norm) / sqrt(gemitt_y));
            NormalizedParticlesData_set_zeta_norm(norm_part, ii,
                (w_inv[24] * x_norm + w_inv[25] * px_norm + w_inv[26] * y_norm +
                w_inv[27] * py_norm + w_inv[28] * zeta_norm + w_inv[29] * pzeta_norm) / sqrt(gemitt_zeta));
            NormalizedParticlesData_set_pzeta_norm(norm_part, ii,
                (w_inv[30] * x_norm + w_inv[31] * px_norm + w_inv[32] * y_norm +
                w_inv[33] * py_norm + w_inv[34] * zeta_norm + w_inv[35] * pzeta_norm) / sqrt(gemitt_zeta));            
        }
        else
        {
            // set everything to xt.particles.LAST_INVALID_STATE = -999999999
            NormalizedParticlesData_set_x_norm(norm_part, ii, -999999999.);
            NormalizedParticlesData_set_px_norm(norm_part, ii, -999999999.);
            NormalizedParticlesData_set_y_norm(norm_part, ii, -999999999.);
            NormalizedParticlesData_set_py_norm(norm_part, ii, -999999999.);
            NormalizedParticlesData_set_zeta_norm(norm_part, ii, -999999999.);
            NormalizedParticlesData_set_pzeta_norm(norm_part, ii, -999999999.);
        }
        // also copy the particle_id, state, at_turn and at_element
        NormalizedParticlesData_set_particle_id(
            norm_part, ii, ParticlesData_get_particle_id(part, ii));
        NormalizedParticlesData_set_state(
            norm_part, ii, ParticlesData_get_state(part, ii));
        NormalizedParticlesData_set_at_turn(
            norm_part, ii, ParticlesData_get_at_turn(part, ii));
        NormalizedParticlesData_set_at_element(
            norm_part, ii, ParticlesData_get_at_element(part, ii));
    } // end_vectorize
}

#endif /* XNLBD_PHYS_TO_NORM_H */