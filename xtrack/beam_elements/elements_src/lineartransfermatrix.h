// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_LINEARTRANSFERMATRIX_H
#define XTRACK_LINEARTRANSFERMATRIX_H

/*gpufun*/
void remove_closed_orbit(
    LocalParticle* part0, double const x_ref, double const px_ref,
    double const y_ref, double const py_ref){

    //start_per_particle_block (part0->part)
        // Remove closed orbit
        LocalParticle_add_to_x(part, -x_ref);
        LocalParticle_add_to_px(part, -px_ref);
        LocalParticle_add_to_y(part, -y_ref);
        LocalParticle_add_to_py(part, -py_ref);
    //end_per_particle_block

}

/*gpufun*/
void add_closed_orbit(
    LocalParticle* part0, double const x_ref, double const px_ref,
    double const y_ref, double const py_ref){

    //start_per_particle_block (part0->part)
        // Add closed orbit
        LocalParticle_add_to_x(part, x_ref);
        LocalParticle_add_to_px(part, px_ref);
        LocalParticle_add_to_y(part, y_ref);
        LocalParticle_add_to_py(part, py_ref);
    //end_per_particle_block

}

/*gpufun*/
void remove_dispersion(
    LocalParticle* part0, double const disp_x_0, double const disp_px_0,
    double const disp_y_0, double const disp_py_0){

    //start_per_particle_block (part0->part)

        // Remove dispersion
        // Symplecticity correction (not working, to be investigated)
        // LocalParticle_add_to_zeta(part, (
        //     disp_px_0 * LocalParticle_get_x(part)
        //     - disp_x_0 * LocalParticle_get_px(part)
        //     + disp_py_0 * LocalParticle_get_y(part)
        //     - disp_y_0 * LocalParticle_get_py(part)
        //     )/LocalParticle_get_rvv(part));
        double const delta = LocalParticle_get_delta(part);
        LocalParticle_add_to_x(part, -disp_x_0 * delta);
        LocalParticle_add_to_px(part, -disp_px_0 * delta);
        LocalParticle_add_to_y(part, -disp_y_0 * delta);
        LocalParticle_add_to_py(part, -disp_py_0 * delta);

    //end_per_particle_block

}

/*gpufun*/
void add_dispersion(
    LocalParticle* part0, double const disp_x_1, double const disp_px_1,
    double const disp_y_1, double const disp_py_1){

    //start_per_particle_block (part0->part)

        // Add dispersion
        // Symplecticity correction (not working, to be investigated)
        // LocalParticle_add_to_zeta(part, (
        //     disp_px_1 * LocalParticle_get_x(part)
        //     - disp_x_1 * LocalParticle_get_px(part)
        //     + disp_py_1 * LocalParticle_get_y(part)
        //     - disp_y_1 * LocalParticle_get_py(part)
        //     )/LocalParticle_get_rvv(part));
        double const delta = LocalParticle_get_delta(part);
        LocalParticle_add_to_x(part, disp_x_1 * delta);
        LocalParticle_add_to_px(part, disp_px_1 * delta);
        LocalParticle_add_to_y(part, disp_y_1 * delta);
        LocalParticle_add_to_py(part, disp_py_1 * delta);

    //end_per_particle_block

}

/*gpufun*/
void transverse_motion(LocalParticle *part0,
    double const qx, double const qy,
    double const chroma_x, double const chroma_y,
    double const detx_x, double const detx_y, double const dety_x, double const dety_y,
    double const alpha_x_0, double const beta_x_0, double const alpha_y_0, double const beta_y_0,
    double const alpha_x_1, double const beta_x_1, double const alpha_y_1, double const beta_y_1){

    int64_t detuning;
    double sin_x, cos_x, sin_y, cos_y;
    if (chroma_x != 0.0 || chroma_y != 0.0 ||
        detx_x != 0.0 || detx_y != 0.0 || dety_x != 0.0 || dety_y != 0.0){
        detuning = 1;
    }
    else{
        detuning = 0;
        sin_x = sin(2 * PI * qx);
        cos_x = cos(2 * PI * qx);
        sin_y = sin(2 * PI * qy);
        cos_y = cos(2 * PI * qy);
    }

    double const sqrt_beta_prod_x = sqrt(beta_x_1 * beta_x_0);
    double const sqrt_beta_prod_y = sqrt(beta_y_1 * beta_y_0);

    double const sqrt_beta_ratio_x = sqrt(beta_x_1 / beta_x_0);
    double const sqrt_beta_ratio_y = sqrt(beta_y_1 / beta_y_0);

    //start_per_particle_block (part0->part)

        if (detuning){
            double const J_x = 0.5 * (
                (1.0 + alpha_x_0 * alpha_x_0) / beta_x_0
                    * LocalParticle_get_x(part)*LocalParticle_get_x(part)
                + 2 * alpha_x_0
                    * LocalParticle_get_x(part)*LocalParticle_get_px(part)
                + beta_x_0
                    * LocalParticle_get_px(part)*LocalParticle_get_px(part));
            double const J_y = 0.5 * (
                (1.0 + alpha_y_0 * alpha_y_0) /beta_y_0
                    * LocalParticle_get_y(part)*LocalParticle_get_y(part)
                + 2*alpha_y_0
                    * LocalParticle_get_y(part)*LocalParticle_get_py(part)
                + beta_y_0
                    * LocalParticle_get_py(part)*LocalParticle_get_py(part));
            double phase = 2*PI*(qx + chroma_x * LocalParticle_get_delta(part)
                                +detx_x * J_x + detx_y * J_y);
            cos_x = cos(phase);
            sin_x = sin(phase);
            phase = 2*PI*(qy + chroma_y * LocalParticle_get_delta(part)
                            +dety_x * J_x + dety_y * J_y);
            cos_y = cos(phase);
            sin_y = sin(phase);
        }

        double const M00_x = sqrt_beta_ratio_x*(cos_x+alpha_x_0*sin_x);
        double const M01_x = sqrt_beta_prod_x*sin_x;
        double const M10_x = ((alpha_x_0-alpha_x_1)*cos_x
                    -(1+alpha_x_0*alpha_x_1)*sin_x
                    )/sqrt_beta_prod_x;
        double const M11_x = (cos_x-alpha_x_1*sin_x)/sqrt_beta_ratio_x;
        double const M00_y = sqrt_beta_ratio_y*(cos_y+alpha_y_0*sin_y);
        double const M01_y = sqrt_beta_prod_y*sin_y;
        double const M10_y = ((alpha_y_0-alpha_y_1)*cos_y
                    -(1+alpha_y_0*alpha_y_1)*sin_y
                    )/sqrt_beta_prod_y;
        double const M11_y = (cos_y-alpha_y_1*sin_y)/sqrt_beta_ratio_y;

        double const x_out = M00_x*LocalParticle_get_x(part) + M01_x * LocalParticle_get_px(part);
        double const px_out = M10_x*LocalParticle_get_x(part) + M11_x * LocalParticle_get_px(part);
        double const y_out = M00_y*LocalParticle_get_y(part) + M01_y * LocalParticle_get_py(part);
        double const py_out = M10_y*LocalParticle_get_y(part) + M11_y * LocalParticle_get_py(part);

        LocalParticle_set_x(part, x_out);
        LocalParticle_set_px(part, px_out);
        LocalParticle_set_y(part, y_out);
        LocalParticle_set_py(part, py_out);
    //end_per_particle_block

}

void longitudinal_motion(LocalParticle *part0,
                         LinearTransferMatrixData el){

    int64_t const mode_flag = LinearTransferMatrixData_get_longitudinal_mode_flag(el);

    if (mode_flag==1){ // linear motion fixed qs
        double const Q_s = LinearTransferMatrixData_get_Q_s(el);
        double const beta_s = LinearTransferMatrixData_get_beta_s(el);
        double const sin_s = sin(2 * PI * Q_s);
        double const cos_s = cos(2 * PI * Q_s);
        //start_per_particle_block (part->part)
            // We set cos_s = 999 if long map is to be skipped
            double const new_zeta = cos_s * LocalParticle_get_zeta(part) - beta_s * sin_s * LocalParticle_get_pzeta(part);
            double const new_pzeta = sin_s * LocalParticle_get_zeta(part) / beta_s + cos_s * LocalParticle_get_pzeta(part);

            LocalParticle_set_zeta(part, new_zeta);
            LocalParticle_update_pzeta(part, new_pzeta);
        //end_per_particle_block

    }
    else if (mode_flag==2){ // non-linear motion

        double const alpha_p =
            LinearTransferMatrixData_get_momentum_compaction_factor(el);
        double const slippage_length =
            LinearTransferMatrixData_get_slippage_length(el);

        //start_per_particle_block (part->part)
            double const gamma0 = LocalParticle_get_gamma0(part);
            double const eta = alpha_p - 1.0 / (gamma0 * gamma0);
            LocalParticle_add_to_zeta(part,
                -0.5 * eta * slippage_length * LocalParticle_get_delta(part));
        //end_per_particle_block

        int64_t const n_rf = LinearTransferMatrixData_len_voltage_rf(el);
        for (int i_rf=0; i_rf<n_rf; i_rf++){

            double const v_rf = LinearTransferMatrixData_get_voltage_rf(el,i_rf);
            double const f_rf = LinearTransferMatrixData_get_frequency_rf(el,i_rf);
            double const lag_rf = LinearTransferMatrixData_get_lag_rf(el,i_rf);

            if (f_rf == 0) continue;

            //start_per_particle_block (part0->part)
                double const K_FACTOR = ( ( double )2.0 *PI ) / C_LIGHT;
                double const   beta0  = LocalParticle_get_beta0(part);
                double const   zeta   = LocalParticle_get_zeta(part);
                double const   q      = fabs(LocalParticle_get_q0(part))
                                        * LocalParticle_get_charge_ratio(part);
                double const   tau    = zeta / beta0;
                double const   phase  = DEG2RAD  * lag_rf - K_FACTOR * f_rf * tau;
                double const energy   = q * v_rf * sin(phase);
                LocalParticle_add_to_energy(part, energy, 1);
            //end_per_particle_block
        }

        //start_per_particle_block (part->part)
            double const gamma0 = LocalParticle_get_gamma0(part);
            double const eta = alpha_p - 1.0 / (gamma0 * gamma0);
            LocalParticle_add_to_zeta(part,
                -0.5 * eta * slippage_length * LocalParticle_get_delta(part));
        //end_per_particle_block
    }
}

/*gpufun*/
void uncorrelated_radiation_damping(LocalParticle *part0,
            double const damping_factor_x, double const damping_factor_y,
            double const damping_factor_s){

    //start_per_particle_block (part0->part)
        LocalParticle_scale_x(part,damping_factor_x);
        LocalParticle_scale_px(part,damping_factor_x);
        LocalParticle_scale_y(part,damping_factor_y);
        LocalParticle_scale_py(part,damping_factor_y);
        LocalParticle_scale_zeta(part,damping_factor_s);
        double delta = LocalParticle_get_delta(part);
        delta *= damping_factor_s;
        LocalParticle_update_delta(part,delta);
    //end_per_particle_block
}

void energy_and_reference_increments(LocalParticle *part0,
    double const energy_increment, double const energy_ref_increment){

    //start_per_particle_block (part0->part)
        // Change energy without change of reference momentum
        if (energy_increment !=0){
        LocalParticle_add_to_energy(part, energy_increment, 1);
        }

        // Change energy reference
        // In the transverse plane de change is smoothed, i.e.
        // both the position and the momentum are scaled,
        // rather than only the momentum.
        if (energy_ref_increment != 0){
            double const new_energy0 = LocalParticle_get_mass0(part)
                *LocalParticle_get_gamma0(part) + energy_ref_increment;
            double const new_p0c = sqrt(new_energy0*new_energy0
            -LocalParticle_get_mass0(part)*LocalParticle_get_mass0(part));
            double const new_beta0 = new_p0c / new_energy0;
            double const new_gamma0 = new_energy0 / LocalParticle_get_mass0(part);
            double const geo_emit_factor = sqrt(LocalParticle_get_beta0(part)
                    *LocalParticle_get_gamma0(part)/new_beta0/new_gamma0);
            LocalParticle_update_p0c(part,new_p0c);
            LocalParticle_scale_x(part,geo_emit_factor);
            LocalParticle_scale_px(part,geo_emit_factor);
            LocalParticle_scale_y(part,geo_emit_factor);
            LocalParticle_scale_py(part,geo_emit_factor);
        }
    //end_per_particle_block

}

void uncorrelated_gaussian_noise(LocalParticle *part0,
                    double const gauss_noise_ampl_x,
                    double const gauss_noise_ampl_px,
                    double const gauss_noise_ampl_y,
                    double const gauss_noise_ampl_py,
                    double const gauss_noise_ampl_zeta,
                    double const gauss_noise_ampl_delta){

        //start_per_particle_block (part0->part)
            double r = RandomNormal_generate(part);
            LocalParticle_add_to_x(part,r*gauss_noise_ampl_x);
            r = RandomNormal_generate(part);
            LocalParticle_add_to_px(part,r*gauss_noise_ampl_px);
            r = RandomNormal_generate(part);
            LocalParticle_add_to_y(part,r*gauss_noise_ampl_y);
            r = RandomNormal_generate(part);
            LocalParticle_add_to_py(part,r*gauss_noise_ampl_py);
            r = RandomNormal_generate(part);
            LocalParticle_add_to_zeta(part,r*gauss_noise_ampl_zeta);
            r = RandomNormal_generate(part);
            double delta = LocalParticle_get_delta(part);
            delta += r*gauss_noise_ampl_delta;
            LocalParticle_update_delta(part,delta);
        //end_per_particle_block

}

/*gpufun*/
void LinearTransferMatrix_track_local_particle(LinearTransferMatrixData el, LocalParticle* part0){


    remove_closed_orbit(part0,
        LinearTransferMatrixData_get_x_ref_0(el),
        LinearTransferMatrixData_get_px_ref_0(el),
        LinearTransferMatrixData_get_y_ref_0(el),
        LinearTransferMatrixData_get_py_ref_0(el));

    remove_dispersion(part0,
        LinearTransferMatrixData_get_disp_x_0(el),
        LinearTransferMatrixData_get_disp_px_0(el),
        LinearTransferMatrixData_get_disp_y_0(el),
        LinearTransferMatrixData_get_disp_py_0(el));

    transverse_motion(part0,
        LinearTransferMatrixData_get_qx(el),
        LinearTransferMatrixData_get_qy(el),
        LinearTransferMatrixData_get_chroma_x(el),
        LinearTransferMatrixData_get_chroma_y(el),
        LinearTransferMatrixData_get_detx_x(el),
        LinearTransferMatrixData_get_detx_y(el),
        LinearTransferMatrixData_get_dety_x(el),
        LinearTransferMatrixData_get_dety_y(el),
        LinearTransferMatrixData_get_alpha_x_0(el),
        LinearTransferMatrixData_get_beta_x_0(el),
        LinearTransferMatrixData_get_alpha_y_0(el),
        LinearTransferMatrixData_get_beta_y_0(el),
        LinearTransferMatrixData_get_alpha_x_1(el),
        LinearTransferMatrixData_get_beta_x_1(el),
        LinearTransferMatrixData_get_alpha_y_1(el),
        LinearTransferMatrixData_get_beta_y_1(el));

    longitudinal_motion(part0, el);

    energy_and_reference_increments(part0,
        LinearTransferMatrixData_get_energy_increment(el),
        LinearTransferMatrixData_get_energy_ref_increment(el));

    if (LinearTransferMatrixData_get_uncorrelated_rad_damping(el) == 1){
        uncorrelated_radiation_damping(part0,
            LinearTransferMatrixData_get_damping_factor_x(el),
            LinearTransferMatrixData_get_damping_factor_y(el),
            LinearTransferMatrixData_get_damping_factor_s(el));
    }

    if (LinearTransferMatrixData_get_uncorrelated_gauss_noise(el) == 1){
        uncorrelated_gaussian_noise(part0,
            LinearTransferMatrixData_get_gauss_noise_ampl_x(el),
            LinearTransferMatrixData_get_gauss_noise_ampl_px(el),
            LinearTransferMatrixData_get_gauss_noise_ampl_y(el),
            LinearTransferMatrixData_get_gauss_noise_ampl_py(el),
            LinearTransferMatrixData_get_gauss_noise_ampl_zeta(el),
            LinearTransferMatrixData_get_gauss_noise_ampl_delta(el));
    }

    add_dispersion(part0,
        LinearTransferMatrixData_get_disp_x_1(el),
        LinearTransferMatrixData_get_disp_px_1(el),
        LinearTransferMatrixData_get_disp_y_1(el),
        LinearTransferMatrixData_get_disp_py_1(el));

    add_closed_orbit(part0,
        LinearTransferMatrixData_get_x_ref_1(el),
        LinearTransferMatrixData_get_px_ref_1(el),
        LinearTransferMatrixData_get_y_ref_1(el),
        LinearTransferMatrixData_get_py_ref_1(el));

    double const length = LinearTransferMatrixData_get_length(el);
    //start_per_particle_block (part0->part)
        LocalParticle_add_to_s(part, length);
    //end_per_particle_block
}

#endif
