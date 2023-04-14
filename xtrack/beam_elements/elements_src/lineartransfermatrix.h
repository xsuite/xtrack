// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_LINEARTRANSFERMATRIX_H
#define XTRACK_LINEARTRANSFERMATRIX_H


/*gpufun*/
void LinearTransferMatrix_track_local_particle(LinearTransferMatrixData el, LocalParticle* part0){

    int64_t const no_detuning = LinearTransferMatrixData_get_no_detuning(el);
    double const q_x = LinearTransferMatrixData_get_q_x(el);
    double const q_y = LinearTransferMatrixData_get_q_y(el);
    double const chroma_x = LinearTransferMatrixData_get_chroma_x(el);
    double const chroma_y = LinearTransferMatrixData_get_chroma_y(el);
    double const detx_x = LinearTransferMatrixData_get_detx_x(el);
    double const detx_y = LinearTransferMatrixData_get_detx_y(el);
    double const dety_y = LinearTransferMatrixData_get_dety_y(el);
    double const dety_x = LinearTransferMatrixData_get_dety_x(el);

    double const cos_s = LinearTransferMatrixData_get_cos_s(el);
    double const sin_s = LinearTransferMatrixData_get_sin_s(el);
    double const beta_s = LinearTransferMatrixData_get_beta_s(el);

    double const length = LinearTransferMatrixData_get_length(el);

    double const beta_x_0 = LinearTransferMatrixData_get_beta_x_0(el);
    double const beta_y_0 = LinearTransferMatrixData_get_beta_y_0(el);
    double const disp_x_0 = LinearTransferMatrixData_get_disp_x_0(el);
    double const disp_y_0 = LinearTransferMatrixData_get_disp_y_0(el);
    double const disp_px_0 = LinearTransferMatrixData_get_disp_px_0(el);
    double const disp_py_0 = LinearTransferMatrixData_get_disp_py_0(el);
    double const alpha_x_0 = LinearTransferMatrixData_get_alpha_x_0(el);
    double const alpha_y_0 = LinearTransferMatrixData_get_alpha_y_0(el);
    double const disp_x_1 = LinearTransferMatrixData_get_disp_x_1(el);
    double const disp_y_1 = LinearTransferMatrixData_get_disp_y_1(el);
    double const disp_px_1 = LinearTransferMatrixData_get_disp_px_1(el);
    double const disp_py_1 = LinearTransferMatrixData_get_disp_py_1(el);
    double const alpha_x_1 = LinearTransferMatrixData_get_alpha_x_1(el);
    double const alpha_y_1 = LinearTransferMatrixData_get_alpha_y_1(el);
    double const beta_x_1 = LinearTransferMatrixData_get_beta_x_1(el);
    double const beta_y_1 = LinearTransferMatrixData_get_beta_y_1(el);

    double const x_ref_0 = LinearTransferMatrixData_get_x_ref_0(el);
    double const x_ref_1 = LinearTransferMatrixData_get_x_ref_1(el);
    double const px_ref_0 = LinearTransferMatrixData_get_px_ref_0(el);
    double const px_ref_1 = LinearTransferMatrixData_get_px_ref_1(el);
    double const y_ref_0 = LinearTransferMatrixData_get_y_ref_0(el);
    double const y_ref_1 = LinearTransferMatrixData_get_y_ref_1(el);
    double const py_ref_0 = LinearTransferMatrixData_get_py_ref_0(el);
    double const py_ref_1 = LinearTransferMatrixData_get_py_ref_1(el);

    double const energy_ref_increment = 
        LinearTransferMatrixData_get_energy_ref_increment(el);

    int64_t const uncorrelated_rad_damping = LinearTransferMatrixData_get_uncorrelated_rad_damping(el);
    int64_t const uncorrelated_gauss_noise = LinearTransferMatrixData_get_uncorrelated_gauss_noise(el);

    double const sqrt_beta_prod_x = sqrt(beta_x_1 * beta_x_0);
    double const sqrt_beta_prod_y = sqrt(beta_y_1 * beta_y_0);

    double const sqrt_beta_ratio_x = sqrt(beta_x_1 / beta_x_0);
    double const sqrt_beta_ratio_y = sqrt(beta_y_1 / beta_y_0);

    //start_per_particle_block (part0->part)

        // Remove closed orbit
        LocalParticle_add_to_x(part, -x_ref_0);
        LocalParticle_add_to_px(part, -px_ref_0);
        LocalParticle_add_to_y(part, -y_ref_0);
        LocalParticle_add_to_py(part, -py_ref_0);

        // removing dispersion
        // Symplecticity correction (not working, to be investigated)
        // LocalParticle_add_to_zeta(part, (
        //     disp_px_0 * LocalParticle_get_x(part)
        //     - disp_x_0 * LocalParticle_get_px(part)
        //     + disp_py_0 * LocalParticle_get_y(part)
        //     - disp_y_0 * LocalParticle_get_py(part)
        //     )/LocalParticle_get_rvv(part));
        LocalParticle_add_to_x(part, -disp_x_0 * LocalParticle_get_delta(part));
        LocalParticle_add_to_px(part, -disp_px_0 * LocalParticle_get_delta(part));
        LocalParticle_add_to_y(part, -disp_y_0 * LocalParticle_get_delta(part));
        LocalParticle_add_to_py(part, -disp_py_0 * LocalParticle_get_delta(part));

        // Symplecticity correction (not working, to be investigated)
        // double rvv = LocalParticle_get_rvv(part);

        double sin_x, cos_x, sin_y, cos_y;

        if (no_detuning){
        // I use this parameters to pass cos_x, sin_x, ...
            cos_x = chroma_x;
            sin_x = q_x;
            cos_y = chroma_y;
            sin_y = q_y;
        }
        else{
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
            double phase = 2*PI*(q_x + chroma_x * LocalParticle_get_delta(part)
                                +detx_x * J_x + detx_y * J_y);
                cos_x = cos(phase);
                sin_x = sin(phase);
                phase = 2*PI*(q_y + chroma_y * LocalParticle_get_delta(part)
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
        if (cos_s < 2){
            // We set cos_s = 999 if long map is to be skipped
            double const new_zeta = cos_s * LocalParticle_get_zeta(part) + beta_s * sin_s * LocalParticle_get_pzeta(part);
            double const new_pzeta = -sin_s * LocalParticle_get_zeta(part) / beta_s + cos_s * LocalParticle_get_pzeta(part);

            LocalParticle_set_zeta(part, new_zeta);
            LocalParticle_update_pzeta(part, new_pzeta);
        }

        // Change energy without change of reference momentume
        double const energy_increment = 
            LinearTransferMatrixData_get_energy_increment(el);
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

        if(uncorrelated_rad_damping == 1) {
            double const damping_factor_x = LinearTransferMatrixData_get_damping_factor_x(el);
            double const damping_factor_y = LinearTransferMatrixData_get_damping_factor_y(el);
            double const damping_factor_s = LinearTransferMatrixData_get_damping_factor_s(el);

            LocalParticle_scale_x(part,damping_factor_x);
            LocalParticle_scale_px(part,damping_factor_x);
            LocalParticle_scale_y(part,damping_factor_y);
            LocalParticle_scale_py(part,damping_factor_y);
            LocalParticle_scale_zeta(part,damping_factor_s);
            double delta = LocalParticle_get_delta(part);
            delta *= damping_factor_s;
            LocalParticle_update_delta(part,delta);
        }

        if(uncorrelated_gauss_noise == 1) {
            double const gauss_noise_ampl_x = LinearTransferMatrixData_get_gauss_noise_ampl_x(el);
            double const gauss_noise_ampl_px = LinearTransferMatrixData_get_gauss_noise_ampl_px(el);
            double const gauss_noise_ampl_y = LinearTransferMatrixData_get_gauss_noise_ampl_y(el);
            double const gauss_noise_ampl_py = LinearTransferMatrixData_get_gauss_noise_ampl_py(el);
            double const gauss_noise_ampl_zeta = LinearTransferMatrixData_get_gauss_noise_ampl_zeta(el);
            double const gauss_noise_ampl_delta = LinearTransferMatrixData_get_gauss_noise_ampl_delta(el);

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
        }

        // Add dispersion
        // Symplecticity correction (not working, to be investigated)
        // LocalParticle_add_to_zeta(part, -(
        //     disp_px_1 * LocalParticle_get_x(part)
        //     - disp_x_1 * LocalParticle_get_px(part)
        //     + disp_py_1 * LocalParticle_get_y(part)
        //     - disp_y_1 * LocalParticle_get_py(part)
        //     )/LocalParticle_get_rvv(part));

        LocalParticle_add_to_x(part, disp_x_1 * LocalParticle_get_delta(part));
        LocalParticle_add_to_px(part, disp_px_1 * LocalParticle_get_delta(part));
        LocalParticle_add_to_y(part, disp_y_1 * LocalParticle_get_delta(part));
        LocalParticle_add_to_py(part, disp_py_1 * LocalParticle_get_delta(part));

        // Add closed orbit
        LocalParticle_add_to_x(part, x_ref_1);
        LocalParticle_add_to_px(part, px_ref_1);
        LocalParticle_add_to_y(part, y_ref_1);
        LocalParticle_add_to_py(part, py_ref_1);

        // Add to s coordinate
        LocalParticle_add_to_s(part, length);

    //end_per_particle_block
}

#endif
