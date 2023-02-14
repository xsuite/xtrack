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

    double const beta_x_0 = LinearTransferMatrixData_get_beta_x_0(el);
    double const beta_y_0 = LinearTransferMatrixData_get_beta_y_0(el);
    double const beta_ratio_x = LinearTransferMatrixData_get_beta_ratio_x(el);
    double const beta_prod_x = LinearTransferMatrixData_get_beta_prod_x(el);
    double const beta_ratio_y = LinearTransferMatrixData_get_beta_ratio_y(el);
    double const beta_prod_y = LinearTransferMatrixData_get_beta_prod_y(el);
    double const disp_x_0 = LinearTransferMatrixData_get_disp_x_0(el);
    double const disp_y_0 = LinearTransferMatrixData_get_disp_y_0(el);
    double const alpha_x_0 = LinearTransferMatrixData_get_alpha_x_0(el);
    double const alpha_y_0 = LinearTransferMatrixData_get_alpha_y_0(el);
    double const disp_x_1 = LinearTransferMatrixData_get_disp_x_1(el);
    double const disp_y_1 = LinearTransferMatrixData_get_disp_y_1(el);
    double const alpha_x_1 = LinearTransferMatrixData_get_alpha_x_1(el);
    double const alpha_y_1 = LinearTransferMatrixData_get_alpha_y_1(el);

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

    //start_per_particle_block (part0->part)

    // Transverse linear uncoupled matrix
    double new_x = LocalParticle_get_x(part);
    double new_y = LocalParticle_get_y(part);
    double new_px = LocalParticle_get_px(part);
    double new_py = LocalParticle_get_py(part);
    double delta = LocalParticle_get_delta(part);

    // removing dispersion and close orbit
    new_x -= disp_x_0 * delta + x_ref_0;
    new_px -= px_ref_0;
    new_y -= disp_y_0 * delta + y_ref_0;
    new_py -= py_ref_0;

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
            (1.0 + alpha_x_0*alpha_x_0)/beta_x_0 * new_x*new_x
            + 2*alpha_x_0 * new_x*new_px
            + beta_x_0 * new_px*new_px);
        double const J_y = 0.5 * (
            (1.0 + alpha_y_0*alpha_y_0)/beta_y_0 * new_y*new_y
            + 2*alpha_y_0 * new_y*new_py
            + beta_y_0 * new_py*new_py);
        double phase = 2*PI*(q_x+chroma_x*delta+detx_x*J_x+detx_y*J_y);
            cos_x = cos(phase);
            sin_x = sin(phase);
            phase = 2*PI*(q_y+chroma_y*delta+dety_y*J_y+dety_x*J_x);
            cos_y = cos(phase);
            sin_y = sin(phase);
    }

    double const M00_x = beta_ratio_x*(cos_x+alpha_x_0*sin_x);
    double const M01_x = beta_prod_x*sin_x;
    double const M10_x = ((alpha_x_0-alpha_x_1)*cos_x
                  -(1+alpha_x_0*alpha_x_1)*sin_x
                   )/beta_prod_x;
    double const M11_x = (cos_x-alpha_x_1*sin_x)/beta_ratio_x;
    double const M00_y = beta_ratio_y*(cos_y+alpha_y_0*sin_y);
    double const M01_y = beta_prod_y*sin_y;
    double const M10_y = ((alpha_y_0-alpha_y_1)*cos_y
                  -(1+alpha_y_0*alpha_y_1)*sin_y
                  )/beta_prod_y;
    double const M11_y = (cos_y-alpha_y_1*sin_y)/beta_ratio_y;

    double tmp = new_x;
    new_x = M00_x*tmp + M01_x*new_px;
    new_px = M10_x*tmp + M11_x*new_px;
    tmp = new_y;
    new_y = M00_y*tmp + M01_y*new_py;
    new_py = M10_y*tmp + M11_y*new_py;

    if (cos_s < 2){
        // We set cos_s = 999 if long map is to be skipped
        double const old_zeta = LocalParticle_get_zeta(part);
        double const old_pzeta = LocalParticle_get_pzeta(part); // Use canonically conjugate variables
        double const new_zeta = cos_s*old_zeta+beta_s*sin_s*old_pzeta;
        double const new_pzeta = -sin_s*old_zeta/beta_s+cos_s*old_pzeta;

        LocalParticle_set_zeta(part, new_zeta);
        LocalParticle_update_pzeta(part, new_pzeta);
    }

    // Change energy without change of reference momentume
    double const energy_increment = 
        LinearTransferMatrixData_get_energy_increment(el);
    if (energy_increment !=0){
      LocalParticle_add_to_energy(part, energy_increment, 1);
    }

    LocalParticle_set_x(part, new_x);
    LocalParticle_set_y(part, new_y);
    LocalParticle_set_px(part, new_px);
    LocalParticle_set_py(part, new_py);

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


        
    // re-adding dispersion and closed orbit
    delta = LocalParticle_get_delta(part);
    LocalParticle_add_to_x(part,disp_x_1 * delta + x_ref_1);
    LocalParticle_add_to_px(part,px_ref_1);
    LocalParticle_add_to_y(part,disp_y_1 * delta + y_ref_1);
    LocalParticle_add_to_py(part,py_ref_1);

    //end_per_particle_block
}

#endif
