#ifndef XTRACK_LINEARTRANSFERMATRIXWITHDETUNING_H
#define XTRACK_LINEARTRANSFERMATRIXWITHDETUNING_H

/*gpufun*/
void LinearTransferMatrixWithDetuning_track_local_particle(LinearTransferMatrixWithDetuningData el, LocalParticle* part){
    double const new_energy0 = LocalParticle_get_mass0(part)*LocalParticle_get_gamma0(part)+LinearTransferMatrixWithDetuningData_get_energy_ref_increment(el);
    double const new_p0c = sqrt(new_energy0*new_energy0-LocalParticle_get_mass0(part)*LocalParticle_get_mass0(part));
    double const new_beta0 = new_p0c / new_energy0;
    double const new_gamma0 = new_energy0 / LocalParticle_get_mass0(part);
    double const geo_emit_factor = sqrt(LocalParticle_get_beta0(part)*LocalParticle_get_gamma0(part)/new_beta0/new_gamma0);

    double const q_x = LinearTransferMatrixWithDetuningData_get_q_x(el);
    double const q_y = LinearTransferMatrixWithDetuningData_get_q_y(el);
    double const cos_s = LinearTransferMatrixWithDetuningData_get_cos_s(el);
    double const sin_s = LinearTransferMatrixWithDetuningData_get_sin_s(el);
    double const beta_ratio_x = LinearTransferMatrixWithDetuningData_get_beta_ratio_x(el);
    double const beta_prod_x = LinearTransferMatrixWithDetuningData_get_beta_prod_x(el);
    double const beta_ratio_y = LinearTransferMatrixWithDetuningData_get_beta_ratio_y(el);
    double const beta_prod_y = LinearTransferMatrixWithDetuningData_get_beta_prod_y(el);
    double const beta_s = LinearTransferMatrixWithDetuningData_get_beta_s(el);
    double const disp_x_0 = LinearTransferMatrixWithDetuningData_get_disp_x_0(el);
    double const disp_y_0 = LinearTransferMatrixWithDetuningData_get_disp_y_0(el);
    double const alpha_x_0 = LinearTransferMatrixWithDetuningData_get_alpha_x_0(el);
    double const alpha_y_0 = LinearTransferMatrixWithDetuningData_get_alpha_y_0(el);
    double const disp_x_1 = LinearTransferMatrixWithDetuningData_get_disp_x_1(el);
    double const disp_y_1 = LinearTransferMatrixWithDetuningData_get_disp_y_1(el);
    double const alpha_x_1 = LinearTransferMatrixWithDetuningData_get_alpha_x_1(el);
    double const alpha_y_1 = LinearTransferMatrixWithDetuningData_get_alpha_y_1(el);

    double const chroma_x = LinearTransferMatrixWithDetuningData_get_chroma_x(el);
    double const chroma_y = LinearTransferMatrixWithDetuningData_get_chroma_y(el);
    double const detx_x = LinearTransferMatrixWithDetuningData_get_detx_x(el);
    double const detx_y = LinearTransferMatrixWithDetuningData_get_detx_x(el);
    double const dety_y = LinearTransferMatrixWithDetuningData_get_detx_x(el);
    double const dety_x = LinearTransferMatrixWithDetuningData_get_detx_x(el);

    int64_t const n_part = LocalParticle_get_num_particles(part);
    for (int ii=0; ii<n_part; ii++){ //only_for_context cpu_serial cpu_openmp
	    part->ipart = ii;            //only_for_context cpu_serial cpu_openmp
        // Transverse linear uncoupled matrix
        double new_x = LocalParticle_get_x(part);
        double new_y = LocalParticle_get_y(part);
        double new_px = LocalParticle_get_px(part);
        double new_py = LocalParticle_get_py(part);
        double new_zeta = LocalParticle_get_zeta(part);
        double new_delta = LocalParticle_get_delta(part);

        // removing dispersion
        new_x -= disp_x_0 * LocalParticle_get_delta(part);
        new_y -= disp_y_0 * LocalParticle_get_delta(part);

        double const J_x = 0.5*(new_x*new_x+new_px*new_px);
        double const J_y = 0.5*(new_y*new_y+new_py*new_py);
        double phase = 2*M_PI*(q_x+chroma_x*new_delta+detx_x*J_x+detx_y*J_y);
        double const cos_x = cos(phase);
        double const sin_x = sin(phase);
        phase = 2*M_PI*(q_y+chroma_y*new_delta+dety_y*J_y+dety_x*J_x);
        double const cos_y = cos(phase);
        double const sin_y = sin(phase);
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
        tmp = new_zeta;
        new_zeta = cos_s*tmp+beta_s*sin_s*new_delta;
        new_delta = -sin_s*tmp/beta_s+cos_s*new_delta;
    	LocalParticle_set_zeta(part, new_zeta);
    	LocalParticle_set_delta(part, new_delta);
        
        // Change energy without change of reference momentume
        //LocalParticle_add_to_energy(part, LinearTransferMatrixWithDetuningData_get_energy_increment(el)); // TODO This function has a bug, it changes the energy even with LocalParticle_add_to_energy(part, 0.0)
        // Change energy with change of reference. In the transverse plane de change is smoothed, i.e. 
        // both the position and the momentum are scaled, rather than only the momentum.
        
        LocalParticle_set_delta(part,LocalParticle_get_delta(part) * LocalParticle_get_p0c(part)/new_p0c);
        new_x *= geo_emit_factor;
        new_px *= geo_emit_factor;
        new_y *= geo_emit_factor;
        new_py *= geo_emit_factor;
        
        // re-adding dispersion
        new_x += disp_x_1 * LocalParticle_get_delta(part);
        new_y += disp_y_1 * LocalParticle_get_delta(part);

    	LocalParticle_set_x(part, new_x);
    	LocalParticle_set_y(part, new_y);
    	LocalParticle_set_px(part, new_px);
    	LocalParticle_set_py(part, new_py);
    } //only_for_context cpu_serial cpu_openmp

    // Can an element change the reference energy ? ///////////
    /* //Not working
    part->p0c = new_p0c;
    part->gamma0 = new_gamma0;
    part->beta0 = new_beta0;
    */
    /* //Not defined in the API
    //LocalParticle_set_p0c(part,new_p0c);
    //LocalParticle_set_gamma0(part,new_gamma0);
    //LocalParticle_set_beta0(part,new_beta0);
    //LocalParticle_set_p0c(part,new_p0c);
    */
    ///////////////////////////////////////////////////////////
}

#endif
