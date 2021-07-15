#ifndef XTRACK_LINEARTRANSFERMATRIX_H
#define XTRACK_LINEARTRANSFERMATRIX_H

/*gpufun*/
void LinearTransferMatrix_track_local_particle(LinearTransferMatrixData el, LocalParticle* part){
    double const new_energy0 = LocalParticle_get_energy0(part)+LinearTransferMatrixData_get_energy_ref_increment(el);
    double const new_p0c = sqrt(new_energy0*new_energy0-LocalParticle_get_mass0(part)*LocalParticle_get_mass0(part));
    double const new_beta0 = new_p0c / new_energy0;
    double const new_gamma0 = new_energy0 / LocalParticle_get_mass0(part);
    double const geo_emit_factor = sqrt(LocalParticle_get_beta0(part)*LocalParticle_get_gamma0(part)/new_beta0/new_gamma0);
    int64_t const n_part = LocalParticle_get_num_particles(part);
    for (int ii=0; ii<n_part; ii++){ //only_for_context cpu_serial cpu_openmp
	    part->ipart = ii;            //only_for_context cpu_serial cpu_openmp
        double new_x = LocalParticle_get_x(part);
        double new_y = LocalParticle_get_y(part);
        double new_px = LocalParticle_get_px(part);
        double new_py = LocalParticle_get_py(part);

        new_x -= LinearTransferMatrixData_get_disp_x_0(el) * LocalParticle_get_delta(part);
        new_y -= LinearTransferMatrixData_get_disp_y_0(el) * LocalParticle_get_delta(part);

        double M00 = LinearTransferMatrixData_get_beta_ratio_x(el)*
                     (LinearTransferMatrixData_get_cos_x(el)+LinearTransferMatrixData_get_alpha_x_0(el)*LinearTransferMatrixData_get_sin_x(el));
        double M01 = LinearTransferMatrixData_get_beta_prod_x(el)*LinearTransferMatrixData_get_sin_x(el);
        double M10 = (
                      (LinearTransferMatrixData_get_alpha_x_0(el)-LinearTransferMatrixData_get_alpha_x_1(el))*LinearTransferMatrixData_get_cos_x(el)
                      -(1+LinearTransferMatrixData_get_alpha_x_0(el)*LinearTransferMatrixData_get_alpha_x_1(el))*LinearTransferMatrixData_get_sin_x(el)
                       )/LinearTransferMatrixData_get_beta_prod_x(el);
        double M11 = (LinearTransferMatrixData_get_cos_x(el)-LinearTransferMatrixData_get_alpha_x_1(el)*LinearTransferMatrixData_get_sin_x(el))
                        /LinearTransferMatrixData_get_beta_ratio_x(el);
        double tmp = new_x;
        new_x = M00*new_x + M01*new_px;
        new_px = M10*tmp + M11*new_px;
        tmp = new_y;
        M00 = LinearTransferMatrixData_get_beta_ratio_y(el)*
                     (LinearTransferMatrixData_get_cos_y(el)+LinearTransferMatrixData_get_alpha_y_0(el)*LinearTransferMatrixData_get_sin_y(el));
        M01 = LinearTransferMatrixData_get_beta_prod_y(el)*LinearTransferMatrixData_get_sin_y(el);
        M10 = (
                      (LinearTransferMatrixData_get_alpha_y_0(el)-LinearTransferMatrixData_get_alpha_y_1(el))*LinearTransferMatrixData_get_cos_y(el)
                      -(1+LinearTransferMatrixData_get_alpha_y_0(el)*LinearTransferMatrixData_get_alpha_y_1(el))*LinearTransferMatrixData_get_sin_y(el)
                       )/LinearTransferMatrixData_get_beta_prod_y(el);
        M11 = (LinearTransferMatrixData_get_cos_y(el)-LinearTransferMatrixData_get_alpha_y_1(el)*LinearTransferMatrixData_get_sin_y(el))
                        /LinearTransferMatrixData_get_beta_ratio_y(el);
        new_y = M00*new_y + M01*new_py;
        new_py = M10*tmp + M11*new_py;

        LocalParticle_add_to_energy(part, LinearTransferMatrixData_get_energy_increment(el));
        LocalParticle_set_delta(part,LocalParticle_get_delta(part) * LocalParticle_get_p0c(part)/new_p0c);
        new_x *= geo_emit_factor;
        new_px *= geo_emit_factor;
        new_y *= geo_emit_factor;
        new_py *= geo_emit_factor;

        new_x += LinearTransferMatrixData_get_disp_x_1(el) * LocalParticle_get_delta(part);
        new_y += LinearTransferMatrixData_get_disp_y_1(el) * LocalParticle_get_delta(part);

    	LocalParticle_set_x(part, new_x);
    	LocalParticle_set_y(part, new_y);
    	LocalParticle_set_px(part, new_px);
    	LocalParticle_set_py(part, new_py);
    } //only_for_context cpu_serial cpu_openmp
    //LocalParticle_set_p0c(part,new_p0c); can one set attributes of the particles object from c?
}

#endif
