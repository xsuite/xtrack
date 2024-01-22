// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_LINESEGMENTMAP_H
#define XTRACK_LINESEGMENTMAP_H

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
    LocalParticle* part0, double const dx_0, double const dpx_0,
    double const dy_0, double const dpy_0){

    //start_per_particle_block (part0->part)

        // Remove dispersion
        // Symplecticity correction (not working, to be investigated)
        // LocalParticle_add_to_zeta(part, (
        //     dpx_0 * LocalParticle_get_x(part)
        //     - dx_0 * LocalParticle_get_px(part)
        //     + dpy_0 * LocalParticle_get_y(part)
        //     - dy_0 * LocalParticle_get_py(part)
        //     )/LocalParticle_get_rvv(part));
        double const delta = LocalParticle_get_delta(part);
        LocalParticle_add_to_x(part, -dx_0 * delta);
        LocalParticle_add_to_px(part, -dpx_0 * delta);
        LocalParticle_add_to_y(part, -dy_0 * delta);
        LocalParticle_add_to_py(part, -dpy_0 * delta);

    //end_per_particle_block

}

/*gpufun*/
void add_dispersion(
    LocalParticle* part0, double const dx_1, double const dpx_1,
    double const dy_1, double const dpy_1){

    //start_per_particle_block (part0->part)

        // Add dispersion
        // Symplecticity correction (not working, to be investigated)
        // LocalParticle_add_to_zeta(part, (
        //     dpx_1 * LocalParticle_get_x(part)
        //     - dx_1 * LocalParticle_get_px(part)
        //     + dpy_1 * LocalParticle_get_y(part)
        //     - dy_1 * LocalParticle_get_py(part)
        //     )/LocalParticle_get_rvv(part));
        double const delta = LocalParticle_get_delta(part);
        LocalParticle_add_to_x(part, dx_1 * delta);
        LocalParticle_add_to_px(part, dpx_1 * delta);
        LocalParticle_add_to_y(part, dy_1 * delta);
        LocalParticle_add_to_py(part, dpy_1 * delta);

    //end_per_particle_block

}

/*gpufun*/
void transverse_motion(LocalParticle *part0,
    double const qx, double const qy,
    double const dqx, double const dqy,
    double const det_xx, double const det_xy, double const det_yx, double const det_yy,
    double const alfx_0, double const betx_0, double const alfy_0, double const bety_0,
    double const alfx_1, double const betx_1, double const alfy_1, double const bety_1){

    int64_t detuning;
    double sin_x = 0;
    double cos_x = 0;
    double sin_y = 0;
    double cos_y = 0;
    if (dqx != 0.0 || dqy != 0.0 ||
        det_xx != 0.0 || det_xy != 0.0 || det_yx != 0.0 || det_yy != 0.0){
        detuning = 1;
    }
    else{
        detuning = 0;
        sin_x = sin(2 * PI * qx);
        cos_x = cos(2 * PI * qx);
        sin_y = sin(2 * PI * qy);
        cos_y = cos(2 * PI * qy);
    }

    double const sqrt_betprod_x = sqrt(betx_1 * betx_0);
    double const sqrt_betprod_y = sqrt(bety_1 * bety_0);

    double const sqrt_betratio_x = sqrt(betx_1 / betx_0);
    double const sqrt_betratio_y = sqrt(bety_1 / bety_0);

    //start_per_particle_block (part0->part)

        if (detuning){
            double const J_x = 0.5 * (
                (1.0 + alfx_0 * alfx_0) / betx_0
                    * LocalParticle_get_x(part)*LocalParticle_get_x(part)
                + 2 * alfx_0
                    * LocalParticle_get_x(part)*LocalParticle_get_px(part)
                + betx_0
                    * LocalParticle_get_px(part)*LocalParticle_get_px(part));
            double const J_y = 0.5 * (
                (1.0 + alfy_0 * alfy_0) /bety_0
                    * LocalParticle_get_y(part)*LocalParticle_get_y(part)
                + 2*alfy_0
                    * LocalParticle_get_y(part)*LocalParticle_get_py(part)
                + bety_0
                    * LocalParticle_get_py(part)*LocalParticle_get_py(part));
            double phase = 2*PI*(qx + dqx * LocalParticle_get_delta(part)
                                +det_xx * J_x + det_xy * J_y);
            cos_x = cos(phase);
            sin_x = sin(phase);
            phase = 2*PI*(qy + dqy * LocalParticle_get_delta(part)
                            +det_yx * J_x + det_yy * J_y);
            cos_y = cos(phase);
            sin_y = sin(phase);
        }

        double const M00_x = sqrt_betratio_x*(cos_x+alfx_0*sin_x);
        double const M01_x = sqrt_betprod_x*sin_x;
        double const M10_x = ((alfx_0-alfx_1)*cos_x
                    -(1+alfx_0*alfx_1)*sin_x
                    )/sqrt_betprod_x;
        double const M11_x = (cos_x-alfx_1*sin_x)/sqrt_betratio_x;
        double const M00_y = sqrt_betratio_y*(cos_y+alfy_0*sin_y);
        double const M01_y = sqrt_betprod_y*sin_y;
        double const M10_y = ((alfy_0-alfy_1)*cos_y
                    -(1+alfy_0*alfy_1)*sin_y
                    )/sqrt_betprod_y;
        double const M11_y = (cos_y-alfy_1*sin_y)/sqrt_betratio_y;

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

/*gpufun*/
void longitudinal_motion(LocalParticle *part0,
                         LineSegmentMapData el){

    int64_t const mode_flag = LineSegmentMapData_get_longitudinal_mode_flag(el);

    if (mode_flag==1){ // linear motion fixed qs
        double const qs = LineSegmentMapData_get_qs(el);
        double const bets = LineSegmentMapData_get_bets(el);
        double const sin_s = sin(2 * PI * qs);
        double const cos_s = cos(2 * PI * qs);
        //start_per_particle_block (part->part)
            // We set cos_s = 999 if long map is to be skipped
            double const new_zeta = cos_s * LocalParticle_get_zeta(part) - bets * sin_s * LocalParticle_get_pzeta(part);
            double const new_pzeta = sin_s * LocalParticle_get_zeta(part) / bets + cos_s * LocalParticle_get_pzeta(part);

            LocalParticle_set_zeta(part, new_zeta);
            LocalParticle_update_pzeta(part, new_pzeta);
        //end_per_particle_block

    }
    else if (mode_flag==2){ // non-linear motion

        double const alfp =
            LineSegmentMapData_get_momentum_compaction_factor(el);
        double const slippage_length =
            LineSegmentMapData_get_slippage_length(el);

        //start_per_particle_block (part->part)
            double const gamma0 = LocalParticle_get_gamma0(part);
            double const eta = alfp - 1.0 / (gamma0 * gamma0);
            LocalParticle_add_to_zeta(part,
                -0.5 * eta * slippage_length * LocalParticle_get_delta(part));
        //end_per_particle_block

        int64_t const n_rf = LineSegmentMapData_len_voltage_rf(el);
        for (int i_rf=0; i_rf<n_rf; i_rf++){

            double const v_rf = LineSegmentMapData_get_voltage_rf(el,i_rf);
            double const f_rf = LineSegmentMapData_get_frequency_rf(el,i_rf);
            double const lag_rf = LineSegmentMapData_get_lag_rf(el,i_rf);

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
            double const eta = alfp - 1.0 / (gamma0 * gamma0);
            LocalParticle_add_to_zeta(part,
                -0.5 * eta * slippage_length * LocalParticle_get_delta(part));
        //end_per_particle_block
    }
    else if (mode_flag == 3){

        double const alfp =
            LineSegmentMapData_get_momentum_compaction_factor(el);
        double const slippage_length =
            LineSegmentMapData_get_slippage_length(el);

        // Assume there is only one RF term (checked in the Python code)
        double const v_rf = LineSegmentMapData_get_voltage_rf(el,0);
        double const f_rf = LineSegmentMapData_get_frequency_rf(el,0);

        //start_per_particle_block (part->part)
            double const gamma0 = LocalParticle_get_gamma0(part);
            double const beta0 = LocalParticle_get_beta0(part);
            double const q0 = LocalParticle_get_q0(part);
            double const mass0 = LocalParticle_get_mass0(part);
            double const eta = alfp - 1.0 / (gamma0 * gamma0);
            double const E0 = mass0 * gamma0;

            double const qs = sqrt(q0 * fabs(eta) * slippage_length * f_rf * v_rf
                        / (2 * PI * beta0 * beta0 * beta0 * E0 * C_LIGHT));
            double const bets = eta * slippage_length / (2 * PI * qs);

            double const sin_s = sin(2 * PI * qs);
            double const cos_s = cos(2 * PI * qs);

            double const new_zeta = cos_s * LocalParticle_get_zeta(part) - bets * sin_s * LocalParticle_get_pzeta(part);
            double const new_pzeta = sin_s * LocalParticle_get_zeta(part) / bets + cos_s * LocalParticle_get_pzeta(part);

            LocalParticle_set_zeta(part, new_zeta);
            LocalParticle_update_pzeta(part, new_pzeta);

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

/*gpufun*/
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
            double const old_px = LocalParticle_get_px(part);
            double const old_py = LocalParticle_get_py(part);
            double const new_energy0 = LocalParticle_get_mass0(part)
                *LocalParticle_get_gamma0(part) + energy_ref_increment;
            double const new_p0c = sqrt(new_energy0*new_energy0
            -LocalParticle_get_mass0(part)*LocalParticle_get_mass0(part));
            double const new_beta0 = new_p0c / new_energy0;
            double const new_gamma0 = new_energy0 / LocalParticle_get_mass0(part);
            double const geo_emit_factor = sqrt(LocalParticle_get_beta0(part)
                    *LocalParticle_get_gamma0(part)/new_beta0/new_gamma0);
            LocalParticle_update_p0c(part, new_p0c); // updates px, py but not in the smoothed way
            LocalParticle_set_px(part, old_px * geo_emit_factor);
            LocalParticle_set_py(part, old_py * geo_emit_factor);
            LocalParticle_scale_x(part, geo_emit_factor);
            LocalParticle_scale_y(part, geo_emit_factor);
        }
    //end_per_particle_block

}

/*gpufun*/
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
void LineSegmentMap_track_local_particle(LineSegmentMapData el, LocalParticle* part0){

    remove_closed_orbit(part0,
        LineSegmentMapData_get_x_ref(el, 0),
        LineSegmentMapData_get_px_ref(el, 0),
        LineSegmentMapData_get_y_ref(el, 0),
        LineSegmentMapData_get_py_ref(el, 0));

    remove_dispersion(part0,
        LineSegmentMapData_get_dx(el, 0),
        LineSegmentMapData_get_dpx(el, 0),
        LineSegmentMapData_get_dy(el, 0),
        LineSegmentMapData_get_dpy(el, 0));

    transverse_motion(part0,
        LineSegmentMapData_get_qx(el),
        LineSegmentMapData_get_qy(el),
        LineSegmentMapData_get_dqx(el),
        LineSegmentMapData_get_dqy(el),
        LineSegmentMapData_get_det_xx(el),
        LineSegmentMapData_get_det_xy(el),
        LineSegmentMapData_get_det_yx(el),
        LineSegmentMapData_get_det_yy(el),
        LineSegmentMapData_get_alfx(el, 0),
        LineSegmentMapData_get_betx(el, 0),
        LineSegmentMapData_get_alfy(el, 0),
        LineSegmentMapData_get_bety(el, 0),
        LineSegmentMapData_get_alfx(el, 1),
        LineSegmentMapData_get_betx(el, 1),
        LineSegmentMapData_get_alfy(el, 1),
        LineSegmentMapData_get_bety(el, 1));

    longitudinal_motion(part0, el);

    energy_and_reference_increments(part0,
        LineSegmentMapData_get_energy_increment(el),
        LineSegmentMapData_get_energy_ref_increment(el));

    if (LineSegmentMapData_get_uncorrelated_rad_damping(el) == 1){
        uncorrelated_radiation_damping(part0,
            LineSegmentMapData_get_damping_factor_x(el),
            LineSegmentMapData_get_damping_factor_y(el),
            LineSegmentMapData_get_damping_factor_s(el));
    }

    if (LineSegmentMapData_get_uncorrelated_gauss_noise(el) == 1){
        uncorrelated_gaussian_noise(part0,
            LineSegmentMapData_get_gauss_noise_ampl_x(el),
            LineSegmentMapData_get_gauss_noise_ampl_px(el),
            LineSegmentMapData_get_gauss_noise_ampl_y(el),
            LineSegmentMapData_get_gauss_noise_ampl_py(el),
            LineSegmentMapData_get_gauss_noise_ampl_zeta(el),
            LineSegmentMapData_get_gauss_noise_ampl_delta(el));
    }

    add_dispersion(part0,
        LineSegmentMapData_get_dx(el, 1),
        LineSegmentMapData_get_dpx(el, 1),
        LineSegmentMapData_get_dy(el, 1),
        LineSegmentMapData_get_dpy(el, 1));

    add_closed_orbit(part0,
        LineSegmentMapData_get_x_ref(el, 1),
        LineSegmentMapData_get_px_ref(el, 1),
        LineSegmentMapData_get_y_ref(el, 1),
        LineSegmentMapData_get_py_ref(el, 1));

    double const length = LineSegmentMapData_get_length(el);
    //start_per_particle_block (part0->part)
        LocalParticle_add_to_s(part, length);
    //end_per_particle_block
}

#endif
