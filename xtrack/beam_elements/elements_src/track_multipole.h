// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_TRACK_MULTIPOLE_H
#define XTRACK_TRACK_MULTIPOLE_H


/*gpufun*/
void multipole_compute_dpx_dpy_single_particle(LocalParticle* part,
    double const* knl,
    double const* ksl,
    int64_t order, double inv_factorial_order_0,
    double delta_tap, double const backtrack_sign,
    double* dpx_out, double* dpy_out){

        double const chi = LocalParticle_get_chi(part);

        int64_t index = order;
        double inv_factorial = inv_factorial_order_0;

        double dpx = backtrack_sign * chi * knl[index] * inv_factorial;
        double dpy = backtrack_sign * chi * ksl[index] * inv_factorial;

        dpx = dpx * (1 + delta_tap);
        dpy = dpy * (1 + delta_tap);

        double const x   = LocalParticle_get_x(part);
        double const y   = LocalParticle_get_y(part);

        while( index > 0 )
        {
            double const zre = dpx * x - dpy * y;
            double const zim = dpx * y + dpy * x;

            inv_factorial *= index;
            index -= 1;

            double this_knl = chi * knl[index];
            double this_ksl = chi * ksl[index];

            this_knl = this_knl * backtrack_sign;
            this_ksl = this_ksl * backtrack_sign;

            this_knl = this_knl * (1 + delta_tap);
            this_ksl = this_ksl * (1 + delta_tap);

            dpx = this_knl*inv_factorial + zre;
            dpy = this_ksl*inv_factorial + zim;
        }

        *dpx_out = -dpx;
        *dpy_out = dpy;
}

/*gpufun*/
void Multipole_track_single_particle(LocalParticle* part,
    double hxl, double length, double weight,
    double const* knl, double const* ksl,
    int64_t const order, double const inv_factorial_order_0,
    double const* knl_2, double const* ksl_2,
    int64_t const order_2, double const inv_factorial_order_2_0,
    double const backtrack_sign,
    double const delta_tap, int64_t const radiation_flag,
    double* dp_record_entry, double* dpx_record_entry, double* dpy_record_entry,
    double* dp_record_exit, double* dpx_record_exit, double* dpy_record_exit,
    SynchrotronRadiationRecordData record, RecordIndex record_index){

        double dpx = 0.;
        double dpy = 0.;

        if (knl){
            double dpx1, dpy1;
            multipole_compute_dpx_dpy_single_particle(part, knl, ksl,
                order, inv_factorial_order_0,
                delta_tap, backtrack_sign,
                &dpx1, &dpy1);
            dpx += dpx1 * weight;
            dpy += dpy1 * weight;
        }

        if (knl_2){
            double dpx2, dpy2;
            multipole_compute_dpx_dpy_single_particle(part, knl_2, ksl_2,
                order_2, inv_factorial_order_2_0,
                delta_tap, backtrack_sign,
                &dpx2, &dpy2);
            dpx += dpx2 * weight;
            dpy += dpy2 * weight;
        }

        #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        // Radiation at entrance
        double const mass0 = LocalParticle_get_mass0(part);
        double const q0 = LocalParticle_get_q0(part);
        double const beta0 = LocalParticle_get_beta0(part);
        double const gamma0 = LocalParticle_get_gamma0(part);

        double const curv = sqrt(dpx*dpx + dpy*dpy) / length;
        double const mass0_kg = mass0 * QELEM / C_LIGHT / C_LIGHT;
        double const Q0_coulomb = q0 * QELEM;
        /*b*/double const P0_J = mass0_kg * beta0 * gamma0 * C_LIGHT;
        double const B_T = curv * P0_J / Q0_coulomb;

        if (radiation_flag > 0 && length > 0){
            double const x      = LocalParticle_get_x(part);
            double const L_path = 0.5 * length * (1 + hxl*x/length);
            if (radiation_flag == 1){
                synrad_average_kick(part, B_T, L_path,
                        dp_record_entry, dpx_record_entry, dpy_record_entry);
            }
            else if (radiation_flag == 2){
                synrad_emit_photons(part, B_T, L_path, record_index, record);
            }
        }
        #endif

        if( ( hxl > 0) || ( hxl < 0 ) )
        {
            double const delta  = LocalParticle_get_delta(part);
            double const chi    = LocalParticle_get_chi(part);
            double const x      = LocalParticle_get_x(part);

            double const hxlx   = x * hxl;

            double const rv0v = 1./LocalParticle_get_rvv(part);

            dpx += (hxl + hxl * delta);

            if( length != 0)
            {
                double knl0 = 0;

                if (knl){
                    knl0 += knl[0];
                }

                if (knl_2){
                    knl0 += knl_2[0];
                }

                double b1l = backtrack_sign * chi * knl0 * weight;

                b1l = b1l * (1 + delta_tap);

                dpx -= b1l * hxlx / length;

            }


            LocalParticle_add_to_zeta(part, -rv0v*chi * hxlx);
        }

        LocalParticle_add_to_px(part, dpx);
        LocalParticle_add_to_py(part, dpy);

        // Radiation at exit
        #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        if (radiation_flag > 0 && length > 0){
            double const x      = LocalParticle_get_x(part);
            double const L_path = 0.5*length * (1 + (hxl*x)/length);
            if (radiation_flag == 1){
                synrad_average_kick(part, B_T, L_path,
                        dp_record_exit, dpx_record_exit, dpy_record_exit);
            }
            else if (radiation_flag == 2){
                synrad_emit_photons(part, B_T, L_path, record_index, record);
            }
        }
        #endif
    }

#endif