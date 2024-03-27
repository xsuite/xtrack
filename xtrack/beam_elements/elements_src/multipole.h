// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_MULTIPOLE_H
#define XTRACK_MULTIPOLE_H


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

/*gpuglmem*/
void Multipole_track_single_particle(LocalParticle* part,
    double hxl, double hyl, double length, double* knl, double* ksl,
    int64_t order, double inv_factorial_order_0, double backtrack_sign,
    double delta_tap, int64_t radiation_flag,
    double dp_record_entry, double dpx_record_entry, double dpy_record_entry,
    double dp_record_exit, double dpx_record_exit, double dpy_record_exit,
    SynchrotronRadiationRecordData record, RecordIndex record_index){

        delta_tap = LocalParticle_get_delta(part);

        double dpx, dpy;
        multipole_compute_dpx_dpy_single_particle(part, knl, ksl,
            order, inv_factorial_order_0,
            delta_tap, backtrack_sign,
            &dpx, &dpy);

        #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        // Radiation at entrance
        double const curv = sqrt(dpx*dpx + dpy*dpy) / length;
        if (radiation_flag > 0 && length > 0){
            double const x      = LocalParticle_get_x(part);
            double const y      = LocalParticle_get_y(part);
            double const L_path = 0.5 * length * (1 + (hxl*x - hyl*y)/length);
            if (radiation_flag == 1){
                synrad_average_kick(part, curv, L_path,
                        &dp_record_entry, &dpx_record_entry, &dpy_record_entry);
            }
            else if (radiation_flag == 2){
                synrad_emit_photons(part, curv, L_path, record_index, record);
            }
        }
        #endif

        if( ( hxl > 0) || ( hyl > 0) || ( hxl < 0 ) || ( hyl < 0 ) )
        {
            double const delta  = LocalParticle_get_delta(part);
            double const chi    = LocalParticle_get_chi(part);
            double const x      = LocalParticle_get_x(part);
            double const y      = LocalParticle_get_y(part);

            double const hxlx   = x * hxl;
            double const hyly   = y * hyl;

            double const rv0v = 1./LocalParticle_get_rvv(part);

            dpx += (hxl + hxl * delta);
            dpy -= (hyl + hyl * delta);

            if( length != 0)
            {
                double b1l = backtrack_sign * chi * knl[0];
                double a1l = backtrack_sign * chi * ksl[0];

                b1l = b1l * (1 + delta_tap);
                a1l = a1l * (1 + delta_tap);

                dpx -= b1l * hxlx / length;
                dpy -= a1l * hyly / length;
            }

            LocalParticle_add_to_zeta(part, rv0v*chi * ( hyly - hxlx ) );
        }

        LocalParticle_add_to_px(part, dpx);
        LocalParticle_add_to_py(part, dpy);

        // Radiation at exit
        #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        if (radiation_flag > 0 && length > 0){
            double const x      = LocalParticle_get_x(part);
            double const y      = LocalParticle_get_y(part);
            double const L_path = 0.5*length * (1 + (hxl*x - hyl*y)/length);
            if (radiation_flag == 1){
                synrad_average_kick(part, curv, L_path,
                        &dp_record_exit, &dpx_record_exit, &dpy_record_exit);
            }
            else if (radiation_flag == 2){
                // printf("L_path = %e curv = %e\n", L_path, curv);
                synrad_emit_photons(part, curv, L_path, record_index, record);
            }
        }
        #endif
    }

/*gpufun*/
void Multipole_track_local_particle(MultipoleData el, LocalParticle* part0){

    #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
    int64_t radiation_flag = MultipoleData_get_radiation_flag(el);

    // Extract record and record_index
    SynchrotronRadiationRecordData record = NULL;
    RecordIndex record_index = NULL;
    if (radiation_flag==2){
        record = (SynchrotronRadiationRecordData) MultipoleData_getp_internal_record(el, part0);
        if (record){
            record_index = SynchrotronRadiationRecordData_getp__index(record);
        }
    }
    double dp_record_entry = 0.;
    double dpx_record_entry = 0.;
    double dpy_record_entry = 0.;
    double dp_record_exit = 0.;
    double dpx_record_exit = 0.;
    double dpy_record_exit = 0.;
    #endif

    #ifdef XTRACK_MULTIPOLE_NO_SYNRAD
    #define delta_taper (0)
    #else
        double delta_taper = MultipoleData_get_delta_taper(el);
    #endif

    int64_t const order = MultipoleData_get_order(el);
    double const inv_factorial_order_0 = MultipoleData_get_inv_factorial_order(el);

    #ifndef XSUITE_BACKTRACK
        double const hxl = MultipoleData_get_hxl(el);
        double const hyl = MultipoleData_get_hyl(el);
    #else
        double const hxl = -MultipoleData_get_hxl(el);
        double const hyl = -MultipoleData_get_hyl(el);
    #endif

    /*gpuglmem*/ double const* knl = MultipoleData_getp1_knl(el, 0);
    /*gpuglmem*/ double const* ksl = MultipoleData_getp1_ksl(el, 0);

    #ifndef XSUITE_BACKTRACK
        double const length = MultipoleData_get_length(el); // m
        double const backtrack_sign = 1;
    #else
        double const length = -MultipoleData_get_length(el); // m
        double const backtrack_sign = -1;
    #endif

    //start_per_particle_block (part0->part)

        #ifdef XTRACK_MULTIPOLE_TAPER
            delta_taper = LocalParticle_get_delta(part);
        #endif

        double dpx, dpy;
        multipole_compute_dpx_dpy_single_particle(part, knl, ksl,
            order, inv_factorial_order_0,
            delta_taper, backtrack_sign,
            &dpx, &dpy);

        #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        // Radiation at entrance
        double const curv = sqrt(dpx*dpx + dpy*dpy) / length;
        if (radiation_flag > 0 && length > 0){
            double const x      = LocalParticle_get_x(part);
            double const y      = LocalParticle_get_y(part);
            double const L_path = 0.5 * length * (1 + (hxl*x - hyl*y)/length);
            if (radiation_flag == 1){
                synrad_average_kick(part, curv, L_path,
                        &dp_record_entry, &dpx_record_entry, &dpy_record_entry);
            }
            else if (radiation_flag == 2){
                synrad_emit_photons(part, curv, L_path, record_index, record);
            }
        }
        #endif

        if( ( hxl > 0) || ( hyl > 0) || ( hxl < 0 ) || ( hyl < 0 ) )
        {
            double const delta  = LocalParticle_get_delta(part);
            double const chi    = LocalParticle_get_chi(part);
            double const x      = LocalParticle_get_x(part);
            double const y      = LocalParticle_get_y(part);

            double const hxlx   = x * hxl;
            double const hyly   = y * hyl;

            double const rv0v = 1./LocalParticle_get_rvv(part);

            dpx += (hxl + hxl * delta);
            dpy -= (hyl + hyl * delta);

            if( length != 0)
            {
                double b1l = backtrack_sign * chi * knl[0];
                double a1l = backtrack_sign * chi * ksl[0];

                b1l = b1l * (1 + delta_taper);
                a1l = a1l * (1 + delta_taper);

                dpx -= b1l * hxlx / length;
                dpy -= a1l * hyly / length;
            }

            LocalParticle_add_to_zeta(part, rv0v*chi * ( hyly - hxlx ) );
        }

        LocalParticle_add_to_px(part, dpx);
        LocalParticle_add_to_py(part, dpy);

        // Radiation at exit
        #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        if (radiation_flag > 0 && length > 0){
            double const x      = LocalParticle_get_x(part);
            double const y      = LocalParticle_get_y(part);
            double const L_path = 0.5*length * (1 + (hxl*x - hyl*y)/length);
            if (radiation_flag == 1){
                synrad_average_kick(part, curv, L_path,
                        &dp_record_exit, &dpx_record_exit, &dpy_record_exit);
            }
            else if (radiation_flag == 2){
                // printf("L_path = %e curv = %e\n", L_path, curv);
                synrad_emit_photons(part, curv, L_path, record_index, record);
            }
        }
        #endif
    //end_per_particle_block
}

#endif
