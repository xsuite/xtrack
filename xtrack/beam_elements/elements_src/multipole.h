// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_MULTIPOLE_H
#define XTRACK_MULTIPOLE_H

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
        #ifndef XTRACK_MULTIPOLE_TAPER
        double const delta_taper = MultipoleData_get_delta_taper(el);
        #endif
    #endif

    //start_per_particle_block (part0->part)
        int64_t order = MultipoleData_get_order(el);
        int64_t index = order;

        double inv_factorial = MultipoleData_get_inv_factorial_order(el);

        double dpx = MultipoleData_get_knl(el, index) * inv_factorial;
        double dpy = MultipoleData_get_ksl(el, index) * inv_factorial;

        #ifdef XSUITE_BACKTRACK
        dpx = -dpx;
        dpy = -dpy;
        #endif

        #ifdef XTRACK_MULTIPOLE_TAPER
        double const delta_taper = LocalParticle_get_delta(part);
        #endif
        dpx = dpx * (1 + delta_taper);
        dpy = dpy * (1 + delta_taper);

        double const x   = LocalParticle_get_x(part);
        double const y   = LocalParticle_get_y(part);
        double const chi = LocalParticle_get_chi(part);

        #ifndef XSUITE_BACKTRACK
        double const hxl = MultipoleData_get_hxl(el);
        double const hyl = MultipoleData_get_hyl(el);
        #else
        double const hxl = -MultipoleData_get_hxl(el);
        double const hyl = -MultipoleData_get_hyl(el);
        #endif

        while( index > 0 )
        {
            double const zre = dpx * x - dpy * y;
            double const zim = dpx * y + dpy * x;

            inv_factorial *= index;
            index -= 1;

            double this_knl = MultipoleData_get_knl(el, index);
            double this_ksl = MultipoleData_get_ksl(el, index);

            #ifdef XSUITE_BACKTRACK
            this_knl = -this_knl;
            this_ksl = -this_ksl;
            #endif

            this_knl = this_knl * (1 + delta_taper);
            this_ksl = this_ksl * (1 + delta_taper);

            dpx = this_knl*inv_factorial + zre;
            dpy = this_ksl*inv_factorial + zim;
        }

        #ifndef XSUITE_BACKTRACK
        double const length = MultipoleData_get_length(el); // m
        #else
        double const length = -MultipoleData_get_length(el); // m
        #endif

        #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        // Radiation at entrance
        double const curv = sqrt(dpx*dpx + dpy*dpy) / length;
        if (radiation_flag > 0 && length > 0){
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

        dpx = -chi * dpx; // rad
        dpy =  chi * dpy; // rad

        if( ( hxl > 0) || ( hyl > 0) || ( hxl < 0 ) || ( hyl < 0 ) )
        {
            double const delta  = LocalParticle_get_delta(part);

            double const hxlx   = x * hxl;
            double const hyly   = y * hyl;

            double const rv0v = 1./LocalParticle_get_rvv(part);

            dpx += (hxl + hxl * delta);
            dpy -= (hyl + hyl * delta);

            if( length != 0)
            {
                double b1l = chi * MultipoleData_get_knl(el, 0 );
                double a1l = chi * MultipoleData_get_ksl(el, 0 );

                #ifdef XSUITE_BACKTRACK
                b1l = -b1l;
                a1l = -a1l;
                #endif

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
