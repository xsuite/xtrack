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
    double dp_record_entry, dpx_record_entry, dpy_record_entry;
    double dp_record_exit, dpx_record_exit, dpy_record_exit;
    #endif

    //start_per_particle_block (part0->part)
        int64_t order = MultipoleData_get_order(el);
        int64_t index = order;

        double inv_factorial = MultipoleData_get_inv_factorial_order(el);

        double dpx = MultipoleData_get_knl(el, index) * inv_factorial;
        double dpy = MultipoleData_get_ksl(el, index) * inv_factorial;

        #ifdef XTRACK_MULTIPOLE_TAPER
        double const delta_taper = LocalParticle_get_delta(part);
        dpx = dpx * (1 + delta_taper);
        dpy = dpy * (1 + delta_taper);
        #endif

        double const x   = LocalParticle_get_x(part);
        double const y   = LocalParticle_get_y(part);
        double const chi = LocalParticle_get_chi(part);

        double const hxl = MultipoleData_get_hxl(el);
        double const hyl = MultipoleData_get_hyl(el);

        while( index > 0 )
        {
            double const zre = dpx * x - dpy * y;
            double const zim = dpx * y + dpy * x;

            inv_factorial *= index;
            index -= 1;

            double this_knl = MultipoleData_get_knl(el, index);
            double this_ksl = MultipoleData_get_ksl(el, index);

            #ifdef XTRACK_MULTIPOLE_TAPER
            this_knl = this_knl * (1 + delta_taper);
            this_ksl = this_ksl * (1 + delta_taper);
            #endif

            dpx = this_knl*inv_factorial + zre;
            dpy = this_ksl*inv_factorial + zim;
        }


        double const length = MultipoleData_get_length(el); // m

        #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        // Radiation at entrance
        double const curv = sqrt(dpx*dpx + dpy*dpy) / length;
        if (radiation_flag > 0 && length > 0){
            double const L_path = 0.5*length*(1 + (hxl*x - hyl*y)/length); //CHECK!!!!
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

            LocalParticle_add_to_zeta(part, rv0v*chi * ( hyly - hxlx ) );

            dpx += hxl + hxl * delta;
            dpy -= hyl + hyl * delta;

            if( length != 0)
            {
                double b1l = chi * MultipoleData_get_knl(el, 0 );
                double a1l = chi * MultipoleData_get_ksl(el, 0 );

                #ifdef XTRACK_MULTIPOLE_TAPER
                b1l = b1l * (1 + delta_taper);
                a1l = b1l * (1 + delta_taper);
                #endif

                dpx -= b1l * hxlx / length;
                dpy += a1l * hyly / length;
            }
        }

        LocalParticle_add_to_px(part, dpx);
        LocalParticle_add_to_py(part, dpy);

        // Radiation at exit
        #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        if (radiation_flag > 0 && length > 0){
            double const L_path = 0.5*length*(1 + (hxl*x - hyl*y)/length); //CHECK!!!!
            if (radiation_flag == 1){
                synrad_average_kick(part, curv, L_path,
                        &dp_record_exit, &dpx_record_exit, &dpy_record_exit);
            }
            else if (radiation_flag == 2){
                synrad_emit_photons(part, curv, L_path, record_index, record);
            }
        }
        #endif
    //end_per_particle_block
}

#endif
