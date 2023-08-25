// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_DIPOLEEDGE_H
#define XTRACK_DIPOLEEDGE_H

/*gpufun*/
void DipoleEdge_track_local_particle(DipoleEdgeData el, LocalParticle* part0){

    int64_t const model = DipoleEdgeData_get_model(el);

    #ifdef XTRACK_MULTIPOLE_NO_SYNRAD
    #define delta_taper (0)
    #else
        #ifndef XTRACK_DIPOLEEDGE_TAPER
        double const delta_taper = DipoleEdgeData_get_delta_taper(el);
        #endif
    #endif

    if (model == 0){
        double r21 = DipoleEdgeData_get_r21(el);
        double r43 = DipoleEdgeData_get_r43(el);

        #ifndef XTRACK_DIPOLEEDGE_TAPER
            r21 = r21 * (1 + delta_taper);
            r43 = r43 * (1 + delta_taper);
        #endif

        #ifdef XSUITE_BACKTRACK
            r21 = -r21;
            r43 = -r43;
        #endif

        //start_per_particle_block (part0->part)
            double const x = LocalParticle_get_x(part);
            double const y = LocalParticle_get_y(part);

            #ifdef XTRACK_DIPOLEEDGE_TAPER
                double const delta_taper = LocalParticle_get_delta(part);
                r21 = r21 * (1 + delta_taper);
                r43 = r43 * (1 + delta_taper);
            #endif

            LocalParticle_add_to_px(part, r21*x);
            LocalParticle_add_to_py(part, r43*y);

        //end_per_particle_block

    }
    else if (model == 1){

        #ifdef XSUITE_BACKTRACK
            //start_per_particle_block (part0->part)
                LocalParticle_kill_particle(part, -32);
            //end_per_particle_block
            return;
        #else

        double const e1 = DipoleEdgeData_get_e1(el);
        double const fint = DipoleEdgeData_get_fint(el);
        double const hgap = DipoleEdgeData_get_hgap(el);
        double const k = DipoleEdgeData_get_k(el);
        int64_t const side = DipoleEdgeData_get_side(el);

        double sin_, cos_, tan_;
        if (fabs(e1) < 10e-10) {
            sin_ = -999.0; cos_ = -999.0; tan_ = -999.0;
        }
        else{
            sin_ = sin(e1); cos_ = cos(e1); tan_ = tan(e1);
        }

        if (side == 0){ // entry
            if (sin_ > -99.){
                //start_per_particle_block (part0->part)
                    YRotation_single_particle(part, sin_, cos_, tan_);
                //end_per_particle_block
            }
            //start_per_particle_block (part0->part)
                Fringe_single_particle(part, fint, hgap, k);
            //end_per_particle_block
            if (sin_ > -99.){
                //start_per_particle_block (part0->part)
                    Wedge_single_particle(part, -e1, k);
                //end_per_particle_block
            }
        }
        else if (side == 1){ // exit
            if (sin_ > -99.){
                //start_per_particle_block (part0->part)
                    Wedge_single_particle(part, -e1, k);
                //end_per_particle_block
            }
            //start_per_particle_block (part0->part)
                Fringe_single_particle(part, fint, hgap, -k);
            //end_per_particle_block
            if (sin_ > -99.){
                //start_per_particle_block (part0->part)
                    YRotation_single_particle(part, sin_, cos_, tan_);
                //end_per_particle_block
            }

        }
        #endif
    }

}

#endif
