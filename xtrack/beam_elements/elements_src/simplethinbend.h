// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_FASTDIPOLE_H
#define XTRACK_FASTDIPOLE_H



/*gpufun*/
void SimpleThinBend_track_local_particle(SimpleThinBendData el, LocalParticle* part0){
        // Horizontal bend

        double knl0 = SimpleThinBendData_get_knl(el, 0);
        double hxl = SimpleThinBendData_get_hxl(el);
        double length = SimpleThinBendData_get_length(el); // m

        #ifdef XSUITE_BACKTRACK
            knl0 = -knl0;
            hxl = -hxl;
            length = -length;
        #endif

        //start_per_particle_block (part0->part)
            double const chi = LocalParticle_get_chi(part);

            double dpx = - chi * knl0;

            if((hxl > 0) || (hxl < 0))
            {
                double const delta = LocalParticle_get_delta(part);
                double const x = LocalParticle_get_x(part);

                double const hxlx = x * hxl;

                double const rv0v = 1./LocalParticle_get_rvv(part);

                LocalParticle_add_to_zeta(part, rv0v*chi * (-hxlx));

                dpx += hxl + hxl * delta;

                if( length != 0)
                {
                    double const b1l = chi * knl0;
                    dpx -= b1l * hxlx / length;
                }
            }

            LocalParticle_add_to_px(part, dpx);
        //end_per_particle_block
}

#endif
