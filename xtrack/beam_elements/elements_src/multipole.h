#ifndef XTRACK_MULTIPOLE_H
#define XTRACK_MULTIPOLE_H

/*gpufun*/
void Multipole_track_local_particle(MultipoleData el, LocalParticle* part0){

   //start_per_particle_block (part0->part)
    	int64_t order = MultipoleData_get_order(el);
    	int64_t index_x = 2 * order;
    	int64_t index_y = index_x + 1;

    	double dpx = MultipoleData_get_bal(el, index_x);
    	double dpy = MultipoleData_get_bal(el, index_y);

    	double const x   = LocalParticle_get_x(part);
    	double const y   = LocalParticle_get_y(part);
    	double const chi = LocalParticle_get_chi(part);

	double const hxl = MultipoleData_get_hxl(el);
    	double const hyl = MultipoleData_get_hyl(el);

    	while( index_x > 0 )
    	{
    	    double const zre = dpx * x - dpy * y;
    	    double const zim = dpx * y + dpy * x;

    	    index_x -= 2;
    	    index_y -= 2;

    	    dpx = MultipoleData_get_bal(el, index_x) + zre;
    	    dpy = MultipoleData_get_bal(el, index_y) + zim;
    	}

    	dpx = -chi * dpx;
    	dpy =  chi * dpy;

    	if( ( hxl > 0) || ( hyl > 0) || ( hxl < 0 ) || ( hyl < 0 ) )
    	{
    	    double const delta  = LocalParticle_get_delta(part);
    	    double const length = MultipoleData_get_length(el);

    	    double const hxlx   = x * hxl;
    	    double const hyly   = y * hyl;

    	    LocalParticle_add_to_zeta(part, chi * ( hyly - hxlx ) );

    	    dpx += hxl + hxl * delta;
    	    dpy -= hyl + hyl * delta;

    	    if( length > 0 )
    	    {
    	        double const b1l = chi * MultipoleData_get_bal(el, 0 );
    	        double const a1l = chi * MultipoleData_get_bal(el, 1 );

    	        dpx -= b1l * hxlx / length;
    	        dpy += a1l * hyly / length;
    	    }
    	}

    	LocalParticle_add_to_px(part, dpx );
    	LocalParticle_add_to_py(part, dpy );

    //end_per_particle_block
}

#endif
