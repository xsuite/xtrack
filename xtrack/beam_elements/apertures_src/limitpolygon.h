// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_LIMITPOLYGON_H
#define XTRACK_LIMITPOLYGON_H

#ifndef NO_LIMITPOLYGON_TRACK_LOCAL_PARTICLE
/*gpufun*/
void LimitPolygon_track_local_particle(LimitPolygonData el,
		LocalParticle* part0){

    int64_t N_edg = LimitPolygonData_len_x_vertices(el);

    //start_per_particle_block (part0->part)

        double const x = LocalParticle_get_x(part);
        double const y = LocalParticle_get_y(part);


	int64_t ii = 0;
        int64_t jj = N_edg-1;
	int64_t is_alive = 0;
        while (ii < N_edg){
	   const double Vx_ii = LimitPolygonData_get_x_vertices(el, ii);
	   const double Vx_jj = LimitPolygonData_get_x_vertices(el, jj);
	   const double Vy_ii = LimitPolygonData_get_y_vertices(el, ii);
	   const double Vy_jj = LimitPolygonData_get_y_vertices(el, jj);
           if (((Vy_ii>y) != (Vy_jj>y)) &&
                  (x < (Vx_jj-Vx_ii) * (y-Vy_ii)
		          / (Vy_jj-Vy_ii) + Vx_ii)){
                
		is_alive = !is_alive;
	   }
           jj = ii;
           ii ++;
	}

	// I assume that if I am in the function is because
	// the particle is alive
    	if (!is_alive){
           LocalParticle_set_state(part, XT_LOST_ON_APERTURE);
	}

    //end_per_particle_block

}
#endif

/*gpukern*/
void LimitPolygon_impact_point_and_normal(
		             LimitPolygonData el,
                /*gpuglmem*/ const double* x_in,
                /*gpuglmem*/ const double* y_in, 
                /*gpuglmem*/ const double* z_in,
                /*gpuglmem*/ const double* x_out,
                /*gpuglmem*/ const double* y_out, 
                /*gpuglmem*/ const double* z_out,
                             const int64_t n_impacts,
                /*gpuglmem*/       double* x_inters,
                /*gpuglmem*/       double* y_inters, 
                /*gpuglmem*/       double* z_inters,
                /*gpuglmem*/       double* Nx_inters,
                /*gpuglmem*/       double* Ny_inters,
                /*gpuglmem*/       int64_t* i_found){ 
			           
    double const tol = 1e-13;

    int64_t N_edg = LimitPolygonData_len_x_vertices(el);
    /*gpuglmem*/ const double* Vx = LimitPolygonData_getp1_x_vertices(el, 0);
    /*gpuglmem*/ const double* Vy = LimitPolygonData_getp1_y_vertices(el, 0);
    /*gpuglmem*/ const double* Nx = LimitPolygonData_getp1_x_normal(el, 0);
    /*gpuglmem*/ const double* Ny = LimitPolygonData_getp1_y_normal(el, 0);
    double resc_fac = LimitPolygonData_get_resc_fac(el);

    for (int64_t i_imp=0; i_imp<n_impacts; i_imp++){ //vectorize_over i_imp n_impacts

        double t_min_curr = 1.;
        int64_t i_found_curr = -1;
        double x_in_curr = x_in[i_imp];
        double y_in_curr = y_in[i_imp];
        double x_out_curr = x_out[i_imp];
        double y_out_curr = y_out[i_imp];

        for (int64_t ii=0; ii<N_edg; ii++){

	    double t_border;
	    double t_ii;
            double const den = ((y_out_curr-y_in_curr)*(Vx[ii+1]-Vx[ii])
			    +(x_in_curr-x_out_curr)*(Vy[ii+1]-Vy[ii]));
            if (den == 0.){
                // it is the case when the normal top the segment is perpendicular to the edge
                // the case case overlapping the edge is not possible (this would not allow Pin inside and Pout outside - a point on the edge is condidered outside)
                // the only case left is segment parallel to tue edge => no intersection
                t_border = -2.;
	    }
            else{
                t_border=((y_out_curr-y_in_curr)*(x_in_curr-Vx[ii])
		         +(x_in_curr-x_out_curr)*(y_in_curr-Vy[ii]))/den;
	    }

            if (t_border>=0.-tol && t_border<=1.+tol){
                t_ii = (Nx[ii]*(Vx[ii]-x_in_curr)
		       +Ny[ii]*(Vy[ii]-y_in_curr)) 
		       /(Nx[ii]*(x_out_curr-x_in_curr)
	                 +Ny[ii]*(y_out_curr-y_in_curr));
                if (t_ii>=0.-tol && t_ii<t_min_curr+tol){
                    t_min_curr=t_ii;
                    i_found_curr = ii;
		}
            }
	}

        t_min_curr=resc_fac*t_min_curr;
        x_inters[i_imp]=t_min_curr*x_out_curr+(1.-t_min_curr)*x_in_curr;
        y_inters[i_imp]=t_min_curr*y_out_curr+(1.-t_min_curr)*y_in_curr;
        z_inters[i_imp]=0;

        if (i_found_curr>=0){
            Nx_inters[i_imp] = Nx[i_found_curr];
            Ny_inters[i_imp] = Ny[i_found_curr];
	}
        i_found[i_imp] = i_found_curr;
    } //end_vectorize
    
}

#endif
