#ifndef XTRACK_LIMITPOLYGON_H
#define XTRACK_LIMITPOLYGON_H

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
           //printf("ii=%d\n", (int)ii); 
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
           LocalParticle_set_state(part, 0);
	}

    //end_per_particle_block

}

#endif
