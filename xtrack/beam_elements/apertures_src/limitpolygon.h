#ifndef XTRACK_LIMITPOLYGON_H
#define XTRACK_LIMITPOLYGON_H

/*gpufun*/
void LimitPolygon_track_local_particle(LimitPolygonData el,
		LocalParticle* part0){

    int64_t N_edg = LimitPolygonData_len_x_vertices(el);

    //start_per_particle_block (part0->part)

        double const x = LocalParticle_get_x(part);
        double const y = LocalParticle_get_y(part);


	int ii = 0;
        int jj = N_edg-1;
	int64_t is_alive = 0;
        while (ii < N_edg){
           if (((Vy[ii]>y_curr) != (Vy[jj]>y_curr)) &&
                  (x_curr < (Vx[jj]-Vx[ii]) * (y_curr-Vy[ii])
		                   / (Vy[jj]-Vy[ii]) + Vx[ii])){
                
		is_alive = !is_alive;
                jj = ii;
                ii ++;

	// I assume that if I am in the function is because
	// the particle is alive
    	if (!is_alive){
           LocalParticle_set_state(part, 0);
	}

    //end_per_particle_block

}

#endif
