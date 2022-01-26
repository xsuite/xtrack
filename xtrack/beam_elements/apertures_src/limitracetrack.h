#ifndef XTRACK_LIMITRACETRACK_H
#define XTRACK_LIMITRACETRACK_H


/*gpufun*/
void LimitRacetrack_track_local_particle(LimitRacetrackData el, LocalParticle* part0){

    double const min_x = LimitRacetrackData_get_min_x(el);
    double const max_x = LimitRacetrackData_get_max_x(el);
    double const min_y = LimitRacetrackData_get_min_y(el);
    double const max_y = LimitRacetrackData_get_max_y(el);
    double const a = LimitRacetrackData_get_a(el);
    double const b = LimitRacetrackData_get_b(el);


    //start_per_particle_block (part0->part)

        double const x = LocalParticle_get_x(part);
        double const y = LocalParticle_get_y(part);
        double dx, dy;
        int refine;

	    int64_t is_alive = (int64_t)(
                      (x >= min_x) &&
		              (x <= max_x) &&
		              (y >= min_y) &&
		              (y <= max_y) );
        if (is_alive){
            if (((max_x - x) < a) && ((max_y - y) < b)){
                refine = 1;
                dx = x - (max_x - a);
                dy = y - (max_y - b);
            }
            else if (((x - min_x) < a) && ((max_y - y) < b)){
                refine = 1;
                dx = x - (min_x + a);
                dy = y - (max_y - b);
            }
            else if (((x - min_x) < a) && ((y - min_y) < b)){
                refine = 1;
                dx = x - (min_x + a);
                dy = y - (min_y + b);
            }
            else if (((max_x - x) < a) && ((y - min_y) < b)){
                refine = 1;
                dx = x - (max_x - a);
                dy = y - (min_y + b);
            }
            else{
                refine = 0;
            }
            if (refine){
	            double const temp = dx * dx * b * b + dy * dy * a * a;
                is_alive = (int64_t)( temp <= a*a*b*b );
            }
        }

    	if (!is_alive){
           LocalParticle_set_state(part, 0);
        }

    //end_per_particle_block
}

#endif
