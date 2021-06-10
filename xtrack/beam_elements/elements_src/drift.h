#ifndef XTRACK_DRIFT_H
#define XTRACK_DRIFT_H

/*gpufun*/
void Drift_track_local_particle(DriftData el, LocalParticle* part){

    double const length = DriftData_get_length(el);

    double const n_part = LocalParticle_get_num_particles(part); //only_for_context cpu_serial cpu_openmp
    for (int ii=0; ii<n_part; ii++){ //only_for_context cpu_serial cpu_openmp
	part->ipart = ii;            //only_for_context cpu_serial cpu_openmp


        double const rpp    = LocalParticle_get_rpp(part);
        double const xp     = LocalParticle_get_px(part) * rpp;
        double const yp     = LocalParticle_get_py(part) * rpp;
        double const dzeta  = LocalParticle_get_rvv(part) -
                               ( 1. + ( xp*xp + yp*yp ) / 2. );
    
        LocalParticle_add_to_x(part, xp * length );
        LocalParticle_add_to_y(part, yp * length );
        LocalParticle_add_to_s(part, length);
        LocalParticle_add_to_zeta(part, length * dzeta );
    } //only_for_context cpu_serial cpu_openmp


}

#endif
