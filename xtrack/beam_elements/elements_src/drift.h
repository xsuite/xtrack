#ifndef XTRACK_DRIFT_H
#define XTRACK_DRIFT_H

/*gpufun*/
void Drift_track_local_particle(DriftData el, LocalParticle* part0){

    double const length = DriftData_get_length(el);

    int64_t const n_part = LocalParticle_get_num_particles(part0); //only_for_context cpu_serial cpu_openmp
    #pragma omp parallel for//only_for_context cpu_openmp
    for (int jj=0; jj<n_part; jj+=64){
    for (int ii=jj; ii<jj+64 && ii<n_part; ii++){ //only_for_context cpu_serial cpu_openmp
	LocalParticle lpart = *part0;
	LocalParticle* part = &lpart;
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

}

#endif
