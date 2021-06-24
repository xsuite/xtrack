#ifndef XTRACK_DIPOLEEDGE_H
#define XTRACK_DIPOLEEDGE_H

/*gpufun*/
void DipoleEdge_track_local_particle(DipoleEdgeData el, LocalParticle* part){
    
    double const r21 = DipoleEdgeData_get_r21(el);
    double const r43 = DipoleEdgeData_get_r43(el);
	    
    int64_t const n_part = LocalParticle_get_num_particles(part); //only_for_context cpu_serial cpu_openmp
    for (int ii=0; ii<n_part; ii++){ //only_for_context cpu_serial cpu_openmp
	part->ipart = ii;            //only_for_context cpu_serial cpu_openmp

	double const x = LocalParticle_get_x(part);
	double const y = LocalParticle_get_y(part);

	LocalParticle_add_to_px(part, r21*x);
	LocalParticle_add_to_py(part, r43*y);

    } //only_for_context cpu_serial cpu_openmp

}

#endif
