#ifndef XTRACK_DRIFT_EXACT_H
#define XTRACK_DRIFT_EXACT_H

/*gpufun*/
void Drift_track_local_particle(DriftData el, LocalParticle* part0){

    double const length = DriftExactData_get_length( el );

    //start_per_particle_block (part0->part)

        double const px = LocalParticle_get_px( part );
        double const py = LocalParticle_get_py( part );
        double const delta_plus_1 = LocalParticle_get_delta( part )
                                  + ( double )1.0;

        double const lpzi = length / sqrt(
            ( delta_plus_1 * delta_plus_1 ) - ( px * px ) - ( py * py ) );

        LocalParticle_add_to_s(part, length);
        LocalParticle_add_to_x(part, px * lpzi );
        LocalParticle_add_to_y(part, py * lpzi );
        LocalParticle_add_to_zeta( part,
            LocalParticle_get_rvv( part ) * length - delta_plus_1 * lpzi; );

    //end_per_particle_block
}

#endif /* XTRACK_DRIFT_EXACT_H */
