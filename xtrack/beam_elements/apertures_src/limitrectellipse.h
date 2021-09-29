#ifndef XTRACK_LIMITRECT_H
#define XTRACK_LIMITRECT_H

/*gpufun*/
void LimitRectEllipse_track_local_particle(
    LimitRectEllipseData el, LocalParticle* part0 ){

    double const max_x = LimitRectEllipse_get_max_x( el );
    double const max_y = LimitRectEllipse_get_max_y( el );
    double const a_squ = LimitRectEllipse_get_a_squ( el );
    double const b_squ = LimitRectEllipse_get_b_squ( el );
    double const a_b_squ = LimitRectEllipse_get_a_b_squ( el );

    //start_per_particle_block (part0->part)

        double const x = LocalParticle_get_x( part );
        double const y = LocalParticle_get_y( part );

        if( ( x <= max_x ) && ( x >= -max_x ) &&
            ( y <= max_y ) && ( y >= -max_y ) &&
            ( ( ( x * x * b_squ ) + ( y * y * a_squ ) ) <= a_b_squ ) )
        {
            LocalParticle_set_state( part, 0 );
        }

    //end_per_particle_block
}

#endif /* XTRACK_LIMITRECT_H */
