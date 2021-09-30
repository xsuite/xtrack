#ifndef XTRACK_TRACKER_H
#define XTRACK_TRACKER_H

#if !defined( CPUIMPLEM )
    #define CPUIMPLEM //only_for_context cpu_serial cpu_openmp
#endif /* !defined( CPUIMPLEM ) */

#if defined( XTRACK_GLOBAL_POSLIMIT )
/*gpufun*/
void global_aperture_check( LocalParticle* part0 ){
    //start_per_particle_block (part0->part)
        double const x = LocalParticle_get_x(part);
        double const y = LocalParticle_get_y(part);

        bool const is_alive = (
            ( x >= -( double )XTRACK_GLOBAL_POSLIMIT ) &&
            ( x <=  ( double )XTRACK_GLOBAL_POSLIMIT ) &&
            ( y >= -( double )XTRACK_GLOBAL_POSLIMIT ) &&
            ( y <=  ( double )XTRACK_GLOBAL_POSLIMIT ) );

        if( !is_alive ) LocalParticle_mark_as_lost( part );
    //end_per_particle_block
}

#else /*  !defined( XTRACK_GLOBAL_POSLIMIT ) */
/*gpufun*/ void global_aperture_check( LocalParticle* part0 ){ ( void )part0; }

#endif /* defined( XTRACK_GLOBAL_POSLIMIT ) */

/*gpufun*/
void increment_at_element(LocalParticle* part0){
   //start_per_particle_block (part0->part)
        LocalParticle_add_to_at_element( part, 1 );
   //end_per_particle_block
}

/*gpufun*/
void increment_at_turn(LocalParticle* part0){
    //start_per_particle_block (part0->part)
        LocalParticle_add_to_at_turn(part, 1);
        LocalParticle_set_at_element(part, 0);
    //end_per_particle_block
}

/*gpufun*/
bool check_is_active( LocalParticle* part ) {
    return LocalParticle_are_any_active( part ); }

/*gpufun*/
void sync_locally( void ) {
    __syncthreads(); //only_for_context cuda
    #if defined( __OPENCL_VERSION__ ) /* defined( __OPENCL_VERSION__ ) */
    barrier( CLK_LOCAL_MEM_FENCE ); //only_for_context opencl
    #endif /* defined( __OPENCL_VERSION__ ) */
}

#endif
