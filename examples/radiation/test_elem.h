#ifndef XTRACK_TESTELEM_H
#define XTRACK_TESTELEM_H

/*gpufun*/
void TestElement_track_local_particle(TestElementData el, LocalParticle* part0){

    //start_per_particle_block (part0->part)

        double rr = LocalParticle_generate_random_double(part); 

        LocalParticle_set_x(part, rr);
    //end_per_particle_block

}

#endif
