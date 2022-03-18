#ifndef XTRACK_FIRSTORDERTAYLORMAP_H
#define XTRACK_FIRSTORDERTAYLORMAP_H

#include <stdio.h>

/*gpufun*/
void FirstOrderTaylorMap_track_local_particle(FirstOrderTaylorMapData el, LocalParticle* part0){

    int64_t radiation_flag = FirstOrderTaylorMapData_get_radiation_flag(el);

    //start_per_particle_block (part0->part)
        for(unsigned int i = 0; i<6;++i){
            printf("m0,%i,%E\n",i,FirstOrderTaylorMapData_get_m0(el,i));
        }
        for(unsigned int i = 0; i<6;++i){
            for(unsigned int j = 0;j<6;++j){
                printf("m1,%i/%i,%E\n",i,j,FirstOrderTaylorMapData_get_m1(el));
            }
        }
    //end_per_particle_block
}

#endif
