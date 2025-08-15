// ##################################
// Bunch Monitor
// 
// Author: Philipp Niedermayer, Cristopher Cortes
// Date: 2023-08-14
// Edit: 2024-06-12
// ##################################


#ifndef XTRACK_BUNCH_MONITOR_H
#define XTRACK_BUNCH_MONITOR_H

#if !defined( C_LIGHT )
    #define   C_LIGHT ( 299792458.0 )
#endif /* !defined( C_LIGHT ) */

/*gpufun*/
void BunchMonitor_track_local_particle(BunchMonitorData el, LocalParticle* part0){

    // get parameters
    int64_t const start_at_turn = BunchMonitorData_get_start_at_turn(el);
    int64_t particle_id_start = BunchMonitorData_get_particle_id_start(el);
    int64_t particle_id_stop = particle_id_start + BunchMonitorData_get_num_particles(el);
    int64_t const harmonic = BunchMonitorData_get_harmonic(el);
    double const frev = BunchMonitorData_get_frev(el);
    
    BunchMonitorRecord record = BunchMonitorData_getp_data(el);

    int64_t max_slot = BunchMonitorRecord_len_count(record);

    //start_per_particle_block(part0->part)

        int64_t particle_id = LocalParticle_get_particle_id(part);
        if (particle_id_stop < 0 || (particle_id_start <= particle_id && particle_id < particle_id_stop)){

            // zeta is the absolute path length deviation from the reference particle: zeta = (s - beta0*c*t)
            // but without limits, i.e. it can exceed the circumference (for coasting beams)
            // as the particle falls behind or overtakes the reference particle
            double const zeta = LocalParticle_get_zeta(part);
            double const at_turn = LocalParticle_get_at_turn(part);
            double const beta0 = LocalParticle_get_beta0(part);

            // compute sample index
            int64_t slot = round( harmonic * ( (at_turn-start_at_turn) - frev * zeta/beta0/C_LIGHT ));

            if (slot >= 0 && slot < max_slot){

                double const delta = LocalParticle_get_delta(part);

                /*gpuglmem*/ double* count = BunchMonitorRecord_getp1_count(record, slot);
                atomicAdd(count, 1);

                /*gpuglmem*/ double * zeta_sum = BunchMonitorRecord_getp1_zeta_sum(record, slot);
                atomicAdd(zeta_sum, zeta);

                /*gpuglmem*/ double * delta_sum = BunchMonitorRecord_getp1_delta_sum(record, slot);
                atomicAdd(delta_sum, delta);

                /*gpuglmem*/ double * zeta2_sum = BunchMonitorRecord_getp1_zeta2_sum(record, slot);
                atomicAdd(zeta2_sum, zeta*zeta);

                /*gpuglmem*/ double * delta2_sum = BunchMonitorRecord_getp1_delta2_sum(record, slot);
                atomicAdd(delta2_sum, delta*delta);
            }
        }
        
	//end_per_particle_block
}

#endif

