// ##################################
// Beam Profile Monitor
// 
// Author: Philipp Niedermayer
// Date: 2023-08-15
// ##################################


#ifndef XTRACK_BEAM_PROFILE_MONITOR_H
#define XTRACK_BEAM_PROFILE_MONITOR_H

#if !defined( C_LIGHT )
    #define   C_LIGHT ( 299792458.0 )
#endif /* !defined( C_LIGHT ) */


/*gpufun*/
void BeamProfileMonitor_track_local_particle(BeamProfileMonitorData el, LocalParticle* part0){

    // get parameters
    int64_t const start_at_turn = BeamProfileMonitorData_get_start_at_turn(el);
    int64_t particle_id_start = BeamProfileMonitorData_get_particle_id_start(el);
    int64_t particle_id_stop = particle_id_start + BeamProfileMonitorData_get_num_particles(el);
    double const frev = BeamProfileMonitorData_get_frev(el);
    double const sampling_frequency = BeamProfileMonitorData_get_sampling_frequency(el);
    
    BeamProfileMonitorRecord record = BeamProfileMonitorData_getp_data(el);                 //only_for_context cpu_serial cpu_openmp
    /*gpuglmem*/ BeamProfileMonitorRecord * record = BeamProfileMonitorData_getp_data(el);  //only_for_context opencl cuda
    
    int64_t max_sample = BeamProfileMonitorData_get_sample_size(el);
    int64_t max_slot_x = BeamProfileMonitorRecord_len_counts_x(record);
    int64_t max_slot_y = BeamProfileMonitorRecord_len_counts_y(record);
    int64_t raster_size_x = BeamProfileMonitorData_get_raster_size_x(el);
    int64_t raster_size_y = BeamProfileMonitorData_get_raster_size_y(el);
    double raster_range_x0 = BeamProfileMonitorData_get_raster_range_x0(el);
    double raster_range_dx = BeamProfileMonitorData_get_raster_range_dx(el);
    double raster_range_y0 = BeamProfileMonitorData_get_raster_range_y0(el);
    double raster_range_dy = BeamProfileMonitorData_get_raster_range_dy(el);


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
            int64_t sample = round(sampling_frequency * ( (at_turn-start_at_turn)/frev - zeta/beta0/C_LIGHT ));

            if (sample >= 0 && sample < max_sample){
                double x = LocalParticle_get_x(part);
                double y = LocalParticle_get_y(part);

                // compute bin index in raster
                int64_t bin_x = floor((x - raster_range_x0) / raster_range_dx);
                int64_t bin_y = floor((y - raster_range_y0) / raster_range_dy);

                if (bin_x >= 0 && bin_x < raster_size_x){
                    int64_t slot_x = sample * raster_size_x + bin_x;

                    if (slot_x >= 0 && slot_x < max_slot_x){
                        /*gpuglmem*/ int64_t * counts_x = BeamProfileMonitorRecord_getp1_counts_x(record, slot_x);
                        #pragma omp atomic capture    //only_for_context cpu_openmp
                        (*counts_x) += 1;             //only_for_context cpu_serial cpu_openmp
                        atomic_add(counts_x, 1);      //only_for_context opencl
                        atomicAdd(counts_x, 1);       //only_for_context cuda
                    }
                }

                if (bin_y >= 0 && bin_y < raster_size_y){
                    int64_t slot_y = sample * raster_size_y + bin_y;

                    if (slot_y >= 0 && slot_y < max_slot_y){
                        /*gpuglmem*/ int64_t * counts_y = BeamProfileMonitorRecord_getp1_counts_y(record, slot_y);
                        #pragma omp atomic capture    //only_for_context cpu_openmp
                        (*counts_y) += 1;             //only_for_context cpu_serial cpu_openmp
                        atomic_add(counts_y, 1);      //only_for_context opencl
                        atomicAdd(counts_y, 1);       //only_for_context cuda
                    }
                }

            }
        }

	//end_per_particle_block

}

#endif

