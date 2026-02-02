// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //

#ifndef XTRACK_MULTI_ELEMENT_MONITOR_H
#define XTRACK_MULTI_ELEMENT_MONITOR_H

#include <headers/track.h>

GPUFUN
void MultiElementMonitor_track_local_particle(MultiElementMonitorData el,
                       LocalParticle* part0){

    int64_t const start_at_turn = MultiElementMonitorData_get_start_at_turn(el);
    int64_t const stop_at_turn = MultiElementMonitorData_get_stop_at_turn(el);
    int64_t const part_id_start = MultiElementMonitorData_get_part_id_start(el);
    int64_t const part_id_end = MultiElementMonitorData_get_part_id_end(el);

    int64_t mapping_len = MultiElementMonitorData_len_at_element_mapping(el);

    START_PER_PARTICLE_BLOCK(part0, part);
        int64_t const at_turn = LocalParticle_get_at_turn(part);
        int64_t const particle_id = LocalParticle_get_particle_id(part);
        if ((at_turn >= start_at_turn && at_turn < stop_at_turn)
            && (particle_id >= part_id_start && particle_id < part_id_end)){
            int64_t const at_element = LocalParticle_get_at_element(part);

            int64_t store_at = -1;
            if (at_element < mapping_len){
                store_at = MultiElementMonitorData_get_at_element_mapping(el, at_element);
            }

            if (store_at >=0) {
                int64_t const turn_index = at_turn - start_at_turn;
                int64_t const particle_index = particle_id - part_id_start;

                double const x = LocalParticle_get_x(part);
                double const px = LocalParticle_get_px(part);
                double const y = LocalParticle_get_y(part);
                double const py = LocalParticle_get_py(part);
                double const zeta = LocalParticle_get_zeta(part);
                double const delta = LocalParticle_get_delta(part);

                MultiElementMonitorData_set_data(
                    el, turn_index, particle_index, 0, store_at, x);
                MultiElementMonitorData_set_data(
                    el, turn_index, particle_index, 1, store_at, px);
                MultiElementMonitorData_set_data(
                    el, turn_index, particle_index, 2, store_at, y);
                MultiElementMonitorData_set_data(
                    el, turn_index, particle_index, 3, store_at, py);
                MultiElementMonitorData_set_data(
                    el, turn_index, particle_index, 4, store_at, zeta);
                MultiElementMonitorData_set_data(
                    el, turn_index, particle_index, 5, store_at, delta);
            }
        }
    END_PER_PARTICLE_BLOCK;
}

#endif
