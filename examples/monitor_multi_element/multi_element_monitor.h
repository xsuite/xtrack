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

    ArrNInt64 const at_element_mapping = MultiElementMonitorData_getp_at_element_mapping(el);
    ArrNxMxOxPFloat64 const data = MultiElementMonitorData_getp_data(el);

    int64_t const map_len = ArrNInt64_len(at_element_mapping);
    int64_t data_shape[4];
    ArrNxMxOxPFloat64_shape(data, data_shape);
    int64_t const n_locations = data_shape[3];

    START_PER_PARTICLE_BLOCK(part0, part);
        int64_t const at_turn = LocalParticle_get_at_turn(part);
        if (at_turn >= start_at_turn && at_turn < stop_at_turn){
            int64_t const at_element = LocalParticle_get_at_element(part);
            if (at_element < map_len){
                int64_t const location_index = ArrNInt64_get(at_element_mapping, at_element);
                if (location_index >= 0 && location_index < n_locations){
                    int64_t const particle_id = LocalParticle_get_particle_id(part);
                    if (particle_id >= part_id_start && particle_id < part_id_end){
                        int64_t const turn_index = at_turn - start_at_turn;
                        int64_t const particle_index = particle_id - part_id_start;

                        double const x = LocalParticle_get_x(part);
                        double const px = LocalParticle_get_px(part);
                        double const y = LocalParticle_get_y(part);
                        double const py = LocalParticle_get_py(part);
                        double const zeta = LocalParticle_get_zeta(part);
                        double const delta = LocalParticle_get_delta(part);

                        ArrNxMxOxPFloat64_set(data, turn_index, particle_index, 0, location_index, x);
                        ArrNxMxOxPFloat64_set(data, turn_index, particle_index, 1, location_index, px);
                        ArrNxMxOxPFloat64_set(data, turn_index, particle_index, 2, location_index, y);
                        ArrNxMxOxPFloat64_set(data, turn_index, particle_index, 3, location_index, py);
                        ArrNxMxOxPFloat64_set(data, turn_index, particle_index, 4, location_index, zeta);
                        ArrNxMxOxPFloat64_set(data, turn_index, particle_index, 5, location_index, delta);
                    }
                }
            }
        }
    END_PER_PARTICLE_BLOCK;
}

#endif
