// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //

#ifndef XTRACK_MULTI_ELEMENT_MONITOR_H
#define XTRACK_MULTI_ELEMENT_MONITOR_H

#include "xtrack/headers/track.h"

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

                // Truncation constant part of TPSA map or takes scalar directly
                // if scalar tracking.
                double const x = xt_num_truncate_to_double(LocalParticle_get_x(part));
                double const px = xt_num_truncate_to_double(LocalParticle_get_px(part));
                double const y = xt_num_truncate_to_double(LocalParticle_get_y(part));
                double const py = xt_num_truncate_to_double(LocalParticle_get_py(part));
                double const zeta = xt_num_truncate_to_double(LocalParticle_get_zeta(part));
                double const delta = xt_num_truncate_to_double(LocalParticle_get_delta(part));
                double const s = xt_num_truncate_to_double(LocalParticle_get_s(part));

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
                MultiElementMonitorData_set_data(
                    el, turn_index, particle_index, 6, store_at, s);

#ifdef XTRACK_TPSA_TRACK
                // `data` only holds the constant part, so the map is recorded on
                // the side: either whole into preallocated series, or as the
                // requested coefficients. Never both.
                int64_t const num_slots =
                    MultiElementMonitorData_len_monomial_indices(el);
                if (MultiElementMonitorData_len_tpsa_addresses(el) > 0){
                    #define XT_MONITOR_STORE_MAP(INDEX, NAME)                  \
                        mad_tpsa_copy(part->NAME, (tpsa_t*)(uintptr_t)         \
                            MultiElementMonitorData_get_tpsa_addresses(        \
                                el, turn_index, store_at, INDEX));

                    XT_MONITOR_STORE_MAP(0, x)
                    XT_MONITOR_STORE_MAP(1, px)
                    XT_MONITOR_STORE_MAP(2, y)
                    XT_MONITOR_STORE_MAP(3, py)
                    XT_MONITOR_STORE_MAP(4, zeta)
                    XT_MONITOR_STORE_MAP(5, delta)

                    #undef XT_MONITOR_STORE_MAP
                } else if (num_slots > 0){
                    tpsa_t* const series[6] = {part->x, part->px, part->y,
                                               part->py, part->zeta, part->delta};
                    for (int64_t slot = 0; slot < num_slots; slot++){
                        int64_t const coord =
                            MultiElementMonitorData_get_coord_indices(el, slot);
                        int64_t const coefficient_index =
                            MultiElementMonitorData_get_monomial_indices(el, slot);
                        MultiElementMonitorData_set_coefficients(
                            el, turn_index, store_at, slot,
                            mad_tpsa_geti(series[coord], coefficient_index));
                    }
                }
#endif
            }
        }
    END_PER_PARTICLE_BLOCK;
}

#endif
