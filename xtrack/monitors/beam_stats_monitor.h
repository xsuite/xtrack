// copyright ############################### //
// This file is part of the Xtrack Package.  //
// copyright ############################### //

#ifndef XTRACK_BEAM_STATS_MONITOR_H
#define XTRACK_BEAM_STATS_MONITOR_H

#include "xtrack/headers/track.h"


GPUFUN
void BeamStatsMonitor_track_local_particle(
        BeamStatsMonitorData el,
        LocalParticle* part0
) {
    int64_t const start_at_turn =
        BeamStatsMonitorData_get_start_at_turn(el);
    int64_t const stop_at_turn =
        BeamStatsMonitorData_get_stop_at_turn(el);
    int64_t const every_n_turns =
        BeamStatsMonitorData_get_every_n_turns(el);
    int64_t const mode = BeamStatsMonitorData_get__mode(el);
    int64_t const n_records =
        BeamStatsMonitorData_get__num_records(el);
    int64_t const n_selected =
        BeamStatsMonitorData_get__num_selected_slots(el);
    int64_t const n_slices =
        BeamStatsMonitorData_get__num_slices(el);
    double const z_min_edge =
        BeamStatsMonitorData_get__z_min_edge(el);
    double const dzeta =
        BeamStatsMonitorData_get__dzeta(el);
    double const bunch_spacing_zeta =
        BeamStatsMonitorData_get__bunch_spacing_zeta(el);

    BeamStatsMonitorRecord data = BeamStatsMonitorData_getp_data(el);

    GPUGLMEM double* num_particles =
        BeamStatsMonitorRecord_getp1_num_particles(data, 0);
    GPUGLMEM double* sum_beta0_gamma0 =
        BeamStatsMonitorRecord_getp1_sum_beta0_gamma0(data, 0);

    GPUGLMEM double* sum_x =
        (BeamStatsMonitorRecord_len_sum_x(data) > 0)
        ? BeamStatsMonitorRecord_getp1_sum_x(data, 0) : NULL;
    GPUGLMEM double* sum_px =
        (BeamStatsMonitorRecord_len_sum_px(data) > 0)
        ? BeamStatsMonitorRecord_getp1_sum_px(data, 0) : NULL;
    GPUGLMEM double* sum_y =
        (BeamStatsMonitorRecord_len_sum_y(data) > 0)
        ? BeamStatsMonitorRecord_getp1_sum_y(data, 0) : NULL;
    GPUGLMEM double* sum_py =
        (BeamStatsMonitorRecord_len_sum_py(data) > 0)
        ? BeamStatsMonitorRecord_getp1_sum_py(data, 0) : NULL;
    GPUGLMEM double* sum_zeta =
        (BeamStatsMonitorRecord_len_sum_zeta(data) > 0)
        ? BeamStatsMonitorRecord_getp1_sum_zeta(data, 0) : NULL;
    GPUGLMEM double* sum_delta =
        (BeamStatsMonitorRecord_len_sum_delta(data) > 0)
        ? BeamStatsMonitorRecord_getp1_sum_delta(data, 0) : NULL;

    GPUGLMEM double* sum_x_x =
        (BeamStatsMonitorRecord_len_sum_x_x(data) > 0)
        ? BeamStatsMonitorRecord_getp1_sum_x_x(data, 0) : NULL;
    GPUGLMEM double* sum_x_px =
        (BeamStatsMonitorRecord_len_sum_x_px(data) > 0)
        ? BeamStatsMonitorRecord_getp1_sum_x_px(data, 0) : NULL;
    GPUGLMEM double* sum_x_y =
        (BeamStatsMonitorRecord_len_sum_x_y(data) > 0)
        ? BeamStatsMonitorRecord_getp1_sum_x_y(data, 0) : NULL;
    GPUGLMEM double* sum_x_py =
        (BeamStatsMonitorRecord_len_sum_x_py(data) > 0)
        ? BeamStatsMonitorRecord_getp1_sum_x_py(data, 0) : NULL;
    GPUGLMEM double* sum_x_zeta =
        (BeamStatsMonitorRecord_len_sum_x_zeta(data) > 0)
        ? BeamStatsMonitorRecord_getp1_sum_x_zeta(data, 0) : NULL;
    GPUGLMEM double* sum_x_delta =
        (BeamStatsMonitorRecord_len_sum_x_delta(data) > 0)
        ? BeamStatsMonitorRecord_getp1_sum_x_delta(data, 0) : NULL;
    GPUGLMEM double* sum_px_px =
        (BeamStatsMonitorRecord_len_sum_px_px(data) > 0)
        ? BeamStatsMonitorRecord_getp1_sum_px_px(data, 0) : NULL;
    GPUGLMEM double* sum_px_y =
        (BeamStatsMonitorRecord_len_sum_px_y(data) > 0)
        ? BeamStatsMonitorRecord_getp1_sum_px_y(data, 0) : NULL;
    GPUGLMEM double* sum_px_py =
        (BeamStatsMonitorRecord_len_sum_px_py(data) > 0)
        ? BeamStatsMonitorRecord_getp1_sum_px_py(data, 0) : NULL;
    GPUGLMEM double* sum_px_zeta =
        (BeamStatsMonitorRecord_len_sum_px_zeta(data) > 0)
        ? BeamStatsMonitorRecord_getp1_sum_px_zeta(data, 0) : NULL;
    GPUGLMEM double* sum_px_delta =
        (BeamStatsMonitorRecord_len_sum_px_delta(data) > 0)
        ? BeamStatsMonitorRecord_getp1_sum_px_delta(data, 0) : NULL;
    GPUGLMEM double* sum_y_y =
        (BeamStatsMonitorRecord_len_sum_y_y(data) > 0)
        ? BeamStatsMonitorRecord_getp1_sum_y_y(data, 0) : NULL;
    GPUGLMEM double* sum_y_py =
        (BeamStatsMonitorRecord_len_sum_y_py(data) > 0)
        ? BeamStatsMonitorRecord_getp1_sum_y_py(data, 0) : NULL;
    GPUGLMEM double* sum_y_zeta =
        (BeamStatsMonitorRecord_len_sum_y_zeta(data) > 0)
        ? BeamStatsMonitorRecord_getp1_sum_y_zeta(data, 0) : NULL;
    GPUGLMEM double* sum_y_delta =
        (BeamStatsMonitorRecord_len_sum_y_delta(data) > 0)
        ? BeamStatsMonitorRecord_getp1_sum_y_delta(data, 0) : NULL;
    GPUGLMEM double* sum_py_py =
        (BeamStatsMonitorRecord_len_sum_py_py(data) > 0)
        ? BeamStatsMonitorRecord_getp1_sum_py_py(data, 0) : NULL;
    GPUGLMEM double* sum_py_zeta =
        (BeamStatsMonitorRecord_len_sum_py_zeta(data) > 0)
        ? BeamStatsMonitorRecord_getp1_sum_py_zeta(data, 0) : NULL;
    GPUGLMEM double* sum_py_delta =
        (BeamStatsMonitorRecord_len_sum_py_delta(data) > 0)
        ? BeamStatsMonitorRecord_getp1_sum_py_delta(data, 0) : NULL;
    GPUGLMEM double* sum_zeta_zeta =
        (BeamStatsMonitorRecord_len_sum_zeta_zeta(data) > 0)
        ? BeamStatsMonitorRecord_getp1_sum_zeta_zeta(data, 0) : NULL;
    GPUGLMEM double* sum_zeta_delta =
        (BeamStatsMonitorRecord_len_sum_zeta_delta(data) > 0)
        ? BeamStatsMonitorRecord_getp1_sum_zeta_delta(data, 0) : NULL;
    GPUGLMEM double* sum_delta_delta =
        (BeamStatsMonitorRecord_len_sum_delta_delta(data) > 0)
        ? BeamStatsMonitorRecord_getp1_sum_delta_delta(data, 0) : NULL;

    START_PER_PARTICLE_BLOCK(part0, part);
        if (LocalParticle_get_state(part) > 0) {
            int64_t const at_turn = LocalParticle_get_at_turn(part);
            int64_t const turn_offset = at_turn - start_at_turn;

            if (at_turn >= start_at_turn && at_turn < stop_at_turn
                    && turn_offset % every_n_turns == 0) {
                int64_t const i_record = turn_offset / every_n_turns;

                if (i_record >= 0 && i_record < n_records) {
                    int64_t index = i_record;
                    int64_t accepted = 1;

                    if (mode > 0) {
                        double const zeta = LocalParticle_get_zeta(part);
                        int64_t slot = 0;
                        int64_t i_selected = 0;

                        if (bunch_spacing_zeta != 0.0) {
                            slot = -floor((zeta - z_min_edge)
                                          / bunch_spacing_zeta);
                            if (slot >= 0
                                    && slot < BeamStatsMonitorData_len__slot_to_selected(el)) {
                                i_selected =
                                    BeamStatsMonitorData_get__slot_to_selected(
                                        el, slot);
                            } else {
                                i_selected = -1;
                            }
                        } else if (n_selected == 1) {
                            slot = BeamStatsMonitorData_get__selected_slots(
                                el, 0);
                            i_selected = 0;
                        } else {
                            i_selected = -1;
                        }

                        if (i_selected < 0) {
                            accepted = 0;
                        } else if (mode == 1) {
                            index = i_record * n_selected + i_selected;
                        } else {
                            double const z_min_edge_bunch =
                                z_min_edge - slot * bunch_spacing_zeta;
                            int64_t const i_slice = floor(
                                (zeta - z_min_edge_bunch) / dzeta);
                            if (i_slice < 0 || i_slice >= n_slices) {
                                accepted = 0;
                            } else {
                                index = (
                                    (i_record * n_selected + i_selected)
                                    * n_slices + i_slice);
                            }
                        }
                    }

                    if (accepted) {
                        double const weight = LocalParticle_get_weight(part);
                        double const x = LocalParticle_get_x(part);
                        double const px = LocalParticle_get_px(part);
                        double const y = LocalParticle_get_y(part);
                        double const py = LocalParticle_get_py(part);
                        double const zeta = LocalParticle_get_zeta(part);
                        double const delta = LocalParticle_get_delta(part);
                        double const beta0_gamma0 =
                            LocalParticle_get_beta0(part)
                            * LocalParticle_get_gamma0(part);

                        atomicAdd(&num_particles[index], weight);
                        atomicAdd(&sum_beta0_gamma0[index],
                                  weight * beta0_gamma0);

                        if (sum_x) atomicAdd(&sum_x[index], weight * x);
                        if (sum_px) atomicAdd(&sum_px[index], weight * px);
                        if (sum_y) atomicAdd(&sum_y[index], weight * y);
                        if (sum_py) atomicAdd(&sum_py[index], weight * py);
                        if (sum_zeta) {
                            atomicAdd(&sum_zeta[index], weight * zeta);
                        }
                        if (sum_delta) {
                            atomicAdd(&sum_delta[index], weight * delta);
                        }

                        if (sum_x_x) atomicAdd(&sum_x_x[index], weight * x * x);
                        if (sum_x_px) atomicAdd(&sum_x_px[index], weight * x * px);
                        if (sum_x_y) atomicAdd(&sum_x_y[index], weight * x * y);
                        if (sum_x_py) atomicAdd(&sum_x_py[index], weight * x * py);
                        if (sum_x_zeta) atomicAdd(&sum_x_zeta[index], weight * x * zeta);
                        if (sum_x_delta) atomicAdd(&sum_x_delta[index], weight * x * delta);
                        if (sum_px_px) atomicAdd(&sum_px_px[index], weight * px * px);
                        if (sum_px_y) atomicAdd(&sum_px_y[index], weight * px * y);
                        if (sum_px_py) atomicAdd(&sum_px_py[index], weight * px * py);
                        if (sum_px_zeta) atomicAdd(&sum_px_zeta[index], weight * px * zeta);
                        if (sum_px_delta) atomicAdd(&sum_px_delta[index], weight * px * delta);
                        if (sum_y_y) atomicAdd(&sum_y_y[index], weight * y * y);
                        if (sum_y_py) atomicAdd(&sum_y_py[index], weight * y * py);
                        if (sum_y_zeta) atomicAdd(&sum_y_zeta[index], weight * y * zeta);
                        if (sum_y_delta) atomicAdd(&sum_y_delta[index], weight * y * delta);
                        if (sum_py_py) atomicAdd(&sum_py_py[index], weight * py * py);
                        if (sum_py_zeta) atomicAdd(&sum_py_zeta[index], weight * py * zeta);
                        if (sum_py_delta) atomicAdd(&sum_py_delta[index], weight * py * delta);
                        if (sum_zeta_zeta) {
                            atomicAdd(&sum_zeta_zeta[index],
                                      weight * zeta * zeta);
                        }
                        if (sum_zeta_delta) {
                            atomicAdd(&sum_zeta_delta[index],
                                      weight * zeta * delta);
                        }
                        if (sum_delta_delta) {
                            atomicAdd(&sum_delta_delta[index],
                                      weight * delta * delta);
                        }
                    }
                }
            }
        }
    END_PER_PARTICLE_BLOCK;
}

#endif
