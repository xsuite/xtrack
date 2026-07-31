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
    int64_t const coasting = (mode == 3);

    BeamStatsMonitorRecordData data = BeamStatsMonitorData_getp_data(el);
    BeamStatsMonitorTouchedRecordsData touched_records_data =
        BeamStatsMonitorData_getp_touched_records(el);
    BeamStatsMonitorProfileRecordData profile_data =
        BeamStatsMonitorData_getp__profile_data(el);

    GPUGLMEM double* num_particles =
        BeamStatsMonitorRecordData_getp1_num_particles(data, 0);
    GPUGLMEM int64_t* touched_records =
        BeamStatsMonitorTouchedRecordsData_getp1_value(
            touched_records_data, 0);
    GPUGLMEM double* sum_beta0_gamma0 =
        BeamStatsMonitorRecordData_getp1_sum_beta0_gamma0(data, 0);

    GPUGLMEM double* sum_x =
        (BeamStatsMonitorRecordData_len_sum_x(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_x(data, 0) : NULL;
    GPUGLMEM double* sum_px =
        (BeamStatsMonitorRecordData_len_sum_px(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_px(data, 0) : NULL;
    GPUGLMEM double* sum_y =
        (BeamStatsMonitorRecordData_len_sum_y(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_y(data, 0) : NULL;
    GPUGLMEM double* sum_py =
        (BeamStatsMonitorRecordData_len_sum_py(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_py(data, 0) : NULL;
    GPUGLMEM double* sum_zeta =
        (BeamStatsMonitorRecordData_len_sum_zeta(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_zeta(data, 0) : NULL;
    GPUGLMEM double* sum_delta =
        (BeamStatsMonitorRecordData_len_sum_delta(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_delta(data, 0) : NULL;
    GPUGLMEM double* sum_pzeta =
        (BeamStatsMonitorRecordData_len_sum_pzeta(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_pzeta(data, 0) : NULL;

    GPUGLMEM double* sum_x_x =
        (BeamStatsMonitorRecordData_len_sum_x_x(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_x_x(data, 0) : NULL;
    GPUGLMEM double* sum_x_px =
        (BeamStatsMonitorRecordData_len_sum_x_px(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_x_px(data, 0) : NULL;
    GPUGLMEM double* sum_x_y =
        (BeamStatsMonitorRecordData_len_sum_x_y(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_x_y(data, 0) : NULL;
    GPUGLMEM double* sum_x_py =
        (BeamStatsMonitorRecordData_len_sum_x_py(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_x_py(data, 0) : NULL;
    GPUGLMEM double* sum_x_zeta =
        (BeamStatsMonitorRecordData_len_sum_x_zeta(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_x_zeta(data, 0) : NULL;
    GPUGLMEM double* sum_x_delta =
        (BeamStatsMonitorRecordData_len_sum_x_delta(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_x_delta(data, 0) : NULL;
    GPUGLMEM double* sum_x_pzeta =
        (BeamStatsMonitorRecordData_len_sum_x_pzeta(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_x_pzeta(data, 0) : NULL;
    GPUGLMEM double* sum_px_px =
        (BeamStatsMonitorRecordData_len_sum_px_px(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_px_px(data, 0) : NULL;
    GPUGLMEM double* sum_px_y =
        (BeamStatsMonitorRecordData_len_sum_px_y(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_px_y(data, 0) : NULL;
    GPUGLMEM double* sum_px_py =
        (BeamStatsMonitorRecordData_len_sum_px_py(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_px_py(data, 0) : NULL;
    GPUGLMEM double* sum_px_zeta =
        (BeamStatsMonitorRecordData_len_sum_px_zeta(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_px_zeta(data, 0) : NULL;
    GPUGLMEM double* sum_px_delta =
        (BeamStatsMonitorRecordData_len_sum_px_delta(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_px_delta(data, 0) : NULL;
    GPUGLMEM double* sum_px_pzeta =
        (BeamStatsMonitorRecordData_len_sum_px_pzeta(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_px_pzeta(data, 0) : NULL;
    GPUGLMEM double* sum_y_y =
        (BeamStatsMonitorRecordData_len_sum_y_y(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_y_y(data, 0) : NULL;
    GPUGLMEM double* sum_y_py =
        (BeamStatsMonitorRecordData_len_sum_y_py(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_y_py(data, 0) : NULL;
    GPUGLMEM double* sum_y_zeta =
        (BeamStatsMonitorRecordData_len_sum_y_zeta(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_y_zeta(data, 0) : NULL;
    GPUGLMEM double* sum_y_delta =
        (BeamStatsMonitorRecordData_len_sum_y_delta(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_y_delta(data, 0) : NULL;
    GPUGLMEM double* sum_y_pzeta =
        (BeamStatsMonitorRecordData_len_sum_y_pzeta(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_y_pzeta(data, 0) : NULL;
    GPUGLMEM double* sum_py_py =
        (BeamStatsMonitorRecordData_len_sum_py_py(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_py_py(data, 0) : NULL;
    GPUGLMEM double* sum_py_zeta =
        (BeamStatsMonitorRecordData_len_sum_py_zeta(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_py_zeta(data, 0) : NULL;
    GPUGLMEM double* sum_py_delta =
        (BeamStatsMonitorRecordData_len_sum_py_delta(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_py_delta(data, 0) : NULL;
    GPUGLMEM double* sum_py_pzeta =
        (BeamStatsMonitorRecordData_len_sum_py_pzeta(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_py_pzeta(data, 0) : NULL;
    GPUGLMEM double* sum_zeta_zeta =
        (BeamStatsMonitorRecordData_len_sum_zeta_zeta(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_zeta_zeta(data, 0) : NULL;
    GPUGLMEM double* sum_zeta_delta =
        (BeamStatsMonitorRecordData_len_sum_zeta_delta(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_zeta_delta(data, 0) : NULL;
    GPUGLMEM double* sum_zeta_pzeta =
        (BeamStatsMonitorRecordData_len_sum_zeta_pzeta(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_zeta_pzeta(data, 0) : NULL;
    GPUGLMEM double* sum_delta_delta =
        (BeamStatsMonitorRecordData_len_sum_delta_delta(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_delta_delta(data, 0) : NULL;
    GPUGLMEM double* sum_pzeta_pzeta =
        (BeamStatsMonitorRecordData_len_sum_pzeta_pzeta(data) > 0)
        ? BeamStatsMonitorRecordData_getp1_sum_pzeta_pzeta(data, 0) : NULL;

    int64_t const n_profiles =
        BeamStatsMonitorProfileRecordData_len_num_bins(profile_data);
    GPUGLMEM double* profile_counts =
        (BeamStatsMonitorProfileRecordData_len_counts(profile_data) > 0)
        ? BeamStatsMonitorProfileRecordData_getp1_counts(profile_data, 0)
        : NULL;
    GPUGLMEM int64_t* profile_offsets =
        (n_profiles > 0)
        ? BeamStatsMonitorProfileRecordData_getp1_offsets(profile_data, 0)
        : NULL;
    GPUGLMEM int64_t* profile_num_bins =
        (n_profiles > 0)
        ? BeamStatsMonitorProfileRecordData_getp1_num_bins(profile_data, 0)
        : NULL;
    GPUGLMEM int64_t* profile_coord_id =
        (n_profiles > 0)
        ? BeamStatsMonitorProfileRecordData_getp1_coord_id(profile_data, 0)
        : NULL;
    GPUGLMEM double* profile_min =
        (n_profiles > 0)
        ? BeamStatsMonitorProfileRecordData_getp1_min(profile_data, 0)
        : NULL;
    GPUGLMEM double* profile_bin_width =
        (n_profiles > 0)
        ? BeamStatsMonitorProfileRecordData_getp1_bin_width(profile_data, 0)
        : NULL;

    START_PER_PARTICLE_BLOCK(part0, part);
        if (LocalParticle_get_state(part) > 0) {
            int64_t effective_turn = LocalParticle_get_at_turn(part);
            double zeta = LocalParticle_get_zeta(part);
            int64_t coasting_slice = 0;
            int64_t accepted = 1;

            if (coasting) {
                double const line_length = part->line_length;
                if (line_length <= 0.0) {
                    accepted = 0;
                } else {
                    double const u = (
                        (double)effective_turn - zeta / line_length);
                    effective_turn = (int64_t)floor(u + 0.5);
                    double relative_turn_fraction =
                        u - (double)effective_turn;
                    double slice_position =
                        (relative_turn_fraction + 0.5) * (double)n_slices;
                    coasting_slice = (int64_t)floor(slice_position);

                    if (coasting_slice < 0 || coasting_slice >= n_slices) {
                        accepted = 0;
                    }
                    zeta = -relative_turn_fraction * line_length;
                }
            }

            int64_t const turn_offset = effective_turn - start_at_turn;

            if (accepted
                    && effective_turn >= start_at_turn
                    && effective_turn < stop_at_turn
                    && turn_offset % every_n_turns == 0) {
                int64_t const i_record = turn_offset / every_n_turns;

                if (i_record >= 0 && i_record < n_records) {
                    int64_t index = i_record;

                    if (coasting) {
                        index = (
                            (i_record * n_selected) * n_slices
                            + coasting_slice);
                    } else if (mode > 0) {
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
                        touched_records[i_record] = 1;

                        double const weight = LocalParticle_get_weight(part);
                        double const x = LocalParticle_get_x(part);
                        double const px = LocalParticle_get_px(part);
                        double const y = LocalParticle_get_y(part);
                        double const py = LocalParticle_get_py(part);
                        double const delta = LocalParticle_get_delta(part);
                        double const pzeta = LocalParticle_get_pzeta(part);
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
                        if (sum_pzeta) {
                            atomicAdd(&sum_pzeta[index], weight * pzeta);
                        }

                        if (sum_x_x) atomicAdd(&sum_x_x[index], weight * x * x);
                        if (sum_x_px) atomicAdd(&sum_x_px[index], weight * x * px);
                        if (sum_x_y) atomicAdd(&sum_x_y[index], weight * x * y);
                        if (sum_x_py) atomicAdd(&sum_x_py[index], weight * x * py);
                        if (sum_x_zeta) atomicAdd(&sum_x_zeta[index], weight * x * zeta);
                        if (sum_x_delta) atomicAdd(&sum_x_delta[index], weight * x * delta);
                        if (sum_x_pzeta) atomicAdd(&sum_x_pzeta[index], weight * x * pzeta);
                        if (sum_px_px) atomicAdd(&sum_px_px[index], weight * px * px);
                        if (sum_px_y) atomicAdd(&sum_px_y[index], weight * px * y);
                        if (sum_px_py) atomicAdd(&sum_px_py[index], weight * px * py);
                        if (sum_px_zeta) atomicAdd(&sum_px_zeta[index], weight * px * zeta);
                        if (sum_px_delta) atomicAdd(&sum_px_delta[index], weight * px * delta);
                        if (sum_px_pzeta) atomicAdd(&sum_px_pzeta[index], weight * px * pzeta);
                        if (sum_y_y) atomicAdd(&sum_y_y[index], weight * y * y);
                        if (sum_y_py) atomicAdd(&sum_y_py[index], weight * y * py);
                        if (sum_y_zeta) atomicAdd(&sum_y_zeta[index], weight * y * zeta);
                        if (sum_y_delta) atomicAdd(&sum_y_delta[index], weight * y * delta);
                        if (sum_y_pzeta) atomicAdd(&sum_y_pzeta[index], weight * y * pzeta);
                        if (sum_py_py) atomicAdd(&sum_py_py[index], weight * py * py);
                        if (sum_py_zeta) atomicAdd(&sum_py_zeta[index], weight * py * zeta);
                        if (sum_py_delta) atomicAdd(&sum_py_delta[index], weight * py * delta);
                        if (sum_py_pzeta) atomicAdd(&sum_py_pzeta[index], weight * py * pzeta);
                        if (sum_zeta_zeta) {
                            atomicAdd(&sum_zeta_zeta[index],
                                      weight * zeta * zeta);
                        }
                        if (sum_zeta_delta) {
                            atomicAdd(&sum_zeta_delta[index],
                                      weight * zeta * delta);
                        }
                        if (sum_zeta_pzeta) {
                            atomicAdd(&sum_zeta_pzeta[index],
                                      weight * zeta * pzeta);
                        }
                        if (sum_delta_delta) {
                            atomicAdd(&sum_delta_delta[index],
                                      weight * delta * delta);
                        }
                        if (sum_pzeta_pzeta) {
                            atomicAdd(&sum_pzeta_pzeta[index],
                                      weight * pzeta * pzeta);
                        }

                        if (profile_counts) {
                            for (int64_t i_profile = 0;
                                    i_profile < n_profiles; i_profile++) {
                                double value = 0.0;
                                int64_t const coord_id =
                                    profile_coord_id[i_profile];
                                if (coord_id == 0) {
                                    value = x;
                                } else if (coord_id == 1) {
                                    value = px;
                                } else if (coord_id == 2) {
                                    value = y;
                                } else if (coord_id == 3) {
                                    value = py;
                                } else if (coord_id == 4) {
                                    value = zeta;
                                } else if (coord_id == 5) {
                                    value = delta;
                                } else if (coord_id == 6) {
                                    value = pzeta;
                                }

                                int64_t const n_profile_bins =
                                    profile_num_bins[i_profile];
                                int64_t const i_profile_bin = floor(
                                    (value - profile_min[i_profile])
                                    / profile_bin_width[i_profile]);

                                if (i_profile_bin >= 0
                                        && i_profile_bin < n_profile_bins) {
                                    int64_t const profile_index =
                                        profile_offsets[i_profile]
                                        + index * n_profile_bins
                                        + i_profile_bin;
                                    atomicAdd(
                                        &profile_counts[profile_index],
                                        weight);
                                }
                            }
                        }
                    }
                }
            }
        }
    END_PER_PARTICLE_BLOCK;
}

#endif
