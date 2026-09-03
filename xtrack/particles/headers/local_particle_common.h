// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //

#ifndef XTRACK_PARTICLES_LOCAL_PARTICLE_COMMON_H
#define XTRACK_PARTICLES_LOCAL_PARTICLE_COMMON_H

// Helpers over LocalParticle metadata shared by scalar and TPSA tracking.
GPUFUN
GPUGLMEM int8_t* LocalParticle_get_io_buffer(LocalParticle* part) {
    return part->io_buffer;
}

GPUFUN
uint64_t LocalParticle_check_track_flag(LocalParticle* part, uint8_t index) {
    return (part->track_flags >> index) & 1;
}

// Swap all per-particle scalar arrays when reorganizing native particles. A TPSA map
// represents one particle and therefore has nothing to exchange.
GPUFUN
void LocalParticle_exchange(LocalParticle* part, int64_t i1, int64_t i2) {
#ifndef XTRACK_TPSA_TRACK
    #define EXCHANGE_NUM_FIELD(NAME) \
        {                                              \
            xt_num_t temp = part->NAME[i2];            \
            part->NAME[i2] = part->NAME[i1];           \
            part->NAME[i1] = temp;                     \
        }
        XT_LP_SCALAR_NUM_FIELDS(EXCHANGE_NUM_FIELD)
    #undef EXCHANGE_NUM_FIELD

    #define EXCHANGE_FIELD(TYPE, NAME) \
        {                                                \
            TYPE temp = part->NAME[i2];                  \
            part->NAME[i2] = part->NAME[i1];             \
            part->NAME[i1] = temp;                       \
        }
        XT_LP_INT_FIELDS(EXCHANGE_FIELD)
        XT_LP_UINT32_FIELDS(EXCHANGE_FIELD)
    #undef EXCHANGE_FIELD
#else
    (void)part;
    (void)i1;
    (void)i2;
#endif
}

// Transverse slopes derived from canonical momenta.
GPUFUN
xt_num_t LocalParticle_get_xp(LocalParticle* part) {
    return LocalParticle_get_px(part) * LocalParticle_get_rpp(part);
}

GPUFUN
xt_num_t LocalParticle_get_yp(LocalParticle* part) {
    return LocalParticle_get_py(part) * LocalParticle_get_rpp(part);
}

GPUFUN
xt_num_t LocalParticle_get_exact_xp(LocalParticle* part) {
    xt_num_t const px = LocalParticle_get_px(part);
    xt_num_t const py = LocalParticle_get_py(part);
    xt_num_t const one_plus_delta = 1.0 + LocalParticle_get_delta(part);
    xt_num_t const rpp = 1.0 / sqrt(one_plus_delta * one_plus_delta - px * px - py * py);
    return px * rpp;
}

GPUFUN
xt_num_t LocalParticle_get_exact_yp(LocalParticle* part) {
    xt_num_t const px = LocalParticle_get_px(part);
    xt_num_t const py = LocalParticle_get_py(part);
    xt_num_t const one_plus_delta = 1.0 + LocalParticle_get_delta(part);
    xt_num_t const rpp = 1.0 / sqrt(one_plus_delta * one_plus_delta - px * px - py * py);
    return py * rpp;
}

GPUFUN
void LocalParticle_set_xp(LocalParticle* part, xt_num_arg_t xp) {
    LocalParticle_set_px(part, xp / LocalParticle_get_rpp(part));
}

GPUFUN
void LocalParticle_set_yp(LocalParticle* part, xt_num_arg_t yp) {
    LocalParticle_set_py(part, yp / LocalParticle_get_rpp(part));
}

GPUFUN
void LocalParticle_set_exact_xp(LocalParticle* part, xt_num_arg_t xp) {
    xt_num_t const yp = LocalParticle_get_exact_yp(part);
    xt_num_t rpp = LocalParticle_get_rpp(part);
    rpp *= sqrt(1.0 + xp * xp + yp * yp);
    LocalParticle_set_px(part, xp / rpp);
}

GPUFUN
void LocalParticle_set_exact_yp(LocalParticle* part, xt_num_arg_t yp) {
    xt_num_t const xp = LocalParticle_get_exact_xp(part);
    xt_num_t rpp = LocalParticle_get_rpp(part);
    rpp *= sqrt(1.0 + xp * xp + yp * yp);
    LocalParticle_set_py(part, yp / rpp);
}

GPUFUN
void LocalParticle_add_to_xp(LocalParticle* part, xt_num_arg_t xp) {
    LocalParticle_set_xp(part, LocalParticle_get_xp(part) + xp);
}

GPUFUN
void LocalParticle_add_to_yp(LocalParticle* part, xt_num_arg_t yp) {
    LocalParticle_set_yp(part, LocalParticle_get_yp(part) + yp);
}

GPUFUN
void LocalParticle_add_to_exact_xp(LocalParticle* part, xt_num_arg_t xp) {
    LocalParticle_set_exact_xp(part, LocalParticle_get_exact_xp(part) + xp);
}

GPUFUN
void LocalParticle_add_to_exact_yp(LocalParticle* part, xt_num_arg_t yp) {
    LocalParticle_set_exact_yp(part, LocalParticle_get_exact_yp(part) + yp);
}

GPUFUN
void LocalParticle_scale_xp(LocalParticle* part, xt_num_arg_t value) {
    LocalParticle_set_xp(part, LocalParticle_get_xp(part) * value);
}

GPUFUN
void LocalParticle_scale_yp(LocalParticle* part, xt_num_arg_t value) {
    LocalParticle_set_yp(part, LocalParticle_get_yp(part) * value);
}

GPUFUN
void LocalParticle_scale_exact_xp(LocalParticle* part, xt_num_arg_t value) {
    LocalParticle_set_exact_xp(part, LocalParticle_get_exact_xp(part) * value);
}

GPUFUN
void LocalParticle_scale_exact_yp(LocalParticle* part, xt_num_arg_t value) {
    LocalParticle_set_exact_yp(part, LocalParticle_get_exact_yp(part) * value);
}

GPUFUN
void LocalParticle_set_xp_yp(LocalParticle* part, xt_num_arg_t xp, xt_num_arg_t yp) {
    xt_num_t const rpp = LocalParticle_get_rpp(part);
    LocalParticle_set_px(part, xp / rpp);
    LocalParticle_set_py(part, yp / rpp);
}

GPUFUN
void LocalParticle_set_exact_xp_yp(LocalParticle* part, xt_num_arg_t xp, xt_num_arg_t yp) {
    xt_num_t rpp = LocalParticle_get_rpp(part);
    rpp *= sqrt(1.0 + xp * xp + yp * yp);
    LocalParticle_set_px(part, xp / rpp);
    LocalParticle_set_py(part, yp / rpp);
}

GPUFUN
void LocalParticle_add_to_xp_yp(LocalParticle* part, xt_num_arg_t xp, xt_num_arg_t yp) {
    LocalParticle_set_xp_yp(
        part, LocalParticle_get_xp(part) + xp, LocalParticle_get_yp(part) + yp);
}

GPUFUN
void LocalParticle_add_to_exact_xp_yp(
        LocalParticle* part, xt_num_arg_t xp, xt_num_arg_t yp) {
    LocalParticle_set_exact_xp_yp(
        part, LocalParticle_get_exact_xp(part) + xp, LocalParticle_get_exact_yp(part) + yp);
}

GPUFUN
void LocalParticle_scale_xp_yp(
        LocalParticle* part, xt_num_arg_t value_x, xt_num_arg_t value_y) {
    LocalParticle_set_xp_yp(
        part, LocalParticle_get_xp(part) * value_x, LocalParticle_get_yp(part) * value_y);
}

GPUFUN
void LocalParticle_scale_exact_xp_yp(
        LocalParticle* part, xt_num_arg_t value_x, xt_num_arg_t value_y) {
    LocalParticle_set_exact_xp_yp(
        part,
        LocalParticle_get_exact_xp(part) * value_x,
        LocalParticle_get_exact_yp(part) * value_y);
}

// Reference-energy and longitudinal-coordinate conversions. The algebra is the
// established scalar implementation from before TPSA tracking was introduced.
GPUFUN
double LocalParticle_get_energy0(LocalParticle* part) {
    double const p0c = LocalParticle_get_p0c(part);
    double const mass0 = LocalParticle_get_mass0(part);
    return sqrt(p0c * p0c + mass0 * mass0);
}

GPUFUN
void LocalParticle_update_ptau(LocalParticle* part, xt_num_arg_t new_ptau_value) {
    double const beta0 = LocalParticle_get_beta0(part);
    xt_num_t const ptau = new_ptau_value;
    xt_num_t const irpp = sqrt(ptau * ptau + 2.0 * ptau / beta0 + 1.0);
    xt_num_t const new_rpp = 1.0 / irpp;
    xt_num_t const new_rvv = irpp / (1.0 + beta0 * ptau);

    LocalParticle_set_delta(part, irpp - 1.0);
    LocalParticle_set_rvv(part, new_rvv);
    LocalParticle_set_ptau(part, ptau);
    LocalParticle_set_rpp(part, new_rpp);
}

GPUFUN
void LocalParticle_update_delta(LocalParticle* part, xt_num_arg_t new_delta_value) {
    double const beta0 = LocalParticle_get_beta0(part);
    xt_num_t const delta_beta0 = new_delta_value * beta0;
    xt_num_t const ptau_beta0 = sqrt(
        delta_beta0 * delta_beta0 + 2.0 * delta_beta0 * beta0 + 1.0) - 1.0;
    xt_num_t const one_plus_delta = 1.0 + new_delta_value;
    xt_num_t const rvv = one_plus_delta / (1.0 + ptau_beta0);
    xt_num_t const rpp = 1.0 / one_plus_delta;
    xt_num_t const ptau = ptau_beta0 / beta0;

    LocalParticle_set_delta(part, new_delta_value);
    LocalParticle_set_rvv(part, rvv);
    LocalParticle_set_rpp(part, rpp);
    LocalParticle_set_ptau(part, ptau);
}

GPUFUN
xt_num_t LocalParticle_get_pzeta(LocalParticle* part) {
    return LocalParticle_get_ptau(part) / LocalParticle_get_beta0(part);
}

GPUFUN
void LocalParticle_update_pzeta(LocalParticle* part, xt_num_arg_t new_pzeta_value) {
    LocalParticle_update_ptau(part, LocalParticle_get_beta0(part) * new_pzeta_value);
}

// Turn and element counters operate over the context-specific LocalParticle block.
GPUFUN
void increment_at_element(LocalParticle* part0, int64_t const increment) {
    START_PER_PARTICLE_BLOCK(part0, part);
        LocalParticle_add_to_at_element(part, increment);
    END_PER_PARTICLE_BLOCK;
}

GPUFUN
void increment_at_turn(LocalParticle* part0, int flag_reset_s) {
    START_PER_PARTICLE_BLOCK(part0, part);
        LocalParticle_add_to_at_turn(part, 1);
        LocalParticle_set_at_element(part, 0);
        if (flag_reset_s > 0) {
            LocalParticle_set_s(part, 0.0);
        }
    END_PER_PARTICLE_BLOCK;
}

GPUFUN
void increment_at_turn_backtrack(
        LocalParticle* part0, int flag_reset_s, double line_length, int64_t num_elements) {
    START_PER_PARTICLE_BLOCK(part0, part);
        LocalParticle_add_to_at_turn(part, -1);
        LocalParticle_set_at_element(part, num_elements);
        if (flag_reset_s > 0) {
            LocalParticle_set_s(part, line_length);
        }
    END_PER_PARTICLE_BLOCK;
}

// Native CPU contexts reorganize particle arrays. TPSA and GPU contexts track one
// LocalParticle directly and only need its state.
#if defined(XTRACK_TPSA_TRACK)

GPUFUN
int64_t check_is_active(LocalParticle* part) {
    return LocalParticle_get_state(part) > 0;
}

#elif defined(XO_CONTEXT_CPU_SERIAL)

GPUFUN
int64_t check_is_active(LocalParticle* part) {
    int64_t ipart = 0;
    while (ipart < part->_num_active_particles) {
        #ifdef XSUITE_RESTORE_LOSS
            ipart++;
        #else
            if (part->state[ipart] < 1) {
                LocalParticle_exchange(part, ipart, part->_num_active_particles - 1);
                part->_num_active_particles--;
                part->_num_lost_particles++;
            } else {
                ipart++;
            }
        #endif
    }
    return part->_num_active_particles > 0;
}

#elif defined(XO_CONTEXT_CPU_OPENMP)

GPUFUN
int64_t check_is_active(LocalParticle* part) {
#ifndef SKIP_SWAPS
    int64_t left = part->ipart;
    int64_t right = part->endpart - 1;
    int64_t swap_made = 0;
    int64_t has_alive = 0;

    if (left == right) {
        return part->state[left] > 0;
    }
    while (left < right) {
        if (part->state[left] > 0) {
            left++;
            has_alive = 1;
        } else if (part->state[right] <= 0) {
            right--;
        } else {
            LocalParticle_exchange(part, left, right);
            left++;
            right--;
            swap_made = 1;
        }
    }
    return swap_made || has_alive;
#else
    return 1;
#endif
}

GPUFUN
void count_reorganized_particles(LocalParticle* part) {
    int64_t num_active = 0;
    int64_t num_lost = 0;
    for (int64_t i = part->ipart; i < part->endpart; i++) {
        if (part->state[i] <= -999999999) {
            break;
        } else if (part->state[i] > 0) {
            num_active++;
        } else {
            num_lost++;
        }
    }
    part->_num_active_particles = num_active;
    part->_num_lost_particles = num_lost;
}

#else

GPUFUN
int64_t check_is_active(LocalParticle* part) {
    return LocalParticle_get_state(part) > 0;
}

#endif

// Energy kicks and reference-momentum updates.
GPUFUN
void LocalParticle_add_to_energy(LocalParticle* part, xt_num_arg_t delta_energy, int pz_only) {
    xt_num_t ptau = LocalParticle_get_ptau(part);
    double const p0c = LocalParticle_get_p0c(part);
    double const charge_ratio = LocalParticle_get_charge_ratio(part);
    double const chi = LocalParticle_get_chi(part);
    double const mass_ratio = charge_ratio / chi;

    ptau += delta_energy / p0c / mass_ratio;
    xt_num_t const old_rpp = LocalParticle_get_rpp(part);
    LocalParticle_update_ptau(part, ptau);

    if (!pz_only) {
        xt_num_t const new_rpp = LocalParticle_get_rpp(part);
        xt_num_t const factor = old_rpp / new_rpp;
        LocalParticle_scale_px(part, factor);
        LocalParticle_scale_py(part, factor);
    }
}

GPUFUN
void LocalParticle_update_p0c(LocalParticle* part, double new_p0c_value) {
    double const mass0 = LocalParticle_get_mass0(part);
    double const old_p0c = LocalParticle_get_p0c(part);
    xt_num_t const old_delta = LocalParticle_get_delta(part);
    double const old_beta0 = LocalParticle_get_beta0(part);

    xt_num_t const ppc = old_p0c * old_delta + old_p0c;
    xt_num_t const new_delta = (ppc - new_p0c_value) / new_p0c_value;
    double const new_energy0 = sqrt(new_p0c_value * new_p0c_value + mass0 * mass0);
    double const new_beta0 = new_p0c_value / new_energy0;
    double const new_gamma0 = new_energy0 / mass0;

    LocalParticle_set_p0c(part, new_p0c_value);
    LocalParticle_set_gamma0(part, new_gamma0);
    LocalParticle_set_beta0(part, new_beta0);
    LocalParticle_update_delta(part, new_delta);
    LocalParticle_scale_px(part, old_p0c / new_p0c_value);
    LocalParticle_scale_py(part, old_p0c / new_p0c_value);
    LocalParticle_scale_zeta(part, new_beta0 / old_beta0);
}

// Loss-state helpers.
GPUFUN
void LocalParticle_kill_particle(LocalParticle* part, int64_t kill_state) {
    LocalParticle_set_x(part, 1e30);
    LocalParticle_set_px(part, 1e30);
    LocalParticle_set_y(part, 1e30);
    LocalParticle_set_py(part, 1e30);
    LocalParticle_set_zeta(part, 1e30);
    LocalParticle_update_delta(part, -1);
    LocalParticle_set_state(part, kill_state);
}

#ifdef XTRACK_GLOBAL_XY_LIMIT

GPUFUN
void global_aperture_check(LocalParticle* part0) {
    if (LocalParticle_check_track_flag(part0, XS_FLAG_IGNORE_GLOBAL_APERTURE)) {
        return;
    }

    START_PER_PARTICLE_BLOCK(part0, part);
        xt_num_t const x = LocalParticle_get_x(part);
        xt_num_t const y = LocalParticle_get_y(part);
        double const x0 = xt_num_truncate_to_double(x);
        double const y0 = xt_num_truncate_to_double(y);
        int64_t const is_within_global_aperture = (int64_t)(
            x0 >= -XTRACK_GLOBAL_XY_LIMIT && x0 <= XTRACK_GLOBAL_XY_LIMIT
            && y0 >= -XTRACK_GLOBAL_XY_LIMIT && y0 <= XTRACK_GLOBAL_XY_LIMIT);
        if (LocalParticle_get_state(part) > 0 && !is_within_global_aperture) {
            LocalParticle_set_state(part, -1);
        }
    END_PER_PARTICLE_BLOCK;
}

#endif

#endif /* XTRACK_PARTICLES_LOCAL_PARTICLE_COMMON_H */
