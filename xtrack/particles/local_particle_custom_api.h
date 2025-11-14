// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //

#ifndef XTRACK_LOCAL_PARTICLE_CUSTOM_API_H
#define XTRACK_LOCAL_PARTICLE_CUSTOM_API_H

#include "xtrack/headers/track.h"


GPUFUN
double LocalParticle_get_energy0(LocalParticle* part) {
    double const p0c = LocalParticle_get_p0c(part);
    double const m0  = LocalParticle_get_mass0(part);
    return sqrt(p0c * p0c + m0 * m0);
}


GPUFUN
void LocalParticle_update_ptau(LocalParticle* part, double new_ptau_value) {
    double const beta0 = LocalParticle_get_beta0(part);
    double const ptau = new_ptau_value;
    double const irpp = sqrt(ptau * ptau + 2 * ptau / beta0 + 1);
    double const new_rpp = 1. / irpp;
    double const new_rvv = irpp / (1 + beta0 * ptau);

    LocalParticle_set_delta(part, irpp - 1);
    LocalParticle_set_rvv(part, new_rvv);
    LocalParticle_set_ptau(part, ptau);
    LocalParticle_set_rpp(part, new_rpp);
}


GPUFUN
void LocalParticle_update_delta(LocalParticle* part, double new_delta_value) {
    double const beta0 = LocalParticle_get_beta0(part);
    double const delta_beta0 = new_delta_value * beta0;
    double const ptau_beta0 = sqrt(delta_beta0 * delta_beta0 + 2 * delta_beta0 * beta0 + 1) - 1;
    double const one_plus_delta = 1 + new_delta_value;
    double const rvv = (one_plus_delta) / (1 + ptau_beta0);
    double const rpp = 1 / one_plus_delta;
    double const ptau = ptau_beta0 / beta0;

    LocalParticle_set_delta(part, new_delta_value);
    LocalParticle_set_rvv(part, rvv);
    LocalParticle_set_rpp(part, rpp);
    LocalParticle_set_ptau(part, ptau);
}


GPUFUN
double LocalParticle_get_pzeta(LocalParticle* part) {
    double const ptau = LocalParticle_get_ptau(part);
    double const beta0 = LocalParticle_get_beta0(part);
    return ptau / beta0;
}


GPUFUN
void LocalParticle_update_pzeta(LocalParticle* part, double new_pzeta_value) {
    double const beta0 = LocalParticle_get_beta0(part);
    LocalParticle_update_ptau(part, beta0 * new_pzeta_value);
}


GPUFUN
void increment_at_element(LocalParticle* part0, int64_t const increment) {
    START_PER_PARTICLE_BLOCK(part0, part);
        LocalParticle_add_to_at_element(part, increment);
    END_PER_PARTICLE_BLOCK;
}


GPUFUN
void increment_at_turn(LocalParticle* part0, int flag_reset_s){
    START_PER_PARTICLE_BLOCK(part0, part);
        LocalParticle_add_to_at_turn(part, 1);
        LocalParticle_set_at_element(part, 0);
        if (flag_reset_s > 0) {
            LocalParticle_set_s(part, 0.);
        }
    END_PER_PARTICLE_BLOCK;
}


GPUFUN
void increment_at_turn_backtrack(
        LocalParticle* part0,
        int flag_reset_s,
        double const line_length,
        int64_t const num_elements
) {
    START_PER_PARTICLE_BLOCK(part0, part);
        LocalParticle_add_to_at_turn(part, -1);
        LocalParticle_set_at_element(part, num_elements);
        if (flag_reset_s > 0) {
            LocalParticle_set_s(part, line_length);
        }
    END_PER_PARTICLE_BLOCK;
}


/* check_is_active has different implementation on CPU and GPU */
#if defined(XO_CONTEXT_CPU_SERIAL)

    GPUFUN
    int64_t check_is_active(LocalParticle* part) {
        int64_t ipart = 0;
        while (ipart < part->_num_active_particles){
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

        if (part->_num_active_particles == 0){
            return 0; //All particles lost
        } else {
            return 1; //Some stable particles are still present
        }
    }

#elif defined(XO_CONTEXT_CPU_OPENMP)

    GPUFUN
    int64_t check_is_active(LocalParticle* part) {
    #ifndef SKIP_SWAPS
        int64_t ipart = part->ipart;
        int64_t endpart = part->endpart;
        int64_t left = ipart;
        int64_t right = endpart - 1;
        int64_t swap_made = 0;
        int64_t has_alive = 0;

        if (left == right)
            return part->state[left] > 0;

        while (left < right) {
            if (part->state[left] > 0) {
                left++;
                has_alive = 1;
            }
            else if (part->state[right] <= 0)
                right--;
            else {
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
            if (part->state[i] <= -999999999)
                break;
            else if (part->state[i] > 0)
                num_active++;
            else
                num_lost++;
        }

        part->_num_active_particles = num_active;
        part->_num_lost_particles = num_lost;
    }

#else // not XO_CONTEXT_CPU_SERIAL and not XO_CONTEXT_CPU_OPENMP

    GPUFUN
    int64_t check_is_active(LocalParticle* part) {
        return LocalParticle_get_state(part) > 0;
    };

#endif


GPUFUN
void LocalParticle_add_to_energy(LocalParticle* part, double delta_energy, int pz_only )
{
    double ptau = LocalParticle_get_ptau(part);
    double const p0c = LocalParticle_get_p0c(part);
    double const charge_ratio = LocalParticle_get_charge_ratio(part);
    double const chi = LocalParticle_get_chi(part);
    double const mass_ratio = charge_ratio / chi;

    ptau += delta_energy / p0c / mass_ratio;

    double const old_rpp = LocalParticle_get_rpp(part);

    LocalParticle_update_ptau(part, ptau);

    if (!pz_only) {
        double const new_rpp = LocalParticle_get_rpp(part);
        double const f = old_rpp / new_rpp;
        LocalParticle_scale_px(part, f);
        LocalParticle_scale_py(part, f);
    }
}


GPUFUN
void LocalParticle_update_p0c(LocalParticle* part, double new_p0c_value)
{
    double const mass0 = LocalParticle_get_mass0(part);
    double const old_p0c = LocalParticle_get_p0c(part);
    double const old_delta = LocalParticle_get_delta(part);
    double const old_beta0 = LocalParticle_get_beta0(part);

    double const ppc = old_p0c * old_delta + old_p0c;
    double const new_delta = (ppc - new_p0c_value) / new_p0c_value;

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

GPUFUN
void LocalParticle_kill_particle(LocalParticle* part, int64_t kill_state) {
    LocalParticle_set_x(part, 1e30);
    LocalParticle_set_px(part, 1e30);
    LocalParticle_set_y(part, 1e30);
    LocalParticle_set_py(part, 1e30);
    LocalParticle_set_zeta(part, 1e30);
    LocalParticle_update_delta(part, -1);  // zero energy
    LocalParticle_set_state(part, kill_state);
}


#ifdef XTRACK_GLOBAL_XY_LIMIT

    GPUFUN
    void global_aperture_check(LocalParticle* part0)
    {
        if (LocalParticle_check_track_flag(part0, XS_FLAG_IGNORE_GLOBAL_APERTURE))
        {
            return;
        }

        START_PER_PARTICLE_BLOCK(part0, part);
            double const x = LocalParticle_get_x(part);
            double const y = LocalParticle_get_y(part);

            int64_t const is_alive = (int64_t)(
                (x >= -XTRACK_GLOBAL_XY_LIMIT) &&
                (x <=  XTRACK_GLOBAL_XY_LIMIT) &&
                (y >= -XTRACK_GLOBAL_XY_LIMIT) &&
                (y <=  XTRACK_GLOBAL_XY_LIMIT)
            );

            // I assume that if I am in the function is because
            if (!is_alive) {
               LocalParticle_set_state(part, -1);
            }
        END_PER_PARTICLE_BLOCK;
    }

#endif

#endif /* XTRACK_LOCAL_PARTICLE_CUSTOM_API_H */
