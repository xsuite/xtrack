# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #
import pytest
import numpy as np
from scipy import constants

import xpart as xp
import xtrack as xt
from xobjects.test_helpers import fix_random_seed, for_all_test_contexts


def do_checks(test_context,part,n_part_per_bunch,sigma_z,sigma_delta,
        filled_slots,bunch_numbers,bunch_spacing):
        
    for i_bunch,bunch_number in enumerate(bunch_numbers):
        zeta_avg = np.average(
        test_context.nparray_from_context_array(
            part.zeta[i_bunch*n_part_per_bunch:
                   (i_bunch+1)*n_part_per_bunch]))
        delta_avg = np.average(
        test_context.nparray_from_context_array(
            part.delta[i_bunch*n_part_per_bunch:
                   (i_bunch+1)*n_part_per_bunch]))
        delta_rms = np.std(
            test_context.nparray_from_context_array(
                part.delta[i_bunch*n_part_per_bunch:
                           (i_bunch+1)*n_part_per_bunch]))
        zeta_rms = np.std(
            test_context.nparray_from_context_array(
                part.zeta[i_bunch*n_part_per_bunch:
                          (i_bunch+1)*n_part_per_bunch]))

        assert np.isclose((zeta_avg+bunch_spacing*filled_slots[bunch_number])/sigma_z, 0.0, atol=1e-2)
        assert np.isclose(delta_avg/sigma_delta, 0.0, atol=1e-2)
        assert np.isclose(zeta_rms, sigma_z, rtol=1e-2, atol=1e-15)
        assert np.isclose(delta_rms, sigma_delta, rtol=0.2, atol=1e-15)


@for_all_test_contexts
@fix_random_seed(64237673)
@pytest.mark.parametrize(
    'arc_type', ['arc_linear_fixed_qs', 'arc_linear_fixed_rf', 'arc_nonlinear']
)
def test_multi_bunch_gaussian_generation(test_context, arc_type):
    bunch_intensity = 1e11
    n_part_per_bunch = int(1e5)
    nemitt_x = 2e-6
    nemitt_y = 2.5e-6
    beta_x = 1.0
    beta_y = 1.0
    p0c = 7e12
    gamma = p0c/xp.PROTON_MASS_EV
    Q_x = 0.31
    Q_y = 0.32
    sigma_z = 0.08
    momentumCompaction = 3.483575072011584e-04
    eta = momentumCompaction - 1.0 / gamma ** 2
    circumference = 26658.883199999
    frev = constants.c/circumference
    f_RF = 400.8E6
    h = f_RF / frev
    betar = np.sqrt(1 - 1 / gamma ** 2)
    p0 = constants.m_p * betar * gamma * constants.c
    voltage = 16E6
    Qs = np.sqrt(constants.e * voltage * eta * h / (2 * np.pi * betar * constants.c * p0))
    beta_s = eta * circumference / (2 * np.pi * Qs);
    sigma_delta = sigma_z / beta_s
    bucket_length = circumference/h
    bunch_spacing_in_buckets = 10
    bunch_spacing = bunch_spacing_in_buckets * bucket_length
    filling_scheme = np.zeros(int(np.amin(h)/bunch_spacing_in_buckets))
    n_bunches_tot = 10
    filling_scheme[0:int(n_bunches_tot/2)] = 1
    filling_scheme[n_bunches_tot:int(3*n_bunches_tot/2)] = 1
    filled_slots = filling_scheme.nonzero()[0]
    n_procs = 2
    bunch_numbers_per_rank = xp.matched_gaussian.split_scheme(
        filling_scheme=filling_scheme,
        n_chunk=n_procs)

    if arc_type == 'arc_linear_fixed_qs':
        arc = xt.LineSegmentMap(
            betx=beta_x,qx=Q_x,
            bety=beta_y,qy=Q_y,
            bets=-beta_s,qs=Qs,bucket_length=bucket_length/constants.c,length=circumference)
    elif arc_type == 'arc_linear_fixed_rf':
        arc = xt.LineSegmentMap(_context=test_context,
            betx=beta_x,qx=Q_x,
            bety=beta_y,qy=Q_y,
            voltage_rf = voltage,
            longitudinal_mode = 'linear_fixed_rf',
            frequency_rf = f_RF,
            lag_rf = 180.0,
            slippage_length = circumference,
            momentum_compaction_factor = momentumCompaction,length=circumference)
    elif arc_type == 'arc_nonlinear':
        arc = xt.LineSegmentMap(_context=test_context,
            betx=beta_x,qx=Q_x,
            bety=beta_y,qy=Q_y,
            voltage_rf = voltage,
            longitudinal_mode = 'nonlinear',
            frequency_rf = f_RF,
            lag_rf = 180.0,
            slippage_length = circumference,
            momentum_compaction_factor = momentumCompaction,
            length=circumference)
    else:
        raise ValueError(f'Undefined test parameter for arc type {arc_type}')

    line = xt.Line([arc])
    line.particle_ref = xp.Particles(
        p0c=p0c,
        _context=test_context  # for testing purposes
    )
    line.build_tracker(_context=test_context)

    for rank in range(n_procs):
        part = xp.generate_matched_gaussian_multibunch_beam(
            _context=test_context,
            filling_scheme=filling_scheme,
            bunch_num_particles=n_part_per_bunch,
            bunch_intensity_particles=bunch_intensity,
            nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
            line=line, bunch_spacing_buckets=10,
            bunch_selection=bunch_numbers_per_rank[rank],
            particle_ref=line.particle_ref,
            bucket_length=bucket_length,
        )

        do_checks(test_context,part,n_part_per_bunch,sigma_z,sigma_delta,
            filled_slots,bunch_numbers_per_rank[rank],bunch_spacing)
        line.track(part,num_turns=100)
        do_checks(test_context,part,n_part_per_bunch,sigma_z,sigma_delta,
            filled_slots,bunch_numbers_per_rank[rank],bunch_spacing)
