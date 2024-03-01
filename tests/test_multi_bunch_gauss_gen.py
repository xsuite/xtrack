# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json

import numpy as np

import xobjects as xo
import xpart as xp
import xtrack as xt

from xobjects.test_helpers import for_all_test_contexts

from scipy import constants

class DummyCommunicator:
    def __init__(self, n_procs, rank):
        self.n_procs = n_procs
        self.rank = rank

    def Get_size(self):
        return self.n_procs

    def Get_rank(self):
        return self.rank

#@for_all_test_contexts
#def test_multi_bunch_gaussian_generation(test_context):
if __name__ == '__main__':
    test_context = xo.ContextCpu()
    bunch_intensity = 1e11
    n_part_per_bunch = int(1e5)
    nemitt_x = 2e-6
    nemitt_y = 2.5e-6
    beta_x = 1.0
    beta_y = 1.0
    p0c = 7e12
    gamma = p0c/xp.PROTON_MASS_EV
    physemit_x = nemitt_x/gamma
    physemit_y = nemitt_y/gamma
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
    bucket_length = 0.7480045791245511

    arc = xt.LineSegmentMap(
        betx=beta_x,qx=Q_x,
        bety=beta_y,qy=Q_y,
        bets=-beta_s,qs=Qs,bucket_length=bucket_length/constants.c,length=circumference)
    arc_linear_fixed_rf = xt.LineSegmentMap(_context=test_context,
        betx=beta_x,qx=Q_x,
        bety=beta_y,qy=Q_y,
        voltage_rf = voltage,
        longitudinal_mode = 'linear_fixed_rf',
        frequency_rf = f_RF,
        lag_rf = 180.0,
        slippage_length = circumference,
        momentum_compaction_factor = momentumCompaction,length=circumference)
    arc_nonlinear = xt.LineSegmentMap(_context=test_context,
        betx=beta_x,qx=Q_x,
        bety=beta_y,qy=Q_y,
        voltage_rf = voltage,
        longitudinal_mode = 'nonlinear',
        frequency_rf = f_RF,
        lag_rf = 180.0,
        slippage_length = circumference,
        momentum_compaction_factor = momentumCompaction,
        length=circumference)

    line = xt.Line([arc])
    line.particle_ref = xp.Particles(
        p0c=p0c,
        _context=test_context  # for testing purposes
    )
    line.build_tracker(_context=test_context)

    part_on_co = line.find_closed_orbit()

    h_list = [35640]
    bunch_spacing_in_buckets = 10
    filling_scheme = np.zeros(int(np.amin(h_list)/bunch_spacing_in_buckets))
    # build a dummy filling scheme
    n_bunches_tot = 10
    filling_scheme[0:int(n_bunches_tot/2)] = 1
    filling_scheme[n_bunches_tot:int(3*n_bunches_tot/2)] = 1

    # build a dummy communicator made of two ranks and use rank 0
    n_procs = 2
    rank = 0
    communicator = DummyCommunicator(n_procs, rank)

    first_bunch, n_bunches = xp.split_scheme(filling_scheme=filling_scheme,
                                             communicator=communicator)

    part = xp.generate_matched_gaussian_multibunch_beam(
        _context=test_context,
        filling_scheme=filling_scheme,
        num_particles=n_part_per_bunch,
        total_intensity_particles=bunch_intensity,
        nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
        line=line, bunch_spacing_buckets=10,
        i_bunch_0=first_bunch, num_bunches=n_bunches,
        particle_ref=line.particle_ref,
        bucket_length = bucket_length
    )

    tw = line.twiss()

    # CHECKS
    for i_bunch in range(n_bunches):
        y_rms = np.std(
            test_context.nparray_from_context_array(
                part.y[i_bunch*n_part_per_bunch:
                       (i_bunch+1)*n_part_per_bunch]))
        x_rms = np.std(
            test_context.nparray_from_context_array(
                part.x[i_bunch*n_part_per_bunch:
                       (i_bunch+1)*n_part_per_bunch]))
        delta_rms = np.std(
            test_context.nparray_from_context_array(
                part.delta[i_bunch*n_part_per_bunch:
                           (i_bunch+1)*n_part_per_bunch]))
        zeta_rms = np.std(
            test_context.nparray_from_context_array(
                part.zeta[i_bunch*n_part_per_bunch:
                          (i_bunch+1)*n_part_per_bunch]))

        part_on_co.move(_context=xo.ContextCpu())

        gemitt_x = nemitt_x/part_on_co.beta0/part_on_co.gamma0
        gemitt_y = nemitt_y/part_on_co.beta0/part_on_co.gamma0
        assert np.isclose(zeta_rms, sigma_z, rtol=1e-2, atol=1e-15)
        assert np.isclose(
            x_rms,
            np.sqrt(tw['betx'][0]*gemitt_x + tw['dx'][0]**2*delta_rms**2),
            rtol=1e-2, atol=1e-15)
        assert np.isclose(
            y_rms,
            np.sqrt(tw['bety'][0]*gemitt_y + tw['dy'][0]**2*delta_rms**2),
            rtol=1e-2, atol=1e-15)

    # build a dummy communicator made of two ranks and use rank 1
    rank = 1
    communicator = DummyCommunicator(n_procs, rank)

    first_bunch, n_bunches = xp.split_scheme(filling_scheme=filling_scheme,
                                             communicator=communicator)

    part = xp.generate_matched_gaussian_multibunch_beam(
        filling_scheme=filling_scheme,  # engine='linear',
        num_particles=n_part_per_bunch,
        total_intensity_particles=bunch_intensity,
        nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
        line=line, bunch_spacing_buckets=10,
        i_bunch_0=first_bunch, num_bunches=n_bunches,
        particle_ref=line.particle_ref,
        bucket_length = bucket_length
    )

    # CHECKS
    for i_bunch in range(n_bunches):
        y_rms = np.std(
            test_context.nparray_from_context_array(
                part.y[i_bunch*n_part_per_bunch:
                       (i_bunch+1)*n_part_per_bunch]))
        x_rms = np.std(
            test_context.nparray_from_context_array(
                part.x[i_bunch*n_part_per_bunch:
                       (i_bunch+1)*n_part_per_bunch]))
        delta_rms = np.std(
            test_context.nparray_from_context_array(
                part.delta[i_bunch*n_part_per_bunch:
                           (i_bunch+1)*n_part_per_bunch]))
        zeta_rms = np.std(
            test_context.nparray_from_context_array(
                part.zeta[i_bunch*n_part_per_bunch:
                          (i_bunch+1)*n_part_per_bunch]))

        part_on_co.move(_context=xo.ContextCpu())

        gemitt_x = nemitt_x/part_on_co.beta0/part_on_co.gamma0
        gemitt_y = nemitt_y/part_on_co.beta0/part_on_co.gamma0
        assert np.isclose(zeta_rms, sigma_z, rtol=1e-2, atol=1e-15)
        assert np.isclose(
            x_rms,
            np.sqrt(tw['betx'][0]*gemitt_x + tw['dx'][0]**2*delta_rms**2),
            rtol=1e-2, atol=1e-15
        )
        assert np.isclose(
            y_rms,
            np.sqrt(tw['bety'][0]*gemitt_y + tw['dy'][0]**2*delta_rms**2),
            rtol=1e-2, atol=1e-15
        )

