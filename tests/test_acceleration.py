# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import pathlib
import json
import numpy as np
import pandas as pd
from scipy.constants import c as clight

import xpart as xp
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

from cpymad.madx import Madx

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()


@for_all_test_contexts
def test_acceleration(test_context):
    Delta_p0c = 450e9/10*23e-6  # ramp rate 450GeV/10s

    fname_line = test_data_folder.joinpath(
        'sps_w_spacecharge/line_no_spacecharge_and_particle.json')

    with open(fname_line, 'r') as fid:
        input_data = json.load(fid)
    line = xt.Line.from_dict(input_data['line'])

    energy_increase = xt.ReferenceEnergyIncrease(Delta_p0c=Delta_p0c)
    line.append_element(energy_increase, 'energy_increase')

    line.build_tracker(_context=test_context)

    # Assume only first cavity is active
    frequency = line.get_elements_of_type(xt.Cavity)[0][0].frequency
    voltage = line.get_elements_of_type(xt.Cavity)[0][0].voltage
    # Assuming proton and beta=1
    stable_z = np.arcsin(Delta_p0c/voltage)/frequency/2/np.pi*clight

    p_co = line.find_closed_orbit(particle_ref=xp.Particles.from_dict(
        input_data['particle']))

    assert np.isclose(p_co._xobject.zeta[0], stable_z, atol=0, rtol=1e-2)


@for_all_test_contexts
def test_energy_program(test_context):

    df = pd.read_csv(test_data_folder / 'psb_chicane/Ramp_and_RF_functions.dat',
                     sep='\t', skiprows=2,
                     names=['t_s', 'E_kin_GeV', 'V1_MV', 'phi1_rad', 'V2_MV', 'phi2_rad'])
    E_kin_GeV = df.E_kin_GeV.values
    t_s = df.t_s.values

    # Shift to enhance change in revolution frequency
    E_min = np.min(E_kin_GeV)
    E_max = np.max(E_kin_GeV)
    E_kin_GeV = E_min/100 + (E_kin_GeV - E_min)
    # Shift the time scale for testing purposes
    t_s = t_s

    # Load mad model and apply element shifts
    mad = Madx(stdout=False)
    mad.call(str(test_data_folder / 'psb_chicane/psb.seq'))
    mad.call(str(test_data_folder / 'psb_chicane/psb_fb_lhc.str'))
    mad.input('''
        beam, particle=PROTON, pc=0.5708301551893517;
        use, sequence=psb1;
        twiss;
    ''')

    line = xt.Line.from_madx_sequence(mad.sequence.psb1, allow_thick=True,
                                      deferred_expressions=True)
    line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV,
                                     gamma0=mad.sequence.psb1.beam.gamma)

    line.build_tracker(_context=test_context)

    # Attach energy program
    ep = xt.EnergyProgram(t_s=t_s, kinetic_energy0=E_kin_GeV*1e9)
    line.energy_program = ep

    line['br1.acwf7l1.1'].voltage = 2e6
    line['br1.acwf7l1.1'].frequency = 1e3

    # test to_dict and from_dict
    line_dict = line.to_dict()
    line = xt.Line.from_dict(line_dict)

    # test copy method
    line = line.copy()

    line.build_tracker(_context=test_context)

    p_test = line.build_particles()
    line.enable_time_dependent_vars = True
    n_turn_test = 5000
    monitor = xt.ParticlesMonitor(num_particles=len(p_test.zeta), start_at_turn=0,
                                  stop_at_turn=n_turn_test,
                                  _context=test_context)
    for ii in range(n_turn_test):
        line.track(p_test, turn_by_turn_monitor=monitor)

    beta_at_turn = monitor.beta0[0, :]
    gamma_at_turn = 1 / np.sqrt(1 - beta_at_turn**2)

    t_turn_ref = np.cumsum(line.get_length()/clight/beta_at_turn)
    t_turn_ref = t_turn_ref - t_turn_ref[0]
    E_kin_turn = line.particle_ref.mass0 * (monitor.gamma0[0, :] - 1)

    t_check = np.linspace(0, 20e-3, 1000)
    E_check = np.interp(t_check, t_turn_ref, E_kin_turn)
    E_check_ref = np.interp(t_check, t_s, E_kin_GeV*1e9)
    assert np.allclose(E_check, E_check_ref, atol=0, rtol=2e-3)

    t_turn_check = line.energy_program.get_t_s_at_turn(np.arange(n_turn_test))
    assert np.allclose(t_turn_check, t_turn_ref, atol=0, rtol=6e-4)

    p0c_check = line.energy_program.get_p0c_at_t_s(t_check)
    p0c_ref = np.interp(t_check,
                        t_turn_check,
                        line.particle_ref.mass0 * gamma_at_turn * beta_at_turn)
    assert np.allclose(p0c_check, p0c_ref, atol=0, rtol=1e-3)

    kinetic_energy0_check = line.energy_program.get_kinetic_energy0_at_t_s(t_check)
    kinetic_energy0_ref = np.interp(t_check,
                        t_turn_check,
                        line.particle_ref.mass0 * (gamma_at_turn - 1))
    assert np.allclose(kinetic_energy0_check, kinetic_energy0_ref, atol=0, rtol=2e-3)

    beta0_check = line.energy_program.get_beta0_at_t_s(t_check)
    beta0_ref = np.interp(t_check, t_turn_check, beta_at_turn)
    assert np.allclose(beta0_check, beta0_ref, atol=0, rtol=1e-3)

    frev_check = line.energy_program.get_frev_at_t_s(t_check)
    frev_ref = np.interp(t_check, t_turn_check[:-1], 1/np.diff(t_turn_ref))
    assert np.allclose(frev_check, frev_ref, atol=0, rtol=4e-5)

    p0c_increse_per_turn_check = line.energy_program.get_p0c_increse_per_turn_at_t_s(
        t_check)
    p0c_increse_per_turn_ref = np.interp(
        t_check, t_turn_check[:-1], np.diff(monitor.p0c[0, :]))
    assert np.allclose(p0c_increse_per_turn_check - p0c_increse_per_turn_ref, 0,
                       atol=5e-5 * p0c_ref[0], rtol=0)

    line.enable_time_dependent_vars = False
    line.vars['t_turn_s'] = 20e-3

    E_kin_expected = np.interp(line.vv['t_turn_s'], t_s, E_kin_GeV*1e9)
    E_tot_expected = E_kin_expected + line.particle_ref.mass0
    assert np.isclose(
        E_tot_expected, line.particle_ref.energy0[0], rtol=1e-4, atol=0)
    assert np.isclose(
        E_kin_expected, line.particle_ref.kinetic_energy0[0], rtol=1e-4, atol=0)

    tw = line.twiss(method='6d')
    # To check that it does not change
    assert np.isclose(tw.zeta[0], -13.48, rtol=0, atol=1e-4)
    assert np.isclose(line.particle_ref.mass0 * tw.gamma0, E_tot_expected,
                      atol=0, rtol=1e-12)

    line.vars['t_turn_s'] = 0
    line.vars['on_chicane_k0'] = 0
    tw = line.twiss(method='6d')
    assert np.allclose(tw.zeta[0], 0, rtol=0, atol=1e-12)
    assert np.allclose(line.particle_ref.mass0 * tw.gamma0, line.particle_ref.mass0 + E_kin_turn[0],
                       rtol=1e-10, atol=0)
