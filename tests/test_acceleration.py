# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json
import pathlib

import numpy as np
import pandas as pd
from cpymad.madx import Madx
from scipy.constants import c as clight

import xobjects as xo
import xpart as xp
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

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

    xo.assert_allclose(p_co._xobject.zeta[0], stable_z, atol=0, rtol=1e-2)


@for_all_test_contexts(excluding=('ContextPyopencl',))
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
    lbefore = line
    line_dict = line.to_dict()
    line = xt.Line.from_dict(line_dict)
    assert np.all(line.vars.get_table().name == lbefore.vars.get_table().name)

    line = line.copy()
    assert np.all(line.vars.get_table().name == lbefore.vars.get_table().name)

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
    xo.assert_allclose(E_check, E_check_ref, atol=0, rtol=2e-3)

    t_turn_check = line.energy_program.get_t_s_at_turn(np.arange(n_turn_test))
    xo.assert_allclose(t_turn_check, t_turn_ref, atol=0, rtol=6e-4)

    p0c_check = line.energy_program.get_p0c_at_t_s(t_check)
    p0c_ref = np.interp(t_check,
                        t_turn_check,
                        line.particle_ref.mass0 * gamma_at_turn * beta_at_turn)
    xo.assert_allclose(p0c_check, p0c_ref, atol=0, rtol=1e-3)

    kinetic_energy0_check = line.energy_program.get_kinetic_energy0_at_t_s(t_check)
    kinetic_energy0_ref = np.interp(t_check,
                        t_turn_check,
                        line.particle_ref.mass0 * (gamma_at_turn - 1))
    xo.assert_allclose(kinetic_energy0_check, kinetic_energy0_ref, atol=0, rtol=2e-3)

    beta0_check = line.energy_program.get_beta0_at_t_s(t_check)
    beta0_ref = np.interp(t_check, t_turn_check, beta_at_turn)
    xo.assert_allclose(beta0_check, beta0_ref, atol=0, rtol=1e-3)

    frev_check = line.energy_program.get_frev_at_t_s(t_check)
    frev_ref = np.interp(t_check, t_turn_check[:-1], 1/np.diff(t_turn_ref))
    xo.assert_allclose(frev_check, frev_ref, atol=0, rtol=4e-5)

    p0c_increse_per_turn_check = line.energy_program.get_p0c_increse_per_turn_at_t_s(
        t_check)
    p0c_increse_per_turn_ref = np.interp(
        t_check, t_turn_check[:-1], np.diff(monitor.p0c[0, :]))
    xo.assert_allclose(p0c_increse_per_turn_check - p0c_increse_per_turn_ref, 0,
                       atol=5e-5 * p0c_ref[0], rtol=0)

    line.enable_time_dependent_vars = False
    line.vars['t_turn_s'] = 20e-3

    E_kin_expected = np.interp(line.vv['t_turn_s'], t_s, E_kin_GeV*1e9)
    E_tot_expected = E_kin_expected + line.particle_ref.mass0
    xo.assert_allclose(
        E_tot_expected, line.particle_ref.energy0[0], rtol=1e-4, atol=0)
    xo.assert_allclose(
        E_kin_expected, line.particle_ref.kinetic_energy0[0], rtol=1e-4, atol=0)

    tw = line.twiss(method='6d')
    # To check that it does not change
    xo.assert_allclose(tw.zeta[0], -13.48, rtol=0, atol=1e-4)
    xo.assert_allclose(line.particle_ref.mass0 * tw.gamma0, E_tot_expected,
                      atol=0, rtol=1e-12)

    line.vars['t_turn_s'] = 0
    line.vars['on_chicane_k0'] = 0
    tw = line.twiss(method='6d')
    xo.assert_allclose(tw.zeta[0], 0, rtol=0, atol=1e-12)
    xo.assert_allclose(line.particle_ref.mass0 * tw.gamma0, line.particle_ref.mass0 + E_kin_turn[0],
                       rtol=1e-10, atol=0)

@for_all_test_contexts(excluding=('ContextPyopencl',))
def test_acceleration_transverse_shrink(test_context):

    mad = Madx(stdout=False)

    # Load mad model and apply element shifts
    mad.input(f'''
    call, file = '{str(test_data_folder)}/psb_chicane/psb.seq';
    call, file = '{str(test_data_folder)}/psb_chicane/psb_fb_lhc.str';
    beam;
    use, sequence=psb1;
    ''')

    line = xt.Line.from_madx_sequence(mad.sequence.psb1,
                                        deferred_expressions=True)
    e_kin_start_eV = 160e6
    line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1.,
                                    energy0=xt.PROTON_MASS_EV + e_kin_start_eV)

    # Slice to gain some tracking speed
    line.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(slicing=xt.Teapot(1)),
        ])
    line.build_tracker(_context=test_context)

    # User-defined energy ramp
    t_s = np.array([0., 0.0006, 0.0008, 0.001 , 0.0012, 0.0014, 0.0016, 0.0018,
                    0.002 , 0.0022, 0.0024, 0.0026, 0.0028, 0.003, 0.01, 0.1])
    E_kin_GeV = np.array([0.16000000,0.16000000,
        0.16000437, 0.16001673, 0.16003748, 0.16006596, 0.16010243, 0.16014637,
        0.16019791, 0.16025666, 0.16032262, 0.16039552, 0.16047524, 0.16056165,
        0.163586, 0.20247050000000014])

    # Enhance energy swing to better see the effect of energy on beam size
    E_kin_GeV -= 0.140

    # Go away from half integer
    opt = line.match(
        #verbose=True,
        method='4d',
        solve=False,
        vary=[
            xt.Vary('kbrqfcorr', step=1e-4),
            xt.Vary('kbrqdcorr', step=1e-4),
        ],
        targets = [
            xt.Target('qx', value=4.15, tol=1e-5, scale=1),
            xt.Target('qy', value=4.18, tol=1e-5, scale=1),
        ]
    )
    opt.solve()


    # Attach energy program to the line
    line.energy_program = xt.EnergyProgram(t_s=t_s, kinetic_energy0=E_kin_GeV*1e9)

    # Plot energy and revolution frequency vs time
    t_plot = np.linspace(0, 10e-3, 20)
    E_kin_plot = line.energy_program.get_kinetic_energy0_at_t_s(t_plot)
    f_rev_plot = line.energy_program.get_frev_at_t_s(t_plot)

    # import matplotlib.pyplot as plt
    # plt.close('all')
    # plt.figure(1, figsize=(6.4 * 1.5, 4.8))
    # ax1 = plt.subplot(2,2,1)
    # plt.plot(t_plot * 1e3, E_kin_plot * 1e-6)
    # plt.ylabel(r'$E_{kin}$ [MeV]')
    # ax2 = plt.subplot(2,2,3, sharex=ax1)
    # plt.plot(t_plot * 1e3, f_rev_plot * 1e-3)
    # plt.ylabel(r'$f_{rev}$ [kHz]')
    # plt.xlabel('t [ms]')

    # Setup frequency of the RF cavity  to stay on the second harmonic of the
    # revolution frequency during the acceleration

    t_rf = np.linspace(0, 3e-3, 100) # time samples for the frequency program
    f_rev = line.energy_program.get_frev_at_t_s(t_rf)
    h_rf = 2 # harmonic number
    f_rf = h_rf * f_rev # frequency program

    # Build a function with these samples and link it to the cavity
    line.functions['fun_f_rf'] = xt.FunctionPieceWiseLinear(x=t_rf, y=f_rf)
    line.element_refs['br1.acwf5l1.1'].frequency = line.functions['fun_f_rf'](
                                                            line.vars['t_turn_s'])

    # Setup voltage and lag
    line.element_refs['br1.acwf5l1.1'].voltage = 3000 # V
    line.element_refs['br1.acwf5l1.1'].lag = 0 # degrees (below transition energy)

    # When setting line.vars['t_turn_s'] the reference energy and the rf frequency
    # are updated automatically
    line.vars['t_turn_s'] = 0
    line.particle_ref.kinetic_energy0 # is 160.00000 MeV
    line['br1.acwf5l1.1'].frequency # is 1983931.935 Hz

    line.vars['t_turn_s'] = 3e-3
    line.particle_ref.kinetic_energy0 # is 160.56165 MeV
    line['br1.acwf5l1.1'].frequency # is 1986669.0559674294

    # Back to zero for tracking!
    line.vars['t_turn_s'] = 0

    # Track a few particles to visualize the longitudinal phase space
    p_test = line.build_particles(x_norm=0, zeta=np.linspace(0, line.get_length(), 101))

    # Enable time-dependent variables (t_turn_s and all the variables that depend on
    # it are automatically updated at each turn)
    line.enable_time_dependent_vars = True

    # Track
    line.track(p_test, num_turns=9000, turn_by_turn_monitor=True, with_progress=True)
    mon = line.record_last_track

    # Plot
    # plt.subplot2grid((2,2), (0,1), rowspan=2)
    # plt.plot(mon.zeta[:, -2000:].T, mon.delta[:, -2000:].T, color='C0')
    # plt.xlabel(r'$\zeta$ [m]')
    # plt.ylabel('$\delta$')
    # plt.xlim(-40, 30)
    # plt.ylim(-0.0025, 0.0025)
    # plt.title('Last 2000 turns')
    # plt.subplots_adjust(left=0.08, right=0.95, wspace=0.26)


    # Check transverse beam size reduction
    line['t_turn_s'] = 0
    line.enable_time_dependent_vars = False

    n_part_test = 500
    # Generate Gaussian distribution with fixed rng seed
    rng = np.random.default_rng(seed=123)
    x_norm = rng.normal(loc=0, scale=1, size=n_part_test)
    px_norm = rng.normal(loc=0, scale=1, size=n_part_test)
    y_norm = rng.normal(loc=0, scale=1, size=n_part_test)
    py_norm = rng.normal(loc=0, scale=1, size=n_part_test)

    # rescale to have exact std dev.
    x_norm = x_norm / np.std(x_norm)
    px_norm = px_norm / np.std(px_norm)
    y_norm = y_norm / np.std(y_norm)
    py_norm = py_norm / np.std(py_norm)

    p_test2 = line.build_particles(x_norm=x_norm, px_norm=px_norm,
                                y_norm=x_norm, py_norm=px_norm,
                                nemitt_x=1e-6, nemitt_y=1e-6,
                                delta=0)

    line.enable_time_dependent_vars = True
    line.track(p_test2, num_turns=10_000, turn_by_turn_monitor=True, with_progress=True)
    mon2 = line.record_last_track

    std_y = np.std(mon2.y, axis=0)
    std_x = np.std(mon2.x, axis=0)

    # Apply moving average filter
    from scipy.signal import savgol_filter
    std_y_smooth = savgol_filter(std_y, 10000, 2)
    std_x_smooth = savgol_filter(std_x, 10000, 2)

    i_turn_match = 1000
    std_y_expected = std_y_smooth[i_turn_match] * np.sqrt(
        mon2.gamma0[0, i_turn_match]* mon2.beta0[0, i_turn_match]
        / mon2.gamma0[0, :] / mon2.beta0[0, :])
    std_x_expected = std_x_smooth[i_turn_match] * np.sqrt(
        mon2.gamma0[0, i_turn_match]* mon2.beta0[0, i_turn_match]
        / mon2.gamma0[0, :] / mon2.beta0[0, :])

    d_sigma_x = std_x_expected[0] - std_x_expected[-1]
    d_sigma_y = std_y_expected[0] - std_y_expected[-1]

    import xobjects as xo
    xo.assert_allclose(std_y_expected[8000:9000].mean(),
                    std_y_smooth[8000:9000].mean(),
                    rtol=0, atol=0.03 * d_sigma_y)
    xo.assert_allclose(std_x_expected[8000:9000].mean(),
                    std_x_smooth[8000:9000].mean(),
                    rtol=0, atol=0.03 * d_sigma_x)

    # plt.figure(2)
    # ax1 = plt.subplot(2,1,1)
    # plt.plot(std_x, label='raw')
    # plt.plot(std_x_smooth, label='smooth')
    # plt.plot(std_x_expected, label='expected')
    # plt.legend()

    # ax2 = plt.subplot(2,1,2, sharex=ax1)
    # plt.plot(std_y, label='raw')
    # plt.plot(std_y_smooth, label='smooth')
    # plt.plot(std_y_expected, label='expected')
    # plt.show()