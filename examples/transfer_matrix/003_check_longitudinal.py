from itertools import product
import numpy as np

import matplotlib.pyplot as plt
plt.close('all')

import xtrack as xt
import xpart as xp
import xobjects as xo


# machine = 'sps'
# line = xt.Line.from_json(
#     '../../test_data/sps_w_spacecharge/line_no_spacecharge_and_particle.json')
# # I put the cavity at the end of the ring to get closer to the kick-drift model
# line.cycle('actb.31739_aper', inplace=True)
# configurations = ['above transition', 'below transition']
# num_turns = 250
# cavity_name = 'acta.31637'
# sigmaz=0.20

machine = 'psb'
line = xt.Line.from_json('../../test_data/psb_injection/line_and_particle.json')
configurations = ['below transition']
num_turns = 1000
cavity_name = 'br.c02'
sigmaz = 22.

line.build_tracker()

for i_case, (configuration, longitudinal_mode) in enumerate(
    product(configurations,
            ['linear_fixed_qs', 'linear_fixed_rf', 'nonlinear'])):

    print(f'Case {i_case}: {configuration}, {longitudinal_mode}')

    if machine == 'sps':
        if configuration == 'above transition':
            line[cavity_name].lag = 180.
            line.particle_ref = xt.Particles(p0c=450e9, q0=1.0)
        else:
            line[cavity_name].lag = 0.
            line.particle_ref = xt.Particles(p0c=16e9, q0=1.0)

    # Build corresponding matrix
    tw = line.twiss()
    circumference = tw.circumference

    if longitudinal_mode == 'nonlinear':
        matrix = xt.LineSegmentMap(
            qx=tw.qx, qy=tw.qy,
            dqx=tw.dqx, dqy=tw.dqy,
            betx=tw.betx[0], alfx=tw.alfx[0],
            bety=tw.bety[0], alfy=tw.alfy[0],
            dx=tw.dx[0], dpx=tw.dpx[0],
            dy=tw.dy[0], dpy=tw.dpy[0],
            voltage_rf=line[cavity_name].voltage,
            frequency_rf=line[cavity_name].frequency,
            lag_rf=line[cavity_name].lag,
            momentum_compaction_factor=tw.momentum_compaction_factor,
            length=circumference)
    elif longitudinal_mode == 'linear_fixed_rf':
        matrix = xt.LineSegmentMap(
            longitudinal_mode='linear_fixed_rf',
            qx=tw.qx, qy=tw.qy,
            dqx=tw.dqx, dqy=tw.dqy,
            betx=tw.betx[0], alfx=tw.alfx[0],
            bety=tw.bety[0], alfy=tw.alfy[0],
            dx=tw.dx[0], dpx=tw.dpx[0],
            dy=tw.dy[0], dpy=tw.dpy[0],
            voltage_rf=line[cavity_name].voltage,
            frequency_rf=line[cavity_name].frequency,
            lag_rf=line[cavity_name].lag,
            momentum_compaction_factor=tw.momentum_compaction_factor,
            length=circumference)
    elif longitudinal_mode == 'linear_fixed_qs':
        eta = tw.slip_factor # > 0 above transition
        qs = tw.qs
        circumference = line.get_length()
        bet_s = eta * circumference / (2 * np.pi * qs)
        matrix = xt.LineSegmentMap(
            qx=tw.qx, qy=tw.qy,
            dqx=tw.dqx, dqy=tw.dqy,
            betx=tw.betx[0], alfx=tw.alfx[0],
            bety=tw.bety[0], alfy=tw.alfy[0],
            dx=tw.dx[0], dpx=tw.dpx[0],
            dy=tw.dy[0], dpy=tw.dpy[0],
            bets=bet_s, qs=qs,
            length=circumference)

    line_matrix = xt.Line(elements=[matrix])
    line_matrix.particle_ref = line.particle_ref.copy()
    line_matrix.build_tracker()

    # Compare tracking longitudinal tracking on one particle
    particle0_line = line.build_particles(x_norm=0, y_norm=0, zeta=1e-3)
    line.track(particle0_line.copy(), num_turns=num_turns, turn_by_turn_monitor=True)
    mon = line.record_last_track
    particle0_matrix = line_matrix.build_particles(x_norm=0, y_norm=0, zeta=1e-3)
    line_matrix.track(particle0_matrix.copy(), num_turns=num_turns, turn_by_turn_monitor=True)
    mon_matrix = line_matrix.record_last_track

    assert np.allclose(np.max(mon.zeta), np.max(mon_matrix.zeta), rtol=1e-2, atol=0)
    assert np.allclose(np.max(mon.pzeta), np.max(mon_matrix.pzeta), rtol=1e-2, atol=0)
    assert np.allclose(np.max(mon.x), np.max(mon_matrix.x), rtol=1e-2, atol=0)

    assert np.allclose(mon.zeta, mon_matrix.zeta, rtol=0, atol=5e-2*np.max(mon.zeta.T))
    assert np.allclose(mon.pzeta, mon_matrix.pzeta, rtol=0, atol=5e-2*np.max(mon.pzeta[:]))
    assert np.allclose(mon.x, mon_matrix.x, rtol=0, atol=5e-2*np.max(mon.x.T)) # There is some phase difference...

    # Match Gaussian distributions
    p_line = xp.generate_matched_gaussian_bunch(num_particles=1000000,
        nemitt_x=1e-6, nemitt_y=1e-6, sigma_z=5e-2, line=line, engine='linear')
    p_matrix = xp.generate_matched_gaussian_bunch(num_particles=1000000,
        nemitt_x=1e-6, nemitt_y=1e-6, sigma_z=5e-2, line=line_matrix, engine='linear')

    assert np.isclose(np.std(p_line.zeta), np.std(p_matrix.zeta), rtol=1e-2)
    assert np.isclose(np.std(p_line.pzeta), np.std(p_matrix.pzeta), rtol=2e-2)
    assert np.isclose(np.std(p_line.x), np.std(p_matrix.x), rtol=1e-2)
    assert np.isclose(np.std(p_line.px), np.std(p_matrix.px), rtol=1e-2)
    assert np.isclose(np.std(p_line.y), np.std(p_matrix.y), rtol=1e-2)
    assert np.isclose(np.std(p_line.py), np.std(p_matrix.py), rtol=1e-2)

    # Compare twiss
    tw_line = line.twiss()
    tw_matrix = line_matrix.twiss()

    if configuration == 'above transition':
        assert tw_line.bets0 > 0
        assert tw_matrix.bets0 > 0
        assert tw_line.slip_factor > 0
        assert tw_matrix.slip_factor > 0
    elif configuration == 'below transition':
        assert tw_line.bets0 < 0
        assert tw_matrix.bets0 < 0
        assert tw_line.slip_factor < 0
        assert tw_matrix.slip_factor < 0
    else:
        raise ValueError('Unknown configuration')

    line_frac_qx = np.mod(tw_line.qx, 1)
    line_frac_qy = np.mod(tw_line.qy, 1)
    matrix_frac_qx = np.mod(tw_matrix.qx, 1)
    matrix_frac_qy = np.mod(tw_matrix.qy, 1)

    assert np.isclose(line_frac_qx, matrix_frac_qx, atol=1e-5, rtol=0)
    assert np.isclose(line_frac_qy, matrix_frac_qy, atol=1e-5, rtol=0)
    assert np.isclose(tw_line.betx[0], tw_matrix.betx[0], atol=1e-5, rtol=0)
    assert np.isclose(tw_line.alfx[0], tw_matrix.alfx[0], atol=1e-5, rtol=0)
    assert np.isclose(tw_line.bety[0], tw_matrix.bety[0], atol=1e-5, rtol=0)
    assert np.isclose(tw_line.alfy[0], tw_matrix.alfy[0], atol=1e-5, rtol=0)
    assert np.isclose(tw_line.dx[0], tw_matrix.dx[0], atol=1e-5, rtol=0)
    assert np.isclose(tw_line.dpx[0], tw_matrix.dpx[0], atol=1e-5, rtol=0)
    assert np.isclose(tw_line.dy[0], tw_matrix.dy[0], atol=1e-5, rtol=0)
    assert np.isclose(tw_line.dpy[0], tw_matrix.dpy[0], atol=1e-5, rtol=0)

    assert tw_matrix.s[0] == 0
    assert np.isclose(tw_matrix.s[-1], tw_line.circumference, rtol=0, atol=1e-6)
    assert np.allclose(tw_matrix.bets0, tw_line.bets0, rtol=1e-2, atol=0)

    assert np.allclose(np.squeeze(mon.zeta), np.squeeze(mon_matrix.zeta),
                       rtol=0, atol=2e-2*np.max(np.squeeze(mon.zeta)))
    assert np.allclose(np.squeeze(mon.pzeta), np.squeeze(mon_matrix.pzeta),
                          rtol=0, atol=3e-2*np.max(np.squeeze(mon.pzeta)))

    fig1 = plt.figure(1 + i_case * 10)
    fig1.suptitle(machine + ' - ' + configuration + ' - ' + longitudinal_mode)
    ax1 = fig1.add_subplot(311)
    ax2 = fig1.add_subplot(312, sharex=ax1)
    ax3 = fig1.add_subplot(313, sharex=ax1)
    ax1.set_ylabel('zeta')
    ax2.set_ylabel('pzeta')
    ax2.set_xlabel('turn')
    ax1.plot(mon.zeta.T, label='lattice')
    ax1.plot(mon_matrix.zeta.T, label='matrix')
    ax2.plot(mon.pzeta.T)
    ax2.plot(mon_matrix.pzeta.T)
    ax3.plot(mon.x.T)
    ax3.plot(mon_matrix.x.T)
    ax1.legend()

    fig1.subplots_adjust(left=0.2)

    particles_dp0 = line.build_particles(x_norm=0, y_norm=0,
            delta=np.linspace(-5e-3, 5e-3, 41))
    line_matrix.track(particles_dp0.copy(), num_turns=500, turn_by_turn_monitor=True)
    mon_matrix_dp = line_matrix.record_last_track

    fig2 = plt.figure(2 + i_case*10)
    fig2.suptitle(machine + ' - ' + configuration + ' - ' + longitudinal_mode)
    plt.plot(mon_matrix_dp.zeta.T, mon_matrix_dp.pzeta.T)

    particles_matrix = xp.generate_matched_gaussian_bunch(num_particles=1000000,
        nemitt_x=1e-6, nemitt_y=1e-6, sigma_z=sigmaz, line=line_matrix)

    particles_line = xp.generate_matched_gaussian_bunch(num_particles=1000000,
        nemitt_x=1e-6, nemitt_y=1e-6, sigma_z=sigmaz, line=line)

    particles_matrix.move(_context=xo.context_default)
    particles_line.move(_context=xo.context_default)

    assert np.isclose(np.std(particles_matrix.zeta), np.std(particles_line.zeta),
                      atol=0, rtol=2e-2)
    assert np.isclose(np.std(particles_matrix.pzeta), np.std(particles_line.pzeta),
        atol=0, rtol=(25e-2 if longitudinal_mode.startswith('linear') else 2e-2))

plt.show()