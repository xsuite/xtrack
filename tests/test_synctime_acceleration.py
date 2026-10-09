import numpy as np
import pytest
from scipy.constants import c as clight

import xobjects as xo
import xtrack as xt
import xtrack.synctime as st
from xobjects.test_helpers import allow_kernel_compilation, for_all_test_contexts


@pytest.fixture(autouse=True)
def compile_current_particle_layout():
    # Development kernels installed for the released Particles layout cannot
    # be reused after adding scalar clock fields.
    with xo.settings.override(force_kernel_compilation=True):
        yield


def test_reference_change_preserves_arrival_time_of_waiting_particles():
    p = xt.Particles(p0c=1e9, at_frame=12, t_frame=1e-5,
                     s=[0, 40, 80], zeta=[2, -30, 4],
                     px=[.001, .002, .003], py=[.003, .002, .001],
                     delta=[.01, -.2, .03], state=[1, -1000001, -1],
                     mass_ratio=[1, 2, 1], charge_ratio=[1, 1, 1])
    before = p.copy()
    time = p.t_frame + (p.s - p.zeta) / (p.beta0 * clight)
    mask = (p.state > 0) | (p.state == -1000001)
    p.update_p0c_and_energy_deviations(1.2e9, update_pxpy=True, _mask=mask)
    xo.assert_allclose(p.energy, before.energy, rtol=2e-15, atol=1e-6)
    xo.assert_allclose(p.px * p.p0c, before.px * before.p0c, rtol=2e-15)
    xo.assert_allclose(p.py * p.p0c, before.py * before.p0c, rtol=2e-15)
    xo.assert_allclose(p.t_frame + (p.s - p.zeta) / (p.beta0 * clight),
                       time, rtol=2e-15, atol=1e-20)
    xo.assert_allclose(p.p0c[mask], 1.2e9, rtol=0, atol=0)
    lost = p.state == -1
    for name in ('s', 'zeta', 'p0c', 'delta', 'px', 'py'):
        xo.assert_allclose(getattr(p, name)[lost], getattr(before, name)[lost],
                           rtol=0, atol=0)


def make_line(context, ramp):
    line = xt.Line(elements={
        'd0': xt.Drift(length=25),
        'rf1': xt.Cavity(voltage=1500, frequency=2.1e6,
                         phase=.3, absolute_time=True),
        'd1': xt.Drift(length=40),
        'rf2': xt.Cavity(voltage=1700, frequency=1.6e6,
                         phase=-.2, absolute_time=True),
        'd2': xt.Drift(length=35),
    })
    line.particle_ref = xt.Particles(mass0=56 * xt.PROTON_MASS_EV,
                                    q0=25, p0c=20e9)
    if ramp:
        times = np.linspace(0, 2e-4, 2001)
        line.energy_program = xt.EnergyProgram(
            t_s=times, p0c=20e9 * (1 + .15 * times / times[-1]))
    line.enable_time_dependent_vars = True
    st.install_sync_time_at_collective_elements(line, frame_clock=True)
    line.build_tracker(_context=context, use_prebuilt_kernels=False)
    return line


def make_particles(line, context):
    mass_ratio = np.array([1., 209 / 56])
    charge_ratio = np.array([1., 68 / 25])
    return xt.Particles(_context=context, mass0=line.particle_ref.mass0,
                        q0=line.particle_ref.q0, p0c=20e9,
                        mass_ratio=mass_ratio, charge_ratio=charge_ratio,
                        delta=charge_ratio / mass_ratio - 1, zeta=[0., -10.])


def event_reference(initial, turns, s):
    """Independent laboratory-time tracking; no frames or reference updates."""
    energy = float(initial.energy[0])
    mass = float(initial.mass0 * initial.mass_ratio[0])
    charge = float(initial.q0 * initial.charge_ratio[0])
    time = float(-initial.zeta[0] / (initial.beta0[0] * clight))
    distance = 0.
    stop = int(turns) * 100 + s
    for turn in range(int(turns) + 1):
        for pos, voltage, frequency, phase in (
                (25, 1500, 2.1e6, .3), (65, 1700, 1.6e6, -.2)):
            target = turn * 100 + pos
            if target >= stop - 1e-10:
                continue
            beta = np.sqrt(1 - (mass / energy)**2)
            time += (target - distance) / (beta * clight)
            energy += charge * voltage * np.sin(2*np.pi*frequency*time + phase)
            distance = target
    beta = np.sqrt(1 - (mass / energy)**2)
    time += (stop - distance) / (beta * clight)
    return time, energy


@pytest.mark.parametrize('ramp', [False, True])
@allow_kernel_compilation
@for_all_test_contexts(excluding='ContextPyopencl')
def test_two_species_against_lab_time_tracking(test_context, ramp):
    line = make_line(test_context, ramp)
    p = make_particles(line, test_context)
    initial = p.copy(_context=xo.ContextCpu())
    previous_clock = 0.
    saw_waiting = False
    for frame in range(45):
        line.track(p, num_turns=1)
        snapshot = p.copy(_context=xo.ContextCpu())
        assert snapshot.at_frame == frame + 1
        assert snapshot.t_frame > previous_clock
        xo.assert_allclose(line['t_turn_s'], previous_clock, atol=1e-20)
        previous_clock = snapshot.t_frame
        saw_waiting |= bool(np.any(snapshot.state < -1000000))
        for ii, id_ in enumerate(snapshot.particle_id):
            expected_t, expected_e = event_reference(
                initial.filter(initial.particle_id == id_),
                snapshot.at_turn[ii], snapshot.s[ii])
            actual_t = (snapshot.t_frame + (snapshot.s[ii] - snapshot.zeta[ii])
                        / (snapshot.beta0[ii] * clight))
            xo.assert_allclose(actual_t, expected_t, rtol=0, atol=5e-17)
            xo.assert_allclose(snapshot.energy[ii], expected_e,
                               rtol=2e-14, atol=1e-3)
        if ramp:
            # Paused particles must use the same reference as active ones.
            xo.assert_allclose(snapshot.p0c,
                line.energy_program.get_p0c_at_t_s(line['t_turn_s']), rtol=1e-15)
    assert saw_waiting
    assert np.ptp(snapshot.at_turn) > 5


@allow_kernel_compilation
@for_all_test_contexts(excluding='ContextPyopencl')
def test_frame_clock_restart_and_independent_ensembles(test_context):
    line = make_line(test_context, ramp=True)
    p = make_particles(line, test_context)
    line.track(p, num_turns=17)
    checkpoint = p.to_dict()
    line.track(p, num_turns=28)
    # Use this same line to track another ensemble, disturbing all cached knobs.
    other = make_particles(line, test_context)
    line.track(other, num_turns=4)
    restored = xt.Particles.from_dict(checkpoint, _context=test_context)
    line.track(restored, num_turns=28)
    p = p.copy(_context=xo.ContextCpu())
    restored = restored.copy(_context=xo.ContextCpu())
    assert restored.at_frame == p.at_frame == 45
    assert restored.t_frame == p.t_frame
    for name in ('s', 'zeta', 'delta', 'p0c', 'at_turn', 'state', 'particle_id'):
        xo.assert_allclose(getattr(restored, name), getattr(p, name),
                           rtol=0, atol=0)


@allow_kernel_compilation
@for_all_test_contexts(excluding='ContextPyopencl')
def test_frame_cavity_phase_and_legacy_time(test_context):
    cavity = xt.Cavity(_context=test_context, voltage=3000, frequency=2e6,
                       phase=.2, absolute_time=True)
    p = xt.Particles(_context=test_context, p0c=1e9, at_frame=8, t_frame=4e-6,
                     at_turn=[3, 5], s=[25., 65.], zeta=[-10., 5.])
    before = p.copy(_context=xo.ContextCpu())
    time = before.t_frame + (before.s - before.zeta) / (before.beta0 * clight)
    cavity.track(p)
    after = p.copy(_context=xo.ContextCpu())
    xo.assert_allclose(after.energy - before.energy,
        3000 * np.sin(2*np.pi*2e6*time + .2), rtol=0, atol=1e-6)

    # Chirped RF: integrate f, rather than replacing f in 2*pi*f*t.
    t_frame, f0, fdot = .01, 2e6, 1e8
    cavity.frequency = f0 + fdot * t_frame
    cavity.phase = .2 - np.pi * fdot * t_frame**2
    p = xt.Particles(_context=test_context, p0c=1e9, at_frame=123,
                     t_frame=t_frame, at_turn=[43, 61],
                     s=[25., 65.], zeta=[-10., 5.])
    before = p.copy(_context=xo.ContextCpu())
    dt = (before.s - before.zeta) / (before.beta0 * clight)
    time = before.t_frame + dt
    cavity.track(p)
    after = p.copy(_context=xo.ContextCpu())
    exact_phase = .2 + 2*np.pi*(f0*time + .5*fdot*time**2)
    error = np.abs(after.energy - before.energy - 3000*np.sin(exact_phase))
    # The within-frame linearization omits pi*fdot*dt**2.
    assert np.all(error <= 3000*np.pi*fdot*dt**2 + 2e-6)
    # Ordinary absolute-time tracking keeps its established local phase convention.
    cavity.frequency, cavity.phase = 2e6, .2
    p = xt.Particles(_context=test_context, p0c=1e9, t_sim=2e-6,
                     at_turn=[3, 5], s=[25., 65.], zeta=[-10., 5.])
    before = p.copy(_context=xo.ContextCpu())
    time = before.at_turn * before.t_sim - before.zeta / (before.beta0 * clight)
    cavity.track(p)
    after = p.copy(_context=xo.ContextCpu())
    xo.assert_allclose(after.energy - before.energy,
        3000 * np.sin(2*np.pi*2e6*time + .2), rtol=0, atol=1e-6)


@allow_kernel_compilation
@for_all_test_contexts(excluding='ContextPyopencl')
@pytest.mark.parametrize('ramping', [False, True])
def test_coasting_monitors_use_frame_arrival_time(test_context, ramping):
    # Two species with unrelated revolution counters arrive in known bins at
    # a pickup away from s=0. Record each species separately, by particle id.
    monitors = [xt.BeamStatsMonitor(
        start_at_turn=0, stop_at_turn=6, coasting=True, num_slices=8,
        coasting_reference_turn=2.1 if ramping else -1.,
        particle_id_range=ids, stats=['num_particles'])
        for ids in [(0, 2), (10, 12)]]
    line = xt.Line(elements=[xt.Drift(length=25), *monitors,
                             xt.Drift(length=75)])
    line.build_tracker(_context=test_context, use_prebuilt_kernels=False)
    # A changed reference speed and an independent clock origin must keep
    # the same fixed zeta slices around the accumulated reference turn.
    beta0 = .21 if ramping else .15
    period = 100 / (beta0 * clight)
    frame_time = 9*period if ramping else 2.1*period
    reference_turn = np.array([2.0625, 2.3125, 3.1875, 4.0625])
    arrival = frame_time + (reference_turn - 2.1)*period
    p = xt.Particles(_context=test_context, beta0=beta0, at_frame=23,
                     t_frame=frame_time, s=25,
                     particle_id=[0, 1, 10, 11], at_turn=[1, 1, 7, 7],
                     mass_ratio=[1, 1, 1/12, 1/12],
                     charge_ratio=[1, 1, 1/6, 1/6],
                     delta=[0, 0, 1, 1],
                     weight=[1, 2, 3, 4],
                     zeta=25 - beta0*clight*(arrival - frame_time))
    line.track(p, ele_start=1, ele_stop=3)
    expected = np.zeros((2, 6, 8))
    expected[0, 2, 4] = 1
    expected[0, 2, 6] = 2
    expected[1, 3, 5] = 3
    expected[1, 4, 4] = 4
    for ii, monitor in enumerate(monitors):
        xo.assert_allclose(monitor.num_particles, expected[ii], rtol=0, atol=0)
        tt = monitor.time_centers(line_length=100, beta0=beta0)
        recorded = tt[expected[ii] > 0]
        xo.assert_allclose(recorded, reference_turn[2*ii:2*ii+2]*period,
                           atol=1e-20, rtol=1e-14)
        restored = xt.BeamStatsMonitor.from_dict(monitor.to_dict())
        assert restored.coasting_reference_turn == monitor.coasting_reference_turn


def test_coasting_time_centers_follow_energy_program():
    monitor = xt.BeamStatsMonitor(coasting=True, num_slices=8,
                                 start_at_turn=0, stop_at_turn=5)
    line = xt.Line(elements=[xt.Drift(length=100)])
    line.particle_ref = xt.Particles(p0c=1e9)
    line.energy_program = xt.EnergyProgram(t_s=np.linspace(0, 1e-5, 100),
                                          p0c=np.linspace(1e9, 2e9, 100))
    turns = -monitor.zeta_centers_unwrapped(line_length=100)/100
    actual = monitor.time_centers(line_length=100, energy_program=line.energy_program)
    expected = line.energy_program.get_t_s_at_turn(turns)
    expected[turns < 0] = turns[turns < 0]*100/(line.particle_ref.beta0[0]*clight)
    xo.assert_allclose(actual, expected, rtol=1e-14, atol=0)
    assert np.all(np.diff(actual.ravel()) > 0)
    assert np.diff(actual[-1]).mean() < np.diff(actual[0]).mean()
