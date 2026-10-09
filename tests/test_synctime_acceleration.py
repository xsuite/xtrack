import numpy as np
import pytest
from scipy.constants import c as clight

import xobjects as xo
import xtrack as xt
import xtrack.synctime as st
from xobjects.test_helpers import allow_kernel_compilation, for_all_test_contexts


def test_reference_change_preserves_arrival_time_of_waiting_particles():
    p = xt.Particles(p0c=1e9, time_s=1e-5,
                     s=[0, 40, 80], zeta=[2, -30, 4],
                     px=[.001, .002, .003], py=[.003, .002, .001],
                     delta=[.01, -.2, .03], state=[1, -1000001, -1],
                     mass_ratio=[1, 2, 1], charge_ratio=[1, 1, 1])
    before = p.copy()
    time = p.time_s + (p.s - p.zeta) / (p.beta0 * clight)
    mask = (p.state > 0) | (p.state == -1000001)
    p.update_p0c_and_energy_deviations(1.2e9, update_pxpy=True, _mask=mask)
    xo.assert_allclose(p.energy, before.energy, rtol=2e-15, atol=1e-6)
    xo.assert_allclose(p.px * p.p0c, before.px * before.p0c, rtol=2e-15)
    xo.assert_allclose(p.py * p.p0c, before.py * before.p0c, rtol=2e-15)
    xo.assert_allclose(p.time_s + (p.s - p.zeta) / (p.beta0 * clight),
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
    st.install_sync_time_at_collective_elements(line)
    line.build_tracker(_context=context)
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
        assert snapshot.time_s > previous_clock
        xo.assert_allclose(line['t_turn_s'], previous_clock, atol=1e-20)
        previous_clock = snapshot.time_s
        saw_waiting |= bool(np.any(snapshot.state < -1000000))
        for ii, id_ in enumerate(snapshot.particle_id):
            expected_t, expected_e = event_reference(
                initial.filter(initial.particle_id == id_),
                snapshot.at_turn[ii], snapshot.s[ii])
            actual_t = (snapshot.time_s + (snapshot.s[ii] - snapshot.zeta[ii])
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
def test_clock_restart_and_independent_ensembles(test_context):
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
    assert restored.time_s == p.time_s
    for name in ('s', 'zeta', 'delta', 'p0c', 'at_turn', 'state', 'particle_id'):
        xo.assert_allclose(getattr(restored, name), getattr(p, name),
                           rtol=0, atol=0)


@allow_kernel_compilation
@for_all_test_contexts(excluding='ContextPyopencl')
def test_cavity_uses_particle_clock(test_context):
    cavity = xt.Cavity(_context=test_context, voltage=3000, frequency=2e6,
                       phase=.2, absolute_time=True)
    p = xt.Particles(_context=test_context, p0c=1e9, time_s=4e-6,
                     at_turn=[3, 5], s=[25., 65.], zeta=[-10., 5.])
    before = p.copy(_context=xo.ContextCpu())
    time = before.time_s + (before.s - before.zeta) / (before.beta0 * clight)
    cavity.track(p)
    after = p.copy(_context=xo.ContextCpu())
    xo.assert_allclose(after.energy - before.energy,
        3000 * np.sin(2*np.pi*2e6*time + .2), rtol=0, atol=1e-6)

    # Chirped RF: integrate f, rather than replacing f in 2*pi*f*t.
    time_s, f0, fdot = .01, 2e6, 1e8
    cavity.frequency = f0 + fdot * time_s
    cavity.phase = .2 - np.pi * fdot * time_s**2
    p = xt.Particles(_context=test_context, p0c=1e9,
                     time_s=time_s, at_turn=[43, 61],
                     s=[25., 65.], zeta=[-10., 5.])
    before = p.copy(_context=xo.ContextCpu())
    dt = (before.s - before.zeta) / (before.beta0 * clight)
    time = before.time_s + dt
    cavity.track(p)
    after = p.copy(_context=xo.ContextCpu())
    exact_phase = .2 + 2*np.pi*(f0*time + .5*fdot*time**2)
    error = np.abs(after.energy - before.energy - 3000*np.sin(exact_phase))
    # The within-frame linearization omits pi*fdot*dt**2.
    assert np.all(error <= 3000*np.pi*fdot*dt**2 + 2e-6)
    # Default time_s=0 is also a clock origin, regardless of at_turn or t_sim.
    cavity.frequency, cavity.phase = 2e6, .2
    p = xt.Particles(_context=test_context, p0c=1e9, t_sim=2e-6,
                     at_turn=[3, 5], s=[25., 65.], zeta=[-10., 5.])
    before = p.copy(_context=xo.ContextCpu())
    time = before.time_s + (before.s - before.zeta) / (before.beta0 * clight)
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
        particle_id_range=ids, stats=['num_particles'])
        for ids in [(0, 2), (10, 12)]]
    line = xt.Line(elements=[xt.Drift(length=25), *monitors,
                             xt.Drift(length=75)])
    line.build_tracker(_context=test_context)
    # A changed reference speed and an independent clock origin must keep
    # the same fixed zeta slices around the accumulated reference turn.
    beta0 = .21 if ramping else .15
    period = 100 / (beta0 * clight)
    frame_time = 2.1*period
    if ramping:
        line.particle_ref = xt.Particles(beta0=.15)
        line.energy_program = xt.EnergyProgram(
            t_s=np.linspace(0, 100*period, 1001),
            p0c=np.linspace(line.particle_ref.p0c[0], 3*line.particle_ref.p0c[0], 1001))
        line.enable_time_dependent_vars = True
        frame_time = float(line.energy_program.get_t_s_at_turn(np.array(2.1)))
        beta0 = float(line.energy_program.get_beta0_at_t_s(frame_time))
        period = 100 / (beta0 * clight)
    reference_turn = np.array([2.0625, 2.3125, 3.1875, 4.0625])
    arrival = frame_time + (reference_turn - 2.1)*period
    p = xt.Particles(_context=test_context, beta0=beta0,
                     time_s=frame_time, s=25,
                     particle_id=[0, 1, 10, 11], at_turn=[1, 1, 7, 7],
                     mass_ratio=[1, 1, 1/12, 1/12],
                     charge_ratio=[1, 1, 1/6, 1/6],
                     delta=[0, 0, 1, 1],
                     weight=[1, 2, 3, 4],
                     zeta=25 - beta0*clight*(arrival - frame_time))
    if ramping:
        # Isolate the pickup acquisition while retaining the tracker's ramp map.
        line.tracker._track_no_collective(p, ele_start=1, ele_stop=3)
    else:
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
        assert restored.to_dict() == monitor.to_dict()


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


@pytest.mark.parametrize('num_threads', [0, 6])
def test_clock_partial_turns_losses_and_empty_ensemble(num_threads):
    context = xo.ContextCpu(omp_num_threads=num_threads)
    line = xt.Line(elements=[xt.Drift(length=40),
                            xt.LimitRect(min_x=-.1, max_x=.1),
                            xt.Drift(length=60)])
    line.build_tracker(_context=context)
    p = xt.Particles(_context=context, p0c=1e9, time_s=3e-6,
                     x=[.2, 0., .3, 0.], at_turn=[7, 7, 7, 7])
    period = 100 / (float(p._xobject.beta0[0]) * clight)
    line.track(p, ele_stop=1)
    assert p.time_s == 3e-6  # no turn boundary crossed
    line.track(p, ele_start=1)
    xo.assert_allclose(p.time_s, 3e-6 + period, rtol=1e-15)
    assert np.count_nonzero(p.state > 0) == 2
    line.track(p, num_turns=19)
    xo.assert_allclose(p.time_s, 3e-6 + 20*period, rtol=1e-15)
    p.state[:] = -1
    p.reorganize()
    line.track(p, num_turns=23)
    xo.assert_allclose(p.time_s, 3e-6 + 43*period, rtol=1e-15)


@for_all_test_contexts
def test_clock_chunking_backtracking_and_absolute_rf(test_context):
    line = xt.Line(elements=[xt.Drift(length=25),
                            xt.Cavity(voltage=1500, frequency=2.1e6,
                                      phase=.3, absolute_time=True),
                            xt.Drift(length=75)])
    line.build_tracker(_context=test_context)
    p = xt.Particles(_context=test_context, p0c=1e9, time_s=2e-6,
                     zeta=[0., -10.], delta=[0., .02])
    initial = p.copy()
    chunked = p.copy()
    line.track(p, num_turns=31)
    for turns in [2, 13, 16]:
        line.track(chunked, num_turns=turns)
    xo.assert_allclose(p.time_s, chunked.time_s, rtol=2e-15)
    for name in ['zeta', 'delta', 's', 'at_turn']:
        xo.assert_allclose(getattr(p, name), getattr(chunked, name),
                           rtol=1e-11, atol=1e-10)
    line.track(p, num_turns=31, backtrack=True)
    xo.assert_allclose(p.time_s, initial.time_s, rtol=2e-14)
    for name in ['zeta', 'delta', 's', 'at_turn']:
        xo.assert_allclose(getattr(p, name), getattr(initial, name),
                           rtol=1e-10, atol=1e-9)


@for_all_test_contexts
def test_particle_clock_and_machine_time_ownership(test_context):
    line = xt.Line(elements=[xt.Drift(length=100)])
    line['t_turn_s'] = .123
    line.build_tracker(_context=test_context)
    p = xt.Particles(_context=test_context, p0c=1e9, time_s=2e-6,
                     at_turn=41)
    period = 100 / (float(p._xobject.beta0[0]) * clight)
    line.track(p, num_turns=3)
    assert line['t_turn_s'] == .123
    xo.assert_allclose(p.time_s, 2e-6 + 3*period, rtol=1e-15)
    line.enable_time_dependent_vars = True
    before = p.time_s
    line.track(p, num_turns=2)
    xo.assert_allclose(line['t_turn_s'], before + period, rtol=1e-15)
    line.enable_time_dependent_vars = False
    line['t_turn_s'] = .456
    before = p.time_s
    line.track(p)
    assert line['t_turn_s'] == .456
    xo.assert_allclose(p.time_s, before + period, rtol=1e-15)


@for_all_test_contexts
def test_energy_program_clock_preserves_physical_flight_time(test_context):
    line = xt.Line(elements=[xt.Drift(length=100)])
    line.particle_ref = xt.Particles(p0c=1e9)
    line.energy_program = xt.EnergyProgram(
        t_s=np.linspace(0, 1e-3, 1001), p0c=np.linspace(1e9, 2e9, 1001))
    line.enable_time_dependent_vars = True
    line.build_tracker(_context=test_context)
    p = xt.Particles(_context=test_context, p0c=1e9, zeta=[0., -10.],
                     delta=[0., .02])
    initial = p.copy(_context=xo.ContextCpu())
    nturns = 40
    line.track(p, num_turns=nturns)
    actual = p.copy(_context=xo.ContextCpu())
    expected_clock = line.energy_program.get_t_s_at_turn(np.array(nturns))
    xo.assert_allclose(actual.time_s, expected_clock, rtol=2e-14)
    # With no RF, each particle keeps its energy and physical velocity while
    # the reference momentum and frame duration change.
    flight_time = (initial.time_s - initial.zeta/(initial.beta0*clight)
                  + nturns*100/(initial.beta0*initial.rvv*clight))
    arrival = actual.time_s + (actual.s-actual.zeta)/(actual.beta0*clight)
    xo.assert_allclose(arrival, flight_time, rtol=3e-14, atol=1e-19)
    xo.assert_allclose(actual.energy, initial.energy, rtol=3e-14)
    # Selecting a frozen machine snapshot does not reset the beam's clock.
    line.enable_time_dependent_vars = False
    line['t_turn_s'] = 5e-4
    before = p.time_s
    period = 100/(float(p._xobject.beta0[0])*clight)
    line.track(p)
    assert line['t_turn_s'] == 5e-4
    xo.assert_allclose(p.time_s, before + period, rtol=1e-15)


@pytest.mark.parametrize('num_threads', [0, 6])
def test_sync_time_clock_advances_while_all_particles_wait(num_threads):
    context = xo.ContextCpu(omp_num_threads=num_threads)
    line = xt.Line(elements=[xt.Drift(length=100)])
    st.install_sync_time_at_collective_elements(line, frame_relative_length=.5)
    line.build_tracker(_context=context)
    p = xt.Particles(_context=context, p0c=1e9, zeta=[-200., -220.])
    period = 100 / (float(p._xobject.beta0[0]) * clight)
    arrival = -p.zeta / (p.beta0 * clight)
    line.track(p, num_turns=2)
    xo.assert_allclose(p.time_s, period, rtol=1e-15)
    assert np.all(p.state < -st.COAST_STATE_RANGE_START)
    xo.assert_allclose(p.time_s + (p.s-p.zeta)/(p.beta0*clight),
                       arrival[p.particle_id], rtol=2e-15)
    line.track(p, num_turns=5)
    assert np.any(p.at_turn > 0)
    xo.assert_allclose(p.time_s, 3.5*period, rtol=2e-15)


def test_sparse_energy_program_resolves_reference_clock_speed():
    line = xt.Line(elements=[xt.Drift(length=100)])
    line.particle_ref = xt.Particles(p0c=1e8)
    line.energy_program = xt.EnergyProgram(t_s=[0., .1], p0c=[1e8, 2e8])
    program = line.energy_program
    times = np.linspace(.001, .099, 101)
    momentum = 1e8 + 1e9 * times
    mass = line.particle_ref.mass0
    # Integral of beta(t) for a linear momentum ramp, independent of the map.
    turns = clight / (100 * 1e9) * (
        np.sqrt(momentum**2 + mass**2) - np.sqrt(1e16 + mass**2))
    xo.assert_allclose(program.get_t_s_at_turn(turns), times, rtol=1e-9, atol=1e-12)
    local_turns = program.get_turn_at_t_s(times)
    periods = program.get_t_s_at_turn(local_turns + .01) - times
    beta = momentum / np.sqrt(momentum**2 + mass**2)
    xo.assert_allclose(periods / .01, 100/(beta*clight), rtol=1e-5)
    xo.assert_allclose(program.get_p0c_at_t_s(times), momentum, rtol=1e-15)


@pytest.mark.parametrize('momenta', [
    [1e8, 2e8, 4e8],        # accelerating
    [4e8, 2e8, 1e8],        # decelerating
    [1e8, 1e8, 1e8],        # constant momentum
    [1e8, 1e8*(1+1e-12), 1e8],  # almost constant momentum
    [0., 1e8, 2e8],         # starting from rest
    [2e8, 1e8, 0.],         # stopping
])
def test_analytical_energy_program_against_quadrature(momenta):
    from scipy.integrate import quad
    line = xt.Line(elements=[xt.Drift(length=100)])
    line.particle_ref = xt.Particles(p0c=1e8)
    knots = np.array([0., .01, .03])
    program = xt.EnergyProgram(t_s=knots, p0c=momenta)
    # Exercise the map at rest without assigning zero reference momentum to
    # tracking particles, whose relative momentum coordinates require P>0.
    program.complete_init(line)
    times = np.unique(np.concatenate((knots, np.linspace(0., .03, 51))))
    mass = line.particle_ref.mass0

    def frev(t):
        p = np.interp(t, knots, momenta)
        return clight/100 * p/np.hypot(p, mass)

    expected = np.array([quad(frev, 0., t, points=knots[knots < t],
                              epsabs=1e-10, epsrel=1e-13)[0] for t in times])
    turns = program.get_turn_at_t_s(times)
    xo.assert_allclose(turns, expected, rtol=3e-14, atol=1e-11)
    xo.assert_allclose(program.get_t_s_at_turn(turns), times, rtol=3e-14, atol=1e-16)
    assert isinstance(program.get_turn_at_t_s(.01), float)
    assert isinstance(program.get_t_s_at_turn(0), float)
    assert len(program.p0c_interpolator.x) == len(knots)
    assert len(program._i_turn_at_t_s) == len(knots)
    matrix = times[:6].reshape(2, 3)
    xo.assert_allclose(program.get_t_s_at_turn(program.get_turn_at_t_s(matrix)),
                       matrix, rtol=3e-14, atol=1e-16)


def test_analytical_energy_program_rest_intervals_and_bounds():
    line = xt.Line(elements=[xt.Drift(length=100)])
    line.particle_ref = xt.Particles(p0c=1e8)
    program = xt.EnergyProgram(
        t_s=[0., 1., 2., 3., 4., 5.], p0c=[0., 0., 1e8, 0., 0., 1e8])
    program.complete_init(line)
    assert program.get_turn_at_t_s(.5) == 0
    assert program.get_t_s_at_turn(0) == 0  # earliest time on the initial plateau
    plateau = program.get_turn_at_t_s(3.5)
    assert program.get_t_s_at_turn(plateau) == 3.
    times = np.array([1.1, 1.9, 2.1, 2.9, 4.1, 4.9])
    xo.assert_allclose(program.get_t_s_at_turn(program.get_turn_at_t_s(times)), times,
                       rtol=1e-14, atol=1e-14)
    with pytest.raises(ValueError, match='from rest'):
        program.get_t_s_at_turn(-1)
    with pytest.raises(ValueError, match='outside program range'):
        program.get_turn_at_t_s(5.1)
    with pytest.raises(ValueError, match='outside program range'):
        program.get_t_s_at_turn(program.get_turn_at_t_s(5.) + 1)


def test_analytical_energy_program_serialization_and_injection_extension():
    line = xt.Line(elements=[xt.Drift(length=100)])
    line.particle_ref = xt.Particles(p0c=1e8)
    line.energy_program = xt.EnergyProgram(t_s=[0., .1], p0c=[1e8, 2e8])
    program = line.energy_program
    times = np.array([-.001, 0., .001, .025, .099, .1])
    turns = program.get_turn_at_t_s(times)
    xo.assert_allclose(program.get_t_s_at_turn(turns), times, rtol=1e-14, atol=1e-16)
    xo.assert_allclose(turns[0], times[0]*program.get_frev_at_t_s(0), rtol=1e-15)
    for restored in [line.copy(), xt.Line.from_dict(line.to_dict())]:
        xo.assert_allclose(restored.energy_program.get_turn_at_t_s(times), turns,
                           rtol=0, atol=0)
    # Old files contain an approximate time map; reconstruct from momentum knots.
    legacy = program.to_dict()
    legacy['t_at_turn_interpolator'] = xt.FunctionPieceWiseLinear(
        x=[0., 1.], y=[0., .1]).to_dict()
    restored = xt.EnergyProgram.from_dict(legacy)
    restored.line = line
    xo.assert_allclose(restored.get_turn_at_t_s(times), turns, rtol=0, atol=0)
