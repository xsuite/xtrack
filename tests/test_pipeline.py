import numpy as np
import pandas as pd
import pytest
from scipy.constants import c as clight

import xtrack as xt
import xpart as xp
import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts

from xtrack import PipelineStatus


@for_all_test_contexts
def test_multitracker(test_context):
    class DummyPipelinedElement:
        def __init__(self, n_hold=0):
            self.n_hold = n_hold
            self.iscollective = True
            self.i_hold = 0

        def track(self, particles):
            self.i_hold += 1
            if self.i_hold < self.n_hold:
                return PipelineStatus(on_hold=True,
                    info=f'stopped by internal counter {self.i_hold}/{self.n_hold}')
            else:
                self.i_hold = 0

    # We build two trackers, each with an element that puts the simulation on hold

    line1 = xt.Line(elements=[xt.Drift(length=1),
                            DummyPipelinedElement(n_hold=2),
                            xt.Drift(length=1)],
                    element_names=['d11', 'pipelnd_el1', 'd12'])
    line1.build_tracker(_context=test_context, enable_pipeline_hold=True)
    line1.reset_s_at_end_turn = False

    line2 = xt.Line(elements=[xt.Drift(length=1),
                            DummyPipelinedElement(n_hold=3),
                            xt.Drift(length=1)],
                    element_names=['d11', 'pipelnd_el2', 'd12'])
    line2.build_tracker(_context=test_context, enable_pipeline_hold=True)
    line2.reset_s_at_end_turn = False

    # We use a multitracker to track one two particles with the first tracker
    # and one particle with the second one

    p1 = xp.Particles(p0c=7e12, x=[0,0,0], _context=test_context)
    p2 = xp.Particles(p0c=7e12, x=[0,0,0], _context=test_context)
    p3 = xp.Particles(p0c=7e12, x=[0,0,0], _context=test_context)

    multitracker = xt.PipelineMultiTracker(
        branches=[xt.PipelineBranch(line=line1, particles=p1),
                xt.PipelineBranch(line=line1, particles=p2),
                xt.PipelineBranch(line=line2, particles=p3),
                ],
        enable_debug_log=True)

    multitracker.track(num_turns=4)

    # The simulation switched multiple times from one branch to the other (due to the
    # pipelined elements as it can be seen by the debug log (it is convenient to use
    # a pandas dataframe to view the debug log).

    log_df = pd.DataFrame(multitracker.debug_log)

    # Contains
    #     branch  track_session_turn held_by_element                             info
    # 0        0                   0     pipelnd_el1  stopped by internal counter 1/2
    # 1        1                   1     pipelnd_el1  stopped by internal counter 1/2
    # 2        2                   0     pipelnd_el2  stopped by internal counter 1/3
    # 3        0                   1     pipelnd_el1  stopped by internal counter 1/2
    # 4        1                   2     pipelnd_el1  stopped by internal counter 1/2
    # 5        2                   0     pipelnd_el2  stopped by internal counter 2/3
    # 6        0                   2     pipelnd_el1  stopped by internal counter 1/2
    # [...]

    # Check that all particles have been tracked for 4 turns
    p1.move(_context=xo.ContextCpu()) # to use numpy for the tests
    p2.move(_context=xo.ContextCpu()) # to use numpy for the tests
    p3.move(_context=xo.ContextCpu()) # to use numpy for the tests
    assert np.all(p1.at_turn == 4)
    assert np.all(p2.at_turn == 4)
    assert np.all(p3.at_turn == 4)
    assert np.all(p1.s == 8.)
    assert np.all(p2.s == 8.)
    assert np.all(p3.s == 8.)

    ############  A simpler case for some extra checks

    # Reset the counters
    line1['pipelnd_el1'].i_hold = 0
    line2['pipelnd_el2'].i_hold = 0

    p1 = xp.Particles(p0c=7e12, x=[0,0,0], _context=test_context)
    p2 = xp.Particles(p0c=7e12, x=[0,0,0], _context=test_context)

    multitracker = xt.PipelineMultiTracker(
        branches=[xt.PipelineBranch(line=line1, particles=p1),
                  xt.PipelineBranch(line=line2, particles=p2)],
        enable_debug_log=True)

    multitracker.track(num_turns=4)

    # The debug log can be loaded in a dataframe
    log_df = pd.DataFrame(multitracker.debug_log)

    # Looks like this
    #     branch  track_session_turn held_by_element                             info
    # 0        0                   0     pipelnd_el1  stopped by internal counter 1/2
    # 1        1                   0     pipelnd_el2  stopped by internal counter 1/3
    # 2        0                   1     pipelnd_el1  stopped by internal counter 1/2
    # 3        1                   0     pipelnd_el2  stopped by internal counter 2/3
    # 4        0                   2     pipelnd_el1  stopped by internal counter 1/2
    # 5        1                   1     pipelnd_el2  stopped by internal counter 1/3
    # 6        0                   3     pipelnd_el1  stopped by internal counter 1/2
    # 7        1                   1     pipelnd_el2  stopped by internal counter 2/3
    # 8        1                   2     pipelnd_el2  stopped by internal counter 1/3
    # 9        1                   2     pipelnd_el2  stopped by internal counter 2/3
    # 10       1                   3     pipelnd_el2  stopped by internal counter 1/3
    # 11       1                   3     pipelnd_el2  stopped by internal counter 2/3

    # Some checks
    assert len(multitracker.debug_log) == 12
    p1.move(_context=xo.ContextCpu()) # to use numpy for the tests
    p2.move(_context=xo.ContextCpu()) # to use numpy for the tests
    assert p1.at_turn[0] == 4
    assert p2.at_turn[0] == 4
    assert p1.s[0] == 8.
    assert p2.s[0] == 8.
    assert np.all(log_df['branch'] == np.array(4*[0, 1] + 4*[1]))
    assert np.all(log_df[log_df['branch']==0]['held_by_element'] == 'pipelnd_el1')
    assert np.all(log_df[log_df['branch']==1]['held_by_element'] == 'pipelnd_el2')


class HoldEveryTurn:
    """Suspend each particle ensemble once per traversal of this element."""

    iscollective = True

    def __init__(self):
        self.holds = set()

    def track(self, particles):
        key = id(particles)
        if key not in self.holds:
            self.holds.add(key)
            return PipelineStatus(on_hold=True)
        self.holds.remove(key)


@pytest.mark.parametrize('layout', ['single', 'shared_env', 'shared_line'])
@pytest.mark.parametrize('update_interval', [0., 1e-6])
@for_all_test_contexts
def test_resume_with_time_dependent_vars(test_context, layout, update_interval):
    """Preserve the sampled kick, reference clock and log through holds."""
    env = xt.Environment()
    env.new('arc', xt.Drift, length=120.)
    env['kick'] = '100 * t_turn_s'
    env.new('corrector', xt.Multipole, ksl=['kick'])
    env.elements['hold_a'] = HoldEveryTurn()
    env.elements['hold_end'] = HoldEveryTurn()
    # Suspend both before and after the kick within each turn.
    line = env.new_line(components=['arc', 'hold_a', 'corrector', 'hold_end'])
    lines = [line]
    if layout == 'shared_env':
        env.elements['hold_b'] = HoldEveryTurn()
        lines.append(env.new_line(components=[
            'arc', 'hold_b', 'corrector', 'hold_end']))
    elif layout == 'shared_line':
        lines.append(line)

    buffer = None
    for ll in dict.fromkeys(lines):
        ll.enable_time_dependent_vars = True
        ll.dt_update_time_dependent_vars = update_interval
        ll.build_tracker(_context=test_context, _buffer=buffer,
                         enable_pipeline_hold=True)
        buffer = ll.tracker._buffer

    particles = [xt.Particles(_context=test_context, p0c=7e9, time_s=t0)
                 for t0 in (2e-6, 4e-6)[:len(lines)]]
    histories = {id(p): [] for p in particles}

    def sample(tracked_line, p):
        histories[id(p)].append(tracked_line['t_turn_s'])
        return tracked_line['t_turn_s']

    multitracker = xt.PipelineMultiTracker([
        xt.PipelineBranch(ll, p) for ll, p in zip(lines, particles)])
    num_turns = 5
    multitracker.track(num_turns=2, log=xt.Log(sample=sample))
    multitracker.track(num_turns=3, log=xt.Log(sample=sample))

    beta0 = 7e9 / np.hypot(7e9, xt.PROTON_MASS_EV)
    period = 120. / (beta0 * clight)
    # For the 1 us update interval, the machine samples every third turn.
    sample_turns = np.arange(num_turns)
    if update_interval:
        sample_turns = 3 * (sample_turns // 3)

    for p, t0 in zip(particles, (2e-6, 4e-6)):
        expected_times = t0 + sample_turns * period
        np.testing.assert_allclose(histories[id(p)], expected_times, rtol=1e-14)
        p = p.copy(_context=xo.ContextCpu())
        np.testing.assert_allclose(float(p.py[0]), 100. * expected_times.sum(),
                                   rtol=1e-14, atol=0.)
        assert int(p.at_turn[0]) == num_turns
        np.testing.assert_allclose(p.time_s, t0 + num_turns * period, rtol=1e-14)


@pytest.mark.parametrize('t0', [0., 2e-6])
@for_all_test_contexts
def test_interleaved_lines_with_held_machine_time(test_context, t0):
    """Maintain each line's sampled kick between scheduled updates."""
    env = xt.Environment()
    env.new('arc', xt.Drift, length=120.)
    env['kick'] = '100 * t_turn_s'
    env.new('corrector', xt.Multipole, ksl=['kick'])
    line = env.new_line(components=['arc', 'corrector'])
    sibling = env.new_line(components=['arc'])
    line.enable_time_dependent_vars = sibling.enable_time_dependent_vars = True
    line.dt_update_time_dependent_vars = 10e-6
    line.build_tracker(_context=test_context)
    sibling.build_tracker(_buffer=line.tracker._buffer)
    p = xt.Particles(_context=test_context, p0c=7e9, time_s=t0)
    other = xt.Particles(_context=test_context, p0c=7e9, time_s=4e-6)

    line.track(p)
    sibling.track(other)
    line.track(p)

    assert env['t_turn_s'] == t0
    p = p.copy(_context=xo.ContextCpu())
    np.testing.assert_allclose(float(p.py[0]), 200. * t0, rtol=1e-14, atol=0.)

    # A rewind must also discard the held sample if it was at time zero.
    fresh = xt.Particles(_context=test_context, p0c=7e9, time_s=-1e-6)
    line.track(fresh)
    assert env['t_turn_s'] == -1e-6
    fresh = fresh.copy(_context=xo.ContextCpu())
    np.testing.assert_allclose(float(fresh.py[0]), -1e-4, rtol=1e-14, atol=0.)


@pytest.mark.parametrize('epochs, update_interval', [
    ((2e-6, 2e-6), 2e-6),
    ((2e-6, 4e-6), 0.),
], ids=['same_epoch_held', 'different_epochs_per_turn'])
@for_all_test_contexts(excluding=('ContextPyopencl',))
def test_pipeline_reference_momentum(test_context, epochs, update_interval):
    """Apply a shared energy program to each beam, preserving its energy."""
    line = xt.Line(elements=[xt.Drift(length=120.), HoldEveryTurn()])
    line.set_particle_ref('proton', p0c=0.6e9)
    line.energy_program = xt.EnergyProgram(
        t_s=[0., 10e-6], p0c=[0.6e9, 1.4e9])
    line.enable_time_dependent_vars = True
    line.build_tracker(_context=test_context, enable_pipeline_hold=True)
    line.dt_update_time_dependent_vars = update_interval
    if update_interval:
        # A new beam requires its own sample even within a previous beam's
        # sampling interval.
        warmup = xt.Particles(_context=test_context, p0c=0.6e9, time_s=1e-6)
        line.tracker.resume(line.track(warmup))

    particles = [xt.Particles(_context=test_context, p0c=0.6e9, time_s=t0,
                              state=[1, -1])
                 for t0 in epochs]
    multitracker = xt.PipelineMultiTracker([
        xt.PipelineBranch(line, p) for p in particles])
    multitracker.track(num_turns=1)
    multitracker.track(num_turns=2)

    mass = xt.PROTON_MASS_EV
    physical_energy = np.hypot(0.6e9, mass)
    beta = 0.6e9 / physical_energy
    dp0c_dt = (1.4e9 - 0.6e9) / 10e-6
    for p, t0 in zip(particles, epochs):
        p = p.copy(_context=xo.ContextCpu())
        initial_p0c = 0.6e9 + dp0c_dt * t0
        if update_interval:
            # The three-turn run fits inside the held sampling interval.
            expected_p0c = initial_p0c
        else:
            initial_ref_energy = np.hypot(initial_p0c, mass)
            # A linear momentum ramp gives a constant energy increase per turn.
            last_ref_energy = initial_ref_energy + 2 * dp0c_dt * 120. / clight
            expected_p0c = np.sqrt(last_ref_energy**2 - mass**2)
        np.testing.assert_allclose(float(p.p0c[0]), expected_p0c, rtol=1e-14)
        np.testing.assert_array_equal(p.p0c[p.state < 0], 0.6e9)
        np.testing.assert_allclose(float(p.energy[0]), physical_energy, rtol=1e-14)
        arrival = p.time_s - float(p.zeta[0]) / (float(p.beta0[0]) * clight)
        np.testing.assert_allclose(arrival, t0 + 3 * 120. / (beta * clight), rtol=1e-14)


@for_all_test_contexts
def test_pipeline_keeps_user_machine_time_when_updates_disabled(test_context):
    """Preserve externally prescribed machine time across a hold."""
    env = xt.Environment()
    env.new('arc', xt.Drift, length=120.)
    env['kick'] = '100 * t_turn_s'
    env.new('corrector', xt.Multipole, ksl=['kick'])
    env.elements['hold'] = HoldEveryTurn()
    line = env.new_line(components=['arc', 'hold', 'corrector'])
    line.build_tracker(_context=test_context, enable_pipeline_hold=True)
    p = xt.Particles(_context=test_context, p0c=7e9)
    env['t_turn_s'] = 2e-6
    hold = line.track(p)
    env['t_turn_s'] = 4e-6
    line.tracker.resume(hold)

    assert env['t_turn_s'] == 4e-6
    p = p.copy(_context=xo.ContextCpu())
    np.testing.assert_allclose(float(p.py[0]), 4e-4, rtol=1e-14, atol=0.)


@for_all_test_contexts
def test_single_pipeline_branch_preserves_line_sampling(test_context):
    """Continue the line's sampling with a single branch given as an iterator."""
    env = xt.Environment()
    env.new('arc', xt.Drift, length=120.)
    env['kick'] = '100 * t_turn_s'
    env.new('corrector', xt.Multipole, ksl=['kick'])
    env.new('collective', xt.Marker)
    env.get('collective').iscollective = True
    line = env.new_line(components=['arc', 'collective', 'corrector'])
    line.enable_time_dependent_vars = True
    line.dt_update_time_dependent_vars = 1e-6
    line.build_tracker(_context=test_context)
    p = xt.Particles(_context=test_context, p0c=7e9, time_s=2e-6)
    multi = xt.PipelineMultiTracker(iter([xt.PipelineBranch(line, p)]))

    multi.track(num_turns=2)
    line.track(p, num_turns=2)
    multi.track(num_turns=1)

    beta0 = 7e9 / np.hypot(7e9, xt.PROTON_MASS_EV)
    period = 120. / (beta0 * clight)
    expected = 100. * (3 * 2e-6 + 2 * (2e-6 + 3 * period))
    p = p.copy(_context=xo.ContextCpu())
    np.testing.assert_allclose(float(p.py[0]), expected, rtol=1e-14, atol=0.)


@pytest.mark.parametrize('hold', [False, True])
@for_all_test_contexts(excluding=('ContextPyopencl',))
def test_pipeline_reference_momentum_with_element_updates(test_context, hold):
    """Preserve element-driven reference updates when splitting a track call."""
    marker = HoldEveryTurn() if hold else xt.Marker()
    marker.iscollective = True
    line = xt.Line(elements=[
        xt.Drift(length=120.),
        xt.ReferenceEnergyIncrease(Delta_p0c=1e7), marker])
    line.set_particle_ref('proton', p0c=0.6e9)
    line.energy_program = xt.EnergyProgram(
        t_s=[0., 10e-6], p0c=[0.6e9, 0.6e9])
    line.enable_time_dependent_vars = True
    line.dt_update_time_dependent_vars = 10e-6
    line.build_tracker(_context=test_context, enable_pipeline_hold=True)

    # Both ways of tracking must accumulate two element-driven increments.
    for chunks in ((2,), (1, 1)):
        p = xt.Particles(_context=test_context, p0c=0.6e9)
        multi = xt.PipelineMultiTracker([xt.PipelineBranch(line, p)])
        for num_turns in chunks:
            multi.track(num_turns=num_turns)

        p = p.copy(_context=xo.ContextCpu())
        np.testing.assert_allclose(p.p0c, 0.62e9, rtol=0., atol=0.)
