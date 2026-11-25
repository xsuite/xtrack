# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

"""Total physical hardware ownership after summing nominal and feedback."""

from types import SimpleNamespace

import numpy as np
import pytest
import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

from xtrack.feedback.hardware import PhysicalActuator
from xtrack.feedback.output import KnobOutput
from xtrack.feedback.runtime import FeedbackController, FeedbackSystem


class Source:
    def __init__(self):
        self.context = xo.ContextCpu()

    def read_record(self, turn, *, completed_turn):
        return SimpleNamespace(value=np.array(0.0), valid=np.array(True))


def make_hardware(*, nominal=0.2, tau=0.0, slew=None):
    env = xt.Environment()
    env['nominal'] = nominal
    env['applied'] = nominal
    env.elements['corrector'] = xt.Multipole(ksl=[0.0])
    env.ref['corrector'].ksl[0] = env.ref['applied']
    hardware = PhysicalActuator(
        env,
        target='applied',
        baseline='nominal',
        lower=-1.0,
        upper=1.0,
        response_time_constant_seconds=tau,
        maximum_slew_per_second=slew,
    )
    return env, hardware


def make_system(hardware, *, contribution=0.3, **kwargs):
    def program(previous):
        return {
            'memory': {'calls': previous['calls'] + 1},
            'outputs': {'command': np.array([contribution])},
        }

    controller = FeedbackController(
        'orbit',
        sources={'y': Source()},
        program=program,
        output=hardware,
        memory={'calls': np.array(0)},
        update_turns=5,
        first_calculation_turn=1,
        **kwargs,
    )
    return controller, FeedbackSystem([controller])


@pytest.mark.parametrize('disabled', [False, True])
def test_nominal_changes_without_due_calculation_pass_through_one_response(disabled):
    """Advance the old total and then latch a changed nominal setpoint."""
    env, hardware = make_hardware(tau=1.0)
    controller, system = make_system(hardware)
    first = system.step(1, 1.0)[0]
    assert first['executed']
    np.testing.assert_allclose(first['target_output'], [0.5])
    np.testing.assert_allclose(first['applied_output'], [0.2])
    if disabled:
        controller.disable()
    env['nominal'] = 0.4
    second = system.step(2, 2.0)[0]
    assert not second['executed']
    expected = 0.5 + (0.2 - 0.5) * np.exp(-1.0)
    np.testing.assert_allclose(second['applied_output'], [expected])
    np.testing.assert_allclose(second['target_output'], [0.7])
    np.testing.assert_allclose(controller.actuator.contribution_target, [0.3])
    assert env['corrector'].ksl[0] == pytest.approx(expected)
    assert env['applied'] == pytest.approx(expected)
    assert controller.memory['calls'] == 1
    third = system.step(3, 3.0)[0]
    expected = 0.7 + (expected - 0.7) * np.exp(-1.0)
    np.testing.assert_allclose(third['applied_output'], [expected])


def test_slew_applies_to_total_nominal_and_feedback_change_once():
    """Limit the change of the combined field to one slew increment."""
    env, hardware = make_hardware(nominal=0.0, slew=0.1)
    controller, system = make_system(hardware)
    system.step(1, 1.0)
    env['nominal'] = 0.4
    values = [
        float(system.step(turn, float(turn))[0]['applied_output'][0])
        for turn in range(2, 6)
    ]
    np.testing.assert_allclose(values, [0.1, 0.2, 0.3, 0.4])
    assert controller.memory['calls'] == 1
    assert env['corrector'].ksl[0] == pytest.approx(0.4)


def test_baseline_limit_checked_when_disabled_and_retry_preserves_hardware_state():
    """Reject an illegal total and keep the previous field history."""
    env, hardware = make_hardware(tau=1.0)
    controller, system = make_system(hardware)
    system.step(1, 1.0)
    controller.disable()
    old = hardware.snapshot()
    env['nominal'] = 0.8  # Individually legal, total request is 1.1.
    with pytest.raises(ValueError, match='limits'):
        system.step(2, 2.0)
    np.testing.assert_array_equal(hardware.snapshot(), old)
    np.testing.assert_array_equal(controller.actuator.target, [0.5])
    assert controller.actuator.time_seconds == 1.0
    assert len(controller.log) == 1
    env['nominal'] = 0.4
    system.step(2, 2.0)
    np.testing.assert_array_equal(controller.actuator.target, [0.7])
    assert controller.memory['calls'] == 1


def test_readbacks_distinguish_total_applied_from_held_contribution():
    """Keep the installed startup field and report the exact sampled nominal."""
    env, hardware = make_hardware(nominal=0.1, tau=1.0)
    env['applied'] = 0.2
    readings = []

    def program(inputs):
        readings.append(
            {
                name: inputs[name].copy()
                for name in [
                    'applied_output',
                    'held_target',
                    'pending_output',
                    'total_target',
                    'nominal_target',
                ]
            }
        )
        return {'outputs': {'command': np.array([0.3])}}

    controller = FeedbackController(
        'orbit', sources={'y': Source()}, program=program, output=hardware
    )
    system = FeedbackSystem([controller])
    system.step(1, 1.0)
    system.step(2, 2.0)
    np.testing.assert_array_equal(readings[0]['applied_output'], [0.2])
    np.testing.assert_array_equal(readings[0]['total_target'], [0.2])
    np.testing.assert_array_equal(readings[1]['applied_output'], [0.2])
    np.testing.assert_array_equal(readings[1]['held_target'], [0.3])
    np.testing.assert_array_equal(readings[1]['pending_output'], [0.3])
    np.testing.assert_array_equal(readings[1]['total_target'], [0.4])
    np.testing.assert_array_equal(readings[1]['nominal_target'], [0.1])


def test_baseline_cannot_bypass_response_at_setup_or_after_graph_changes():
    """Reject a nominal path into the field or a changed bound write set."""
    env, _ = make_hardware()
    env.ref['corrector'].ksl[0] = env.ref['applied'] + env.ref['nominal']
    with pytest.raises(ValueError, match='bypass'):
        PhysicalActuator(env, target='applied', baseline='nominal')
    env, hardware = make_hardware()
    controller, system = make_system(hardware)
    env.elements['other'] = xt.Multipole(ksl=[0.0])
    env.ref['other'].ksl[0] = env.ref['nominal']
    # A changed nominal graph is rejected even if its existing value is finite.
    with pytest.raises(RuntimeError, match='dependency graph changed'):
        system.step(1, 1.0)
    assert controller.memory['calls'] == 0


def test_actuator_cannot_drive_its_nominal_expression():
    """Keep a nominal expression outside the actuator's feedback write set."""
    env, _ = make_hardware()
    env['nominal'] = '0.2 + applied'
    with pytest.raises(ValueError, match='external ownership'):
        PhysicalActuator(env, target='applied', baseline='nominal')


def test_deferred_configuration_binds_after_serialization_without_fake_state():
    """Round-trip settings and read the actual applied value on binding."""
    config = PhysicalActuator(
        target='applied',
        baseline='nominal',
        lower=-1,
        upper=1,
        response_time_constant_seconds=0.1,
    )
    loaded = PhysicalActuator(**config.to_dict())
    assert loaded.owner is None
    assert loaded.to_dict() == config.to_dict()
    with pytest.raises(RuntimeError, match='bound to a Line/Environment'):
        loaded.make_actuator(xp=np, initial_time_seconds=0)
    env, _ = make_hardware()
    bound = loaded.copy_for(env)
    assert bound.owner is env
    np.testing.assert_array_equal(bound.snapshot(), [0.2])
    np.testing.assert_array_equal(bound.validate([0.3]), [0.3])
    with pytest.raises(ValueError, match='nonnegative'):
        PhysicalActuator(target='applied', response_time_constant_seconds=-1)


def test_external_applied_write_requires_reconciliation_before_advancing():
    """Stop at the boundary if another writer changed the applied knob."""
    env, hardware = make_hardware()
    controller, system = make_system(hardware)
    system.step(1, 1.0)
    env['applied'] = 0.4
    with pytest.raises(RuntimeError, match='outside its response owner'):
        system.step(2, 2.0)
    assert env['applied'] == 0.4
    assert controller.actuator.time_seconds == 1.0


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('derived', [False, True])
def test_other_controller_cannot_write_a_physical_actuator_nominal_input(
    reverse, derived,
):
    """Keep the nominal input independent of the controlled write set."""
    env, hardware = make_hardware()
    knob = 'nominal'
    if derived:
        knob = 'recipe'
        env[knob] = 0.2
        env['nominal'] = '2 * recipe'
        hardware = hardware.copy_for(env)
    physical = FeedbackController(
        'orbit', sources={'y': Source()}, output=hardware,
        program=lambda: {'outputs': {'command': [0.0]}},
    )
    nominal = FeedbackController(
        'nominal',
        sources={'y': Source()},
        program=lambda: {'outputs': {'command': [0.2]}},
        output=KnobOutput(env, knob),
    )
    controllers = [physical, nominal]
    with pytest.raises(ValueError, match='external ownership'):
        FeedbackSystem(controllers[::-1] if reverse else controllers)


def test_nominal_setpoint_cannot_bypass_another_actuator():
    """Reject a nominal path around a second actuator's field dynamics."""
    env, hardware = make_hardware()
    env['other_applied'] = 0.0
    env.new('other', xt.Multipole, ksl=[0.0])
    env.ref['other'].ksl[0] = env.ref['other_applied'] + env.ref['nominal']
    hardware = hardware.copy_for(env)
    controller1 = FeedbackController(
        'first', sources={'y': Source()}, output=hardware,
        program=lambda: {'outputs': {'command': [0.0]}},
    )
    controller2 = FeedbackController(
        'second', sources={'y': Source()},
        output=PhysicalActuator(env, target='other_applied'),
        program=lambda: {'outputs': {'command': [0.0]}},
    )
    with pytest.raises(ValueError, match='external ownership'):
        FeedbackSystem([controller1, controller2])


def test_pending_contribution_is_rechecked_against_current_nominal():
    """Keep a queued request and field state when the delivery total is illegal."""
    env, hardware = make_hardware(tau=1.0)
    controller, system = make_system(hardware, command_delay_seconds=0.5)
    system.step(1, 1.0)
    controller.disable()
    env['nominal'] = 0.8
    with pytest.raises(ValueError, match='limits'):
        system.step(2, 2.0)
    assert len(controller.queue) == 1
    assert env['corrector'].ksl[0] == 0.2
    assert controller.actuator.time_seconds == 1.0

    env['nominal'] = 0.4
    entry, = system.step(2, 2.0)
    assert len(controller.queue) == 0
    np.testing.assert_allclose(entry['target_output'], [0.7])
    np.testing.assert_allclose(entry['applied_output'], [0.2])


def test_unbound_actuator_rejects_duplicate_nominal_knobs():
    """Validate channel identities before the configuration is bound."""
    with pytest.raises(ValueError, match='distinct knob'):
        PhysicalActuator(target=['x', 'y'], baseline=['nominal', 'nominal'])


def test_controller_cannot_override_physical_actuator_slew_mode():
    """Keep the slew policy with the hardware that owns the response."""
    _, hardware = make_hardware()
    with pytest.raises(ValueError, match='on the physical actuator owner'):
        FeedbackController(
            'orbit', sources={'y': Source()}, output=hardware,
            program=lambda: {'outputs': {'command': [0.0]}},
            slew_mode='independent',
        )


@for_all_test_contexts(excluding=('ContextPyopencl',))
def test_vector_hardware_applies_independent_response_after_nominal_summation(
    test_context,
):
    """Apply the fast channel immediately and evolve the slow channel on context."""
    env = xt.Environment()
    for name, value in [('a', 0.0), ('b', 0.0), ('na', 0.1), ('nb', -0.1)]:
        env[name] = value
    hardware = PhysicalActuator(
        env,
        target=['a', 'b'],
        baseline=['na', 'nb'],
        response_time_constant_seconds=[0.0, 1.0],
    )
    xp = test_context.nplike_lib
    to_host = test_context.nparray_from_context_array
    response = hardware.make_actuator(xp=xp, initial_time_seconds=0.0)
    candidate = response.preview(1.0, target=[0.2, 0.2])
    hardware.apply(candidate.applied)
    response.commit(candidate)
    assert isinstance(response.applied, xp.ndarray)
    np.testing.assert_allclose(to_host(response.target), [0.3, 0.1])
    np.testing.assert_allclose(to_host(response.applied), [0.3, 0.0])
    candidate = response.preview(2.0)
    np.testing.assert_allclose(
        to_host(candidate.applied), [0.3, 0.1 * (1 - np.exp(-1))])
