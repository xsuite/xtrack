# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import madng_tpsa
import numpy as np
import pytest
import xtrack as xt
from numpy.testing import assert_allclose, assert_array_equal
from xobjects.test_helpers import (
    allow_kernel_compilation, for_all_test_contexts,
)
from xdeps.tasks import FunctionTask

from xtrack.feedback.output import KnobOutput


def make_line(*, expressions=True, context=None):
    """Build a thin corrector with an optional baseline and trim graph."""
    line = xt.Line(
        elements={
            'kick': xt.Multipole(knl=[0.0], ksl=[0.0], _context=context),
            'drift': xt.Drift(length=1.0, _context=context),
        },
        element_names=['kick', 'drift'],
    )
    line['feedback'] = 0.0
    line['setpoint'] = 0.0
    line['baseline'] = 1e-6
    line['total'] = 'setpoint + feedback'
    if expressions:
        line.ref['kick'].ksl[0] = line.ref['baseline'] + 2 * line.ref['total']
    return line


@for_all_test_contexts
@allow_kernel_compilation
def test_knob_output_propagates_actual_field_and_tracked_kick(
    test_context, sandbox_cwd,
):
    """Propagate a context command through Xdeps to the actual particle kick."""
    line = make_line()
    output = KnobOutput(line, ['feedback'], lower=-1e-3, upper=1e-3)
    line.build_tracker(_context=test_context, use_prebuilt_kernels=False)
    command = test_context.nparray_to_context_array(np.array([2e-6]))
    output.apply(command)
    assert line['feedback'] == 2e-6
    assert line['total'] == 2e-6
    assert float(line['kick'].ksl[0]) == pytest.approx(5e-6, abs=1e-20)
    particles = xt.Particles(_context=test_context, p0c=1e9)
    line.track(particles)
    py = test_context.nparray_from_context_array(particles.py)
    assert_allclose(py, [5e-6], atol=1e-16, rtol=0)
    assert_array_equal(output.snapshot(), [2e-6])


def test_knob_output_validates_whole_vector_before_writes():
    """Leave every contribution and the field intact for an invalid vector."""
    line = make_line()
    line['other'] = 0.0
    output = KnobOutput(line, ['feedback', 'other'], lower=-1.0, upper=1.0)
    invalid_commands = (
        [0.25, np.nan], [0.25, np.inf], [0.25, 2.0], [[0.25, 0.5]], [1.0],
    )
    for invalid in invalid_commands:
        with pytest.raises(ValueError):
            output.apply(invalid)
        assert_array_equal(output.read(), [0.0, 0.0])
        assert line['kick'].ksl[0] == 1e-6
    with pytest.raises(TypeError):
        output.apply(np.array([0.1 + 1j, 0.2]))
    assert_array_equal(output.read(), [0.0, 0.0])


@pytest.mark.parametrize('operation', ['validate', 'apply'])
def test_output_checks_graph_after_command_transfer(operation):
    """Catch a graph change that occurs during the command's host transfer."""
    line = make_line()
    output = KnobOutput(line, 'feedback')

    class ChangedDuringTransfer:
        def get(self):
            line.ref['kick'].knl[0] = line.ref['feedback']
            return np.array([1e-6])

    with pytest.raises(RuntimeError, match='dependency graph changed'):
        getattr(output, operation)(ChangedDuringTransfer())
    assert line['feedback'] == 0.0
    assert line['kick'].ksl[0] == 1e-6


def test_knob_output_rejects_expression_targets_and_late_graph_change():
    """Keep expression-defined, absent and duplicate knobs out of bindings."""
    line = make_line()
    with pytest.raises(ValueError, match='expression-defined'):
        KnobOutput(line, 'total')
    with pytest.raises(KeyError, match='does not exist'):
        KnobOutput(line, 'missing')
    with pytest.raises(ValueError, match='unique'):
        KnobOutput(line, ['feedback', 'feedback'])
    output = KnobOutput(line, 'feedback')
    line['feedback'] = 'setpoint * 2'
    with pytest.raises(ValueError, match='expression-defined'):
        output.apply(1e-6)
    assert line.vars['feedback']._expr is not None


def test_knob_output_field_ownership_follows_intermediate_variables():
    """Resolve two knobs that reach the same field through a shared variable."""
    line = make_line(expressions=False)
    line['other'] = 0.0
    line['combined'] = 'feedback + other'
    line.ref['kick'].ksl[0] = line.ref['combined']
    first = KnobOutput(line, 'feedback')
    second = KnobOutput(line.env, 'other')
    assert first.dependent_field_keys == second.dependent_field_keys
    assert not set(first.ownership_keys).intersection(second.ownership_keys)
    same_knob = KnobOutput(line.env, 'feedback')
    assert same_knob.ownership_keys == first.ownership_keys


def test_knob_output_detects_distinct_wrappers_sharing_physical_storage():
    """Identify a shared field even when two wrappers expose its storage."""
    element = xt.Multipole(ksl=[0.0])
    alias = xt.Multipole(_xobject=element._xobject)
    env = xt.Environment()
    env.elements['one'] = element
    env.elements['two'] = alias
    env['u'] = 0.0
    env['v'] = 0.0
    env.ref['one'].ksl[0] = env.ref['u']
    env.ref['two'].ksl[0] = env.ref['v']
    first = KnobOutput(env, 'u')
    second = KnobOutput(env, 'v')
    assert element is not alias
    assert first.dependent_field_keys == second.dependent_field_keys
    with pytest.raises(NotImplementedError, match='same physical field'):
        KnobOutput(env, ['u', 'v'])


def test_knob_output_rejects_uninspectable_dependency_tasks():
    """Reject graph tasks whose writes cannot be traced as expressions."""
    line = make_line()
    source = line.vars['feedback']
    task = FunctionTask(
        'side_effect',
        lambda: None,
        targets={line.ref['kick'].knl[0]},
        dependencies={source},
    )
    source._manager.register(task)
    with pytest.raises(NotImplementedError, match='ExprTasks'):
        KnobOutput(line, 'feedback')
    line['independent'] = 0.0
    target = line.vars['independent']
    source._manager.register(FunctionTask(
        'write_knob', lambda: target._set_value(1.0), targets={target},
        dependencies={source},
    ))
    with pytest.raises(ValueError, match='has an Xdeps writer'):
        KnobOutput(line, 'independent')


@for_all_test_contexts
def test_knob_output_rolls_back_graph_failure(test_context):
    """Restore the exact pre-write knobs, intermediate variable and fields."""
    line = make_line(expressions=False, context=test_context)
    line['other'] = 0.0

    def guard(feedback, other):
        if other > feedback or other > 0.5:
            raise ValueError('deliberate graph failure')
        return other

    line.functions['guard'] = guard
    line.ref['kick'].ksl[0] = line.env._xdeps_fref['guard'](
        line.ref['feedback'], line.ref['other'])
    line['combined'] = '2 * feedback'
    line.ref['kick'].knl[0] = line.ref['combined']
    # Bypass Xdeps to install a field value that differs from its expression.
    line.get('kick').knl[0] = 0.125
    output = KnobOutput(line, ['feedback', 'other'])
    snapshot = output.snapshot()
    with pytest.raises(ValueError, match='deliberate graph failure'):
        output.apply([0.75, 0.75])
    assert_array_equal(output.read(), snapshot)
    assert line['combined'] == 0.0
    assert line['kick'].knl[0] == 0.125
    assert line['kick'].ksl[0] == 0.0

    output.apply([0.6, 0.4])
    assert_array_equal(output.read(), [0.6, 0.4])
    assert line['combined'] == 1.2
    assert line['kick'].knl[0] == 1.2
    assert line['kick'].ksl[0] == 0.4


def test_knob_output_rolls_back_finite_command_that_overflows_actual_field():
    """Restore a finite field after propagation overflows a dependence."""
    line = make_line()
    output = KnobOutput(line, 'feedback')
    with np.errstate(over='ignore'):
        with pytest.raises(ValueError, match='nonfinite field'):
            output.apply([1e308])
    assert line['feedback'] == 0.0
    assert line['total'] == 0.0
    assert line['kick'].ksl[0] == 1e-6
    line['baseline'] = np.inf
    with pytest.raises(ValueError, match='nonfinite field'):
        output.snapshot()


def test_knob_output_rejects_nonfinite_derived_variable():
    """A finite physical kick cannot hide a nonfinite intermediate value."""
    line = make_line(expressions=False)
    line['feedback'] = 1.0
    line['ratio'] = '1 / feedback'
    line.ref['kick'].knl[0] = 0.125 / line.ref['ratio']
    output = KnobOutput(line, 'feedback')
    with pytest.raises(ValueError, match="finite real scalar at vars"):
        output.apply(1e-309)
    assert line['feedback'] == 1.0
    assert line['ratio'] == 1.0
    assert line['kick'].knl[0] == 0.125


def test_knob_output_checks_setter_storage():
    """Restore a bend's curvature link and reject unregistered setters."""
    env = xt.Environment()
    env.new('bend', xt.RBend, length_straight=1.0, angle=0.1)
    env.new('rf', xt.Cavity, harmonic=10)
    env['trim'] = 0.0
    env['frequency'] = 0.0
    env.ref['bend'].k0 = 0.1 + env.ref['trim']
    env.ref['rf'].frequency = env.ref['frequency']
    # Restore the native curvature link as the pre-command state.
    env.get('bend').k0 = 'from_h'
    output = KnobOutput(env, ['trim', 'frequency'])
    assert len(output.dependent_field_keys) == 3
    with pytest.raises(ValueError, match='non-zero frequency'):
        output.apply([0.01, 1e3])
    assert env['trim'] == 0.0
    assert env['frequency'] == 0.0
    assert env.get('bend').k0_from_h
    assert env.get('bend')._k0 == env.get('bend').h

    env['bend_geometry'] = 0.0
    env.ref['bend'].rbend_angle_diff = env.ref['bend_geometry']
    with pytest.raises(NotImplementedError, match='RBend.rbend_angle_diff'):
        KnobOutput(env, 'bend_geometry')


def test_knob_output_preserves_scalar_mode():
    """Reject a series-valued strength without changing scalar tracking data."""
    env = xt.Environment()
    env.new('quad', xt.Quadrupole, length=0.5, k1s=0.02)
    env['trim'] = 0.0
    series = madng_tpsa.Descriptor(6, 1).var(1, 0.4)
    env.functions['gradient'] = lambda trim: series if trim > 0 else 0.3 + trim
    env.ref['quad'].k1 = env._xdeps_fref['gradient'](env.ref['trim'])
    output = KnobOutput(env, 'trim')

    with pytest.raises(NotImplementedError, match='requires scalar mode'):
        output.apply(0.25)
    assert env['trim'] == 0.0
    assert env.get('quad').k1 == 0.3
    assert env.get('quad').k1s == 0.02
    output.apply(-0.1)
    assert env.get('quad').k1 == pytest.approx(0.2)


def test_knob_output_rejects_complex_initial_values_and_bad_limits():
    """Check initial knob type, bounds and channel count at binding."""
    env = xt.Environment()
    env['a'] = 0.0
    env['complex'] = 1.0 + 2j
    with pytest.raises(TypeError, match='real numerical'):
        KnobOutput(env, 'complex')
    for lower, upper in [(np.nan, 1.0), (1.0, -1.0), ([0.0, 0.0], [1.0, 1.0])]:
        with pytest.raises(ValueError):
            KnobOutput(env, 'a', lower=lower, upper=upper)


def test_knob_output_rejects_replica_wrapper_writes():
    """Bind the physical parent field used by tracking through a Replica."""
    env = xt.Environment()
    env.new('cavity', xt.Cavity, voltage=10.0)
    env.new('alias', 'cavity', mode='replica')
    env['rf'] = 20.0
    env.ref['alias'].voltage = env.ref['rf']
    with pytest.raises(NotImplementedError, match='Replica parent'):
        KnobOutput(env, 'rf')
    alias = env.get('alias')
    assert alias.voltage == 20.0
    assert alias.resolve(env).voltage == 10.0

    env['physical_rf'] = 10.0
    env.ref['cavity'].voltage = env.ref['physical_rf']
    KnobOutput(env, 'physical_rf').apply(30.0)
    assert alias.resolve(env).voltage == 30.0


def test_knob_output_rejects_linked_variable_setters():
    """Reject an Xdeps write that would propagate to a second environment."""
    line = make_line()
    output = KnobOutput(line, 'feedback')
    linked = xt.Environment()
    linked['feedback'] = 0.0
    line.vars.vars_to_update.add(linked.vars)
    with pytest.raises(NotImplementedError, match='linked variable setters'):
        KnobOutput(line, 'feedback')
    with pytest.raises(NotImplementedError, match='linked variable setters'):
        output.apply(1e-6)
    assert line['feedback'] == 0.0
    assert linked['feedback'] == 0.0
    assert line['kick'].ksl[0] == 1e-6


def test_field_ownership_checks_metadata_without_reading_values(mocker):
    """Resolve scalar and indexed destinations without fetching field values."""
    env = xt.Environment()
    env.new('cavity', xt.Cavity, voltage=0.0)
    env.new('corrector', xt.Multipole, ksl=[0.0])
    env['command'] = 0.0
    env.ref['cavity'].voltage = env.ref['command']
    env.ref['corrector'].ksl[0] = env.ref['command']
    output = KnobOutput(env, 'command')

    cavity = env.get('cavity')
    corrector = env.get('corrector')
    voltage = cavity._xobject._get_offset('voltage')
    kick = corrector._xobject.ksl._get_offset(0)
    cavity_reads = mocker.spy(cavity._buffer, 'to_bytearray')
    corrector_reads = mocker.spy(corrector._buffer, 'to_bytearray')

    assert len(output.dependent_field_keys) == 2
    assert all(call.args[0] != voltage for call in cavity_reads.call_args_list)
    assert all(call.args[0] != kick for call in corrector_reads.call_args_list)
