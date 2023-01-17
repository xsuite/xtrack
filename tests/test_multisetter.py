import numpy as np

import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_custom_setter_array_element(test_context):
    line = xt.Line(elements=dict(
                qf1=xt.Multipole(knl=[0, 0.1]),
                qf2=xt.Multipole(knl=[0, 0.2]),
                qf3=xt.Multipole(knl=[0, 0.3]),
                dr=xt.Drift(length=1.)),
            element_names=['qf1', 'qf2', 'qf3', 'dr'])

    tracker = line.build_tracker(_context=test_context)

    elements_to_trim = [nn for nn in line.element_names if nn.startswith('qf')]

    qf_setter = xt.MultiSetter(tracker, elements_to_trim,
                                field='knl', index=1 # we want to change knl[1]
                                )

    ctx2np = test_context.nparray_from_context_array

    values = qf_setter.get_values()
    assert np.all(ctx2np(values) == np.array([0.1, 0.2, 0.3]))

    qf_setter.set_values(np.array([10., 100., 1000.]))
    assert np.all(ctx2np(qf_setter.get_values()) == np.array([10., 100., 1000.]))
    assert line['qf1'].knl[1] == 10.
    assert line['qf2'].knl[1] == 100.
    assert line['qf3'].knl[1] == 1000.


@for_all_test_contexts
def test_custom_setter_scalar(test_context):
    line = xt.Line(elements=dict(
                qf1=xt.Multipole(knl=[0, 1], length=0.1),
                qf2=xt.Multipole(knl=[0, 1], length=0.2),
                qf3=xt.Multipole(knl=[0, 1], length=0.3),
                dr=xt.Drift(length=1.)),
            element_names=['qf1', 'qf2', 'qf3', 'dr'])

    tracker = line.build_tracker(_context=test_context)

    elements_to_trim = [nn for nn in line.element_names if nn.startswith('qf')]

    qf_setter = xt.MultiSetter(tracker, elements_to_trim,
                                field='length',
                                )

    ctx2np = test_context.nparray_from_context_array

    values = qf_setter.get_values()
    assert np.all(ctx2np(values) == np.array([0.1, 0.2, 0.3]))

    qf_setter.set_values(np.array([10., 100., 1000.]))
    assert np.all(ctx2np(qf_setter.get_values()) == np.array([10., 100., 1000.]))
    assert line['qf1'].length == 10.
    assert line['qf2'].length == 100.
    assert line['qf3'].length == 1000.


@for_all_test_contexts
def test_custom_setter_fast_quadrupole(test_context):
    line = xt.Line(elements=dict(
                qf1=xt.SimpleThinQuadrupole(knl=[0, 1]),
                qf2=xt.SimpleThinQuadrupole(knl=[0, 1]),
                qf3=xt.SimpleThinQuadrupole(knl=[0, 1]),
                dr=xt.Drift(length=1.)),
            element_names=['qf1', 'qf2', 'qf3', 'dr'])

    tracker = line.build_tracker(_context=test_context)

    elements_to_trim = [nn for nn in line.element_names if nn.startswith('qf')]

    qf_setter = xt.MultiSetter(tracker, elements_to_trim,
                               field='knl', index=1)

    ctx2np = test_context.nparray_from_context_array

    values = qf_setter.get_values()
    assert np.all(ctx2np(values) == np.array([1, 1, 1]))

    qf_setter.set_values(np.array([1.1, 1.2, 1.3]))
    assert np.all(ctx2np(qf_setter.get_values()) == np.array([1.1, 1.2, 1.3]))
    assert line['qf1'].knl[1] == 1.1
    assert line['qf2'].knl[1] == 1.2
    assert line['qf3'].knl[1] == 1.3


@for_all_test_contexts
def test_custom_setter_fast_dipole(test_context):
    line = xt.Line(elements=dict(
                df1=xt.SimpleThinBend(knl=[1], length=0.1, hxl=0.1),
                df2=xt.SimpleThinBend(knl=[1], length=0.2, hxl=0.1),
                df3=xt.SimpleThinBend(knl=[1], length=0.3, hxl=0.1),
                dr=xt.Drift(length=1.)),
            element_names=['df1', 'df2', 'df3', 'dr'])

    tracker = line.build_tracker(_context=test_context)

    elements_to_trim = [nn for nn in line.element_names if nn.startswith('df')]

    qf_setter = xt.MultiSetter(tracker, elements_to_trim,
                               field='knl', index=0)

    ctx2np = test_context.nparray_from_context_array

    values = qf_setter.get_values()
    assert np.all(ctx2np(values) == np.array([1, 1, 1]))

    qf_setter.set_values(np.array([1.1, 1.2, 1.3]))
    assert np.all(ctx2np(qf_setter.get_values()) == np.array([1.1, 1.2, 1.3]))
    assert line['df1'].knl[0] == 1.1
    assert line['df2'].knl[0] == 1.2
    assert line['df3'].knl[0] == 1.3
