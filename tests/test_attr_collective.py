import numpy as np
import pytest

import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts


class PythonDrift:
    iscollective = True
    isthick = True

    def __init__(self, length):
        self.length = length

    def track(self, particles):
        pass


class PythonPropertyDrift(PythonDrift):
    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value


@for_all_test_contexts
@pytest.mark.parametrize('kind', ['python', 'property', 'buffered'])
@pytest.mark.parametrize('isthick', [False, True])
def test_collective_length(test_context, kind, isthick):
    if kind == 'buffered':
        element_class = xt.Drift if isthick else xt.Multipole
        element = element_class(length=2., _context=test_context)
    elif kind == 'property':
        element = PythonPropertyDrift(length=2.)
    else:
        element = PythonDrift(length=2.)
    element.iscollective = True
    if kind != 'buffered':
        element.isthick = isthick
    line = xt.Line(elements=[element, xt.Drift(length=1.)])
    line.build_tracker(_context=test_context)

    # Reuse the attribute cache after changing the original element.
    for length in [2., 4.]:
        element.length = length
        expected_s = [0., length if isthick else 0.,
                      (length if isthick else 0.) + 1.]
        np.testing.assert_array_equal(line.attr['length'], [length, 1.])
        table = line.get_table(attr=True)
        np.testing.assert_array_equal(table.length, [length, 1., 0.])
        np.testing.assert_array_equal(table.s, expected_s)
        np.testing.assert_array_equal(line.survey().Z, expected_s)


@for_all_test_contexts
@pytest.mark.parametrize('buffered', [False, True])
def test_collective_strengths(test_context, buffered):
    kwargs = dict(length=2., k1=0.3, knl=[0.1, 0.2],
                  ksl=[0.4, 0.5], rot_s_rad=0.2)
    if buffered:
        element = xt.Quadrupole(_context=test_context, **kwargs)
    else:
        element = PythonDrift(length=kwargs['length'])
        for name, value in kwargs.items():
            setattr(element, name, value)
    element.iscollective = True
    line = xt.Line(elements=[element, xt.Drift(length=1.)])
    reference = xt.Line(elements=[xt.Quadrupole(**kwargs), xt.Drift(length=1.)])
    line.build_tracker(_context=test_context)
    reference.build_tracker(_context=test_context)

    for k1 in [0.3, 0.7]:
        element.k1 = k1
        reference['e0'].k1 = k1
        element.knl[1] = k1
        reference['e0'].knl[1] = k1
        table = line.get_table(attr=True)
        expected = reference.get_table(attr=True)
        for name in ['length', 'rot_s_rad', 'k0l', 'k1l', 'k2l',
                     'k0sl', 'k1sl', 'k2sl']:
            np.testing.assert_allclose(table[name], expected[name], rtol=0, atol=0)


def test_rft_element_survey():
    rft = pytest.importorskip('RF_Track')
    line = xt.Line(elements=[xt.RFT_Element(rft.Drift(2.)), xt.Drift(length=1.)])
    np.testing.assert_array_equal(line.get_table(attr=True).length, [2., 1., 0.])
    np.testing.assert_array_equal(line.survey().Z, [0., 2., 3.])


@for_all_test_contexts
@pytest.mark.parametrize('element_class, kwargs', [
    (xt.Bend, dict(length=2., angle=0.2, k1=0.1, rot_s_rad=0.3)),
    (xt.RBend, dict(length_straight=2., angle=0.2, k1=0.1, rot_s_rad=0.3)),
    (xt.Multipole, dict(length=2., knl=[0., 2.], knl_rel=[0.1, 0.2],
                        main_order=1)),
])
def test_buffered_collective_attributes(test_context, element_class, kwargs):
    element = element_class(_context=test_context, **kwargs)
    reference = xt.Line(elements=[element.copy(), xt.Drift(length=1.)])
    element.iscollective = True
    line = xt.Line(elements=[element, xt.Drift(length=1.)])
    line.build_tracker(_context=test_context)
    reference.build_tracker(_context=test_context)

    actual = line.get_table(attr=True)
    expected = reference.get_table(attr=True)
    for name in line.attr.keys():
        if name == 'angle_rad':
            continue  # Deprecated alias, already covered by `angle`.
        np.testing.assert_allclose(actual[name], expected[name], rtol=0, atol=1e-14)
    np.testing.assert_allclose(line.survey().XYZ, reference.survey().XYZ,
                               rtol=0, atol=1e-14)
