import numpy as np
from numpy.polynomial import Polynomial
import pytest
import xobjects as xo
import xtrack as xt
from xtrack.prebuilt_kernel_definitions import ONLY_XTRACK_ELEMENTS


def make_element(h=0.3, pkin_const=False):
    return xt.BFieldExpansion(
        length=0.8, h=h, s_start=0.17, nstep=160, num_phi=5,
        pkin_const=pkin_const,
        knc=[[0.1, 0.08, -0.03], [0.04, -0.02, 0.01]],
        ksc=[[0.02, -0.01, 0.02]], ksol=[0.15, 0.03, 0.])


def slice_names(line):
    return [name for name in line.element_names
            if isinstance(line[name], xt.ThickSliceBFieldExpansion)]


def check_strengths(line):
    parent = line.get('e')
    table = line.get_table(attr=True)
    for name in slice_names(line):
        element = line.get(name)
        length = parent.length * element.weight
        assert table['length', name] == pytest.approx(length)
        assert table['angle', name] == pytest.approx(length * parent.h)
        for source, integrated_name in [('knc', 'knl'), ('ksc', 'ksl'), ('ksol', 'ksoll')]:
            coefficients = np.asarray(getattr(parent, source)).reshape(-1, parent.deg + 1)
            expected = []
            for order, row in enumerate(coefficients):
                integral = Polynomial(row).integ()
                strength = integral(element.s_start + length) - integral(element.s_start)
                expected.append(strength)
                column = ('ksoll' if source == 'ksol' else
                          f'k{order}{"s" if source == "ksc" else ""}l')
                assert table[column, name] == pytest.approx(strength, abs=1e-14)
            xo.assert_allclose(getattr(element, integrated_name), expected, atol=1e-14, rtol=0)
            with pytest.raises(ValueError):
                getattr(element, integrated_name)[0] = 0.
    for column, expected in [('k0l', parent.knl[0]), ('k1l', parent.knl[1]),
                             ('k0sl', parent.ksl[0]), ('ksoll', parent.ksoll[0])]:
        assert np.sum(table[column]) == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize('h', [0., 0.3])
@pytest.mark.parametrize('pkin_const', [False, True])
@pytest.mark.parametrize('custom', [False, True])
def test_bfieldexpansion_slicing_tracking_and_backtracking(h, pkin_const, custom):
    parent = make_element(h, pkin_const)
    reference_line = xt.Line(elements={'e': parent.copy()})
    line = xt.Line(elements={'e': parent})
    scheme = xt.slicing.Custom([0.2, 0.5]) if custom else xt.Uniform(4, mode='thick')
    line.slice_thick_elements([xt.Strategy(scheme)])
    names = slice_names(line)
    assert len(names) == (3 if custom else 4)
    assert xt.ThickSliceBFieldExpansion in ONLY_XTRACK_ELEMENTS

    offset = 0.
    for name in names:
        element = line.get(name)
        assert element.parent_name == 'e'
        assert element._parent._xobject._offset == parent._xobject._offset
        assert not {'knc', 'ksc', 'ksol', '_c', '_V', '_D1', '_D2', '_Q'} & element._xofields.keys()
        assert element._xobject._size < parent._xobject._size
        assert element.slice_offset == pytest.approx(offset)
        assert element.s_start == pytest.approx(parent.s_start + offset)
        field = element.get_field(x=0.01, y=0.007, s_local=0.05)
        expected = parent.get_field(x=0.01, y=0.007, s_local=element.slice_offset + 0.05)
        np.testing.assert_array_equal(field, expected)
        offset += element.weight * parent.length
    check_strengths(line)

    initial = xt.Particles(p0c=1e9, x=[0.01, -0.005], y=[0.007, 0.01], px=0.002)
    particles = initial.copy()
    reference = initial.copy()
    parent.track(reference)
    line.track(particles, _force_no_end_turn_actions=True)
    for coord in ('x', 'px', 'y', 'py', 'zeta', 'delta', 's', 'ax', 'ay'):
        xo.assert_allclose(getattr(particles, coord), getattr(reference, coord), rtol=0, atol=2e-13)

    line.track(particles, backtrack=True, _force_no_end_turn_actions=True)
    for coord in ('x', 'px', 'y', 'py', 'zeta', 'delta', 's'):
        xo.assert_allclose(getattr(particles, coord), getattr(initial, coord), rtol=0, atol=2e-12)

    survey = line.survey()
    expected_survey = reference_line.survey()
    for column in ('s', 'X', 'Y', 'Z', 'theta', 'phi', 'psi', 'E_matrix'):
        xo.assert_allclose(survey[column][-1], expected_survey[column][-1], rtol=0, atol=1e-14)


@pytest.mark.parametrize('h', [0., 0.3])
def test_bfieldexpansion_slice_parent_updates_and_serialization(h):
    env = xt.Environment()
    env.elements['e'] = make_element(h)
    line = env.new_line(components=['e'])
    line.slice_thick_elements([xt.Strategy(xt.Uniform(4, mode='thick'))])
    # Populate the attribute readers before changing the shared parent.
    check_strengths(line)
    env['strength'] = 0.1
    env['e'].knc[0, 1] = 'strength'
    env['strength'] = 0.2
    env.set('e', length=1.2, s_start=-0.1, nstep=200,
            ksc=[[0.01, 0.05, -0.02]], ksol=[0.2, 0.04, 0.])
    if h:
        env.set('e', h=0.4)
    for i, name in enumerate(slice_names(line)):
        assert line[name].s_start == pytest.approx(-0.1 + i * 0.3)
        assert line[name].nstep == 50
    check_strengths(line)

    restored = xt.Line.from_dict(line.to_dict())
    restored.build_tracker(compile=False)
    copied = restored.copy()
    copied.build_tracker(compile=False)
    for candidate in (line, restored, copied):
        check_strengths(candidate)
        particles = xt.Particles(p0c=1e9, x=0.01, y=0.007)
        reference = particles.copy()
        candidate.get('e').track(reference)
        candidate.track(particles, _force_no_end_turn_actions=True)
        for coord in ('x', 'px', 'y', 'py', 'zeta', 'delta', 's'):
            xo.assert_allclose(getattr(particles, coord), getattr(reference, coord), rtol=0, atol=2e-13)

    line.particle_ref = xt.Particles(p0c=1e9)
    twiss = line.twiss4d(betx=1., bety=1., strengths=True)
    table = line.get_table(attr=True)
    for column in ('k0l', 'k1l', 'k0sl', 'ksoll'):
        xo.assert_allclose(twiss[column], table[column], rtol=0, atol=1e-14)


def test_bfieldexpansion_slice_again_and_insert():
    line = xt.Line(elements={'e': make_element()})
    line.slice_thick_elements([xt.Strategy(xt.Uniform(4, mode='thick'))])
    line.slice_thick_elements([
        xt.Strategy(xt.Uniform(2, mode='thick'), element_type=xt.ThickSliceBFieldExpansion)])
    names = slice_names(line)
    assert len(names) == 8
    for i, name in enumerate(names):
        assert line[name].parent_name == 'e'
        assert line[name].slice_offset == pytest.approx(i * 0.1)
        assert line[name].weight == pytest.approx(0.125)
    check_strengths(line)
    line.insert('marker', xt.Marker(), at=0.35)
    names = slice_names(line)
    assert len(names) == 9
    table = line.get_table()
    assert table['s', 'marker'] == pytest.approx(0.35)
    for name in names:
        assert line[name].slice_offset == pytest.approx(table['s', name])
    check_strengths(line)


def test_bfieldexpansion_slices_reject_unsupported_modes():
    line = xt.Line(elements={'e': make_element()})
    with pytest.raises(NotImplementedError, match="mode='thick'"):
        line.slice_thick_elements([xt.Strategy(xt.Teapot(4))])
    assert line.element_names == ['e']
    line.slice_thick_elements([xt.Strategy(xt.Uniform(4, mode='thick'))])
    particles = xt.Particles(p0c=1e9, x=0.01)
    line.build_tracker(compile=False)
    line.configure_spin('auto')
    with pytest.raises(NotImplementedError, match='spin tracking'):
        line.track(particles)
    assert particles.s[0] == 0.
    line.configure_spin(None)
    element = line.get(slice_names(line)[0])
    element.radiation_flag = 1
    element.track(particles)
    xo.assert_allclose(particles.s, element.weight * element._parent.length,
                       rtol=0, atol=1e-14)
    xo.assert_allclose(particles.delta, 0., rtol=0, atol=0)
    element.radiation_flag = 0
    particles.spin_z = 1.
    with pytest.raises(NotImplementedError, match='spin tracking'):
        element.track(particles)


def test_bfieldexpansion_slice_has_at_least_one_step():
    parent = make_element()
    parent.nstep = 1
    line = xt.Line(elements={'e': parent})
    line.slice_thick_elements([xt.Strategy(xt.Uniform(4, mode='thick'))])
    for name in slice_names(line):
        assert line[name].nstep == 1
    particles = xt.Particles(p0c=1e9, x=0.01, y=0.007)
    line.track(particles, _force_no_end_turn_actions=True)
    assert particles.s[0] == pytest.approx(parent.length)
    assert np.all(np.isfinite(particles.x))
