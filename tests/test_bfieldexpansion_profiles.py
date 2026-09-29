import numpy as np
from numpy.polynomial import Polynomial
import pytest
import xobjects as xo
import xtrack as xt


@pytest.mark.parametrize('h', [0., 0.3])
@pytest.mark.parametrize('profiles, normalized', [
    (
        dict(knc=[[0.1, 0.2], [0.03]], ksc=[[0.04]],
             ksolc=[0.5, 0.01, -0.02, 0.003]),
        dict(knc=[[0.1, 0.2], [0.03, 0.]], ksc=[[0.04]],
             ksolc=[0.5, 0.01, -0.02, 0.003]),
    ),
    (
        dict(knc=[[0.1]], ksc=[[0.04, 0.01, 0., 0.003], [0.02]]),
        dict(knc=[[0.1]], ksc=[[0.04, 0.01, 0., 0.003], [0.02, 0., 0., 0.]],
             ksolc=[]),
    ),
    (
        dict(knc=[[], [0.03]], ksc=[[0.02, 0.001]], ksolc=[]),
        dict(knc=[[0.], [0.03]], ksc=[[0.02, 0.001]], ksolc=[]),
    ),
])
def test_independent_profile_shapes(h, profiles, normalized):
    element = xt.BFieldExpansion(length=0.4, h=h, s_start=0.1,
                                pkin_const=True, **profiles)
    for name, values in normalized.items():
        np.testing.assert_array_equal(getattr(element, name), values)
    width = max(np.shape(normalized['knc'])[1], np.shape(normalized['ksc'])[1],
                len(normalized['ksolc']))
    padded = {
        name: np.pad(values, ((0, 0), (0, width - np.shape(values)[1])))
        for name, values in normalized.items() if name != 'ksolc'}
    padded['ksolc'] = np.pad(normalized['ksolc'], (0, width - len(normalized['ksolc'])))
    reference = xt.BFieldExpansion(length=0.4, h=h, s_start=0.1,
                                  num_phi=element.num_phi, pkin_const=True, **padded)
    coords = dict(x=[0.01, -0.02], y=[0.03, -0.01], s_local=[0., 0.4])
    expected = reference.get_field(**coords)
    for candidate in (element, element.copy(),
                      xt.BFieldExpansion.from_dict(element.to_dict())):
        for name, values in normalized.items():
            assert getattr(candidate, name).shape == np.shape(values)
        actual = candidate.get_field(**coords)
        for name in actual.dtype.names:
            xo.assert_allclose(actual[name], expected[name], rtol=0, atol=2e-13)
        xo.assert_allclose(candidate.get_total_knl_ksl(), reference.get_total_knl_ksl(),
                           rtol=0, atol=2e-14)
        xo.assert_allclose(candidate.ksoll, reference.ksoll, rtol=0, atol=2e-14)
    # Independent on-axis checks avoid relying only on padded equivalence.
    field = element.get_field(0., 0., np.array([0., 0.4]))
    for component, name in [('Bx', 'ksc'), ('By', 'knc'), ('Bs', 'ksolc')]:
        row = normalized[name] if name == 'ksolc' else normalized[name][0]
        value = Polynomial(row)(np.array([0.1, 0.5])) if len(row) else 0.
        xo.assert_allclose(field[component], value, rtol=0, atol=2e-13)
    particles = xt.Particles(p0c=1e9, delta=0., x=coords['x'], y=coords['y'],
                             chi=[0.7, -1.], mass_ratio=[2., 1.])
    expected_particles = particles.copy()
    element.track(particles)
    reference.track(expected_particles)
    for name in ('x', 'px', 'y', 'py', 'zeta', 'delta', 'ax', 'ay', 'state'):
        xo.assert_allclose(getattr(particles, name), getattr(expected_particles, name),
                           rtol=0, atol=2e-13)


@pytest.mark.parametrize('h', [0., 0.3])
def test_profile_expressions_and_shared_slice(h):
    env = xt.Environment()
    env['strength'] = 0.
    env.new('e', xt.BFieldExpansion, length=0.4, h=h,
            knc=[[0.1, 'strength'], ['2*strength']], ksc=[['strength']],
            ksolc=[0., 0., 'strength'], knl=None)
    element = env.get('e')
    parent_xobject = element._xobject
    cache_offset = element._xobject._c._offset
    sliced = xt.ThickSliceBFieldExpansion(_parent=element, weight=0.5,
                                         slice_offset=0.2, _buffer=element._buffer)
    for strength in (0.2, 0., -0.1):
        env['strength'] = strength
        field = sliced.get_field(0., 0., 0.1)
        assert field['By'] == pytest.approx(0.1 + strength * 0.3)
        assert field['Bx'] == pytest.approx(strength)
        assert field['Bs'] == pytest.approx(strength * 0.3**2)
        assert element._xobject is parent_xobject
        assert element._xobject._c._offset == cache_offset
        assert element._eval_degree == (3 if strength else 0)
    env.set('e', knc=[['strength', 0.], ['3*strength']])
    env['strength'] = 0.4
    assert element.knc[1, 0] == pytest.approx(1.2)
    assert element.knc[1, 1] == 0.
    assert element.knl.size == 0
    assert element.knc.shape == (2, 2)
    assert element.ksc.shape == (1, 1)
    assert element.ksolc.shape == (3,)


@pytest.mark.parametrize('h', [0., 0.3])
def test_unused_field_capacity_does_not_expand_evaluation(h):
    element = xt.BFieldExpansion(length=0.4, h=h, knc=[[0.1] + [0.] * 8])
    assert element._potential_degree == 8  # No unused solenoid integral slot.
    assert element.ksc.size == element.ksolc.size == 0
    assert element.knl.size == element.ksl.size == 0
    assert element._eval_degree == 0
    assert element._eval_mmax == 0
    assert element._eval_num_phi <= 2
    coords = dict(x=[-0.02, 0., 0.01], y=[0.03, -0.01, 0.02], s_local=0.2)
    for value in (0., 0.003, 0.):
        element.knc[0, -1] = value
        dense = element.copy()
        dense._eval_degree = dense._potential_degree
        dense._eval_num_phi = dense.num_phi
        dense._eval_mmin, dense._eval_mmax = dense._mmin, dense._mmax
        actual, expected = element.get_field(**coords), dense.get_field(**coords)
        for name in actual.dtype.names:
            xo.assert_allclose(actual[name], expected[name], rtol=0, atol=2e-14)
        assert element._eval_degree == (8 if value else 0)
    element.kscale = 0.
    assert element._eval_num_phi == -1
    element.kscale = 1.
    assert element._eval_num_phi <= 2
    empty = xt.BFieldExpansion(length=0.4, h=h)
    assert empty._eval_num_phi == -1
    empty.k3s = 0.5
    assert empty.get_field(0.02, 0., 0.)['Bx'] == pytest.approx(0.5 * 0.02**3 / 6,
                                                            abs=1e-14)
    table = xt.Line(elements={'e': empty}).get_table(attr=True)
    assert table['k3sl', 'e'] == pytest.approx(0.5 * empty.length)
    empty.k3s = 0.
    assert empty._eval_num_phi == -1


def test_get_field_workspace_does_not_scale_with_expansion_capacity(monkeypatch):
    context = xo.ContextCpu()
    element = xt.BFieldExpansion(_context=context, length=0.4,
                                knc=[[0.1] + [0.] * 20])
    element.get_field(0., 0., 0.)  # Compile before measuring evaluation allocations.
    n_points = 2000
    allocate = context.zeros
    allocated = []

    def bounded_zeros(shape, *args, **kwargs):
        size = int(np.prod(shape))
        assert size <= 13 * n_points
        allocated.append(size)
        return allocate(shape, *args, **kwargs)

    monkeypatch.setattr(context, 'zeros', bounded_zeros)
    field = element.get_field(np.linspace(-0.02, 0.02, n_points), 0.01, 0.2)
    assert sum(allocated) <= 13 * n_points
    xo.assert_allclose(field['By'], 0.1, rtol=0, atol=1e-14)
