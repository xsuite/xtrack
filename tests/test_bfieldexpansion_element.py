import xtrack as xt
import numpy as np
import xobjects as xo
import pytest
from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts(excluding='ContextCpu')
@pytest.mark.parametrize('h', [0., 0.3])
@pytest.mark.parametrize('pkin_const', [False, True])
def test_bfieldexpansion_gpu_matches_cpu(test_context, h, pkin_const):
    kwargs = dict(length=0.3, h=h, num_phi=5, num_integration_steps=12, s_start=0.1,
                  pkin_const=pkin_const, kscale=0.7,
                  ksc=[[0.04, 0.2, 0.], [0.03, 0., 0.]],
                  knc=[[0.05, 0.1, 0.], [0.02, 0., 0.]],
                  ksol=[0.1, 0.02, 0.], knl=[0.01, 0.003, 0.001],
                  ksl=[0.002, 0.001])
    reference = xt.BFieldExpansion(**kwargs)
    element = xt.BFieldExpansion(_context=test_context, **kwargs)
    initial = xt.Particles(
        p0c=1e9, x=[0., -0.007, 0.003], px=[0.001, -0.002, 0.0005],
        y=[0.007, -0.004, 0.002], py=[-0.0003, 0.001, -0.002],
        zeta=[0.001, -0.002, 0.003], delta=[0., 0.01, -0.02])

    def compare_particles(actual, expected):
        for name in ('x', 'px', 'y', 'py', 'zeta', 'delta', 'ptau', 's',
                     'ax', 'ay'):
            xo.assert_allclose(getattr(actual, name), getattr(expected, name),
                               rtol=0, atol=1e-12)
        xo.assert_allclose(actual.state, expected.state, rtol=0, atol=0)
        xo.assert_allclose(actual.state, 1, rtol=0, atol=0)

    # Exercise both GPU construction and rebuilding after coefficient updates.
    for updated in (False, True):
        if updated:
            for ee in (reference, element):
                ee.knc[0, 1] += 0.02
                ee.ksc[0, 0] -= 0.01
                ee.ksol[0] += 0.03
                ee.knl[1] += 0.002
                ee.ksl[0] -= 0.001
                ee.kscale = -1.2
        xo.assert_allclose(element._c, reference._c, rtol=0, atol=1e-13)
        field = element.get_field(initial.x, initial.y, [0.1, 0.2, 0.4])
        expected_field = reference.get_field(initial.x, initial.y, [0.1, 0.2, 0.4])
        for name in field.dtype.names:
            xo.assert_allclose(field[name], expected_field[name], rtol=0, atol=1e-12)
        expected = initial.copy()
        actual = initial.copy(_context=test_context)
        reference.track(expected)
        element.track(actual)
        compare_particles(actual, expected)

    # Building a sliced GPU tracker also exercises moving the parent from CPU.
    lines = []
    for context in (reference._context, test_context):
        line = xt.Line(elements={'e': reference.copy()})
        line.slice_thick_elements([xt.Strategy(xt.Uniform(2, mode='thick'))])
        line.build_tracker(_context=context)
        lines.append(line)
    expected = initial.copy()
    actual = initial.copy(_context=test_context)
    for backtrack in (False, True):
        for line, particles in zip(lines, (expected, actual)):
            line.track(particles, backtrack=backtrack,
                       _force_no_end_turn_actions=True)
        compare_particles(actual, expected)


@pytest.mark.parametrize('h', [0., 0.3])
def test_bfieldexpansion_names_and_coefficients(h):
    from xtrack.prebuilt_kernel_definitions import ONLY_XTRACK_ELEMENTS

    ksc = np.array([[0.04, 0.2, 0.], [0.03, 0., 0.]])
    knc = np.array([[0.05, 0.1, 0.], [0.02, 0., 0.]])
    ksol = np.array([0.1, 0.02, 0.])
    element = xt.BFieldExpansion(
        length=0.3, h=h, ksc=ksc, knc=knc, ksol=ksol, num_phi=5)

    assert type(element) is xt.BFieldExpansion
    assert element.straight == int(h == 0)
    assert xt.BFieldExpansion in ONLY_XTRACK_ELEMENTS
    for name, coefficients in [('ksc', ksc), ('knc', knc), ('ksol', ksol)]:
        xo.assert_allclose(getattr(element, name), coefficients, rtol=0, atol=0)

    serialized = element.to_dict()
    assert serialized['__class__'] == 'BFieldExpansion'
    assert {'ksc', 'knc', 'ksol'} <= serialized.keys()
    assert not {'a', 'b', 'bs', 'k0s', 'k0c', 'kn', 'ks'} & serialized.keys()
    assert not hasattr(element, 'kn')
    assert not hasattr(element, 'ks')
    assert element.knc.shape == knc.shape
    assert element.ksc.shape == ksc.shape
    assert tuple(element._xobject.knc._shape) == knc.shape
    assert tuple(element._xobject.ksc._shape) == ksc.shape

    s = np.array([0., 0.1, 0.3])
    field = element.get_field(x=0., y=0., s_local=s)
    xo.assert_allclose(field['Bx'], 0.04 + 0.2*s, rtol=0, atol=1e-14)
    xo.assert_allclose(field['By'], 0.05 + 0.1*s, rtol=0, atol=1e-14)
    xo.assert_allclose(field['Bs'], 0.1 + 0.02*s, rtol=0, atol=1e-14)


@pytest.mark.parametrize('h', [0., 0.3])
@pytest.mark.parametrize('empty', ['omitted', 'none', 'list', 'array'])
@pytest.mark.parametrize('provided, sizes', [
    pytest.param({}, (1, 1, 1, 1, 1), id='all-empty'),
    pytest.param({'knc': [[0.1, 0.02, 0.], [0.03, 0., 0.]]},
                 (2, 2, 3, 2, 2), id='normal-profile'),
    pytest.param({'ksc': [[0.01, 0.02], [0.03, 0.], [0.004, 0.]]},
                 (3, 3, 2, 3, 3), id='skew-profile'),
    pytest.param({'ksol': [0.1, 0.02, 0.]},
                 (1, 1, 3, 1, 1), id='solenoid-profile'),
    pytest.param({'knl': [0., 0.01, 0.002]},
                 (3, 3, 1, 3, 3), id='normal-hard-edge'),
    pytest.param({'ksl': [0.001, 0.002]},
                 (2, 2, 1, 2, 2), id='skew-hard-edge'),
    pytest.param({'knc': [[0.1, 0.02]], 'ksc': [[0.01, 0.], [0.005, 0.]],
                  'knl': [0.01, 0.003, 0.004]},
                 (1, 2, 2, 3, 2), id='unequal-nonempty-shapes'),
])
def test_bfieldexpansion_empty_coefficients(h, empty, provided, sizes):
    nb, na, width, nnl, nsl = sizes
    expected = dict(knc=np.zeros((nb, width)), ksc=np.zeros((na, width)),
                    ksol=np.zeros(width), knl=np.zeros(nnl), ksl=np.zeros(nsl))
    kwargs = {}
    if empty != 'omitted':
        for name in expected:
            kwargs[name] = (None if empty == 'none' else [] if empty == 'list'
                            else np.zeros((0, 0) if name in ('knc', 'ksc') else 0))
    kwargs.update(provided)
    expected.update(provided)
    element = xt.BFieldExpansion(length=0.4, h=h, s_start=0.1, **kwargs)
    reference = xt.BFieldExpansion(length=0.4, h=h, s_start=0.1, **expected)
    for name, values in expected.items():
        assert getattr(element, name).shape == np.shape(values)
        xo.assert_allclose(getattr(element, name), values, rtol=0, atol=0)
    assert element.num_phi == reference.num_phi
    field = element.get_field(x=[0.01, -0.02], y=0.003, s_local=[0., 0.4])
    reference_field = reference.get_field(x=[0.01, -0.02], y=0.003, s_local=[0., 0.4])
    np.testing.assert_array_equal(field, reference_field)
    particles = xt.Particles(p0c=1e9, x=0.01, y=0.003, px=0.001)
    reference_particles = particles.copy()
    element.track(particles)
    reference.track(reference_particles)
    for name in ('x', 'px', 'y', 'py', 'zeta', 'delta', 's'):
        xo.assert_allclose(getattr(particles, name), getattr(reference_particles, name),
                           rtol=0, atol=1e-14)
    _check_integrated_bfieldexpansion_strengths(element)
    restored = xt.BFieldExpansion.from_dict(element.to_dict())
    for name in expected:
        xo.assert_allclose(getattr(restored, name), getattr(element, name), rtol=0, atol=0)


@pytest.mark.parametrize('h', [0., 0.3])
@pytest.mark.parametrize('empty', [False, True])
def test_bfieldexpansion_empty_coefficients_environment(h, empty):
    env = xt.Environment()
    env['normal'] = 0.02
    kwargs = dict(ksc=[], ksol=[], knl=[], ksl=[]) if empty else {}
    env.new('profile', 'BFieldExpansion', length=0.4, h=h,
            knc=[[0.1, 0., 0.], ['normal', 0.01, 0.]], **kwargs)
    env.new('hard_edge', 'BFieldExpansion', length=0.4, h=h, knl=[0., 'normal'])
    env.new('zero', 'BFieldExpansion', length=0.4, h=h)
    env['normal'] = 0.03
    # The inferred zero arrays remain writable, with correctly allocated caches.
    env.set('profile', ksc=[[0.01, 0., 0.], [0., 0.02, 0.]],
            ksol=[0.1, 0.02, 0.], knl=[0., '2*normal'], ksl=[0.001, 0.])
    env.set('hard_edge', knc=[[0.], [0.01]], ksc=[[0.], [0.02]])
    env['normal'] = 0.04
    for name in ('profile', 'hard_edge', 'zero'):
        element = env.get(name)
        assert element.knc.size and element.ksc.size and element.ksol.size
        reference = xt.BFieldExpansion(length=0.4, h=h, **{
            field: np.asarray(getattr(element, field))
            for field in ('knc', 'ksc', 'ksol', 'knl', 'ksl')})
        xo.assert_allclose(element._c, reference._c, rtol=0, atol=0)
        _check_integrated_bfieldexpansion_strengths(element)
    table = env.new_line(components=['profile', 'hard_edge', 'zero']).get_table(attr=True)
    assert table['k1l', 'profile'] == pytest.approx(0.04 * 0.4 + 0.01 * 0.4**2 / 2 + 0.08)
    assert table['k1l', 'hard_edge'] == pytest.approx(0.01 * 0.4 + 0.04)
    assert table['k1l', 'zero'] == 0.


@pytest.mark.parametrize('kwargs, message', [
    ({'knc': [0.1]}, 'two-dimensional'),
    ({'ksc': [0.1]}, 'two-dimensional'),
    ({'ksol': [[0.1, 0.]]}, 'one-dimensional'),
    ({'knc': [[0.1, 0.]], 'ksc': [[0.]]}, 'longitudinal coefficients'),
    ({'knc': [[0.1, 0.]], 'ksol': [0.]}, 'longitudinal coefficients'),
])
def test_bfieldexpansion_nonempty_coefficient_shapes(kwargs, message):
    with pytest.raises(ValueError, match=message):
        xt.BFieldExpansion(length=0.4, **kwargs)


@pytest.mark.parametrize('h', [0., 0.3])
def test_bfieldexpansion_serialization(h):
    element = xt.BFieldExpansion(
        length=0.3, h=h, ksc=[[0.04, 0.2, 0.]],
        knc=[[0.05, 0.1, 0.], [0.02, 0., 0.]], ksol=[0.1, 0.02, 0.],
        num_phi=5, s_start=0.1, num_integration_steps=12)
    restored = xt.BFieldExpansion.from_dict(element.to_dict())
    assert type(restored) is xt.BFieldExpansion
    assert restored.num_phi == element.num_phi == 5
    assert restored.s_start == element.s_start == 0.1
    assert {'num_phi', 's_start'} <= element.to_dict().keys()
    assert not {'ny', 'sstart'} & element.to_dict().keys()
    assert restored.straight == element.straight
    assert restored.h == element.h
    assert restored.angle == element.angle
    for name in ('knc', 'ksc', 'ksol', 'knl', 'ksl', 'ksoll', '_c'):
        xo.assert_allclose(getattr(restored, name), getattr(element, name),
                           rtol=0, atol=0)
    restored.knc[0, 1] += 0.01
    assert restored.knc[0, 1] != element.knc[0, 1]
    _check_integrated_bfieldexpansion_strengths(restored)


@pytest.mark.parametrize('h', [0., 0.3])
@pytest.mark.parametrize('pkin_const', [False, True])
def test_bfieldexpansion_kscale(h, pkin_const):
    coefficients = dict(
        knc=np.array([[0.1, 0.02, 0.003], [0.04, 0.001, 0.]]),
        ksc=np.array([[0.01, -0.002, 0.], [0.003, 0., 0.]]),
        ksol=np.array([0.05, 0.002, 0.]),
        knl=np.array([0.01, 0.003, 0.001]), ksl=np.array([0.002, 0.001]))
    kwargs = dict(length=0.4, h=h, s_start=0.1, num_integration_steps=20, pkin_const=pkin_const)
    unscaled = xt.BFieldExpansion(**kwargs, **coefficients)
    assert unscaled.kscale == unscaled._xobject.kscale == 1.
    element = xt.BFieldExpansion(**kwargs, **coefficients, kscale=0.5)
    assert element.kscale == element._xobject.kscale == 0.5
    coordinates = dict(x=[0.01, -0.015], y=[0.007, -0.003], s_local=[0., 0.3])
    unscaled_field = unscaled.get_field(**coordinates)
    for scale in (0.5, 0., -0.7, 1.):
        # The first iteration checks scaling at construction, then updates.
        if scale != 0.5:
            element.kscale = scale
        reference = xt.BFieldExpansion(**kwargs, **{
            name: values * scale for name, values in coefficients.items()})
        field = element.get_field(**coordinates)
        reference_field = reference.get_field(**coordinates)
        for name in field.dtype.names:
            xo.assert_allclose(field[name], scale * unscaled_field[name], rtol=0, atol=1e-13)
            xo.assert_allclose(field[name], reference_field[name], rtol=0, atol=1e-13)
        particles = xt.Particles(p0c=1e9, x=0.01, y=0.007, px=0.002, delta=0.01)
        expected = particles.copy()
        element.track(particles)
        reference.track(expected)
        for name in ('x', 'px', 'y', 'py', 'zeta', 'delta', 's', 'ax', 'ay'):
            xo.assert_allclose(getattr(particles, name), getattr(expected, name),
                               rtol=0, atol=1e-13)
        _check_integrated_bfieldexpansion_strengths(element)
        for name, values in coefficients.items():
            xo.assert_allclose(getattr(element, name), values, rtol=0, atol=0)
        assert element.h == h
        assert element.angle == h * kwargs['length']

    # Every cache rebuild must apply the scale once, after any other update.
    element.kscale = -0.8
    for name in coefficients:
        getattr(element, name)[...] *= 1.1
    element.length = 0.5
    element.s_start = -0.1
    if h:
        element.h = 0.4
    reference = xt.BFieldExpansion(
        length=element.length, h=element.h, s_start=element.s_start,
        num_integration_steps=20, pkin_const=pkin_const, **{
            name: np.asarray(getattr(element, name)) * element.kscale
            for name in coefficients})
    reference_field = reference.get_field(**coordinates)
    for candidate in (element, element.copy(), xt.BFieldExpansion.from_dict(element.to_dict())):
        assert candidate.kscale == candidate._xobject.kscale == -0.8
        _check_integrated_bfieldexpansion_strengths(candidate)
        field = candidate.get_field(**coordinates)
        for name in field.dtype.names:
            xo.assert_allclose(field[name], reference_field[name], rtol=0, atol=1e-13)


@pytest.mark.parametrize('h', [0., 0.3])
def test_bfieldexpansion_kscale_environment(h):
    env = xt.Environment()
    env['scale'] = 0.
    env.new('source', 'BFieldExpansion', length=0.4, h=h, kscale='scale',
            knc=[[0.1, 0.02]], ksc=[[0.01, 0.]], ksol=[0.05, 0.],
            knl=[0.002], ksl=[0.001])
    env.new('clone', 'source')
    env.new('linked', 'source', kscale=env.ref['source'].kscale)
    env.set('source', kscale='2*scale')
    for scale in (0.7, 0., -1.2):
        env['scale'] = scale
        for name, factor in [('source', 2 * scale), ('clone', scale), ('linked', 2 * scale)]:
            element = env.get(name)
            assert element.kscale == factor
            field = element.get_field(0., 0., 0.2)
            xo.assert_allclose(field['By'], factor * (0.1 + 0.02 * 0.2 + 0.002 / 0.4),
                               rtol=0, atol=1e-14)
            xo.assert_allclose(field['Bx'], factor * (0.01 + 0.001 / 0.4),
                               rtol=0, atol=1e-14)
            xo.assert_allclose(field['Bs'], factor * 0.05, rtol=0, atol=1e-14)
            _check_integrated_bfieldexpansion_strengths(element)


@pytest.mark.parametrize('na, nb, degree', [(1, 1, 0), (2, 3, 3),
                                         (3, 2, 4), (1, 1, 7)])
def test_bfieldexpansion_auto_complete_straight_field(na, nb, degree):
    rng = np.random.default_rng(2026)
    coefficients = dict(ksc=rng.normal(size=(na, degree + 1)),
                        knc=rng.normal(size=(nb, degree + 1)),
                        ksol=rng.normal(size=degree + 1))
    coefficients['ksol'][-1] = 0.  # Room for the scalar-potential integral.
    element = xt.BFieldExpansion(length=0.4, s_start=0.13, **coefficients)
    reference = xt.BFieldExpansion(
        length=element.length, s_start=element.s_start,
        num_phi=element.num_phi + 4, **coefficients)
    x, y, s_local = [0.2, -0.1], [-0.3, 0.25], [0., 0.4]
    field = element.get_field(x, y, s_local)
    expected = reference.get_field(x, y, s_local)
    for name in field.dtype.names:
        xo.assert_allclose(field[name], expected[name], rtol=0, atol=1e-13)

    particles = xt.Particles(p0c=1e9, x=0.02, y=-0.03, px=0.01, py=0.02)
    tracked_reference = particles.copy()
    element.track(particles)
    reference.track(tracked_reference)
    for name in ('x', 'px', 'y', 'py', 'zeta', 'delta'):
        xo.assert_allclose(getattr(particles, name), getattr(tracked_reference, name),
                           rtol=0, atol=1e-13)


def test_bfieldexpansion_auto_linear_curvature():
    # A quadrupole with k1(s)=s has phi_1=-x*s. To first order in h,
    # phi_3=h*s, hence Ax(x=0)=h*y**4/24. The extra two orders must retain
    # this term; truncating at num_phi=2 misses it entirely.
    coefficients = dict(ksc=[[0., 0.]], knc=[[0., 0.], [0., 1.]], ksol=[0., 0.])
    straight = xt.BFieldExpansion(length=1., **coefficients)
    h, y, s_local = 0.1, 0.4, 0.3
    curved = xt.BFieldExpansion(length=1., h=h, **coefficients)
    assert curved.num_phi == straight.num_phi + 2
    # Auto also reserves scalar-octupole capacity, retaining higher powers
    # of h. Isolate the leading term with an explicit fourth-order expansion.
    linear_curvature = xt.BFieldExpansion(length=1., h=h, num_phi=4,
                                         **coefficients)
    field = linear_curvature.get_field(0., y, s_local)
    xo.assert_allclose(field['Ax'], h * y**4 / 24, rtol=0, atol=1e-14)
    # The automatic order also reserves capacity for independent scalar
    # octupoles. Explicitly truncate below the y**4 vector-potential term.
    truncated = xt.BFieldExpansion(length=1., h=h, num_phi=2,
                                   **coefficients)
    assert abs(truncated.get_field(0., y, s_local)['Ax'] - field['Ax']) > 1e-5

    # All field and potential components agree with the first-order analytic
    # expansion up to O(h**2), including the Frenet metric factors.
    x = 0.2
    errors = []
    for curvature in (0.04, 0.02, 0.01):
        curved.h = curvature
        field = curved.get_field(x, y, s_local)
        expected = dict(
            phi=-x*s_local*y + curvature*s_local*y**3/6,
            Bx=s_local*y,
            By=x*s_local - curvature*s_local*y**2/2,
            Bs=x*y - curvature*(x**2*y + y**3/6),
            Ax=(-x + curvature*x**2)*y**2/2 + curvature*y**4/24,
            Ay=0., As=s_local*(y**2-x**2)/2 + curvature*s_local*x**3/6)
        errors.append(max(abs(field[name] - value) for name, value in expected.items()))
    xo.assert_allclose(np.array(errors[:-1]) / errors[1:], 4., rtol=0.03, atol=0)


@pytest.mark.parametrize('h', [0., 0.3])
def test_bfieldexpansion_auto_environment_updates(h):
    env = xt.Environment()
    env['strength'] = 0.
    knc = [[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 'strength']]
    env.new('e', 'BFieldExpansion', length=0.4, h=h, num_phi='auto',
            ksc=np.zeros((1, 4)), knc=knc, ksol=np.zeros(4))
    element = env.get('e')
    original_order = element.num_phi
    env['strength'] = 0.1
    env.set('e', num_phi='auto', s_start=0.2, ksc=[[0.1, 0.2, 0.3, 0.4]])
    assert element.num_phi == original_order
    assert element.knc[2, 3] == 0.1
    with pytest.raises(ValueError, match='fixed at construction'):
        element.num_phi += 2
    assert element.num_phi == original_order
    restored = xt.BFieldExpansion.from_dict(element.to_dict())
    assert restored.num_phi == original_order
    assert restored.s_start == element.s_start
    reference = xt.BFieldExpansion(length=element.length, h=h, s_start=element.s_start,
        num_phi=original_order + 4, knc=element.knc, ksc=element.ksc, ksol=element.ksol)
    # Straight auto covers the whole polynomial; y=0 avoids higher-h
    # corrections in the curved comparison.
    field = element.get_field(0.02, 0. if h else 0.1, 0.15)
    expected = reference.get_field(0.02, 0. if h else 0.1, 0.15)
    for name in field.dtype.names:
        xo.assert_allclose(field[name], expected[name], rtol=0, atol=1e-13)


@pytest.mark.parametrize('h', [0., 0.3])
def test_bfieldexpansion_get_field_local_origin(h):
    element = xt.BFieldExpansion(length=0.5, h=h, s_start=0.4,
        ksc=[[0.1, 0.2, 0.3]], knc=[[0.4, 0.5, 0.6]], ksol=[0.7, 0.8, 0.])
    s_local = np.array([-0.1, 0., 0.2])
    polynomial_s = element.s_start + s_local
    field = element.get_field(x=0., y=0., s_local=s_local)
    xo.assert_allclose(field['Bx'], 0.1 + 0.2*polynomial_s + 0.3*polynomial_s**2,
                       rtol=0, atol=1e-14)
    xo.assert_allclose(field['By'], 0.4 + 0.5*polynomial_s + 0.6*polynomial_s**2,
                       rtol=0, atol=1e-14)
    xo.assert_allclose(field['Bs'], 0.7 + 0.8*polynomial_s, rtol=0, atol=1e-14)


def test_bfieldexpansion_mixed_geometry_line():
    env = xt.Environment()
    for name, h in [('straight', 0.), ('curved', 0.3)]:
        env.new(name, 'BFieldExpansion', length=0.3, h=h, num_phi=5,
                ksc=[[0.04, 0.2, 0.]], knc=[[0.05, 0.1, 0.]],
                ksol=[0.1, 0.02, 0.])
    line = env.new_line(components=['straight', 'curved'])
    line = xt.Line.from_dict(line.to_dict())
    particles = xt.Particles(p0c=1e9, x=[0.01, -0.01], y=0.007)
    reference = particles.copy()
    env.get('straight').track(reference)
    env.get('curved').track(reference)
    line.track(particles, _force_no_end_turn_actions=True)
    assert line.tracker.line_element_classes == {
        xt.BFieldExpansion._XoStruct, xt.ThickSliceBFieldExpansion._XoStruct}
    for coord in ('x', 'px', 'y', 'py', 'zeta', 'delta', 's'):
        xo.assert_allclose(getattr(particles, coord), getattr(reference, coord),
                           rtol=0, atol=1e-14)


@pytest.mark.parametrize('h', [0., 0.3])
def test_bfieldexpansion_geometry_updates(h):
    coefficients = dict(
        ksc=np.array([[0.04, 0.2, 0.], [0.03, 0., 0.]]),
        knc=np.array([[0.05, 0.1, 0.], [0.02, 0., 0.]]),
        ksol=np.array([0.1, 0.02, 0.]), num_phi=5, num_integration_steps=12,
    )
    element = xt.BFieldExpansion(length=0.3, h=h, **coefficients)
    assert element.angle == pytest.approx(0.3 * h)
    assert element._xobject.angle == element.angle
    with pytest.raises(AttributeError):
        element.angle = 0.7

    for length in [0., -0.2, 0.6]:
        element.length = length
        assert element.angle == pytest.approx(length * h)
        assert element._xobject.angle == element.angle
        assert element.ds == pytest.approx(length / element.num_integration_steps)

    if h:
        for new_h in [0.4, 0.6, h]:
            element.h = new_h
            assert element.angle == pytest.approx(element.length * new_h)
            assert element._xobject.angle == element.angle
            fresh = xt.BFieldExpansion(
                length=element.length, h=new_h, **coefficients)
            xo.assert_allclose(element._c, fresh._c, rtol=0, atol=1e-14)

    # Updating a variable through an element reference must also use the setters.
    line = xt.Line(elements={'expansion': element})
    line.vars['expansion_length'] = element.length
    line.ref['expansion'].length = line.vars['expansion_length']
    line.vars['expansion_length'] = 0.8
    if h:
        line.vars['expansion_h'] = element.h
        line.ref['expansion'].h = line.vars['expansion_h']
        line.vars['expansion_h'] = 0.5

    assert element.angle == pytest.approx(element.length * element.h)
    assert element._xobject.angle == element.angle
    fresh = xt.BFieldExpansion(
        length=element.length, h=element.h, **coefficients)
    field = element.get_field(x=0.02, y=0.01, s_local=0.4)
    expected_field = fresh.get_field(x=0.02, y=0.01, s_local=0.4)
    for name in field.dtype.names:
        xo.assert_allclose(field[name], expected_field[name], rtol=0, atol=1e-14)

    particles = xt.Particles(x=0.01, y=0.007, beta0=0.7)
    expected_particles = particles.copy()
    element.track(particles)
    fresh.track(expected_particles)
    for name in ['x', 'px', 'y', 'py', 'zeta', 'delta', 's']:
        xo.assert_allclose(getattr(particles, name),
                           getattr(expected_particles, name), rtol=0, atol=1e-14)
    xo.assert_allclose(particles.s, element.length, rtol=0, atol=1e-14)

    copied = element.copy()
    assert copied._xobject.angle == element.angle
    copied.length = 0.4
    assert copied.angle == pytest.approx(copied.length * copied.h)
    assert element.length == 0.8


def test_bfieldexpansion_survey_matches_sector_bend():
    expansion = xt.BFieldExpansion(
        length=0.8, h=0.3, ksc=np.array([[0.]]), knc=np.array([[0.]]),
        ksol=np.array([0.]), num_phi=5,
    )
    bend = xt.Bend(length=expansion.length, angle=expansion.angle)
    before, after = 1.25, 0.75
    line = xt.Line(elements={
        'before': xt.Drift(length=before),
        'bend': expansion,
        'after': xt.Drift(length=after),
    })
    reference = xt.Line(elements={
        'before': xt.Drift(length=before),
        'bend': bend,
        'after': xt.Drift(length=after),
    })

    # Repeat on the same lines so cached attribute readers see geometry updates.
    # The field is zero: survey must follow the reference bend, not its strength.
    for attribute, value in [(None, None), ('length', 1.1), ('h', 0.5)]:
        if attribute is not None:
            setattr(expansion, attribute, value)
        angle = expansion.length * expansion.h
        bend.length = expansion.length
        bend.angle = angle

        survey = line.survey()
        expected = reference.survey()
        np.testing.assert_array_equal(survey.name, expected.name)
        for name in ['s', 'X', 'Y', 'Z', 'theta', 'phi', 'psi', 'E_matrix']:
            xo.assert_allclose(survey[name], expected[name], rtol=0, atol=1e-14)

        # Also check the analytical exit pose to catch an ignored bend angle
        # even if both element types were accidentally treated as drifts.
        xo.assert_allclose(
            survey.X[-1], (np.cos(angle) - 1) / expansion.h - after * np.sin(angle),
            rtol=0, atol=1e-14)
        xo.assert_allclose(
            survey.Z[-1], before + np.sin(angle) / expansion.h + after * np.cos(angle),
            rtol=0, atol=1e-14)
        xo.assert_allclose(survey.theta[-1], -angle, rtol=0, atol=1e-14)
        xo.assert_allclose(survey.s[-1], before + expansion.length + after,
                           rtol=0, atol=1e-14)


def test_bfieldexpansion_geometry_is_fixed():
    kwargs = dict(length=0.3, ksc=[[0.]], knc=[[0.1]], ksol=[0.], num_phi=5)
    for h in (-0.3, 1e-9, 1e-4):
        with pytest.raises(ValueError, match='requires h > 1e-4'):
            xt.BFieldExpansion(h=h, **kwargs)

    for h in (0., 0.3):
        element = xt.BFieldExpansion(h=h, **kwargs)
        coefficients = element._c.copy()
        with pytest.raises(ValueError, match='requires h'):
            element.h = 0. if h else 0.3
        with pytest.raises(AttributeError):
            element.straight = 1 - element.straight
        assert element.h == h
        assert element.straight == int(h == 0)
        assert element.angle == pytest.approx(0.3 * h)
        assert element._xobject.angle == element.angle
        xo.assert_allclose(element._c, coefficients, rtol=0, atol=0)


@pytest.mark.parametrize('h', [0., 0.3])
def test_bfieldexpansion_standalone_does_not_radiate(h):
    element = xt.BFieldExpansion(
        length=0.2, h=h, ksc=np.array([[0.]]), knc=np.array([[0.1]]),
        ksol=np.array([0.]), num_phi=5,
    )
    particles = xt.Particles(p0c=1e9, x=0.01)
    reference = particles.copy()
    element.radiation_flag = 1
    element.track(particles)
    element.radiation_flag = 0
    element.track(reference)
    for name in ('x', 'px', 'y', 'py', 'zeta', 'delta', 's'):
        xo.assert_allclose(getattr(particles, name), getattr(reference, name), rtol=0, atol=0)
    xo.assert_allclose(particles.s, element.length, rtol=0, atol=1e-14)
    assert np.all(particles.state > 0)


@pytest.mark.parametrize('h', [0., 0.3])
@pytest.mark.parametrize('model', ['mean', 'quantum', 'quantum-kick'])
@pytest.mark.parametrize('sliced', [False, True])
def test_bfieldexpansion_does_not_radiate(h, model, sliced):
    element = xt.BFieldExpansion(length=0.2, h=h, ksc=[[0.]],
                                 knc=[[0.1]], ksol=[0.])
    line = xt.Line(elements={'e': element})
    if sliced:
        line.slice_thick_elements([xt.Strategy(xt.Uniform(2, mode='thick'))])
    particles = xt.Particles(p0c=5e9, x=0.01, px=0.002, y=0.003, delta=0.01)
    reference = particles.copy()
    line.track(reference, _force_no_end_turn_actions=True)
    line.configure_radiation(model)
    line.track(particles, _force_no_end_turn_actions=True)
    for name in ('x', 'px', 'y', 'py', 'zeta', 'delta', 'ptau', 's', 'state'):
        xo.assert_allclose(getattr(particles, name), getattr(reference, name),
                           rtol=0, atol=1e-14)


def _check_integrated_bfieldexpansion_strengths(element):
    from numpy.polynomial import Polynomial

    knl, ksl = element.get_total_knl_ksl()
    for name, integral_name, total in [('knc', 'knl', knl), ('ksc', 'ksl', ksl),
                                      ('ksol', 'ksoll', element.ksoll)]:
        coefficients = np.asarray(getattr(element, name)).reshape(-1, element.deg + 1)
        expected = np.zeros(len(total))
        for order, row in enumerate(coefficients):
            integral = Polynomial(row).integ()
            expected[order] = (integral(element.s_start + element.length)
                               - integral(element.s_start))
        if name != 'ksol':
            hard_edge = getattr(element, integral_name)
            expected[:len(hard_edge)] += np.asarray(hard_edge)
        expected *= element.kscale
        xo.assert_allclose(total, expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize('h', [0., 0.3])
@pytest.mark.parametrize('use_strings', [False, True])
def test_bfieldexpansion_env_new_and_set(h, use_strings):
    env = xt.Environment()
    env.set('a', 0.1)
    env.set('ll', 0.3)
    env.set('curvature', h)
    env.set('steps', 10)
    name = 'expansion'
    env.new(name, 'BFieldExpansion' if use_strings else xt.BFieldExpansion,
            length='ll', h='curvature', num_integration_steps='steps', num_phi=5, s_start=0.02,
            ksc=[[env.ref['a'], 0.2, 0.], [0.03, 0., 0.]],
            knc=np.array([['2*a', 0.1, 0.], [0.02, 0., 0.]], dtype=object),
            ksol=['3*a', 0.02, 0.])
    assert isinstance(env[name], xt.BFieldExpansion)
    assert env[name].straight == int(h == 0)
    assert env[name].knc[0, 0] == pytest.approx(0.2)
    assert env.ref[name].knc[0, 0].xdeps.expr == "(2.0 * vars['a'])"
    assert env.ref[name].ksc[0, 0].xdeps.expr == "vars['a']"

    element = env.get(name)
    # The normal environment array handling preserves matrix indices.
    env.set(name, length='2*ll', num_integration_steps='2*steps', s_start='-a',
            knc=[[0.04, 0.1, 0.], ['4*a', 0., 0.]],
            ksc=[[0.01, '2*a', 0.], [0.03, 0., 0.]],
            ksol=np.array([env.ref['a'], 0.03, 0.], dtype=object))
    assert env.ref[name].knc[1, 0].xdeps.expr == "(4.0 * vars['a'])"
    assert env.ref[name].knc[0, 0].xdeps.expr is None
    env.set('a', 0.2)
    env.set('ll', 0.4)
    env.set('steps', 6)
    if h:
        env.set('curvature', 0.4)
    assert element.knc[0, 0] == 0.04
    assert element.knc[1, 0] == pytest.approx(0.8)
    assert element.ksc[0, 1] == pytest.approx(0.4)
    assert element.ksol[0] == pytest.approx(0.2)
    assert element.length == 0.8
    assert element.num_integration_steps == 12
    assert element.ds == pytest.approx(0.8 / 12)
    assert element.angle == pytest.approx(element.length * element.h)
    _check_integrated_bfieldexpansion_strengths(element)

    fresh = xt.BFieldExpansion(
        length=element.length, h=element.h, s_start=element.s_start,
        num_integration_steps=element.num_integration_steps, num_phi=element.num_phi,
        knc=np.asarray(element.knc).reshape(element.nb, -1),
        ksc=np.asarray(element.ksc).reshape(element.na, -1),
        ksol=np.asarray(element.ksol))
    xo.assert_allclose(element._c, fresh._c, rtol=0, atol=1e-14)
    field = element.get_field(x=0.02, y=0.01, s_local=0.1)
    expected = fresh.get_field(x=0.02, y=0.01, s_local=0.1)
    for field_name in field.dtype.names:
        xo.assert_allclose(field[field_name], expected[field_name], rtol=0, atol=1e-14)
    particles = xt.Particles(p0c=1e9, x=0.01, y=0.007)
    reference = particles.copy()
    element.track(particles)
    fresh.track(reference)
    for coord in ('x', 'px', 'y', 'py', 'zeta', 'delta', 's'):
        xo.assert_allclose(getattr(particles, coord), getattr(reference, coord),
                           rtol=0, atol=1e-14)

    # Numeric replacements remove expressions, including cells in later rows.
    env.set(name, knc=[[0.05, 0., 0.], [0.06, 0., 0.]],
            ksc=np.zeros((2, 3)), ksol=[0.1, 0., 0.])
    env.set('a', 0.7)
    xo.assert_allclose(element.knc, [[0.05, 0., 0.], [0.06, 0., 0.]], rtol=0, atol=0)
    xo.assert_allclose(element.ksc, 0., rtol=0, atol=0)
    xo.assert_allclose(element.ksol, [0.1, 0., 0.], rtol=0, atol=0)
    _check_integrated_bfieldexpansion_strengths(element)


@pytest.mark.parametrize('h', [0., 0.3])
def test_bfieldexpansion_env_clone_and_array_references(h):
    env = xt.Environment()
    env['a'] = 0.1
    env.new('source', 'BFieldExpansion', length=0.3, h=h, num_phi=5,
            ksc=[[0., 0.]], knc=[['a', 0.]], ksol=[0.2, 0.])
    env.new('clone', 'source')
    env.new('overridden', 'source', knc=[[0.4, 0.]])
    env.new('linked', 'BFieldExpansion', length=0.3, h=h, num_phi=5,
            ksc=env.ref['source'].ksc, knc=env.ref['source'].knc,
            ksol=env.ref['source'].ksol)
    env.new('set_linked', 'source')
    env.set('set_linked', knc=env.ref['source'].knc)
    env['a'] = 0.3
    for name in ('source', 'clone', 'linked', 'set_linked'):
        assert env[name].knc[0, 0] == 0.3
        _check_integrated_bfieldexpansion_strengths(env.get(name))
    assert env['overridden'].knc[0, 0] == 0.4
    env.set('source', knc=[[0.5, 0.]], ksol=[0.4, 0.])
    for name in ('linked', 'set_linked'):
        assert env[name].knc[0, 0] == 0.5
        _check_integrated_bfieldexpansion_strengths(env.get(name))
    assert env['clone'].knc[0, 0] == 0.3
    assert env['linked'].ksol[0] == 0.4


@pytest.mark.parametrize('h', [0., 0.3])
def test_bfieldexpansion_env_rejects_invalid_updates(h):
    env = xt.Environment()
    env.new('e', 'BFieldExpansion', length=0.3, h=h, num_phi=5,
            ksc=[[0.04, 0.]], knc=[[0.1, 0.], [0.2, 0.]], ksol=[0.3, 0.])
    element = env.get('e')
    original = element._c.copy()
    for name, value in [('knc', [0.] * 5), ('knc', np.zeros((1, 4))),
                        ('ksc', np.zeros((2, 1))), ('ksol', [[0., 0.]])]:
        with pytest.raises(ValueError):
            env.set('e', **{name: value})
        assert element.length == 0.3
        xo.assert_allclose(element._c, original, rtol=0, atol=0)
    for name, value in [('angle', 0.2), ('ksoll', [1.])]:
        with pytest.raises((ValueError, AttributeError)):
            env.set('e', **{name: value})
        assert env.get('e') is element
        xo.assert_allclose(element._c, original, rtol=0, atol=0)
    for num_integration_steps in (0, -1, 1.5):
        with pytest.raises(ValueError, match='positive integer'):
            env.set('e', num_integration_steps=num_integration_steps)
        assert element.num_integration_steps == 10
    if h:
        with pytest.raises(ValueError, match='h > 1e-4'):
            env.set('e', h=0.)
        assert element.h == h


@pytest.mark.parametrize('h', [0., 0.3])
def test_bfieldexpansion_coefficient_updates(h):
    element = xt.BFieldExpansion(
        length=0.3, h=h, s_start=0.2, num_phi=5,
        ksc=np.array([[0.04, 0.2, 0.], [0.03, 0., 0.]]),
        knc=np.array([[0.05, 0.1, 0.], [0.02, 0.03, 0.01], [0.01, 0., 0.]]),
        ksol=np.array([0.1, 0.02, 0.]),
    )
    _check_integrated_bfieldexpansion_strengths(element)
    for name in ['knc', 'ksc', 'ksol']:
        view = getattr(element, name)
        original = view.copy()
        for value in [original, np.zeros(len(original) + 1)]:
            with pytest.raises(AttributeError, match='cannot be reassigned'):
                setattr(element, name, value)
        with pytest.raises(AttributeError):
            view.shape = (1, len(view))
        with pytest.raises(ValueError):
            view[:] = np.zeros(original.shape + (2,))
        xo.assert_allclose(view, original, rtol=0, atol=0)

        # A saved slice and a reshaped view must also refresh the caches.
        sliced = view[:2]
        sliced[0] += 0.01
        original[0] += 0.01
        view.reshape(-1, element.deg + 1)[0, 1] *= 1.2
        original.reshape(-1)[1] *= 1.2
        view[:] *= 1.1
        original *= 1.1
        np.add.at(view, [0, 0], 0.001)
        original[0] += 0.002
        xo.assert_allclose(view, original, rtol=0, atol=1e-14)
        _check_integrated_bfieldexpansion_strengths(element)

        detached = np.asarray(view)
        detached[:] = 99.
        xo.assert_allclose(view, original, rtol=0, atol=1e-14)

    # Whole-array augmented assignment changes values without rebinding storage.
    element.knc *= 1.2
    element.ksc += 0.001
    element.ksol *= 0.9
    _check_integrated_bfieldexpansion_strengths(element)

    fresh = xt.BFieldExpansion(
        length=element.length, h=h, s_start=element.s_start, num_phi=element.num_phi,
        knc=np.asarray(element.knc).reshape(element.nb, -1),
        ksc=np.asarray(element.ksc).reshape(element.na, -1),
        ksol=np.asarray(element.ksol),
    )
    xo.assert_allclose(element._c, fresh._c, rtol=0, atol=1e-14)
    field = element.get_field(x=0.02, y=0.01, s_local=0.4)
    expected_field = fresh.get_field(x=0.02, y=0.01, s_local=0.4)
    for name in field.dtype.names:
        xo.assert_allclose(field[name], expected_field[name], rtol=0, atol=1e-14)
    particles = xt.Particles(x=0.01, y=0.007, beta0=0.7)
    expected_particles = particles.copy()
    element.track(particles)
    fresh.track(expected_particles)
    for name in ['x', 'px', 'y', 'py', 'zeta', 'delta', 's']:
        xo.assert_allclose(getattr(particles, name),
                           getattr(expected_particles, name), rtol=0, atol=1e-14)

    for length, s_start in [(0.6, 0.2), (0.6, -0.1), (-0.2, 0.4), (0., 0.4)]:
        element.length = length
        element.s_start = s_start
        _check_integrated_bfieldexpansion_strengths(element)

    # Clear all seeds to check that old expansion terms are not accumulated.
    for name in ['knc', 'ksc', 'ksol']:
        getattr(element, name).fill(0.)
    xo.assert_allclose(element._c, 0., rtol=0, atol=0)
    _check_integrated_bfieldexpansion_strengths(element)


@pytest.mark.parametrize('h', [0., 0.3])
def test_bfieldexpansion_total_integrals_and_twiss_strengths(h):
    knc = np.arange(1, 19).reshape(6, 3) * 1e-4
    ksc = np.arange(1, 10).reshape(3, 3) * 1e-4
    element = xt.BFieldExpansion(
        length=0.2, h=h, s_start=0.1, num_phi=5, knc=knc, ksc=ksc, kscale=0.8,
        ksol=np.array([0.02, 0.01, 0.]),
        knl=[0.001, 0.002], ksl=[0.003, 0., 0., 0.0001],
    )
    for name in ['ksoll']:
        values = getattr(element, name)
        original = values.copy()
        with pytest.raises(AttributeError):
            setattr(element, name, original)
        with pytest.raises(ValueError, match='read-only'):
            values[0] = 9.
        with pytest.raises(ValueError, match='read-only'):
            values[:][0] = 9.
        with pytest.raises(ValueError, match='read-only'):
            values *= 2
        with pytest.raises(ValueError, match='read-only'):
            np.add.at(values, 0, 2.)
        xo.assert_allclose(getattr(element, name), original, rtol=0, atol=0)

    line = xt.Line(elements={
        'expansion': element,
        'replica': xt.Replica(parent_name='expansion'),
        'solenoid': xt.UniformSolenoid(length=0.1, ks=0.02),
        'variable_solenoid': xt.VariableSolenoid(length=0.15, ks_profile=[0.02, 0.04]),
    })
    line.particle_ref = xt.Particles(p0c=1e9)
    line.build_tracker()

    for changed in [False, True]:
        if changed:
            # Exercise coefficient callbacks through the expression system,
            # after the native strength readers have cached their offsets.
            line.vars['normal'] = element.knc[0, 0]
            line.ref['expansion'].knc[0, 0] = line.vars['normal']
            line.vars['normal'] = 0.03
            line['expansion'].ksc[0, 1] = 0.04
            line['expansion'].ksol[0] = 0.05
            line.vars['hard_edge'] = 0.002
            line.ref['expansion'].knl[1] = line.vars['hard_edge']
            line.vars['hard_edge'] = 0.004
            line['expansion'].ksl[3] = 0.0002
            line.vars['scale'] = 0.8
            line.ref['expansion'].kscale = line.vars['scale']
            line.vars['scale'] = -0.5
            element.length = 0.3
            element.s_start = 0.2
        _check_integrated_bfieldexpansion_strengths(element)
        normal, skew = element.get_total_knl_ksl()
        for table in [line.get_table(attr=True), line.get_strengths(),
                      line.twiss(betx=1., bety=1., strengths=True)]:
            for order in range(6):
                for name in ('expansion', 'replica'):
                    assert table[f'k{order}l', name] == pytest.approx(normal[order])
                    assert table[f'k{order}sl', name] == pytest.approx(skew[order])
            assert table['ksoll', 'expansion'] == pytest.approx(element.ksoll[0])
            assert table['ks', 'expansion'] == 0.  # Skew coefficients are not a scalar solenoid strength.
            assert table['ksoll', '_end_point'] == 0.

    copied = element.copy()
    copied.knc[0, 0] = 0.07
    assert element.knc[0, 0] == 0.03
    _check_integrated_bfieldexpansion_strengths(copied)
    serialized = element.to_dict()
    assert 'ksoll' not in serialized  # Reconstructed from ksol on loading.
    for name in ['knc', 'ksc', 'ksol', 'knl', 'ksl']:
        xo.assert_allclose(serialized[name], getattr(element, name), rtol=0, atol=0)


@pytest.mark.parametrize('h', [0., 0.3])
@pytest.mark.parametrize('pkin_const', [False, True])
def test_bfieldexpansion_hard_edge_inputs(h, pkin_const):
    element = xt.BFieldExpansion(
        length=0.4, h=h, s_start=-0.1, num_integration_steps=20, pkin_const=pkin_const,
        knc=[[0.1, 0.02, 0.003]], ksc=[[0.01, -0.005, 0.]],
        ksol=[0.02, 0.001, 0.],
        knl=[0.02, 0.01, 0.004], ksl=[0.001, 0.002])
    original_knc = element.knc.copy()
    original_ksc = element.ksc.copy()
    for changed in (False, True):
        if changed:
            element.knl[1:] *= 1.2
            np.add.at(element.ksl, 0, 0.0001)
            element.knl *= 0.9
            element.knl = [0.01, 0.003, 0.004]
            element.ksl = [0.002, 0.001]
            element.length = 0.7
            element.s_start = 0.15
            if h:
                element.h = 0.4

        combined = {}
        for profile, hard_edge in [('knc', 'knl'), ('ksc', 'ksl')]:
            coefficients = np.asarray(getattr(element, profile))
            integrated = np.asarray(getattr(element, hard_edge))
            values = np.zeros((max(len(coefficients), len(integrated)), element.deg + 1))
            values[:len(coefficients)] = coefficients
            values[:len(integrated), 0] += integrated / element.length
            combined[profile] = values
        reference = xt.BFieldExpansion(
            length=element.length, h=element.h, s_start=element.s_start,
            ksol=element.ksol, num_integration_steps=element.num_integration_steps, pkin_const=pkin_const,
            **combined)
        assert element.num_phi == reference.num_phi
        field = element.get_field(x=[-0.01, 0.02], y=[0.015, -0.007], s_local=[0., 0.3])
        expected_field = reference.get_field(x=[-0.01, 0.02], y=[0.015, -0.007], s_local=[0., 0.3])
        for name in field.dtype.names:
            xo.assert_allclose(field[name], expected_field[name], rtol=0, atol=1e-13)
        particles = xt.Particles(p0c=1e9, x=0.01, px=0.002, y=0.007, delta=0.01)
        expected = particles.copy()
        element.track(particles)
        reference.track(expected)
        for coord in ('x', 'px', 'y', 'py', 'zeta', 'delta', 's', 'ax', 'ay'):
            xo.assert_allclose(getattr(particles, coord), getattr(expected, coord),
                               rtol=0, atol=1e-13)
        _check_integrated_bfieldexpansion_strengths(element)
        for total in element.get_total_knl_ksl():
            total[:] = 99.
        _check_integrated_bfieldexpansion_strengths(element)
        xo.assert_allclose(element.knc, original_knc, rtol=0, atol=0)
        xo.assert_allclose(element.ksc, original_ksc, rtol=0, atol=0)
        restored = xt.BFieldExpansion.from_dict(element.to_dict())
        for name in ('knl', 'ksl', 'knc', 'ksc', '_c'):
            xo.assert_allclose(getattr(restored, name), getattr(element, name),
                               rtol=0, atol=0)


@pytest.mark.parametrize('h', [0., 0.3])
def test_bfieldexpansion_hard_edge_environment(h):
    env = xt.Environment()
    env['normal'] = 0.01
    env['skew'] = 0.002
    env.new('e', 'BFieldExpansion', length=0.4, h=h, knc=[[0.1]],
            ksc=[[0.]], ksol=[0.], num_phi='auto',
            knl=[0., 'normal'], ksl=['skew', 0., 0.])
    env.new('clone', 'e')
    env.new('linked', 'e', knl=env.ref['e'].knl, ksl=env.ref['e'].ksl)
    env['normal'] = 0.02
    env['skew'] = 0.003
    for name in ('e', 'clone', 'linked'):
        assert env[name].knl[1] == 0.02
        assert env[name].ksl[0] == 0.003
        _check_integrated_bfieldexpansion_strengths(env.get(name))
    env.set('e', knl=[0.001, '2*normal'], ksl=['3*skew', 0., 0.001], length=0.5)
    env['normal'] = 0.03
    env['skew'] = 0.004
    reference = xt.BFieldExpansion(
        length=0.5, h=h, knc=[[0.102], [0.12]],
        ksc=[[0.024], [0.], [0.002]], ksol=[0.])
    for name in ('e', 'linked'):
        xo.assert_allclose(env.get(name).knl, [0.001, 0.06], rtol=0, atol=0)
        xo.assert_allclose(env.get(name).ksl, [0.012, 0., 0.001], rtol=0, atol=0)
    xo.assert_allclose(env.get('e')._c, reference._c, rtol=0, atol=1e-13)


def test_bfieldexpansion_hard_edge_validation():
    kwargs = dict(knc=[[0.]], ksc=[[0.]], ksol=[0.])
    for name in ('knl', 'ksl'):
        with pytest.raises(ValueError, match='one-dimensional'):
            xt.BFieldExpansion(length=0.3, **kwargs, **{name: [[0.]]})
        with pytest.raises(ValueError, match='nonzero length'):
            xt.BFieldExpansion(length=0., **kwargs, **{name: [0.1]})
        element = xt.BFieldExpansion(length=0., **kwargs)
        with pytest.raises(ValueError, match='nonzero length'):
            getattr(element, name)[0] = 0.1
        assert getattr(element, name)[0] == 0.
        element.length = 0.3
        getattr(element, name)[0] = 0.1
        with pytest.raises(ValueError, match='nonzero length'):
            element.length = 0.
        assert element.length == 0.3
        with pytest.raises(ValueError, match='allocated shape'):
            setattr(element, name, [0.1, 0.2])
        assert len(getattr(element, name)) == 1


def test_get_field_straight():
    ksc = np.array([[0.04, 0.2, 0.08], [0, 0, 0.1]])
    knc = np.array([[0.05, 0.04, 0.07], [0.01, 0, 0]])
    ksol = np.array([0.1, 0.02, 0])
    element = xt.BFieldExpansion(
        length=1, ksc=ksc, knc=knc, ksol=ksol, num_phi=5)

    x = np.array([[0.01], [0.02]])
    y = np.array([0.005, -0.004, 0.003])
    s = np.array([0.001, 0.4, 0.9])
    x_broadcast, y_broadcast, s_broadcast = np.broadcast_arrays(x, y, s)

    bx_expected = (
        0.1 * s_broadcast**2 * x_broadcast
        + 0.08 * s_broadcast**2
        + 0.2 * s_broadcast
        - y_broadcast**2 * (0.4 * x_broadcast + 0.32) / 4
        + 0.01 * y_broadcast
        + 0.04
    )
    by_expected = (
        0.07 * s_broadcast**2
        + 0.04 * s_broadcast
        + 0.01 * x_broadcast
        + y_broadcast**3 / 15
        - 0.07 * y_broadcast**2
        - y_broadcast * (
            0.2 * s_broadcast**2
            + 0.2 * x_broadcast**2
            + 0.32 * x_broadcast
            + 0.04
        ) / 2
        + 0.05
    )
    bs_expected = (
        0.1 * s_broadcast * x_broadcast**2
        - 0.1 * s_broadcast * y_broadcast**2
        + 0.02 * s_broadcast
        + x_broadcast * (0.16 * s_broadcast + 0.2)
        + y_broadcast * (0.14 * s_broadcast + 0.04)
        + 0.1
    )

    field = element.get_field(x=x, y=y, s_local=s)

    assert isinstance(field, np.ndarray)
    assert field.shape == x_broadcast.shape
    assert field.dtype.names == (
        'phi',
        'Bx', 'By', 'Bs',
        'Ax', 'Ay', 'As',
        'dAx_dx', 'dAx_dy', 'dAx_ds',
        'dAs_dx', 'dAs_dy', 'dAs_ds',
    )
    xo.assert_allclose(field['Bx'], bx_expected, rtol=0, atol=1e-14)
    xo.assert_allclose(field['By'], by_expected, rtol=0, atol=1e-14)
    xo.assert_allclose(field['Bs'], bs_expected, rtol=0, atol=1e-14)
    xo.assert_allclose(field['Ay'], 0, rtol=0, atol=1e-14)
    xo.assert_allclose(field['dAx_dy'], -field['Bs'], rtol=0, atol=1e-14)
    xo.assert_allclose(field['dAs_dy'], field['Bx'], rtol=0, atol=1e-14)
    xo.assert_allclose(
        field['dAx_ds'] - field['dAs_dx'],
        field['By'],
        rtol=0,
        atol=1e-14,
    )

    scalar_field = element.get_field(x=x[0, 0], y=y[0], s_local=s[0])
    assert isinstance(scalar_field, np.ndarray)
    assert scalar_field.shape == ()
    assert scalar_field.dtype == field.dtype
    xo.assert_allclose(
        scalar_field['Bx'], bx_expected[0, 0], rtol=0, atol=1e-14)
    xo.assert_allclose(
        scalar_field['By'], by_expected[0, 0], rtol=0, atol=1e-14)
    xo.assert_allclose(
        scalar_field['Bs'], bs_expected[0, 0], rtol=0, atol=1e-14)


def test_get_field_bent():
    element = xt.BFieldExpansion(
        length=1,
        h=0.1,
        ksc=np.array([[0.0]]),
        knc=np.array([[0.5]]),
        ksol=np.array([0.0]),
        num_phi=5,
    )
    x = np.array([-0.2, 0.0, 0.3])
    y = np.array([0.01, -0.03, 0.02])
    s = np.array([0.0, 0.5, 1.0])

    field = element.get_field(x=x, y=y, s_local=s)

    xo.assert_allclose(field['Bx'], np.zeros(3), rtol=0, atol=1e-14)
    xo.assert_allclose(field['By'], np.full(3, 0.5), rtol=0, atol=1e-14)
    xo.assert_allclose(field['Bs'], np.zeros(3), rtol=0, atol=1e-14)


def test_h_sdep():
    h = 0.1
    ksc = np.array([[1.0, 0.1], [0.2, 0.0], [0.3, 0.1]])
    knc = np.array([[0.1, 0.1], [0.5, 0.0]])
    ksol = np.array([0.1, 0.0])
    num_phi = 5
    length=0.2
    fexp = xt.BFieldExpansion(length=length, h=h, ksc=ksc, knc=knc, ksol=ksol, num_phi=num_phi, num_integration_steps=100, pkin_const=True)

    p0 = xt.Particles(x=0.01, y=0.007, tau=0.002, beta0=0.7)
    line = xt.Line(elements=[fexp])
    line.track(p0, _force_no_end_turn_actions=True)

    assert np.isclose(p0.x[0], 0.00995953)
    assert np.isclose(p0.px[0], -0.00021054)
    assert np.isclose(p0.y[0], 0.02753286)
    assert np.isclose(p0.py[0], 0.2039826)
    assert np.isclose(p0.zeta[0], -0.00020395)
    assert np.isclose(p0.ptau[0], 0)
    assert np.isclose(p0.s[0], length)

def test_sdep():
    ksc = np.array([[1.0, 0.1], [0.2, 0.0], [0.3, 0.1]])
    knc = np.array([[0.1, 0.1], [0.5, 0.0]])
    ksol = np.array([0.1, 0.0])
    num_phi = 5
    length=0.2
    fexp = xt.BFieldExpansion(length=length, ksc=ksc, knc=knc, ksol=ksol, num_phi=num_phi, num_integration_steps=100, pkin_const=True)

    p0 = xt.Particles(x=0.01, y=0.007, tau=0.002, beta0=0.7)
    line = xt.Line(elements=[fexp])
    line.track(p0, _force_no_end_turn_actions=True)

    assert np.isclose(p0.x[0], 0.00792934)
    assert np.isclose(p0.px[0], -0.02026721)
    assert np.isclose(p0.y[0], 0.02750655)
    assert np.isclose(p0.py[0], 0.20395936)
    assert np.isclose(p0.zeta[0], -1.64370788e-05)
    assert np.isclose(p0.ptau[0], 0)

def test_twiss():
    fodo = xt.Line(elements=[
        xt.Drift(length=1.2),
        xt.Quadrupole(k1=7, length=0.1),
        xt.Drift(length=0.5),
        xt.Bend(length=0.2, k0=0.1, angle=0.1*0.2),
        xt.Drift(length=0.5),
        xt.Quadrupole(k1=-7, length=0.1)]
    )
    fodo.particle_ref = xt.Particles(q0=1, mass0=1)
    tw = fodo.twiss4d()

    myfodo = xt.Line(elements=[
        xt.Drift(length=1.2),
        xt.BFieldExpansion(length=0.1, ksc=np.array([[0]]), knc=np.array([[0],[7]]), ksol=np.array([0]), num_phi=5),
        xt.Drift(length=0.5),
        xt.BFieldExpansion(length=0.2, h=0.1, ksc=np.array([[0]]), knc=np.array([[0.1]]), ksol=np.array([0]), num_phi=5),
        xt.Drift(length=0.5),
        xt.BFieldExpansion(length=0.1, ksc=np.array([[0]]), knc=np.array([[0],[-7]]), ksol=np.array([0]), num_phi=5)
    ])
    myfodo.particle_ref = xt.Particles(q0=1, mass0=1)
    mytw = myfodo.twiss4d()

    assert np.allclose(tw.betx, mytw.betx)
    assert np.allclose(tw.bety, mytw.bety)
    assert np.allclose(tw.alfx, mytw.alfx)
    assert np.allclose(tw.alfy, mytw.alfy)
    assert np.allclose(tw.dx, mytw.dx)
    assert np.allclose(tw.dy, mytw.dy)

def test_backtrack():
    h = 0.1
    ksc = np.array([[1.0, 0.1], [0.2, 0.0], [0.3, 0.1]])
    knc = np.array([[0.1, 0.1], [0.5, 0.0]])
    ksol = np.array([0.1, 0.0])
    num_phi = 5
    length=0.2
    fexp = xt.BFieldExpansion(length=length, h=h, ksc=ksc, knc=knc, ksol=ksol, num_phi=num_phi, num_integration_steps=100)

    p0 = xt.Particles(x=0.01, y=0.007, tau=0.002, beta0=0.7)
    line = xt.Line(elements=[fexp])

    p_test = p0.copy()
    line.track(p_test)
    line.track(p_test, backtrack=True)

    assert np.all(p_test.state == 1)
    for coordinate in ['x', 'px', 'y', 'py', 'zeta', 'delta', 's']:
        xo.assert_allclose(
            getattr(p_test, coordinate), getattr(p0, coordinate),
            rtol=0, atol=1e-12)

def test_against_boris():
    p0 = xt.Particles(x=0.01, y=0.005, tau=0.001, px=0.003, py=0.004, ptau=0.002, beta0=0.7)
    p1 = p0.copy()
    length = 1

    ksc = np.array([[0.04, 0.2,  0.08], [0,    0, 0.1]])
    knc = np.array([[0.05, 0.04, 0.07], [0.01, 0, 0]])
    ksol = np.array([0.1,  0.02, 0])

    def fieldvalue(x,y,z):
        # Determined with bpmeth
        return ((0.1*z**2*x + 0.08*z**2 + 0.2*z - y**2*(0.4*x + 0.32)/4 + 0.01*y + 0.04) * p0.rigidity0[0],
                (0.07*z**2 + 0.04*z + 0.01*x + 0.0666666666666667*y**3 - 0.07*y**2 - y*(0.2*z**2 + 0.2*x**2 + 0.32*x + 0.04)/2 + 0.05) * p0.rigidity0[0],
                (0.1*z*x**2 - 0.1*z*y**2 + 0.02*z + x*(0.16*z + 0.2) + y*(0.14*z + 0.04) + 0.1) * p0.rigidity0[0])

    boris = xt.BorisSpatialIntegrator(fieldmap_callable=fieldvalue, s_start=0, s_end=length, n_steps=500)
    fexp = xt.BFieldExpansion(length=length, ksc=ksc, knc=knc, ksol=ksol, num_phi=5, num_integration_steps=50, pkin_const=True)

    boris.track(p0)
    fexp.track(p1)

    assert np.isclose(p0.x, p1.x)
    assert np.isclose(p0.px, p1.px)
    assert np.isclose(p0.y, p1.y)
    assert np.isclose(p0.py, p1.py)
    assert np.isclose(p0.zeta, p1.zeta)
    assert np.isclose(p0.ptau, p1.ptau)
    assert np.isclose(p0.s, p1.s)

def test_straighttocurved():
    def curved_to_straight(p, h):
        return {
            "x" : (1/h + p.x) * np.cos((p.s)*h) - 1/h,
            "s" : (1/h + p.x) * np.sin((p.s)*h),
            "y" : p.y,
        }

    def straight_to_curved(p, h):
        return {
            "x" : np.sqrt((1/h + p.x)**2 + (p.s)**2) - 1/h,
            "s" : 1/h * np.arctan((p.s)/(1/h + p.x)),
            "y" : p.y,
        }

    knc_st = 1
    ksc_st = 0.1
    ksol_st = 0.5
    h = 0.4
    length = 0.5

    knc_cu = np.array([[knc_st, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    ksc_cu = np.array([[ksc_st, 0, - ksc_st/2*h**2, 0, ksc_st/24*h**4, 0, - ksc_st/720*h**6, 0, ksc_st/40320*h**8, 0]])
    ksol_cu = np.array([0, -ksc_st*h, 0, ksc_st/6*h**3, 0, -ksc_st/120*h**5, 0, ksc_st/5040*h**7, 0, -ksc_st/362880*h**9])

    p0 = xt.Particles(x=0.01, y=0.005, tau=0.001, px=0.003, py=0.004, ptau=0.002, beta0=0.7)
    p1 = p0.copy()

    line_straight = xt.Line(elements=[xt.BFieldExpansion(length=length, h=0, ksc=np.array([[ksc_st]]), knc=np.array([[knc_st]]), ksol=np.array([ksol_st]), num_phi=5, num_integration_steps=100)])
    line_straight.track(p0, _force_no_end_turn_actions=True)
    line_curved = xt.Line(elements=[xt.BFieldExpansion(length=straight_to_curved(p0, h)["s"], h=h, ksc=ksc_cu, knc=knc_cu, ksol=ksol_cu, num_phi=5, num_integration_steps=100)])
    line_curved.track(p1, _force_no_end_turn_actions=True)

    assert np.isclose(curved_to_straight(p1, h)["x"], p0.x)
    assert np.isclose(curved_to_straight(p1, h)["s"], p0.s)
    assert np.isclose(curved_to_straight(p1, h)["y"], p0.y)
