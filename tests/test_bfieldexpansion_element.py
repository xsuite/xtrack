import xtrack as xt
import numpy as np
import xobjects as xo
import pytest
from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts(excluding='ContextCpu')
@pytest.mark.parametrize('h', [0., 0.3])
@pytest.mark.parametrize('pkin_const', [False, True])
def test_bfieldexpansion_gpu_matches_cpu(test_context, h, pkin_const):
    kwargs = dict(length=0.3, h=h, ny=5, nstep=12, sstart=0.1,
                  pkin_const=pkin_const,
                  ksc=[[0.04, 0.2, 0.], [0.03, 0., 0.]],
                  knc=[[0.05, 0.1, 0.], [0.02, 0., 0.]],
                  ksol=[0.1, 0.02, 0.])
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
        length=0.3, h=h, ksc=ksc, knc=knc, ksol=ksol, ny=5)

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
    field = element.get_field(x=0., y=0., s=s)
    xo.assert_allclose(field['Bx'], 0.04 + 0.2*s, rtol=0, atol=1e-14)
    xo.assert_allclose(field['By'], 0.05 + 0.1*s, rtol=0, atol=1e-14)
    xo.assert_allclose(field['Bs'], 0.1 + 0.02*s, rtol=0, atol=1e-14)


@pytest.mark.parametrize('h', [0., 0.3])
def test_bfieldexpansion_serialization(h):
    element = xt.BFieldExpansion(
        length=0.3, h=h, ksc=[[0.04, 0.2, 0.]],
        knc=[[0.05, 0.1, 0.], [0.02, 0., 0.]], ksol=[0.1, 0.02, 0.],
        ny=5, sstart=0.1, nstep=12)
    restored = xt.BFieldExpansion.from_dict(element.to_dict())
    assert type(restored) is xt.BFieldExpansion
    assert restored.straight == element.straight
    assert restored.h == element.h
    assert restored.angle == element.angle
    for name in ('knc', 'ksc', 'ksol', 'knl', 'ksl', 'ksoll', '_c'):
        xo.assert_allclose(getattr(restored, name), getattr(element, name),
                           rtol=0, atol=0)
    restored.knc[0, 1] += 0.01
    assert restored.knc[0, 1] != element.knc[0, 1]
    _check_integrated_bfieldexpansion_strengths(restored)


def test_bfieldexpansion_mixed_geometry_line():
    env = xt.Environment()
    for name, h in [('straight', 0.), ('curved', 0.3)]:
        env.new(name, 'BFieldExpansion', length=0.3, h=h, ny=5,
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
        ksol=np.array([0.1, 0.02, 0.]), ny=5, nstep=12,
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
        assert element.ds == pytest.approx(length / element.nstep)

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
    field = element.get_field(x=0.02, y=0.01, s=0.4)
    expected_field = fresh.get_field(x=0.02, y=0.01, s=0.4)
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
        ksol=np.array([0.]), ny=5,
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
    kwargs = dict(length=0.3, ksc=[[0.]], knc=[[0.1]], ksol=[0.], ny=5)
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
def test_bfieldexpansion_rejects_radiation_and_spin_tracking(h):
    element = xt.BFieldExpansion(
        length=0.2, h=h, ksc=np.array([[0.]]), knc=np.array([[0.1]]),
        ksol=np.array([0.]), ny=5,
    )
    line = xt.Line(elements={'expansion': element})
    line.build_tracker(compile=False)
    particles = xt.Particles(p0c=1e9, x=0.01, px=0.002, spin_z=1.)
    initial = particles.copy()

    for mode in ['mean', 'quantum', 'quantum-kick', 'spin', 'compile_flag']:
        line.configure_radiation(model=None)
        line.configure_spin(None)
        if mode == 'spin':
            line.configure_spin('auto')
        elif mode == 'compile_flag':
            # Twiss spin calculations also enable this flag directly.
            line.config.XTRACK_MULTIPOLE_NO_SYNRAD = False
        else:
            line.configure_radiation(model=mode)
        with pytest.raises(NotImplementedError, match='radiation or spin tracking'):
            line.track(particles)
        for name in ['x', 'px', 'y', 'py', 'zeta', 'delta', 's', 'state',
                     'spin_x', 'spin_y', 'spin_z', 'at_element', 'at_turn']:
            xo.assert_allclose(getattr(particles, name), getattr(initial, name),
                               rtol=0, atol=0)

    # Ordinary orbital tracking still works when both features are disabled.
    line.configure_radiation(model=None)
    line.configure_spin(None)
    line.track(particles, _force_no_end_turn_actions=True)
    xo.assert_allclose(particles.s, element.length, rtol=0, atol=1e-14)
    assert np.all(particles.state > 0)

    line.particle_ref = xt.Particles(p0c=1e9)
    with pytest.raises(NotImplementedError, match='radiation or spin tracking'):
        line.twiss4d(betx=1., bety=1., spin=True, spin_x=1.)


@pytest.mark.parametrize('h', [0., 0.3])
def test_bfieldexpansion_rejects_standalone_radiation_and_spin(h):
    element = xt.BFieldExpansion(
        length=0.2, h=h, ksc=np.array([[0.]]), knc=np.array([[0.1]]),
        ksol=np.array([0.]), ny=5,
    )
    for component in ['spin_x', 'spin_y', 'spin_z']:
        particles = xt.Particles(p0c=1e9, x=0.01, **{component: 1.})
        with pytest.raises(NotImplementedError, match='spin tracking'):
            element.track(particles)
        xo.assert_allclose(particles.x, 0.01, rtol=0, atol=0)
        xo.assert_allclose(particles.s, 0., rtol=0, atol=0)
        xo.assert_allclose(getattr(particles, component), 1., rtol=0, atol=0)

    particles = xt.Particles(p0c=1e9, x=0.01)
    element.radiation_flag = 1
    with pytest.raises(NotImplementedError, match='radiation tracking'):
        element.track(particles)
    xo.assert_allclose(particles.s, 0., rtol=0, atol=0)
    element.radiation_flag = 0
    element.track(particles)
    xo.assert_allclose(particles.s, element.length, rtol=0, atol=1e-14)
    assert np.all(particles.state > 0)


def _check_integrated_bfieldexpansion_strengths(element):
    from numpy.polynomial import Polynomial

    for name, integral_name in [('knc', 'knl'), ('ksc', 'ksl'), ('ksol', 'ksoll')]:
        coefficients = np.asarray(getattr(element, name)).reshape(-1, element.deg + 1)
        expected = []
        for row in coefficients:
            integral = Polynomial(row).integ()
            expected.append(integral(element.sstart + element.length)
                            - integral(element.sstart))
        xo.assert_allclose(getattr(element, integral_name), expected, rtol=0, atol=1e-14)


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
            length='ll', h='curvature', nstep='steps', ny=5, sstart=0.02,
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
    env.set(name, length='2*ll', nstep='2*steps', sstart='-a',
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
    assert element.nstep == 12
    assert element.ds == pytest.approx(0.8 / 12)
    assert element.angle == pytest.approx(element.length * element.h)
    _check_integrated_bfieldexpansion_strengths(element)

    fresh = xt.BFieldExpansion(
        length=element.length, h=element.h, sstart=element.sstart,
        nstep=element.nstep, ny=element.ny,
        knc=np.asarray(element.knc).reshape(element.nb, -1),
        ksc=np.asarray(element.ksc).reshape(element.na, -1),
        ksol=np.asarray(element.ksol))
    xo.assert_allclose(element._c, fresh._c, rtol=0, atol=1e-14)
    field = element.get_field(x=0.02, y=0.01, s=0.1)
    expected = fresh.get_field(x=0.02, y=0.01, s=0.1)
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
    env.new('source', 'BFieldExpansion', length=0.3, h=h, ny=5,
            ksc=[[0., 0.]], knc=[['a', 0.]], ksol=[0.2, 0.])
    env.new('clone', 'source')
    env.new('overridden', 'source', knc=[[0.4, 0.]])
    env.new('linked', 'BFieldExpansion', length=0.3, h=h, ny=5,
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
    env.new('e', 'BFieldExpansion', length=0.3, h=h, ny=5,
            ksc=[[0.04, 0.]], knc=[[0.1, 0.], [0.2, 0.]], ksol=[0.3, 0.])
    element = env.get('e')
    original = element._c.copy()
    for name, value in [('knc', [0.] * 5), ('knc', np.zeros((1, 4))),
                        ('ksc', np.zeros((2, 1))), ('ksol', [[0., 0.]])]:
        with pytest.raises(ValueError):
            env.set('e', **{name: value})
        assert element.length == 0.3
        xo.assert_allclose(element._c, original, rtol=0, atol=0)
    for name, value in [('angle', 0.2), ('knl', [1., 2., 3.]),
                        ('ksl', [1.]), ('ksoll', [1.])]:
        with pytest.raises((ValueError, AttributeError)):
            env.set('e', **{name: value})
        assert env.get('e') is element
        xo.assert_allclose(element._c, original, rtol=0, atol=0)
    for nstep in (0, -1, 1.5):
        with pytest.raises(ValueError, match='positive integer'):
            env.set('e', nstep=nstep)
        assert element.nstep == 10
    if h:
        with pytest.raises(ValueError, match='h > 1e-4'):
            env.set('e', h=0.)
        assert element.h == h


@pytest.mark.parametrize('h', [0., 0.3])
def test_bfieldexpansion_coefficient_updates(h):
    element = xt.BFieldExpansion(
        length=0.3, h=h, sstart=0.2, ny=5,
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
        length=element.length, h=h, sstart=element.sstart, ny=element.ny,
        knc=np.asarray(element.knc).reshape(element.nb, -1),
        ksc=np.asarray(element.ksc).reshape(element.na, -1),
        ksol=np.asarray(element.ksol),
    )
    xo.assert_allclose(element._c, fresh._c, rtol=0, atol=1e-14)
    field = element.get_field(x=0.02, y=0.01, s=0.4)
    expected_field = fresh.get_field(x=0.02, y=0.01, s=0.4)
    for name in field.dtype.names:
        xo.assert_allclose(field[name], expected_field[name], rtol=0, atol=1e-14)
    particles = xt.Particles(x=0.01, y=0.007, beta0=0.7)
    expected_particles = particles.copy()
    element.track(particles)
    fresh.track(expected_particles)
    for name in ['x', 'px', 'y', 'py', 'zeta', 'delta', 's']:
        xo.assert_allclose(getattr(particles, name),
                           getattr(expected_particles, name), rtol=0, atol=1e-14)

    for length, sstart in [(0.6, 0.2), (0.6, -0.1), (-0.2, 0.4), (0., 0.4)]:
        element.length = length
        element.sstart = sstart
        _check_integrated_bfieldexpansion_strengths(element)

    # Clear all seeds to check that old expansion terms are not accumulated.
    for name in ['knc', 'ksc', 'ksol']:
        getattr(element, name).fill(0.)
    xo.assert_allclose(element._c, 0., rtol=0, atol=0)
    _check_integrated_bfieldexpansion_strengths(element)


@pytest.mark.parametrize('h', [0., 0.3])
def test_bfieldexpansion_readonly_integrals_and_twiss_strengths(h):
    knc = np.arange(1, 19).reshape(6, 3) * 1e-4
    ksc = np.arange(1, 10).reshape(3, 3) * 1e-4
    element = xt.BFieldExpansion(
        length=0.2, h=h, sstart=0.1, ny=5, knc=knc, ksc=ksc,
        ksol=np.array([0.02, 0.01, 0.]),
    )
    for name in ['knl', 'ksl', 'ksoll']:
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
            element.length = 0.3
            element.sstart = 0.2
        _check_integrated_bfieldexpansion_strengths(element)
        for table in [line.get_table(attr=True), line.get_strengths(),
                      line.twiss(betx=1., bety=1., strengths=True)]:
            for order in range(6):
                assert table[f'k{order}l', 'expansion'] == pytest.approx(element.knl[order])
                skew = element.ksl[order] if order < element.na else 0.
                assert table[f'k{order}sl', 'expansion'] == pytest.approx(skew)
            assert table['ksoll', 'expansion'] == pytest.approx(element.ksoll[0])
            assert table['ks', 'expansion'] == 0.  # Skew coefficients are not a scalar solenoid strength.
            assert table['ksoll', '_end_point'] == 0.

    copied = element.copy()
    copied.knc[0, 0] = 0.07
    assert element.knc[0, 0] == 0.03
    _check_integrated_bfieldexpansion_strengths(copied)
    serialized = element.to_dict()
    for name in ['knc', 'ksc', 'ksol', 'knl', 'ksl', 'ksoll']:
        xo.assert_allclose(serialized[name], getattr(element, name), rtol=0, atol=0)


def test_get_field_straight():
    ksc = np.array([[0.04, 0.2, 0.08], [0, 0, 0.1]])
    knc = np.array([[0.05, 0.04, 0.07], [0.01, 0, 0]])
    ksol = np.array([0.1, 0.02, 0])
    element = xt.BFieldExpansion(
        length=1, ksc=ksc, knc=knc, ksol=ksol, ny=5)

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

    field = element.get_field(x=x, y=y, s=s)

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

    scalar_field = element.get_field(x=x[0, 0], y=y[0], s=s[0])
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
        ny=5,
    )
    x = np.array([-0.2, 0.0, 0.3])
    y = np.array([0.01, -0.03, 0.02])
    s = np.array([0.0, 0.5, 1.0])

    field = element.get_field(x=x, y=y, s=s)

    xo.assert_allclose(field['Bx'], np.zeros(3), rtol=0, atol=1e-14)
    xo.assert_allclose(field['By'], np.full(3, 0.5), rtol=0, atol=1e-14)
    xo.assert_allclose(field['Bs'], np.zeros(3), rtol=0, atol=1e-14)


def test_h_sdep():
    h = 0.1
    ksc = np.array([[1.0, 0.1], [0.2, 0.0], [0.3, 0.1]])
    knc = np.array([[0.1, 0.1], [0.5, 0.0]])
    ksol = np.array([0.1, 0.0])
    ny = 5
    length=0.2
    fexp = xt.BFieldExpansion(length=length, h=h, ksc=ksc, knc=knc, ksol=ksol, ny=ny, nstep=100, pkin_const=True)

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
    ny = 5
    length=0.2
    fexp = xt.BFieldExpansion(length=length, ksc=ksc, knc=knc, ksol=ksol, ny=ny, nstep=100, pkin_const=True)

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
        xt.BFieldExpansion(length=0.1, ksc=np.array([[0]]), knc=np.array([[0],[7]]), ksol=np.array([0]), ny=5),
        xt.Drift(length=0.5),
        xt.BFieldExpansion(length=0.2, h=0.1, ksc=np.array([[0]]), knc=np.array([[0.1]]), ksol=np.array([0]), ny=5),
        xt.Drift(length=0.5),
        xt.BFieldExpansion(length=0.1, ksc=np.array([[0]]), knc=np.array([[0],[-7]]), ksol=np.array([0]), ny=5)
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
    ny = 5
    length=0.2
    fexp = xt.BFieldExpansion(length=length, h=h, ksc=ksc, knc=knc, ksol=ksol, ny=ny, nstep=100)

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
    fexp = xt.BFieldExpansion(length=length, ksc=ksc, knc=knc, ksol=ksol, ny=5, nstep=50, pkin_const=True)

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

    line_straight = xt.Line(elements=[xt.BFieldExpansion(length=length, h=0, ksc=np.array([[ksc_st]]), knc=np.array([[knc_st]]), ksol=np.array([ksol_st]), ny=5, nstep=100)])
    line_straight.track(p0, _force_no_end_turn_actions=True)
    line_curved = xt.Line(elements=[xt.BFieldExpansion(length=straight_to_curved(p0, h)["s"], h=h, ksc=ksc_cu, knc=knc_cu, ksol=ksol_cu, ny=5, nstep=100)])
    line_curved.track(p1, _force_no_end_turn_actions=True)

    assert np.isclose(curved_to_straight(p1, h)["x"], p0.x)
    assert np.isclose(curved_to_straight(p1, h)["s"], p0.s)
    assert np.isclose(curved_to_straight(p1, h)["y"], p0.y)
