import numpy as np
import pytest
import xobjects as xo
import xtrack as xt


@pytest.mark.parametrize('h', [0., 0.3])
def test_bfieldexpansion_lifecycle(h, monkeypatch):
    element = xt.BFieldExpansion(
        length=0.4, h=h, knc=[[0.1, 0.02], [0.03, 0.]], k1=0.2,
        k2s=0.1, kscale=0.7, num_integration_steps=13, pkin_const=True)
    serialized = element.to_dict()
    assert not any(name.startswith('_') and name != '__class__' for name in serialized)
    assert not {'na', 'nb', 'deg', 'ds', 'angle', 'straight', 'ksoll', 'order', 'nstep'} & serialized.keys()
    assert serialized['integrator'] == 'rk4'
    assert serialized['num_integration_steps'] == 13
    # Accept old dictionaries, but never trust derived caches or dimensions.
    legacy = dict(serialized, ds=99., na=123, nb=456, deg=999,
                  angle=99., straight=1, ksoll=[1.], _c=[99.], _V=[99.], _nm=0)
    for restored in (element.copy(), xt.BFieldExpansion.from_dict(serialized),
                     xt.BFieldExpansion.from_dict(legacy)):
        xo.assert_allclose(restored._c, element._c, rtol=0, atol=0)
        assert restored.angle == h * element.length
        assert restored.k1 == element.k1
        assert restored.k2s == element.k2s
        assert restored.num_integration_steps == element.num_integration_steps
        restored.k1 += 0.1
        assert restored.k1 != element.k1

    def unexpected_rebuild(*args, **kwargs):
        raise AssertionError('Wrapping an xobject must not rebuild or allocate caches')

    monkeypatch.setattr(xt.BFieldExpansion, '_update_expansion', unexpected_rebuild)
    wrapped = xt.BFieldExpansion(_xobject=element._xobject)
    assert wrapped._xobject is element._xobject
    assert wrapped.num_phi == element.num_phi
    xo.assert_allclose(wrapped._c, element._c, rtol=0, atol=0)


def test_bfieldexpansion_readonly_metadata_and_integrator():
    element = xt.BFieldExpansion(length=0.4, knc=np.zeros((6, 2)),
                                ksc=np.zeros((2, 2)), num_integration_steps=12)
    assert element.order == 5
    assert (element.na, element.nb, element.deg) == (2, 6, 1)
    for name in ('order', 'na', 'nb', 'deg', 'ds'):
        with pytest.raises(AttributeError):
            setattr(element, name, 7)
    element.knc[5, 0] = 1.
    element.knc[:] = 0.
    assert element.order == 5
    assert element.integrator == 'rk4'
    assert element.get_available_integrators() == ['rk4']
    element.integrator = 'rk4'
    assert not hasattr(element, 'nstep')
    assert element.num_integration_steps == 12
    element.num_integration_steps = 20
    assert element.num_integration_steps == 20
    assert element.ds == pytest.approx(0.02)
    element.length = 0.8
    assert element.ds == pytest.approx(0.04)
    for steps in (0, -1, 1.5):
        with pytest.raises(ValueError, match='positive integer'):
            element.num_integration_steps = steps
    with pytest.raises(ValueError, match='only supports'):
        element.integrator = 'yoshida4'
    with pytest.raises(ValueError, match='only supports'):
        xt.BFieldExpansion(length=1., integrator='yoshida4')
    with pytest.raises(TypeError, match='use num_integration_steps'):
        xt.BFieldExpansion(length=1., nstep=2, num_integration_steps=3)
    with pytest.raises(TypeError, match='use num_integration_steps'):
        xt.BFieldExpansion.from_dict({'length': 1., 'nstep': 2})
    with pytest.raises(ValueError, match='order is read-only'):
        xt.BFieldExpansion(length=1., order=5)
    zero = xt.BFieldExpansion.from_dict(xt.BFieldExpansion(length=0.).to_dict())
    assert zero.length == 0.


@pytest.mark.parametrize('h', [0., 0.3])
@pytest.mark.parametrize('pkin_const', [False, True])
def test_bfieldexpansion_scalar_strengths(h, pkin_const):
    kwargs = dict(length=0.4, h=h, s_start=0.1, num_integration_steps=24,
                  pkin_const=pkin_const, ksol=[0.04, 0.01, 0.],
                  knl=[0.002, 0.001], ksl=[0.003])
    knc = np.array([[0.1, 0.02, 0.], [0.03, -0.01, 0.]])
    ksc = np.array([[0.01, 0.001, 0.]])
    element = xt.BFieldExpansion(**kwargs, knc=knc, ksc=ksc,
                                k0=0.002, k1=0.02, k2=0.03, k3=0.04,
                                k0s=0.001, k1s=0.003, k2s=0.004, k3s=0.005)
    for scale in (1., 0., -0.7):
        element.kscale = scale
        # Change every scalar after allocation, including rows absent in knc/ksc.
        for skew in ('', 's'):
            for i in range(4):
                setattr(element, f'k{i}{skew}', getattr(element, f'k{i}{skew}') + 0.001)
        combined = {}
        for name, skew in [('knc', ''), ('ksc', 's')]:
            array = np.zeros((4, 3))
            original = np.asarray(getattr(element, name))
            array[:len(original)] = original
            array[:, 0] += [getattr(element, f'k{i}{skew}') for i in range(4)]
            combined[name] = array
        reference = xt.BFieldExpansion(**kwargs, **combined, kscale=scale,
                                       num_phi=element.num_phi)
        coords = dict(x=[0.01, -0.008], y=[0.007, -0.009], s_local=[0., 0.4])
        actual_field = element.get_field(**coords)
        expected_field = reference.get_field(**coords)
        for name in actual_field.dtype.names:
            xo.assert_allclose(actual_field[name], expected_field[name], rtol=0, atol=2e-13)
        xo.assert_allclose(element.get_total_knl_ksl(), reference.get_total_knl_ksl(),
                           rtol=0, atol=1e-14)
        initial = xt.Particles(p0c=1e9, x=coords['x'], y=coords['y'], px=0.001)
        particles, expected = initial.copy(), initial.copy()
        element.track(particles)
        reference.track(expected)
        for name in ('x', 'px', 'y', 'py', 'zeta', 'delta', 'ax', 'ay'):
            xo.assert_allclose(getattr(particles, name), getattr(expected, name), rtol=0, atol=2e-13)
    np.testing.assert_array_equal(element.knc, knc)
    np.testing.assert_array_equal(element.ksc, ksc)
    empty = xt.BFieldExpansion(length=0.4, h=h)
    assert empty.k0 == 0.
    assert empty.get_field(0., 0., 0.)['By'] == 0.
    empty.k3s = 0.5  # All scalar orders also work with the smallest profiles.
    assert empty.get_field(0.02, 0., 0.)['Bx'] == pytest.approx(0.5 * 0.02**3 / 6, abs=1e-14)


@pytest.mark.parametrize('h', [0., 0.3])
@pytest.mark.parametrize('sliced', [False, True])
def test_bfieldexpansion_scalar_environment_and_strengths(h, sliced):
    env = xt.Environment()
    env['strength'] = 0.02
    env['scale'] = 0.7
    env.new('e', 'BFieldExpansion', length=0.4, h=h,
            knc=[[0.1, 0.01], [0.03, 0.]], ksc=[[0.001, 0.]],
            knl=[0.001, 0.002], ksl=[0.003],
            k0='strength', k1='2*strength', k2s='3*strength',
            kscale='scale', num_integration_steps=40, integrator='rk4')
    env.new('m', 'Magnet', length=0.1, k0=0.01, k1=0.04, knl=[0.002])
    line = env.new_line(components=['e', 'm'])
    if sliced:
        line.slice_thick_elements([xt.Strategy(None),
            xt.Strategy(xt.Uniform(4, mode='thick'), element_type=xt.BFieldExpansion)])
    line.particle_ref = xt.Particles(p0c=1e9)
    for scale, strength in ((0.7, 0.02), (-0.8, 0.04), (0., 0.03)):
        env['scale'], env['strength'] = scale, strength
        env.set('e', k3='4*strength', num_integration_steps=48, integrator='rk4')
        parent = env.get('e')
        assert parent.num_integration_steps == 48
        names = [name for name in line.element_names
                 if isinstance(line.get(name), (xt.BFieldExpansion, xt.ThickSliceBFieldExpansion))]
        table = line.get_table(attr=True)
        for name in names:
            el = line.get(name)
            totals = el.get_total_knl_ksl()
            for skew, values in zip(('', 's'), totals):
                for i in range(4):
                    assert table[f'k{i}{skew}l', name] == pytest.approx(values[i], abs=1e-14)
        assert table['k0l', 'm'] == pytest.approx(0.003)
        assert table['k1l', 'm'] == pytest.approx(0.004)
        totals = parent.get_total_knl_ksl()
        assert sum(table['k1l', name] for name in names) == pytest.approx(totals[0][1])
        assert totals[0][1] == pytest.approx(scale * ((0.03 + 2 * strength) * 0.4 + 0.002))
        assert totals[1][2] == pytest.approx(scale * 3 * strength * 0.4)
        x, s = 0.01, 0.2
        field = parent.get_field(x, 0., s)
        expected_by = scale * (strength + 0.1 + 0.01*s + 0.001/0.4
                              + (2*strength + 0.03 + 0.002/0.4)*x
                              + 4*strength*x**3/6)
        assert field['By'] == pytest.approx(expected_by, abs=1e-13)
    env['scale'] = 0.7
    twiss = line.twiss4d(betx=1., bety=1., strengths=True)
    table = line.get_table(attr=True)
    for column in ('k0l', 'k1l', 'k3l', 'k2sl'):
        xo.assert_allclose(twiss[column], table[column], atol=1e-14, rtol=0)


@pytest.mark.parametrize('h', [0., 0.3])
@pytest.mark.parametrize('pkin_const', [False, True])
@pytest.mark.parametrize('sliced', [False, True])
def test_bfieldexpansion_standard_misalignments(h, pkin_const, sliced):
    env = xt.Environment()
    env['offset'] = 0.001
    env.new('e', 'BFieldExpansion', length=0.4, h=h, num_integration_steps=80,
            knc=[[0.1, 0.02, 0.], [0.03, 0., 0.]],
            ksc=[[0.01, -0.001, 0.]], ksol=[0.04, 0.01, 0.],
            k1=0.02, k2s=0.003, pkin_const=pkin_const,
            shift_x='offset', shift_y=-0.002, shift_s=0.003,
            rot_x_rad=0.002, rot_y_rad=-0.003, rot_s_rad=0.1,
            rot_s_rad_no_frame=0.02, rot_shift_anchor=0.17)
    line = env.new_line(components=['e'])
    if sliced:
        line.slice_thick_elements([xt.Strategy(xt.Uniform(4, mode='thick'))])
    for updated in (False, True):
        if updated:
            env['offset'] = -0.003
            env.set('e', length=0.6, rot_y_rad=0.001, rot_shift_anchor=0.2)
        parent = env.get('e')
        aligned = parent.copy()
        for name in ('shift_x', 'shift_y', 'shift_s', 'rot_x_rad', 'rot_y_rad',
                     'rot_s_rad', 'rot_s_rad_no_frame'):
            setattr(aligned, name, 0.)
        manual_elements = []
        names = [name for name in line.element_names
                 if isinstance(line.get(name), (xt.BFieldExpansion, xt.ThickSliceBFieldExpansion))]
        for name in names:
            element = line.get(name)
            weight = element.weight if sliced else 1.
            offset = element.slice_offset if sliced else 0.
            parameters = dict(dx=parent.shift_x, dy=parent.shift_y, ds=parent.shift_s,
                              theta=parent.rot_y_rad, phi=parent.rot_x_rad,
                              psi=parent.rot_s_rad_no_frame, tilt=parent.rot_s_rad,
                              anchor=parent.rot_shift_anchor - offset,
                              length=parent.length * weight, angle=parent.angle * weight,
                              h=h)
            body = (xt.ThickSliceBFieldExpansion(_parent=aligned, weight=weight,
                                                slice_offset=offset,
                                                _buffer=aligned._buffer)
                    if sliced else aligned)
            if sliced:
                body.parent_name = 'manual_parent'
            manual_elements.extend([xt.Misalignment(**parameters), body,
                                    xt.Misalignment(**parameters, is_exit=True)])
        manual = xt.Line(elements=manual_elements)
        if sliced:
            manual.env.elements['manual_parent'] = aligned
            manual.build_tracker(_buffer=aligned._buffer)
        initial = xt.Particles(p0c=1e9, x=[0.01, -0.005], y=[0.007, 0.002],
                               px=0.001, py=-0.002, delta=[0.01, -0.02])
        expected, actual = initial.copy(), initial.copy()
        manual.track(expected, _force_no_end_turn_actions=True)
        line.track(actual, _force_no_end_turn_actions=True)
        for field in ('x', 'px', 'y', 'py', 'zeta', 'delta', 's', 'ax', 'ay'):
            xo.assert_allclose(getattr(actual, field), getattr(expected, field),
                               rtol=0, atol=2e-13)
        if sliced:
            unsliced = initial.copy()
            parent.track(unsliced)
            for field in ('x', 'px', 'y', 'py', 'zeta', 'delta', 's'):
                xo.assert_allclose(getattr(actual, field), getattr(unsliced, field),
                                   rtol=0, atol=3e-12)
        # Tables inherit parent alignment also for slices, and clean
        # serialization preserves both alignment inputs and the interval.
        restored = xt.Line.from_dict(line.to_dict())
        restored_particles = initial.copy()
        restored.track(restored_particles, _force_no_end_turn_actions=True)
        for field in ('x', 'px', 'y', 'py', 'zeta', 'delta', 'ax', 'ay'):
            xo.assert_allclose(getattr(actual, field), getattr(restored_particles, field),
                               rtol=0, atol=2e-13)
        table = line.get_table(attr=True)
        for name in names:
            assert table['shift_x', name] == pytest.approx(parent.shift_x)
            assert table['rot_s_rad', name] == pytest.approx(parent.rot_s_rad)
        # get_field is independent of the laboratory-frame placement.
        np.testing.assert_array_equal(parent.get_field(0.01, 0.007, 0.2),
                                      aligned.get_field(0.01, 0.007, 0.2))
        line.track(actual, backtrack=True, _force_no_end_turn_actions=True)
        for field in ('x', 'px', 'y', 'py', 'zeta', 'delta', 's'):
            xo.assert_allclose(getattr(actual, field), getattr(initial, field),
                               rtol=0, atol=3e-12)
