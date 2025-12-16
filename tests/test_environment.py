import xtrack as xt
import xobjects as xo
import xdeps as xd
import numpy as np
import pytest
import json
import pathlib

test_data_folder = pathlib.Path(__file__).parent.joinpath('../test_data').absolute()

@pytest.mark.parametrize('container_type', ['env', 'line'])
def test_vars_and_element_access_modes(container_type):

    env = xt.Environment()

    env.vars({
        'k.1': 1.,
        'a': 2.,
        'b': '2 * a + k.1',
    })

    line = env.new_line([])

    ee = {'env': env, 'line': line}[container_type]

    assert ee.vv['b'] == 2 * 2 + 1

    ee.vars['a'] = ee.vars['k.1']
    assert ee.vv['b'] == 2 * 1 + 1

    ee.vars(a=3.)
    ee.vars({'k.1': 'a'})
    assert ee.vv['k.1'] == 3.
    assert ee.vv['b'] == 2 * 3 + 3.

    ee.vars['k.1'] = 2 * ee.vars['a'] + 5
    assert ee.vv['k.1'] == 2 * 3 + 5
    assert ee.vv['b'] == 2 * 3 + 2 * 3 + 5

    ee.vars.set('a', 4.)
    assert ee.vv['k.1'] == 2 * 4 + 5
    assert ee.vv['b'] == 2 * 4 + 2 * 4 + 5

    ee.vars.set('k.1', '2*a + 5')
    assert ee.vv['k.1'] == 2 * 4 + 5
    assert ee.vv['b'] == 2 * 4 + 2 * 4 + 5

    ee.vars.set('k.1', 3 * ee.vars['a'] + 6)
    assert ee.vv['k.1'] == 3 * 4 + 6
    assert ee.vv['b'] == 2 * 4 + 3 * 4 + 6

    env.set('c', '2*b')
    assert env.vv['c'] == 2 * (2 * 4 + 3 * 4 + 6)
    env.set('d', 6)
    assert env.vv['d'] == 6
    env.set('d', '7')
    assert env.vv['d'] == 7

    ee.set('a', 0.)
    assert ee.vv['k.1'] == 3 * 0 + 6
    assert ee.vv['b'] == 2 * 0 + 3 * 0 + 6

    ee.set('a', 2.)
    ee.set('k.1', '2 * a + 5')
    assert ee.vv['k.1'] == 2 * 2 + 5
    assert ee.vv['b'] == 2 * 2 + 2 * 2 + 5

    ee.set('k.1', 3 * ee.vars['a'] + 6)
    assert ee.vv['k.1'] == 3 * 2 + 6
    assert ee.vv['b'] == 2 * 2 + 3 * 2 + 6

    assert hasattr(ee.ref['k.1'], '_value') # is a Ref

    ee.ref['a'] = 0
    assert ee.vv['k.1'] == 3 * 0 + 6
    assert ee.vv['b'] == 2 * 0 + 3 * 0 + 6

    ee.ref['a'] = 2
    ee.ref['k.1'] = 2 * ee.ref['a'] + 5
    assert ee.vv['k.1'] == 2 * 2 + 5
    assert ee.vv['b'] == 2 * 2 + 2 * 2 + 5

    #--------------------------------------------------

    ee.vars({
        'a': 4.,
        'b': '2 * a + 5',
        'k.1': '2 * a + 5',
    })

    env.new('bb', xt.Bend, k0='2 * b', length=3+env.vars['a'] + env.vars['b'],
            angle=5., model='full')
    assert env['bb'].k0 == 2 * (2 * 4 + 5)
    assert env['bb'].length == 3 + 4 + 2 * 4 + 5
    assert env['bb'].angle == 5.
    assert env['bb'].model == 'full'

    env.vars['a'] = 2.
    assert env['bb'].k0 == 2 * (2 * 2 + 5)
    assert env['bb'].length == 3 + 2 + 2 * 2 + 5
    assert env['bb'].angle == 5.

    env['bb'].model = 'adaptive'
    assert env['bb'].model == 'adaptive'

    line = env.new_line([
        env.new('bb1', 'bb', length=3*env.vars['a'], at='2*a'),
        env.place('bb', at=10 * env.vars['a'], from_='bb1'),
    ])

    assert hasattr(env.ref['bb1'].length, '_value') # is a Ref
    assert not hasattr(env['bb1'].length, '_value') # a number
    assert env.ref['bb1'].length._value == 3 * 2
    assert env['bb1'].length == 3 * 2

    assert hasattr(env.ref['bb1'].length, '_value') # is a Ref
    assert not hasattr(env['bb1'].length, '_value') # a number
    assert env.ref['bb1'].length._value == 3 * 2
    assert env['bb1'].length == 3 * 2

    assert line.get('bb1') is not env.get('bb')
    assert line.get('bb') is env.get('bb')

    a = env.vv['a']
    assert line['bb1'].length == 3 * a
    assert line['bb1'].k0 == 2 * (2 * a + 5)
    assert line['bb1'].angle == 5.

    assert line['bb'].k0 == 2 * (2 * a + 5)
    assert line['bb'].length == 3 + a + 2 * a + 5
    assert line['bb'].angle == 5.

    tt = line.get_table(attr=True)
    tt['s_center'] = tt['s'] + tt['length']/2

    assert np.all(tt.name ==  np.array(['||drift_1', 'bb1', '||drift_2', 'bb', '_end_point']))

    assert tt['s_center', 'bb1'] == 2*a
    assert tt['s_center', 'bb'] - tt['s_center', 'bb1'] == 10*a

    old_a = a
    line.vars['a'] = 3.
    a = line.vv['a']
    assert line['bb1'].length == 3 * a
    assert line['bb1'].k0 == 2 * (2 * a + 5)
    assert line['bb1'].angle == 5.

    assert line['bb'].k0 == 2 * (2 * a + 5)
    assert line['bb'].length == 3 + a + 2 * a + 5
    assert line['bb'].angle == 5.

    tt_new = line.get_table(attr=True)

    # Drifts are not changed:
    tt_new['length', '||drift_1'] == tt['length', '||drift_1']
    tt_new['length', '||drift_2'] == tt['length', '||drift_2']

    lcp = line.copy()
    assert lcp._element_dict is not line._element_dict
    assert lcp.env is not line.env
    assert lcp._xdeps_vref._ownerr is not line._xdeps_vref._owner

    line['a'] = 333
    lcp['a'] = 444

    assert line['a'] == 333
    assert lcp['a'] == 444

    assert env.elements.env is env

    # Check set with multiple targets
    env['x0'] = 0
    env.set(['x1', 'x2', 'x3'], '3*x0')
    env['x0'] = 3.
    ttv = env.vars.get_table()
    for nn in ['x1', 'x2', 'x3']:
        assert ttv['value', nn] == 3 * 3
        assert ttv['expr', nn] == '(3.0 * x0)'

    env.new('qx1', xt.Quadrupole, length=1)
    env.new('qx2', xt.Quadrupole, length=1)
    env.new('qx3', xt.Quadrupole, length=1)

    env.set(['qx1', 'qx2', 'qx3'], k1='3*x0')
    for nn in ['qx1', 'qx2', 'qx3']:
        assert env[nn].k1 == 3 * 3
        assert str(env.ref[nn].k1._expr) == "(3.0 * vars['x0'])"


def test_element_placing_at_s():

    env = xt.Environment()

    env.vars({
        'l.b1': 1.0,
        'l.q1': 0.5,
        's.ip': 10,
        's.left': -5,
        's.right': 5,
        'l.before_right': 1,
        'l.after_left2': 0.5,
    })

    # names, tab_sorted = handle_s_places(seq)
    line = env.new_line(components=[
        env.new('b1', xt.Bend, length='l.b1'),
        env.new('q1', xt.Quadrupole, length='l.q1'),
        env.new('ip', xt.Marker, at='s.ip'),
        env.new('before_before_right', xt.Marker, at='before_right@start'),
        env.new('before_right', xt.Sextupole, length=1, anchor='end', at='right@start'),
        env.new('right',xt.Quadrupole, length=0.8, at='s.right', from_='ip'),
        env.new('after_right', xt.Marker),
        env.new('after_right2', xt.Marker),
        env.new('left', xt.Quadrupole, length=1, at='s.left', from_='ip'),
        env.new('after_left', xt.Marker),
        env.new('after_left2', xt.Bend, length='l.after_left2'),
    ])

    tt = line.get_table(attr=True)
    tt['s_center'] = tt['s'] + tt['length']/2
    assert np.all(tt.name == np.array([
        'b1', 'q1', '||drift_1', 'left', 'after_left', 'after_left2',
       '||drift_2', 'ip', '||drift_3', 'before_before_right',
       'before_right', 'right', 'after_right', 'after_right2',
       '_end_point']))

    xo.assert_allclose(env['b1'].length, 1.0, rtol=0, atol=1e-14)
    xo.assert_allclose(env['q1'].length, 0.5, rtol=0, atol=1e-14)
    xo.assert_allclose(tt['s', 'ip'], 10, rtol=0, atol=1e-14)
    xo.assert_allclose(tt['s', 'before_before_right'], tt['s', 'before_right'],
                    rtol=0, atol=1e-14)
    xo.assert_allclose(tt['s_center', 'before_right'] - tt['s_center', 'right'],
                    -(1 + 0.8)/2, rtol=0, atol=1e-14)
    xo.assert_allclose(tt['s_center', 'right'] - tt['s', 'ip'], 5, rtol=0, atol=1e-14)
    xo.assert_allclose(tt['s_center', 'after_right'] - tt['s_center', 'right'],
                        0.8/2, rtol=0, atol=1e-14)
    xo.assert_allclose(tt['s_center', 'after_right2'] - tt['s_center', 'right'],
                        0.8/2, rtol=0, atol=1e-14)
    xo.assert_allclose(tt['s_center', 'left'] - tt['s_center', 'ip'], -5,
                    rtol=0, atol=1e-14)
    xo.assert_allclose(tt['s_center', 'after_left'] - tt['s_center', 'left'], 1/2,
                        rtol=0, atol=1e-14)
    xo.assert_allclose(tt['s_center', 'after_left2'] - tt['s_center', 'after_left'],
                    0.5/2, rtol=0, atol=1e-14)


    # import matplotlib.pyplot as plt
    # plt.close('all')
    # line.survey().plot()

    # plt.show()

def test_assemble_ring():

    env = xt.Environment()
    env.particle_ref = xt.Particles(p0c=2e9)

    n_bends_per_cell = 6
    n_cells_par_arc = 3
    n_arcs = 3

    n_bends = n_bends_per_cell * n_cells_par_arc * n_arcs

    env.vars({
        'l.mq': 0.5,
        'kqf': 0.027,
        'kqd': -0.0271,
        'l.mb': 10,
        'l.ms': 0.3,
        'k2sf': 0.001,
        'k2sd': -0.001,
        'angle.mb': 2 * np.pi / n_bends,
        'k0.mb': 'angle.mb / l.mb',
        'k0l.corrector': 0,
        'k1sl.corrector': 0,
        'l.halfcell': 38,
    })

    env.new('mb', xt.Bend, length='l.mb', k0='k0.mb', angle='k0.mb * l.mb')
    env.new('mq', xt.Quadrupole, length='l.mq')
    env.new('ms', xt.Sextupole, length='l.ms')
    env.new('corrector', xt.Multipole, knl=[0], length=0.1)

    girder = env.new_line(components=[
        env.place('mq', at=1),
        env.place('ms', at=0.8, from_='mq'),
        env.place('corrector', at=-0.8, from_='mq'),
    ])

    tt_girder = girder.get_table(attr=True)
    assert np.all(tt_girder.name == np.array(
        ['||drift_1', 'corrector', '||drift_2', 'mq', '||drift_3', 'ms',
       '_end_point']))
    tt_girder['s_center'] = tt_girder['s'] + \
        tt_girder['length']/2 * np.float64(tt_girder['isthick'])
    xo.assert_allclose(tt_girder['s_center', 'mq'], 1., atol=1e-14, rtol=0)
    xo.assert_allclose(tt_girder['s_center', 'ms'] - tt_girder['s_center', 'mq'], 0.8,
                    atol=1e-14, rtol=0)
    xo.assert_allclose(
        tt_girder['s_center', 'corrector'] - tt_girder['s_center', 'mq'], -0.8,
        atol=1e-14, rtol=0)


    girder_f = girder.clone(suffix='f')
    girder_d = girder.clone(suffix='d', mirror=True)
    env.set('mq.f', k1='kqf')
    env.set('mq.d', k1='kqd')

    # Check clone
    tt_girder_f = girder_f.get_table(attr=True)
    assert (~(tt_girder_f.isreplica)).all()
    assert np.all(tt_girder_f.name == np.array(
        ['||drift_1', 'corrector.f', '||drift_2', 'mq.f', '||drift_3',
         'ms.f', '_end_point']))
    tt_girder_f['s_center'] = (tt_girder_f['s']
                            + tt_girder_f['length']/2 * np.float64(tt_girder_f['isthick']))
    xo.assert_allclose(tt_girder_f['s_center', 'mq.f'], 1., atol=1e-14, rtol=0)
    xo.assert_allclose(tt_girder_f['s_center', 'ms.f'] - tt_girder_f['s_center', 'mq.f'], 0.8,
                    atol=1e-14, rtol=0)
    xo.assert_allclose(
        tt_girder_f['s_center', 'corrector.f'] -
        tt_girder_f['s_center', 'mq.f'], -0.8,
        atol=1e-14, rtol=0)

    # Check clone mirror
    tt_girder_d = girder_d.get_table(attr=True)
    assert (~(tt_girder_d.isreplica)).all()
    len_girder = tt_girder_d.s[-1]
    assert np.all(tt_girder_d.name == np.array(
        ['ms.d', '||drift_3', 'mq.d', '||drift_2', 'corrector.d',
       '||drift_1', '_end_point']))
    tt_girder_d['s_center'] = (tt_girder_d['s']
                            + tt_girder_d['length']/2 * np.float64(tt_girder_d['isthick']))
    xo.assert_allclose(tt_girder_d['s_center', 'mq.d'],
                    len_girder - 1., atol=1e-14, rtol=0)
    xo.assert_allclose(tt_girder_d['s_center', 'ms.d'] - tt_girder_d['s_center', 'mq.d'],
                    -0.8, atol=1e-14, rtol=0)
    xo.assert_allclose(tt_girder_d['s_center', 'corrector.d'] - tt_girder_d['s_center', 'mq.d'],
                    0.8, atol=1e-14, rtol=0)


    halfcell = env.new_line(components=[

        # End of the half cell (will be mid of the cell)
        env.new('mid', xt.Marker, at='l.halfcell'),

        # Bends
        env.new('mb.2', 'mb', at='l.halfcell / 2'),
        env.new('mb.1', 'mb', at='-l.mb - 1', from_='mb.2'),
        env.new('mb.3', 'mb', at='l.mb + 1', from_='mb.2'),

        # Quadrupoles, sextupoles and correctors
        env.place(girder_d, at=1.2),
        env.place(girder_f, at='l.halfcell - 1.2'),

    ])

    l_hc = env.vv['l.halfcell']
    xo.assert_allclose(l_hc, l_hc, atol=1e-14, rtol=0)
    tt_hc = halfcell.get_table(attr=True)
    assert np.all(tt_hc.name == np.array(
        ['||drift_4', 'ms.d', '||drift_3::0', 'mq.d', '||drift_2::0',
       'corrector.d', '||drift_1::0', '||drift_5', 'mb.1', '||drift_6::0',
       'mb.2', '||drift_6::1', 'mb.3', '||drift_7', '||drift_1::1',
       'corrector.f', '||drift_2::1', 'mq.f', '||drift_3::1', 'ms.f',
       '||drift_8', 'mid', '_end_point']))
    assert np.all(tt_hc.element_type == np.array(
        ['Drift', 'Sextupole', 'Drift', 'Quadrupole', 'Drift', 'Multipole',
        'Drift', 'Drift', 'Bend', 'Drift', 'Bend', 'Drift', 'Bend',
        'Drift', 'Drift', 'Multipole', 'Drift', 'Quadrupole', 'Drift',
        'Sextupole', 'Drift', 'Marker', '']))
    assert np.all(tt_hc.isreplica == False)
    tt_hc['s_center'] = (
        tt_hc['s'] + tt_hc['length'] / 2 * np.float64(tt_hc['isthick']))
    xo.assert_allclose(tt_hc['s_center', 'mq.d'],
                    1.2 - tt_girder_d.s[-1] / 2 +
                    tt_girder_d['s_center', 'mq.d'],
                    atol=1e-14, rtol=0)
    xo.assert_allclose(tt_hc['s_center', 'ms.f'] - tt_hc['s_center', 'mq.f'], 0.8,
                    atol=1e-14, rtol=0)
    xo.assert_allclose(
        tt_hc['s_center', 'corrector.f'] - tt_hc['s_center', 'mq.f'], -0.8,
        atol=1e-14, rtol=0)
    xo.assert_allclose(tt_hc['s_center', 'ms.d'] - tt_hc['s_center', 'mq.d'],
                    -0.8, atol=1e-14, rtol=0)
    xo.assert_allclose(tt_hc['s_center', 'corrector.d'] - tt_hc['s_center', 'mq.d'],
                    0.8, atol=1e-14, rtol=0)
    xo.assert_allclose(tt_hc['s_center', 'mb.2'], l_hc / 2, atol=1e-14, rtol=0)
    xo.assert_allclose(tt_hc['s_center', 'mb.1'], tt_hc['s_center', 'mb.2'] - env.vv['l.mb'] - 1,
                        atol=1e-14, rtol=0)
    xo.assert_allclose(tt_hc['s_center', 'mb.3'], tt_hc['s_center', 'mb.2'] + env.vv['l.mb'] + 1,
                    atol=1e-14, rtol=0)


    hcell_left = halfcell.replicate(suffix='l', mirror=True)
    hcell_right = halfcell.replicate(suffix='r')

    cell = env.new_line(components=[
        env.new('start', xt.Marker),
        hcell_left,
        hcell_right,
        env.new('end', xt.Marker),
    ])

    tt_cell = cell.get_table(attr=True)
    tt_cell['s_center'] = (
        tt_cell['s'] + tt_cell['length'] / 2 * np.float64(tt_cell['isthick']))
    assert np.all(tt_cell.name == np.array(
        ['start', 'mid.l', '||drift_8::0', 'ms.f.l', '||drift_3::0',
       'mq.f.l', '||drift_2::0', 'corrector.f.l', '||drift_1::0',
       '||drift_7::0', 'mb.3.l', '||drift_6::0', 'mb.2.l', '||drift_6::1',
       'mb.1.l', '||drift_5::0', '||drift_1::1', 'corrector.d.l',
       '||drift_2::1', 'mq.d.l', '||drift_3::1', 'ms.d.l', '||drift_4::0',
       '||drift_4::1', 'ms.d.r', '||drift_3::2', 'mq.d.r', '||drift_2::2',
       'corrector.d.r', '||drift_1::2', '||drift_5::1', 'mb.1.r',
       '||drift_6::2', 'mb.2.r', '||drift_6::3', 'mb.3.r', '||drift_7::1',
       '||drift_1::3', 'corrector.f.r', '||drift_2::3', 'mq.f.r',
       '||drift_3::3', 'ms.f.r', '||drift_8::1', 'mid.r', 'end',
       '_end_point']))
    assert np.all(tt_cell.element_type == np.array(
        ['Marker', 'Marker', 'Drift', 'Sextupole', 'Drift', 'Quadrupole',
        'Drift', 'Multipole', 'Drift', 'Drift', 'Bend', 'Drift', 'Bend',
        'Drift', 'Bend', 'Drift', 'Drift', 'Multipole', 'Drift',
        'Quadrupole', 'Drift', 'Sextupole', 'Drift', 'Drift', 'Sextupole',
        'Drift', 'Quadrupole', 'Drift', 'Multipole', 'Drift', 'Drift',
        'Bend', 'Drift', 'Bend', 'Drift', 'Bend', 'Drift', 'Drift',
        'Multipole', 'Drift', 'Quadrupole', 'Drift', 'Sextupole', 'Drift',
        'Marker', 'Marker', '']))
    assert np.all(tt_cell.isreplica == np.array(
        [False,  True, False,  True, False,  True, False,  True, False,
       False,  True, False,  True, False,  True, False, False,  True,
       False,  True, False,  True, False, False,  True, False,  True,
       False,  True, False, False,  True, False,  True, False,  True,
       False, False,  True, False,  True, False,  True, False,  True,
       False, False]))

    tt_cell_stripped = tt_cell.rows[1:-2] # Remove _end_point and markers added in cell
    tt_cell_second_half = tt_cell_stripped.rows[len(tt_cell_stripped)//2 :]
    tt_cell_second_half.s_center -= tt_cell_second_half.s[0]
    tt_hc_stripped = tt_hc.rows[:-1] # Remove _end_point
    xo.assert_allclose(tt_cell_second_half.s_center, tt_hc_stripped.s_center, atol=5e-14, rtol=0)
    tt_cell_first_half = tt_cell_stripped.rows[:len(tt_cell_stripped)//2]
    s_center_mirrored_first_half = (
        tt_cell_stripped['s', len(tt_cell_stripped)//2] - tt_cell_first_half.s_center[::-1])
    xo.assert_allclose(s_center_mirrored_first_half, tt_hc_stripped.s_center, atol=5e-14, rtol=0)

    env.vars({
        'kqf.ss': 0.027 / 2,
        'kqd.ss': -0.0271 / 2,
    })

    halfcell_ss = env.new_line(components=[

        env.new('mid.ss', xt.Marker, at='l.halfcell'),

        env.new('mq.ss.d', 'mq', k1='kqd.ss', at = '0.5 + l.mq / 2'),
        env.new('mq.ss.f', 'mq', k1='kqf.ss', at = 'l.halfcell - l.mq / 2 - 0.5'),

        env.new('corrector.ss.v', 'corrector', at=0.75, from_='mq.ss.d'),
        env.new('corrector.ss.h', 'corrector', at=-0.75, from_='mq.ss.f')
    ])

    hcell_left_ss = halfcell_ss.replicate(suffix='l', mirror=True)
    hcell_right_ss = halfcell_ss.replicate(suffix='r')
    cell_ss = env.new_line(components=[
        env.new('start.ss', xt.Marker),
        hcell_left_ss,
        hcell_right_ss,
        env.new('end.ss', xt.Marker),
    ])

    env.lines['cell.2'] = cell.replicate(suffix='cell.2')
    arc = env.new_line(components=[
        cell.replicate(suffix='cell.1'),
        'cell.2',
        cell.replicate(suffix='cell.3'),
    ])

    assert 'cell.2' in env.lines
    tt_cell2 = env.lines['cell.2'].get_table(attr=True)
    assert np.all([nn for nn in tt_cell2.name[:-1] if not nn.startswith('||drift')]
                   == np.array([nn+'.cell.2' for nn in tt_cell.name[:-1]
                                if not nn.startswith('||drift')]))
    assert np.all(tt_cell2.s == tt_cell.s)
    tt_cell2_nodrift = tt_cell2.rows[~tt_cell2.rows.mask[r'\|\|drift.*']]
    assert tt_cell2_nodrift.isreplica[:-1].all()
    assert tt_cell2['parent_name', 'mq.d.l.cell.2'] == 'mq.d.l'
    assert tt_cell2['parent_name', 'mq.f.l.cell.2'] == 'mq.f.l'
    assert tt_cell['parent_name', 'mq.d.l'] == 'mq.d'
    assert tt_cell['parent_name', 'mq.f.l'] == 'mq.f'

    tt_arc = arc.get_table(attr=True)
    assert len(tt_arc) == 3 * (len(tt_cell)-1) + 1
    n_cell = len(tt_cell) - 1
    assert np.all(tt_arc.env_name[n_cell:2*n_cell]
                  == tt_cell2.env_name[:-1])
    for nn in tt_cell2.env_name[:-1]:
        assert arc.get(nn) is env.get(nn)
        assert arc.get(nn) is env['cell.2'].get(nn)

    ss = env.new_line(components=[
        cell_ss.replicate('cell.1'),
        cell_ss.replicate('cell.2'),
    ])

    env.lines['arc.1'] =  arc.replicate(suffix='arc.1')
    env.lines['ss.1'] =  ss.replicate(suffix='ss.1')
    env.lines['arc.2'] =  arc.replicate(suffix='arc.2')
    env.lines['ss.2'] =  ss.replicate(suffix='ss.2')
    env.lines['arc.3'] =  arc.replicate(suffix='arc.3')
    env.lines['ss.3'] =  ss.replicate(suffix='ss.3')

    ring = env.new_line(components=[
        env['arc.1'],
        env['ss.1'],
        env['arc.2'],
        env['ss.2'],
        env['arc.3'],
        env['ss.3'],
    ])
    tt_ring = ring.get_table(attr=True)
    # Check length
    xo.assert_allclose(tt_ring.s[-1], 2*l_hc * (n_cells_par_arc * n_arcs + 2*n_arcs),
                        atol=1e-12, rtol=0)
    # Check closure
    sv_ring = ring.survey()
    xo.assert_allclose(sv_ring.X[-1], 0, atol=1e-12, rtol=0)
    xo.assert_allclose(sv_ring.Y[-1], 0, atol=1e-12, rtol=0)
    xo.assert_allclose(sv_ring.Z[-1], 0, atol=1e-12, rtol=0)

    xo.assert_allclose(sv_ring.angle.sum(), 2*np.pi, atol=1e-12, rtol=0)

    ## Insertion

    env.vars({
        'k1.q1': 0.025,
        'k1.q2': -0.025,
        'k1.q3': 0.025,
        'k1.q4': -0.02,
        'k1.q5': 0.025,
    })

    half_insertion = env.new_line(components=[

        # Start-end markers
        env.new('ip', xt.Marker),
        env.new('e.insertion', xt.Marker, at=76),

        # Quads
        env.new('mq.1', xt.Quadrupole, k1='k1.q1', length='l.mq', at = 20),
        env.new('mq.2', xt.Quadrupole, k1='k1.q2', length='l.mq', at = 25),
        env.new('mq.3', xt.Quadrupole, k1='k1.q3', length='l.mq', at=37),
        env.new('mq.4', xt.Quadrupole, k1='k1.q4', length='l.mq', at=55),
        env.new('mq.5', xt.Quadrupole, k1='k1.q5', length='l.mq', at=73),

        # Dipole correctors (will use h and v on the same corrector)
        env.new('corrector.ss.1', 'corrector', at=0.75, from_='mq.1'),
        env.new('corrector.ss.2', 'corrector', at=-0.75, from_='mq.2'),
        env.new('corrector.ss.3', 'corrector', at=0.75, from_='mq.3'),
        env.new('corrector.ss.4', 'corrector', at=-0.75, from_='mq.4'),
        env.new('corrector.ss.5', 'corrector', at=0.75, from_='mq.5'),

    ])

    insertion = env.new_line([
        half_insertion.replicate('l', mirror=True),
        half_insertion.replicate('r')])

    ring2 = env.new_line(components=[
        env['arc.1'],
        env['ss.1'],
        env['arc.2'],
        insertion,
        env['arc.3'],
        env['ss.3'],
    ])


    # # Check buffer behavior
    ring2_sliced = ring2.select()
    ring2_sliced.cut_at_s(np.arange(0, ring2.get_length(), 0.5))

    opt = cell.match(
        method='4d',
        vary=xt.VaryList(['kqf', 'kqd'], step=1e-5),
        targets=xt.TargetSet(
            qx=0.333333,
            qy=0.333333,
        ))
    tw_cell = cell.twiss4d()

    opt = cell_ss.match(
        solve=False,
        method='4d',
        vary=xt.VaryList(['kqf.ss', 'kqd.ss'], step=1e-5),
        targets=xt.TargetSet(
            betx=tw_cell.betx[-1], bety=tw_cell.bety[-1], at='start.ss',
        ))
    opt.solve()

    tw_arc = arc.twiss4d()

    opt = half_insertion.match(
        solve=False,
        betx=tw_arc.betx[0], bety=tw_arc.bety[0],
        alfx=tw_arc.alfx[0], alfy=tw_arc.alfy[0],
        init_at='e.insertion',
        start='ip', end='e.insertion',
        vary=xt.VaryList(['k1.q1', 'k1.q2', 'k1.q3', 'k1.q4'], step=1e-5),
        targets=[
            xt.TargetSet(alfx=0, alfy=0, at='ip'),
            xt.Target(lambda tw: tw.betx[0] - tw.bety[0], 0),
            xt.Target(lambda tw: tw.betx.max(), xt.LessThan(400)),
            xt.Target(lambda tw: tw.bety.max(), xt.LessThan(400)),
            xt.Target(lambda tw: tw.betx.min(), xt.GreaterThan(2)),
            xt.Target(lambda tw: tw.bety.min(), xt.GreaterThan(2)),
        ]
    )
    opt.step(40)
    opt.solve()

    # Check that the cell is matched to the rest of the ring
    tw = ring.twiss4d()
    tw_cell_from_ring = tw.rows['start.cell.3.arc.2':'end.cell.3.arc.2']
    xo.assert_allclose(tw_cell_from_ring.betx, tw_cell.betx[:-1], atol=0, rtol=5e-4)
    xo.assert_allclose(tw_cell_from_ring.bety, tw_cell.bety[:-1], atol=0, rtol=5e-4)

    tw2 = ring2.twiss4d()
    tw_cell_from_ring2 = tw2.rows['start.cell.3.arc.2':'end.cell.3.arc.2']
    xo.assert_allclose(tw_cell_from_ring2.betx, tw_cell.betx[:-1], atol=0, rtol=5e-4)
    xo.assert_allclose(tw_cell_from_ring2.bety, tw_cell.bety[:-1], atol=0, rtol=5e-4)

    # Check select
    cell3_select = ring2.select(start='start.cell.3.arc.2', end='end.cell.3.arc.2',
                                name='cell3_copy')
    assert 'cell3_copy' in env.lines
    assert cell3_select.particle_ref is not None
    assert env.lines['cell3_copy'] is cell3_select
    assert cell3_select._element_dict is env._element_dict
    assert cell3_select.element_names[0] == 'start.cell.3.arc.2'
    assert cell3_select.element_names[-1] == 'end.cell.3.arc.2'
    assert (np.array(cell3_select.element_names) == np.array(
        tw.rows['start.cell.3.arc.2':'end.cell.3.arc.2'].env_name)).all()

    # Check that they share the _element_dict
    assert cell._element_dict is env._element_dict
    assert halfcell._element_dict is env._element_dict
    assert halfcell_ss._element_dict is env._element_dict
    assert cell_ss._element_dict is env._element_dict
    assert insertion._element_dict is env._element_dict
    assert ring2._element_dict is env._element_dict

    cell3_select.twiss4d()

    tw2_slice = ring2_sliced.twiss4d()
    xo.assert_allclose(tw2_slice['betx', 'ip.l'], tw2['betx', 'ip.l'], atol=0, rtol=5e-4)
    xo.assert_allclose(tw2_slice['bety', 'ip.l'], tw2['bety', 'ip.l'], atol=0, rtol=5e-4)
    xo.assert_allclose(tw2_slice['alfx', 'ip.l'], 0, atol=1e-6, rtol=0)
    xo.assert_allclose(tw2_slice['alfy', 'ip.l'], 0, atol=1e-6, rtol=0)
    xo.assert_allclose(tw2_slice['dx', 'ip.l'], 0, atol=1e-4, rtol=0)
    xo.assert_allclose(tw2_slice['dpx', 'ip.l'], 0, atol=1e-6, rtol=0)
    xo.assert_allclose(tw2_slice['dy', 'ip.l'], 0, atol=1e-4, rtol=0)
    xo.assert_allclose(tw2_slice['dpy', 'ip.l'], 0, atol=1e-6, rtol=0)

    # import matplotlib.pyplot as plt
    # plt.close('all')
    # for ii, rr in enumerate([ring, ring2_sliced]):

    #     ttww = rr.twiss4d()

    #     fig = plt.figure(ii, figsize=(6.4*1.2, 4.8))
    #     ax1 = fig.add_subplot(2, 1, 1)
    #     pltbet = ttww.plot('betx bety', ax=ax1)
    #     ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    #     pltdx = ttww.plot('dx', ax=ax2)
    #     fig.subplots_adjust(right=.85)
    #     pltbet.move_legend(1.2,1)
    #     pltdx.move_legend(1.2,1)

    # ring2.survey().plot()
    # plt.show()

def test_assemble_ring_builders():

    env = xt.Environment()
    env.particle_ref = xt.Particles(p0c=2e9)

    n_bends_per_cell = 6
    n_cells_par_arc = 3
    n_arcs = 3

    n_bends = n_bends_per_cell * n_cells_par_arc * n_arcs

    env.vars({
        'l.mq': 0.5,
        'kqf': 0.027,
        'kqd': -0.0271,
        'l.mb': 10,
        'l.ms': 0.3,
        'k2sf': 0.001,
        'k2sd': -0.001,
        'angle.mb': 2 * np.pi / n_bends,
        'k0.mb': 'angle.mb / l.mb',
        'k0l.corrector': 0,
        'k1sl.corrector': 0,
        'l.halfcell': 38,
    })

    env.new('mb', xt.Bend, length='l.mb', k0='k0.mb', angle='k0.mb * l.mb')
    env.new('mq', xt.Quadrupole, length='l.mq')
    env.new('ms', xt.Sextupole, length='l.ms')
    env.new('corrector', xt.Multipole, knl=[0], length=0.1)

    girder = env.new_builder()
    girder.place('mq', at=1),
    girder.place('ms', at=0.8, from_='mq'),
    girder.place('corrector', at=-0.8, from_='mq'),
    girder = girder.build()

    tt_girder = girder.get_table(attr=True)
    assert np.all(tt_girder.name == np.array(
        ['||drift_1', 'corrector', '||drift_2', 'mq', '||drift_3', 'ms',
       '_end_point']))
    tt_girder['s_center'] = tt_girder['s'] + \
        tt_girder['length']/2 * np.float64(tt_girder['isthick'])
    xo.assert_allclose(tt_girder['s_center', 'mq'], 1., atol=1e-14, rtol=0)
    xo.assert_allclose(tt_girder['s_center', 'ms'] - tt_girder['s_center', 'mq'], 0.8,
                    atol=1e-14, rtol=0)
    xo.assert_allclose(
        tt_girder['s_center', 'corrector'] - tt_girder['s_center', 'mq'], -0.8,
        atol=1e-14, rtol=0)


    girder_f = girder.clone(suffix='f')
    girder_d = girder.clone(suffix='d', mirror=True)
    env.set('mq.f', k1='kqf')
    env.set('mq.d', k1='kqd')

    # Check clone
    tt_girder_f = girder_f.get_table(attr=True)
    assert (~(tt_girder_f.isreplica)).all()
    assert np.all(tt_girder_f.name == np.array(
        ['||drift_1', 'corrector.f', '||drift_2', 'mq.f', '||drift_3',
       'ms.f', '_end_point']))
    tt_girder_f['s_center'] = (tt_girder_f['s']
                            + tt_girder_f['length']/2 * np.float64(tt_girder_f['isthick']))
    xo.assert_allclose(tt_girder_f['s_center', 'mq.f'], 1., atol=1e-14, rtol=0)
    xo.assert_allclose(tt_girder_f['s_center', 'ms.f'] - tt_girder_f['s_center', 'mq.f'], 0.8,
                    atol=1e-14, rtol=0)
    xo.assert_allclose(
        tt_girder_f['s_center', 'corrector.f'] -
        tt_girder_f['s_center', 'mq.f'], -0.8,
        atol=1e-14, rtol=0)

    # Check clone mirror
    tt_girder_d = girder_d.get_table(attr=True)
    assert (~(tt_girder_d.isreplica)).all()
    len_girder = tt_girder_d.s[-1]
    assert np.all(tt_girder_d.name == np.array(
        ['ms.d', '||drift_3', 'mq.d', '||drift_2', 'corrector.d',
       '||drift_1', '_end_point']))
    tt_girder_d['s_center'] = (tt_girder_d['s']
                            + tt_girder_d['length']/2 * np.float64(tt_girder_d['isthick']))
    xo.assert_allclose(tt_girder_d['s_center', 'mq.d'],
                    len_girder - 1., atol=1e-14, rtol=0)
    xo.assert_allclose(tt_girder_d['s_center', 'ms.d'] - tt_girder_d['s_center', 'mq.d'],
                    -0.8, atol=1e-14, rtol=0)
    xo.assert_allclose(tt_girder_d['s_center', 'corrector.d'] - tt_girder_d['s_center', 'mq.d'],
                    0.8, atol=1e-14, rtol=0)


    halfcell = env.new_builder()
    # End of the half cell (will be mid of the cell)
    halfcell.new('mid', xt.Marker, at='l.halfcell')
    # Bends
    halfcell.new('mb.2', 'mb', at='l.halfcell / 2')
    halfcell.new('mb.1', 'mb', at='-l.mb - 1', from_='mb.2')
    halfcell.new('mb.3', 'mb', at='l.mb + 1', from_='mb.2')
    # Quadrupoles, sextupoles and correctors
    halfcell.place(girder_d, at=1.2)
    halfcell.place(girder_f, at='l.halfcell - 1.2')
    halfcell = halfcell.build()


    l_hc = env.vv['l.halfcell']
    xo.assert_allclose(l_hc, l_hc, atol=1e-14, rtol=0)
    tt_hc = halfcell.get_table(attr=True)
    assert np.all(tt_hc.name == np.array(
        ['||drift_4', 'ms.d', '||drift_3::0', 'mq.d', '||drift_2::0',
       'corrector.d', '||drift_1::0', '||drift_5', 'mb.1', '||drift_6::0',
       'mb.2', '||drift_6::1', 'mb.3', '||drift_7', '||drift_1::1',
       'corrector.f', '||drift_2::1', 'mq.f', '||drift_3::1', 'ms.f',
       '||drift_8', 'mid', '_end_point']))
    assert np.all(tt_hc.element_type == np.array(
        ['Drift', 'Sextupole', 'Drift', 'Quadrupole', 'Drift', 'Multipole',
        'Drift', 'Drift', 'Bend', 'Drift', 'Bend', 'Drift', 'Bend',
        'Drift', 'Drift', 'Multipole', 'Drift', 'Quadrupole', 'Drift',
        'Sextupole', 'Drift', 'Marker', '']))
    assert np.all(tt_hc.isreplica == False)
    tt_hc['s_center'] = (
        tt_hc['s'] + tt_hc['length'] / 2 * np.float64(tt_hc['isthick']))
    xo.assert_allclose(tt_hc['s_center', 'mq.d'],
                    1.2 - tt_girder_d.s[-1] / 2 +
                    tt_girder_d['s_center', 'mq.d'],
                    atol=1e-14, rtol=0)
    xo.assert_allclose(tt_hc['s_center', 'ms.f'] - tt_hc['s_center', 'mq.f'], 0.8,
                    atol=1e-14, rtol=0)
    xo.assert_allclose(
        tt_hc['s_center', 'corrector.f'] - tt_hc['s_center', 'mq.f'], -0.8,
        atol=1e-14, rtol=0)
    xo.assert_allclose(tt_hc['s_center', 'ms.d'] - tt_hc['s_center', 'mq.d'],
                    -0.8, atol=1e-14, rtol=0)
    xo.assert_allclose(tt_hc['s_center', 'corrector.d'] - tt_hc['s_center', 'mq.d'],
                    0.8, atol=1e-14, rtol=0)
    xo.assert_allclose(tt_hc['s_center', 'mb.2'], l_hc / 2, atol=1e-14, rtol=0)
    xo.assert_allclose(tt_hc['s_center', 'mb.1'], tt_hc['s_center', 'mb.2'] - env.vv['l.mb'] - 1,
                        atol=1e-14, rtol=0)
    xo.assert_allclose(tt_hc['s_center', 'mb.3'], tt_hc['s_center', 'mb.2'] + env.vv['l.mb'] + 1,
                    atol=1e-14, rtol=0)


    hcell_left = halfcell.replicate(suffix='l', mirror=True)
    hcell_right = halfcell.replicate(suffix='r')

    cell = env.new_builder()
    cell.new('start', xt.Marker)
    cell.place(hcell_left)
    cell.place(hcell_right)
    cell.new('end', xt.Marker)
    cell = cell.build()

    tt_cell = cell.get_table(attr=True)
    tt_cell['s_center'] = (
        tt_cell['s'] + tt_cell['length'] / 2 * np.float64(tt_cell['isthick']))
    assert np.all(tt_cell.name == np.array(
        ['start', 'mid.l', '||drift_8::0', 'ms.f.l', '||drift_3::0',
       'mq.f.l', '||drift_2::0', 'corrector.f.l', '||drift_1::0',
       '||drift_7::0', 'mb.3.l', '||drift_6::0', 'mb.2.l', '||drift_6::1',
       'mb.1.l', '||drift_5::0', '||drift_1::1', 'corrector.d.l',
       '||drift_2::1', 'mq.d.l', '||drift_3::1', 'ms.d.l', '||drift_4::0',
       '||drift_4::1', 'ms.d.r', '||drift_3::2', 'mq.d.r', '||drift_2::2',
       'corrector.d.r', '||drift_1::2', '||drift_5::1', 'mb.1.r',
       '||drift_6::2', 'mb.2.r', '||drift_6::3', 'mb.3.r', '||drift_7::1',
       '||drift_1::3', 'corrector.f.r', '||drift_2::3', 'mq.f.r',
       '||drift_3::3', 'ms.f.r', '||drift_8::1', 'mid.r', 'end',
       '_end_point']))
    assert np.all(tt_cell.element_type == np.array(
        ['Marker', 'Marker', 'Drift', 'Sextupole', 'Drift', 'Quadrupole',
        'Drift', 'Multipole', 'Drift', 'Drift', 'Bend', 'Drift', 'Bend',
        'Drift', 'Bend', 'Drift', 'Drift', 'Multipole', 'Drift',
        'Quadrupole', 'Drift', 'Sextupole', 'Drift', 'Drift', 'Sextupole',
        'Drift', 'Quadrupole', 'Drift', 'Multipole', 'Drift', 'Drift',
        'Bend', 'Drift', 'Bend', 'Drift', 'Bend', 'Drift', 'Drift',
        'Multipole', 'Drift', 'Quadrupole', 'Drift', 'Sextupole', 'Drift',
        'Marker', 'Marker', '']))
    assert np.all(tt_cell.isreplica == np.array(
        [False,  True, False,  True, False,  True, False,  True, False,
       False,  True, False,  True, False,  True, False, False,  True,
       False,  True, False,  True, False, False,  True, False,  True,
       False,  True, False, False,  True, False,  True, False,  True,
       False, False,  True, False,  True, False,  True, False,  True,
       False, False]))

    tt_cell_stripped = tt_cell.rows[1:-2] # Remove _end_point and markers added in cell
    tt_cell_second_half = tt_cell_stripped.rows[len(tt_cell_stripped)//2 :]
    tt_cell_second_half.s_center -= tt_cell_second_half.s[0]
    tt_hc_stripped = tt_hc.rows[:-1] # Remove _end_point
    xo.assert_allclose(tt_cell_second_half.s_center, tt_hc_stripped.s_center, atol=5e-14, rtol=0)
    tt_cell_first_half = tt_cell_stripped.rows[:len(tt_cell_stripped)//2]
    s_center_mirrored_first_half = (
        tt_cell_stripped['s', len(tt_cell_stripped)//2] - tt_cell_first_half.s_center[::-1])
    xo.assert_allclose(s_center_mirrored_first_half, tt_hc_stripped.s_center, atol=5e-14, rtol=0)

    env.vars({
        'kqf.ss': 0.027 / 2,
        'kqd.ss': -0.0271 / 2,
    })

    halfcell_ss = env.new_line(components=[

        env.new('mid.ss', xt.Marker, at='l.halfcell'),

        env.new('mq.ss.d', 'mq', k1='kqd.ss', at = '0.5 + l.mq / 2'),
        env.new('mq.ss.f', 'mq', k1='kqf.ss', at = 'l.halfcell - l.mq / 2 - 0.5'),

        env.new('corrector.ss.v', 'corrector', at=0.75, from_='mq.ss.d'),
        env.new('corrector.ss.h', 'corrector', at=-0.75, from_='mq.ss.f')
    ])

    hcell_left_ss = halfcell_ss.replicate(suffix='l', mirror=True)
    hcell_right_ss = halfcell_ss.replicate(suffix='r')
    cell_ss = env.new_builder()
    cell_ss.new('start.ss', xt.Marker)
    cell_ss.place(hcell_left_ss)
    cell_ss.place(hcell_right_ss)
    cell_ss.new('end.ss', xt.Marker)
    cell_ss = cell_ss.build()

    arc = env.new_builder()
    arc.new('cell.1', cell, mode='replica')
    arc.new('cell.2', cell, mode='replica')
    arc.new('cell.3', cell, mode='replica')
    arc = arc.build()

    assert 'cell.2' in env.lines
    tt_cell2 = env.lines['cell.2'].get_table(attr=True)
    assert np.all([nn for nn in tt_cell2.name[:-1] if not nn.startswith('||drift')]
                  == np.array([nn + '.cell.2' for nn in tt_cell.name[:-1] if not nn.startswith('||drift')]))
    assert np.all(tt_cell2.s == tt_cell.s)
    tt_cell2_nodrift = tt_cell2.rows[~tt_cell2.rows.mask[r'\|\|drift.*']]
    assert tt_cell2_nodrift.isreplica[:-1].all()
    assert tt_cell2['parent_name', 'mq.d.l.cell.2'] == 'mq.d.l'
    assert tt_cell2['parent_name', 'mq.f.l.cell.2'] == 'mq.f.l'
    assert tt_cell['parent_name', 'mq.d.l'] == 'mq.d'
    assert tt_cell['parent_name', 'mq.f.l'] == 'mq.f'

    tt_arc = arc.get_table(attr=True)
    assert len(tt_arc) == 3 * (len(tt_cell)-1) + 1
    n_cell = len(tt_cell) - 1
    assert np.all(tt_arc.env_name[n_cell:2*n_cell] == tt_cell2.env_name[:-1])
    for nn in tt_cell2.env_name[:-1]:
        assert arc.get(nn) is env.get(nn)
        assert arc.get(nn) is env['cell.2'].get(nn)

    ss = env.new_builder()
    ss.new('ss.cell.1', cell_ss, mode='replica')
    ss.new('ss.cell.2', cell_ss, mode='replica')
    ss = ss.build()

    ring = env.new_builder()
    ring.new('arc.1', arc, mode='replica')
    ring.new('ss.1', ss, mode='replica')
    ring.new('arc.2', arc, mode='replica')
    ring.new('ss.2', ss, mode='replica')
    ring.new('arc.3', arc, mode='replica')
    ring.new('ss.3', ss, mode='replica')
    ring = ring.build()

    tt_ring = ring.get_table(attr=True)
    # Check length
    xo.assert_allclose(tt_ring.s[-1], 2*l_hc * (n_cells_par_arc * n_arcs + 2*n_arcs),
                        atol=1e-12, rtol=0)
    # Check closure
    sv_ring = ring.survey()
    xo.assert_allclose(sv_ring.X[-1], 0, atol=1e-12, rtol=0)
    xo.assert_allclose(sv_ring.Y[-1], 0, atol=1e-12, rtol=0)
    xo.assert_allclose(sv_ring.Z[-1], 0, atol=1e-12, rtol=0)

    xo.assert_allclose(sv_ring.angle.sum(), 2*np.pi, atol=1e-12, rtol=0)

    ## Insertion

    env.vars({
        'k1.q1': 0.025,
        'k1.q2': -0.025,
        'k1.q3': 0.025,
        'k1.q4': -0.02,
        'k1.q5': 0.025,
    })

    half_insertion = env.new_builder()
    # Start-end markers
    half_insertion.new('ip', xt.Marker)
    half_insertion.new('e.insertion', xt.Marker, at=76)
    # Quads
    half_insertion.new('mq.1', xt.Quadrupole, k1='k1.q1', length='l.mq', at = 20)
    half_insertion.new('mq.2', xt.Quadrupole, k1='k1.q2', length='l.mq', at = 25)
    half_insertion.new('mq.3', xt.Quadrupole, k1='k1.q3', length='l.mq', at=37)
    half_insertion.new('mq.4', xt.Quadrupole, k1='k1.q4', length='l.mq', at=55)
    half_insertion.new('mq.5', xt.Quadrupole, k1='k1.q5', length='l.mq', at=73)
    # Dipole correctors (will use h and v on the same corrector)
    half_insertion.new('corrector.ss.1', 'corrector', at=0.75, from_='mq.1')
    half_insertion.new('corrector.ss.2', 'corrector', at=-0.75, from_='mq.2')
    half_insertion.new('corrector.ss.3', 'corrector', at=0.75, from_='mq.3')
    half_insertion.new('corrector.ss.4', 'corrector', at=-0.75, from_='mq.4')
    half_insertion.new('corrector.ss.5', 'corrector', at=0.75, from_='mq.5')
    half_insertion = half_insertion.build()

    insertion = env.new_builder()
    insertion.new('l', half_insertion, mode='replica', mirror=True)
    insertion.new('r', half_insertion, mode='replica')
    insertion = insertion.build()

    ring2 = env.new_builder()
    ring2.place(env['arc.1'])
    ring2.place(env['ss.1'])
    ring2.place(env['arc.2'])
    ring2.place(insertion)
    ring2.place(env['arc.3'])
    ring2.place(env['ss.3'])
    ring2 = ring2.build()

    select_whole = ring2.select()
    assert select_whole.env is ring2.env
    assert select_whole._element_dict is ring2._element_dict
    assert np.all(np.array(select_whole.element_names)
                  == np.array(ring2.element_names))

    shallow_copy = ring2.copy(shallow=True)
    assert shallow_copy.env is ring2.env
    assert shallow_copy._element_dict is ring2._element_dict
    assert np.all(np.array(shallow_copy.element_names)
                    == np.array(ring2.element_names))

    # # Check buffer behavior
    ring2_sliced = ring2.select()
    ring2_sliced.cut_at_s(np.arange(0, ring2.get_length(), 0.5))

    opt = cell.match(
        method='4d',
        vary=xt.VaryList(['kqf', 'kqd'], step=1e-5),
        targets=xt.TargetSet(
            qx=0.333333,
            qy=0.333333,
        ))
    tw_cell = cell.twiss4d()

    opt = cell_ss.match(
        solve=False,
        method='4d',
        vary=xt.VaryList(['kqf.ss', 'kqd.ss'], step=1e-5),
        targets=xt.TargetSet(
            betx=tw_cell.betx[-1], bety=tw_cell.bety[-1], at='start.ss',
        ))
    opt.solve()

    tw_arc = arc.twiss4d()

    opt = half_insertion.match(
        solve=False,
        betx=tw_arc.betx[0], bety=tw_arc.bety[0],
        alfx=tw_arc.alfx[0], alfy=tw_arc.alfy[0],
        init_at='e.insertion',
        start='ip', end='e.insertion',
        vary=xt.VaryList(['k1.q1', 'k1.q2', 'k1.q3', 'k1.q4'], step=1e-5),
        targets=[
            xt.TargetSet(alfx=0, alfy=0, at='ip'),
            xt.Target(lambda tw: tw.betx[0] - tw.bety[0], 0),
            xt.Target(lambda tw: tw.betx.max(), xt.LessThan(400)),
            xt.Target(lambda tw: tw.bety.max(), xt.LessThan(400)),
            xt.Target(lambda tw: tw.betx.min(), xt.GreaterThan(2)),
            xt.Target(lambda tw: tw.bety.min(), xt.GreaterThan(2)),
        ]
    )
    opt.step(40)
    opt.solve()

    # Check that the cell is matched to the rest of the ring
    tw = ring.twiss4d()
    tw_cell_from_ring = tw.rows['start.cell.3.arc.2':'end.cell.3.arc.2']
    xo.assert_allclose(tw_cell_from_ring.betx, tw_cell.betx[:-1], atol=0, rtol=5e-4)
    xo.assert_allclose(tw_cell_from_ring.bety, tw_cell.bety[:-1], atol=0, rtol=5e-4)

    tw2 = ring2.twiss4d()
    tw_cell_from_ring2 = tw2.rows['start.cell.3.arc.2':'end.cell.3.arc.2']
    xo.assert_allclose(tw_cell_from_ring2.betx, tw_cell.betx[:-1], atol=0, rtol=5e-4)
    xo.assert_allclose(tw_cell_from_ring2.bety, tw_cell.bety[:-1], atol=0, rtol=5e-4)

    # Check select
    cell3_select = ring2.select(start='start.cell.3.arc.2', end='end.cell.3.arc.2',
                                name='cell3_copy')
    assert 'cell3_copy' in env.lines
    assert env.lines['cell3_copy'] is cell3_select
    assert cell3_select._element_dict is env._element_dict
    assert cell3_select.element_names[0] == 'start.cell.3.arc.2'
    assert cell3_select.element_names[-1] == 'end.cell.3.arc.2'
    assert (np.array(cell3_select.element_names) == np.array(
        tw.rows['start.cell.3.arc.2':'end.cell.3.arc.2'].env_name)).all()

    # Check that they share the _element_dict
    assert cell._element_dict is env._element_dict
    assert halfcell._element_dict is env._element_dict
    assert halfcell_ss._element_dict is env._element_dict
    assert cell_ss._element_dict is env._element_dict
    assert insertion._element_dict is env._element_dict
    assert ring2._element_dict is env._element_dict

    cell3_select.twiss4d()

    tw2_slice = ring2_sliced.twiss4d()
    xo.assert_allclose(tw2_slice['betx', 'ip.l'], tw2['betx', 'ip.l'], atol=0, rtol=5e-4)
    xo.assert_allclose(tw2_slice['bety', 'ip.l'], tw2['bety', 'ip.l'], atol=0, rtol=5e-4)
    xo.assert_allclose(tw2_slice['alfx', 'ip.l'], 0, atol=1e-6, rtol=0)
    xo.assert_allclose(tw2_slice['alfy', 'ip.l'], 0, atol=1e-6, rtol=0)
    xo.assert_allclose(tw2_slice['dx', 'ip.l'], 0, atol=1e-4, rtol=0)
    xo.assert_allclose(tw2_slice['dpx', 'ip.l'], 0, atol=1e-6, rtol=0)
    xo.assert_allclose(tw2_slice['dy', 'ip.l'], 0, atol=1e-4, rtol=0)
    xo.assert_allclose(tw2_slice['dpy', 'ip.l'], 0, atol=1e-6, rtol=0)

    # import matplotlib.pyplot as plt
    # plt.close('all')
    # for ii, rr in enumerate([ring, ring2_sliced]):

    #     ttww = rr.twiss4d()

    #     fig = plt.figure(ii, figsize=(6.4*1.2, 4.8))
    #     ax1 = fig.add_subplot(2, 1, 1)
    #     pltbet = ttww.plot('betx bety', ax=ax1)
    #     ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    #     pltdx = ttww.plot('dx', ax=ax2)
    #     fig.subplots_adjust(right=.85)
    #     pltbet.move_legend(1.2,1)
    #     pltdx.move_legend(1.2,1)

    # ring2.survey().plot()
    # plt.show()


def test_assemble_ring_repeated_elements():

    env = xt.Environment()
    env.particle_ref = xt.Particles(p0c=2e9)

    n_bends_per_cell = 6
    n_cells_par_arc = 3
    n_arcs = 3

    n_bends = n_bends_per_cell * n_cells_par_arc * n_arcs

    env.vars({
        'l.mq': 0.5,
        'kqf': 0.027,
        'kqd': -0.0271,
        'l.mb': 10,
        'l.ms': 0.3,
        'k2sf': 0.001,
        'k2sd': -0.001,
        'angle.mb': 2 * np.pi / n_bends,
        'k0.mb': 'angle.mb / l.mb',
        'k0l.corrector': 0,
        'k1sl.corrector': 0,
        'l.halfcell': 38,
    })

    env.new('mb', xt.Bend, length='l.mb', k0='k0.mb', angle='k0.mb * l.mb')
    env.new('mq', xt.Quadrupole, length='l.mq')
    env.new('ms', xt.Sextupole, length='l.ms')
    env.new('corrector', xt.Multipole, knl=[0], length=0.1)

    girder = env.new_line(components=[
        env.place('mq', at=1),
        env.place('ms', at=0.8, from_='mq'),
        env.place('corrector', at=-0.8, from_='mq'),
    ])

    tt_girder = girder.get_table(attr=True)
    assert np.all(tt_girder.name == np.array(
        ['||drift_1', 'corrector', '||drift_2', 'mq', '||drift_3', 'ms',
       '_end_point']))
    tt_girder['s_center'] = tt_girder['s'] + \
        tt_girder['length']/2 * np.float64(tt_girder['isthick'])
    xo.assert_allclose(tt_girder['s_center', 'mq'], 1., atol=1e-14, rtol=0)
    xo.assert_allclose(tt_girder['s_center', 'ms'] - tt_girder['s_center', 'mq'], 0.8,
                    atol=1e-14, rtol=0)
    xo.assert_allclose(
        tt_girder['s_center', 'corrector'] - tt_girder['s_center', 'mq'], -0.8,
        atol=1e-14, rtol=0)


    girder_f = girder.clone(suffix='f')
    girder_d = girder.clone(suffix='d', mirror=True)
    env.set('mq.f', k1='kqf')
    env.set('mq.d', k1='kqd')

    # Check clone
    tt_girder_f = girder_f.get_table(attr=True)
    assert (~(tt_girder_f.isreplica)).all()
    assert np.all(tt_girder_f.name == np.array(
        ['||drift_1', 'corrector.f', '||drift_2', 'mq.f', '||drift_3',
       'ms.f', '_end_point']))
    tt_girder_f['s_center'] = (tt_girder_f['s']
                            + tt_girder_f['length']/2 * np.float64(tt_girder_f['isthick']))
    xo.assert_allclose(tt_girder_f['s_center', 'mq.f'], 1., atol=1e-14, rtol=0)
    xo.assert_allclose(tt_girder_f['s_center', 'ms.f'] - tt_girder_f['s_center', 'mq.f'], 0.8,
                    atol=1e-14, rtol=0)
    xo.assert_allclose(
        tt_girder_f['s_center', 'corrector.f'] -
        tt_girder_f['s_center', 'mq.f'], -0.8,
        atol=1e-14, rtol=0)

    # Check clone mirror
    tt_girder_d = girder_d.get_table(attr=True)
    assert (~(tt_girder_d.isreplica)).all()
    len_girder = tt_girder_d.s[-1]
    assert np.all(tt_girder_d.name == np.array(
        ['ms.d', '||drift_3', 'mq.d', '||drift_2', 'corrector.d',
       '||drift_1', '_end_point']))
    tt_girder_d['s_center'] = (tt_girder_d['s']
                            + tt_girder_d['length']/2 * np.float64(tt_girder_d['isthick']))
    xo.assert_allclose(tt_girder_d['s_center', 'mq.d'],
                    len_girder - 1., atol=1e-14, rtol=0)
    xo.assert_allclose(tt_girder_d['s_center', 'ms.d'] - tt_girder_d['s_center', 'mq.d'],
                    -0.8, atol=1e-14, rtol=0)
    xo.assert_allclose(tt_girder_d['s_center', 'corrector.d'] - tt_girder_d['s_center', 'mq.d'],
                    0.8, atol=1e-14, rtol=0)


    halfcell = env.new_line(components=[

        # End of the half cell (will be mid of the cell)
        env.new('mid', xt.Marker, at='l.halfcell'),

        # Bends
        env.new('mb.2', 'mb', at='l.halfcell / 2'),
        env.new('mb.1', 'mb', at='-l.mb - 1', from_='mb.2'),
        env.new('mb.3', 'mb', at='l.mb + 1', from_='mb.2'),

        # Quadrupoles, sextupoles and correctors
        env.place(girder_d, at=1.2),
        env.place(girder_f, at='l.halfcell - 1.2'),

    ])

    l_hc = env.vv['l.halfcell']
    xo.assert_allclose(l_hc, l_hc, atol=1e-14, rtol=0)
    tt_hc = halfcell.get_table(attr=True)
    assert np.all(tt_hc.name == np.array(
        ['||drift_4', 'ms.d', '||drift_3::0', 'mq.d', '||drift_2::0',
       'corrector.d', '||drift_1::0', '||drift_5', 'mb.1', '||drift_6::0',
       'mb.2', '||drift_6::1', 'mb.3', '||drift_7', '||drift_1::1',
       'corrector.f', '||drift_2::1', 'mq.f', '||drift_3::1', 'ms.f',
       '||drift_8', 'mid', '_end_point']))
    assert np.all(tt_hc.element_type == np.array(
        ['Drift', 'Sextupole', 'Drift', 'Quadrupole', 'Drift', 'Multipole',
        'Drift', 'Drift', 'Bend', 'Drift', 'Bend', 'Drift', 'Bend',
        'Drift', 'Drift', 'Multipole', 'Drift', 'Quadrupole', 'Drift',
        'Sextupole', 'Drift', 'Marker', '']))
    assert np.all(tt_hc.isreplica == False)
    tt_hc['s_center'] = (
        tt_hc['s'] + tt_hc['length'] / 2 * np.float64(tt_hc['isthick']))
    xo.assert_allclose(tt_hc['s_center', 'mq.d'],
                    1.2 - tt_girder_d.s[-1] / 2 +
                    tt_girder_d['s_center', 'mq.d'],
                    atol=1e-14, rtol=0)
    xo.assert_allclose(tt_hc['s_center', 'ms.f'] - tt_hc['s_center', 'mq.f'], 0.8,
                    atol=1e-14, rtol=0)
    xo.assert_allclose(
        tt_hc['s_center', 'corrector.f'] - tt_hc['s_center', 'mq.f'], -0.8,
        atol=1e-14, rtol=0)
    xo.assert_allclose(tt_hc['s_center', 'ms.d'] - tt_hc['s_center', 'mq.d'],
                    -0.8, atol=1e-14, rtol=0)
    xo.assert_allclose(tt_hc['s_center', 'corrector.d'] - tt_hc['s_center', 'mq.d'],
                    0.8, atol=1e-14, rtol=0)
    xo.assert_allclose(tt_hc['s_center', 'mb.2'], l_hc / 2, atol=1e-14, rtol=0)
    xo.assert_allclose(tt_hc['s_center', 'mb.1'], tt_hc['s_center', 'mb.2'] - env.vv['l.mb'] - 1,
                        atol=1e-14, rtol=0)
    xo.assert_allclose(tt_hc['s_center', 'mb.3'], tt_hc['s_center', 'mb.2'] + env.vv['l.mb'] + 1,
                    atol=1e-14, rtol=0)

    cell = -halfcell + halfcell
    env['cell'] = cell
    assert 'cell' in env.lines
    assert env['cell'] is cell
    assert cell in env._lines_weakrefs

    tt_cell = cell.get_table(attr=True)
    tt_cell['s_center'] = (
        tt_cell['s'] + tt_cell['length'] / 2 * np.float64(tt_cell['isthick']))
    assert np.all(tt_cell.env_name == np.array(
        ['mid', '||drift_8', 'ms.f', '||drift_3', 'mq.f', '||drift_2',
       'corrector.f', '||drift_1', '||drift_7', 'mb.3', '||drift_6',
       'mb.2', '||drift_6', 'mb.1', '||drift_5', '||drift_1',
       'corrector.d', '||drift_2', 'mq.d', '||drift_3', 'ms.d',
       '||drift_4', '||drift_4', 'ms.d', '||drift_3', 'mq.d', '||drift_2',
       'corrector.d', '||drift_1', '||drift_5', 'mb.1', '||drift_6',
       'mb.2', '||drift_6', 'mb.3', '||drift_7', '||drift_1',
       'corrector.f', '||drift_2', 'mq.f', '||drift_3', 'ms.f',
       '||drift_8', 'mid', '_end_point']))
    assert np.all(tt_cell.element_type == np.array(
        ['Marker', 'Drift', 'Sextupole', 'Drift', 'Quadrupole', 'Drift',
       'Multipole', 'Drift', 'Drift', 'Bend', 'Drift', 'Bend', 'Drift',
       'Bend', 'Drift', 'Drift', 'Multipole', 'Drift', 'Quadrupole',
       'Drift', 'Sextupole', 'Drift', 'Drift', 'Sextupole', 'Drift',
       'Quadrupole', 'Drift', 'Multipole', 'Drift', 'Drift', 'Bend',
       'Drift', 'Bend', 'Drift', 'Bend', 'Drift', 'Drift', 'Multipole',
       'Drift', 'Quadrupole', 'Drift', 'Sextupole', 'Drift', 'Marker', '']))
    assert np.all(tt_cell.isreplica == False)

    tt_cell_stripped = tt_cell.rows[:-1]
    tt_cell_second_half = tt_cell_stripped.rows[len(tt_cell_stripped)//2 :]
    tt_cell_second_half.s_center -= tt_cell_second_half.s[0]
    tt_hc_stripped = tt_hc.rows[:-1] # Remove _end_point
    xo.assert_allclose(tt_cell_second_half.s_center, tt_hc_stripped.s_center, atol=5e-14, rtol=0)
    tt_cell_first_half = tt_cell_stripped.rows[:len(tt_cell_stripped)//2]
    s_center_mirrored_first_half = (
        tt_cell_stripped['s', len(tt_cell_stripped)//2] - tt_cell_first_half.s_center[::-1])
    xo.assert_allclose(s_center_mirrored_first_half, tt_hc_stripped.s_center, atol=5e-14, rtol=0)

    env.vars({
        'kqf.ss': 0.027 / 2,
        'kqd.ss': -0.0271 / 2,
    })

    halfcell_ss = env.new_line(components=[

        env.new('mid.ss', xt.Marker, at='l.halfcell'),

        env.new('mq.ss.d', 'mq', k1='kqd.ss', at = '0.5 + l.mq / 2'),
        env.new('mq.ss.f', 'mq', k1='kqf.ss', at = 'l.halfcell - l.mq / 2 - 0.5'),

        env.new('corrector.ss.v', 'corrector', at=0.75, from_='mq.ss.d'),
        env.new('corrector.ss.h', 'corrector', at=-0.75, from_='mq.ss.f')
    ])


    cell_ss = env.new_line(components=[
        env.new('start.ss', xt.Marker),
        -halfcell_ss,
        halfcell_ss,
        env.new('end.ss', xt.Marker),
    ])


    arc = 3 * cell

    tt_arc = arc.get_table(attr=True)
    assert len(tt_arc) == 3 * (len(tt_cell)-1) + 1
    n_cell = len(tt_cell) - 1
    assert np.all(tt_arc.env_name[n_cell:2*n_cell] == tt_cell.env_name[:-1])
    for nn in tt_cell.env_name[:-1]:
        assert arc.get(nn) is env.get(nn)
        assert arc.get(nn) is env['cell'].get(nn)

    ss = 2 * cell_ss

    ring = 3 * (arc + ss)
    tt_ring = ring.get_table(attr=True)
    # Check length
    xo.assert_allclose(tt_ring.s[-1], 2*l_hc * (n_cells_par_arc * n_arcs + 2*n_arcs),
                        atol=1e-12, rtol=0)
    # Check closure
    sv_ring = ring.survey()
    xo.assert_allclose(sv_ring.X[-1], 0, atol=1e-12, rtol=0)
    xo.assert_allclose(sv_ring.Y[-1], 0, atol=1e-12, rtol=0)
    xo.assert_allclose(sv_ring.Z[-1], 0, atol=1e-12, rtol=0)

    xo.assert_allclose(sv_ring.angle.sum(), 2*np.pi, atol=1e-12, rtol=0)

    ## Insertion

    env.vars({
        'k1.q1': 0.025,
        'k1.q2': -0.025,
        'k1.q3': 0.025,
        'k1.q4': -0.02,
        'k1.q5': 0.025,
    })

    half_insertion = env.new_line(components=[

        # Start-end markers
        env.new('ip', xt.Marker),
        env.new('e.insertion', xt.Marker, at=76),

        # Quads
        env.new('mq.1', xt.Quadrupole, k1='k1.q1', length='l.mq', at = 20),
        env.new('mq.2', xt.Quadrupole, k1='k1.q2', length='l.mq', at = 25),
        env.new('mq.3', xt.Quadrupole, k1='k1.q3', length='l.mq', at=37),
        env.new('mq.4', xt.Quadrupole, k1='k1.q4', length='l.mq', at=55),
        env.new('mq.5', xt.Quadrupole, k1='k1.q5', length='l.mq', at=73),

        # Dipole correctors (will use h and v on the same corrector)
        env.new('corrector.ss.1', 'corrector', at=0.75, from_='mq.1'),
        env.new('corrector.ss.2', 'corrector', at=-0.75, from_='mq.2'),
        env.new('corrector.ss.3', 'corrector', at=0.75, from_='mq.3'),
        env.new('corrector.ss.4', 'corrector', at=-0.75, from_='mq.4'),
        env.new('corrector.ss.5', 'corrector', at=0.75, from_='mq.5'),

    ])

    insertion = -half_insertion + half_insertion

    ring2 = arc + ss + arc + insertion + arc + ss

    opt = cell.match(
        method='4d',
        vary=xt.VaryList(['kqf', 'kqd'], step=1e-5),
        targets=xt.TargetSet(
            qx=0.333333,
            qy=0.333333,
        ))
    tw_cell = cell.twiss4d()

    opt = cell_ss.match(
        solve=False,
        method='4d',
        vary=xt.VaryList(['kqf.ss', 'kqd.ss'], step=1e-5),
        targets=xt.TargetSet(
            betx=tw_cell.betx[-1], bety=tw_cell.bety[-1], at='start.ss',
        ))
    opt.solve()

    tw_arc = arc.twiss4d()

    opt = half_insertion.match(
        solve=False,
        betx=tw_arc.betx[0], bety=tw_arc.bety[0],
        alfx=tw_arc.alfx[0], alfy=tw_arc.alfy[0],
        init_at='e.insertion',
        start='ip', end='e.insertion',
        vary=xt.VaryList(['k1.q1', 'k1.q2', 'k1.q3', 'k1.q4'], step=1e-5),
        targets=[
            xt.TargetSet(alfx=0, alfy=0, at='ip'),
            xt.Target(lambda tw: tw.betx[0] - tw.bety[0], 0),
            xt.Target(lambda tw: tw.betx.max(), xt.LessThan(400)),
            xt.Target(lambda tw: tw.bety.max(), xt.LessThan(400)),
            xt.Target(lambda tw: tw.betx.min(), xt.GreaterThan(2)),
            xt.Target(lambda tw: tw.bety.min(), xt.GreaterThan(2)),
        ]
    )
    opt.step(40)
    opt.solve()
    tw_half_insertion = opt.actions[0].run()

    # Check that the cell is matched to the rest of the ring
    tw_ring = ring.twiss4d()
    xo.assert_allclose(tw_ring.betx[0], tw_cell.betx[0], atol=0, rtol=5e-4)
    xo.assert_allclose(tw_ring.bety[0], tw_cell.bety[0], atol=0, rtol=5e-4)
    tw_ring2 = ring2.twiss4d()
    xo.assert_allclose(tw_ring2.betx[0], tw_cell.betx[0], atol=0, rtol=5e-4)
    xo.assert_allclose(tw_ring2.bety[0], tw_cell.bety[0], atol=0, rtol=5e-4)

    # Check that they share the _element_dict
    assert cell._element_dict is env._element_dict
    assert halfcell._element_dict is env._element_dict
    assert halfcell_ss._element_dict is env._element_dict
    assert cell_ss._element_dict is env._element_dict
    assert insertion._element_dict is env._element_dict
    assert ring2._element_dict is env._element_dict

    xo.assert_allclose(tw_ring2['betx', 'ip::0'], tw_half_insertion['betx', 'ip'], atol=0, rtol=5e-4)
    xo.assert_allclose(tw_ring2['bety', 'ip::0'], tw_half_insertion['bety', 'ip'], atol=0, rtol=5e-4)
    xo.assert_allclose(tw_ring2['alfx', 'ip::0'], 0, atol=1e-6, rtol=0)
    xo.assert_allclose(tw_ring2['alfy', 'ip::0'], 0, atol=1e-6, rtol=0)
    xo.assert_allclose(tw_ring2['dx', 'ip::0'], 0, atol=1e-4, rtol=0)
    xo.assert_allclose(tw_ring2['dpx', 'ip::0'], 0, atol=1e-6, rtol=0)
    xo.assert_allclose(tw_ring2['dy', 'ip::0'], 0, atol=1e-4, rtol=0)
    xo.assert_allclose(tw_ring2['dpy', 'ip::0'], 0, atol=1e-6, rtol=0)

    # Check a line with the same marker at start and end
    assert arc.element_names[0] == 'mid'
    assert arc.element_names[-1] == 'mid'
    twarc = arc.twiss4d()
    xo.assert_allclose(twarc.s[0], 0, atol=1e-12, rtol=0)
    xo.assert_allclose(twarc.s[-1], 228, atol=1e-10, rtol=0)
    twarc_start_end = arc.twiss4d(start=xt.START, end=xt.END, init=twarc)
    xo.assert_allclose(twarc_start_end.betx, twarc.betx, atol=1e-11, rtol=0)

    tw_one_cell_ref = twarc.rows['mid::2':'mid::3']
    tw_one_cell = arc.twiss4d(start='mid::2', end='mid::3', init='periodic')
    tw_one_cell_stripped = tw_one_cell.rows[:-1] # remove _end_point
    xo.assert_allclose(tw_one_cell_stripped.betx, tw_one_cell_ref.betx, atol=0, rtol=5e-4)

    cell_selected = arc.select(start='mid::2', end='mid::3')
    tw_cell_selected = cell_selected.twiss4d()
    tw_cell_selected_stripped = tw_cell_selected.rows[:-1] # remove _end_point
    xo.assert_allclose(tw_cell_selected_stripped.betx, tw_one_cell_ref.betx, atol=0, rtol=5e-4)

    tt_ring2 = ring2.get_table(attr=True)
    assert tt_ring2['name', 39] == 'mq.f::1'
    assert tt_ring2['name', 48] == 'mq.f::2'
    assert ring2.element_names[39] == 'mq.f'
    assert ring2.element_names[48] == 'mq.f'

    ring2.replace_all_repeated_elements()
    tt_ring2_after = ring2.get_table(attr=True)
    assert tt_ring2_after['name', 39] == 'mq.f.1'
    assert tt_ring2_after['name', 48] == 'mq.f.2'
    assert ring2.element_names[39] == 'mq.f.1'
    assert ring2.element_names[48] == 'mq.f.2'
    assert str(ring2.ref['mq.f.1'].k1._expr) == "vars['kqf']"
    assert str(ring2.ref['mq.f.2'].k1._expr) == "vars['kqf']"
    assert ring2.get('mq.f.1') is not ring2.get('mq.f.2')

    # import matplotlib.pyplot as plt
    # plt.close('all')
    # for ii, rr in enumerate([ring, ring2_sliced]):

    #     ttww = rr.twiss4d()

    #     fig = plt.figure(ii, figsize=(6.4*1.2, 4.8))
    #     ax1 = fig.add_subplot(2, 1, 1)
    #     pltbet = ttww.plot('betx bety', ax=ax1)
    #     ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    #     pltdx = ttww.plot('dx', ax=ax2)
    #     fig.subplots_adjust(right=.85)
    #     pltbet.move_legend(1.2,1)
    #     pltdx.move_legend(1.2,1)

    # ring2.survey().plot()
    # plt.show()

@pytest.mark.parametrize('container_type', ['env', 'line'])
def test_element_views(container_type):

    env = xt.Environment()
    line = env.new_line()

    if container_type == 'env':
        ee = env
    elif container_type == 'line':
        ee = line

    ee['a']  = 3.
    ee['b1']  = 3 * ee['a'] # done by value
    ee['b2']  = 3 * ee.ref['a'] # done by reference
    ee['c']  = '4 * a'

    assert isinstance(ee['a'], float)
    assert isinstance(ee['b1'], float)
    assert isinstance(ee['b2'], float)
    assert isinstance(ee['c'], float)

    assert ee['a'] == 3
    assert ee['b1'] == 9
    assert ee['b2'] == 9
    assert ee['c'] == 12

    assert ee.ref['a']._value == 3
    assert ee.ref['b1']._value == 9
    assert ee.ref['b2']._value == 9
    assert ee.ref['c']._value == 12

    assert ee.get('a') == 3
    assert ee.get('b1') == 9
    assert ee.get('b2') == 9
    assert ee.get('c') == 12

    env.new('mb', 'Bend', extra={'kmax': '6*a'},
            k1='3*a', angle=1e-3 * 4*ee.ref['a'], knl=[0, '5*a', 6*ee.ref['a']])
    assert isinstance(ee['mb'].k1, float)
    assert isinstance(ee['mb'].angle, float)
    assert isinstance(ee['mb'].knl[0], float)
    assert ee['mb'].k1 == 9
    assert ee['mb'].angle == 12e-3
    assert ee['mb'].knl[0] == 0
    assert ee['mb'].knl[1] == 15
    assert ee['mb'].knl[2] == 18

    ee['a'] = 4
    assert ee['a'] == 4
    assert ee['b1'] == 9
    assert ee['b2'] == 12
    assert ee['c'] == 16
    assert ee['mb'].k1 == 12
    assert ee['mb'].angle == 16e-3
    assert ee['mb'].knl[0] == 0
    assert ee['mb'].knl[1] == 20
    assert ee['mb'].knl[2] == 24

    ee['mb'].k1 = '30*a'
    ee['mb'].angle = 1e-3* 40 * ee.ref['a']
    ee['mb'].knl[1] = '50*a'
    ee['mb'].knl[2] = 60 * ee.ref['a']
    assert ee['mb'].k1 == 120
    assert ee['mb'].angle == 160e-3
    assert ee['mb'].knl[0] == 0
    assert ee['mb'].knl[1] == 200
    assert ee['mb'].knl[2] == 240

    assert isinstance(ee['mb'].k1, float)
    assert isinstance(ee['mb'].h, float)
    assert isinstance(ee['mb'].knl[0], float)

    assert ee.ref['mb'].k1._value == 120
    assert ee.ref['mb'].angle._value == 160e-3
    assert ee.ref['mb'].knl[0]._value == 0
    assert ee.ref['mb'].knl[1]._value == 200
    assert ee.ref['mb'].knl[2]._value == 240

    assert ee.get('mb').k1 == 120
    assert ee.get('mb').angle == 160e-3
    assert ee.get('mb').knl[0] == 0
    assert ee.get('mb').knl[1] == 200
    assert ee.get('mb').knl[2] == 240

    # Some interesting behavior
    assert type(ee['mb']) is xt.view.View
    assert ee['mb'].__class__ is xt.Bend
    assert isinstance(ee['mb'], xt.Bend)
    assert type(ee.ref['mb']._value) is xt.Bend
    assert type(ee.get('mb')) is xt.Bend

def test_env_new():

    env = xt.Environment()
    env['a'] = 3.

    env.new('m', xt.Bend, k0='3*a')
    assert isinstance(env['m'], xt.Bend)

    env.new('m1', 'm', mode='replica')
    assert isinstance(env['m1'], xt.Replica)
    assert env.get('m1').parent_name == 'm'

    env.new('m2', 'm', mode='clone')
    assert isinstance(env['m2'], xt.Bend)
    str(env.ref['m2'].k0._expr) == "(3.0 * vars['a'])"

    ret = env.new('mm', xt.Bend, k0='3*a', at='4*a', from_='m')
    assert isinstance(ret, xt.Place)
    assert ret.name == 'mm'
    assert ret.at == '4*a'
    assert ret.from_ == 'm'
    assert isinstance(env['mm'], xt.Bend)

    ret = env.new('mm1', 'mm', mode='replica', at='5*a', from_='m1')
    assert isinstance(ret,xt.Place)
    assert isinstance(env['mm1'], xt.Replica)
    assert ret.name == 'mm1'
    assert ret.at == '5*a'
    assert ret.from_ == 'm1'
    assert env['mm1'].parent_name == 'mm'

    ret = env.new('mm2', 'mm', mode='clone', at='6*a', from_='m2')
    assert isinstance(ret,xt.Place)
    assert isinstance(env['mm2'], xt.Bend)
    assert ret.name == 'mm2'
    assert ret.at == '6*a'
    assert ret.from_ == 'm2'
    assert str(env.ref['mm2'].k0._expr) == "(3.0 * vars['a'])"

    env.new('e1', xt.Bend, k0='3*a')
    env.new('e2', xt.Bend)
    line = env.new('ll', xt.Line, components=['e1', 'e2'])
    assert isinstance(line, xt.Line)
    assert line.element_names == ['e1', 'e2']

    line = env.new('ll1', 'Line', components=['e1', 'e2'])
    assert isinstance(line, xt.Line)
    assert line.element_names == ['e1', 'e2']

    env.new('ll2', 'll') # Should be a clone
    assert isinstance(env['ll2'], xt.Line)
    assert env['ll2'].element_names == ['e1.ll2', 'e2.ll2']
    assert isinstance(env['e1.ll2'], xt.Bend)
    assert isinstance(env['e2.ll2'], xt.Bend)
    assert env.ref['e1.ll2'].k0._expr == "(3.0 * vars['a'])"

    env.new('ll3', 'll', mode='replica')
    assert isinstance(env['ll3'], xt.Line)
    assert env['ll3'].element_names == ['e1.ll3', 'e2.ll3']
    assert isinstance(env['e1.ll3'], xt.Replica)
    assert isinstance(env['e2.ll3'], xt.Replica)
    assert env['e1.ll3'].parent_name == 'e1'
    assert env['e2.ll3'].parent_name == 'e2'

    ret = env.new('ll4', 'll', at='5*a', from_='m')
    assert isinstance(ret,xt.Place)
    assert ret.at == '5*a'
    assert ret.from_ == 'm'
    assert isinstance(env['ll4'], xt.Line)
    assert env['ll4'].element_names == ['e1.ll4', 'e2.ll4']
    assert isinstance(env['e1.ll4'], xt.Bend)
    assert isinstance(env['e2.ll4'], xt.Bend)

    ret = env.new('aper', xt.LimitEllipse, a='2*a', b='a')
    assert ret == 'aper'
    assert env[ret].a == 6
    assert env[ret].b == 3


def test_neg_line():

    line = xt.Line(elements=[xt.Bend(k0=0.5), xt.Quadrupole(k1=0.1)])

    line_neg = -line

    assert line_neg.env is line.env

    assert line[0].k0 == 0.5
    assert line[1].k1 == 0.1

    assert line_neg[0].k1 == 0.1
    assert line_neg[1].k0 == 0.5

    assert line in line.env._lines_weakrefs
    assert line_neg in line.env._lines_weakrefs

def test_repeated_elements():

    env = xt.Environment()
    env.new('mb', 'Bend', length=0.5)
    pp = env.place('mb')

    line = env.new_line(components=[
        'mb',
        'mb',
        env.new('ip1', 'Marker', at=10),
        'mb',
        pp,
        env.place('mb', anchor='end', at='ip2@start'),
        env.new('ip2', 'Marker', at=20),
        'mb',
        pp
    ])

    tt = line.get_table()
    assert np.all(tt.env_name == np.array([
        'mb', 'mb', '||drift_1', 'ip1', 'mb', 'mb', '||drift_2', 'mb',
       'ip2', 'mb', 'mb', '_end_point']))
    assert np.all(tt.s == np.array([
        0. ,  0.5,  1. , 10. , 10. , 10.5, 11. , 19.5, 20. , 20. , 20.5, 21. ]))

    l1 = env.new_line(name='l1', components=[
        'mb',
        'mb',
        env.new('mid', 'Marker'),
        'mb',
        'mb',
    ])

    env['s.l1'] = 10
    l_twol1 = env.new_line(components=[
        env.new('ip', 'Marker', at=20),
        env.place('l1', at='s.l1', from_='ip'),
        env.place(l1, at=-env.ref['s.l1'], from_='ip'),
    ])
    tt_twol1 = l_twol1.get_table()
    assert np.all(tt_twol1.env_name == np.array(
        ['||drift_1', 'mb', 'mb', 'mid', 'mb', 'mb', '||drift_1', 'ip',
       '||drift_1', 'mb', 'mb', 'mid', 'mb', 'mb', '_end_point']))
    assert np.all(tt_twol1.s == np.array(
        [ 0. ,  9. ,  9.5, 10. , 10. , 10.5, 11. , 20. , 20. , 29. , 29.5,
        30. , 30. , 30.5, 31. ]))

    l_mult = env.new_line(name='l_from_list', components=[
        2 * l1,
        2 * ['mb']
    ])
    tt_mult = l_mult.get_table()
    assert np.all(tt_mult.name == np.array([
        'mb::0', 'mb::1', 'mid::0', 'mb::2', 'mb::3', 'mb::4', 'mb::5',
       'mid::1', 'mb::6', 'mb::7', 'mb::8', 'mb::9', '_end_point']))
    assert np.all(tt_mult.env_name == np.array([
        'mb', 'mb', 'mid', 'mb', 'mb', 'mb', 'mb', 'mid', 'mb', 'mb', 'mb',
        'mb', '_end_point']))
    assert np.all(tt_mult.s == np.array(
        [0. , 0.5, 1. , 1. , 1.5, 2. , 2.5, 3. , 3. , 3.5, 4. , 4.5, 5. ]))

def test_select_in_multiline():

    # --- Parameters
    seq         = 'lhcb1'
    ip_name     = 'ip1'
    s_marker    = f'e.ds.l{ip_name[-1]}.b1'
    e_marker    = f's.ds.r{ip_name[-1]}.b1'
    #-------------------------------------

    collider_file = test_data_folder / 'hllhc15_collider/collider_00_from_mad.json'

    # Load the machine and select line
    collider= xt.load(collider_file)
    collider.vars['test_vars'] = 3.1416
    line   = collider[seq]
    line_sel    = line.select(s_marker,e_marker)

    assert line_sel._element_dict is line._element_dict
    assert line.get('ip1') is line_sel.get('ip1')

    line_sel['aaa'] = 1e-6
    assert line_sel['aaa'] == 1e-6
    assert line['aaa'] == 1e-6

    line_sel.ref['mcbch.7r1.b1'].knl[0] += line.ref['aaa']
    assert (str(line.ref['mcbch.7r1.b1'].knl[0]._expr)
            == "((-vars['acbch7.r1b1']) + vars['aaa'])")
    assert (str(line_sel.ref['mcbch.7r1.b1'].knl[0]._expr)
            == "((-vars['acbch7.r1b1']) + vars['aaa'])")
    assert line_sel.get('mcbch.7r1.b1').knl[0] == 1e-6
    assert line.get('mcbch.7r1.b1').knl[0] == 1e-6

@pytest.mark.parametrize('container_type', ['env', 'line'])
def test_inpection_methods(container_type):

    env = xt.Environment()

    env.vars({
        'k.1': 1.,
        'a': 2.,
        'b': '2 * a + k.1',
    })

    line = env.new_line([
        env.new('bb', xt.Bend, k0='2 * b', length=3+env.vars['a'] + env.vars['b'],
            angle=0.1, ksl=[0, '3*b']),
    ])

    ee = {'env': env, 'line': line}[container_type]

    # Line/Env methods (get, set, eval, get_expr, new_expr, info)
    assert ee.get('b') == 2 * 2 + 1
    assert ee.get('bb') is env._element_dict['bb']

    assert str(ee.get_expr('b')) == "((2.0 * vars['a']) + vars['k.1'])"

    assert ee.eval('3*a - sqrt(k.1)') == 5

    ne = ee.new_expr('sqrt(3*a + 3)')
    assert xd.refs.is_ref(ne)
    assert str(ne) == "f.sqrt(((3.0 * vars['a']) + 3.0))"

    ee.info('bb') # Check that it works
    ee.info('b')
    ee.info('a')

    ee.set('c', '6*a')
    assert ee.get('c') == 6 * 2

    # Line/Env containers (env[...], env.ref[...]
    assert ee['b'] == 2 * 2 + 1
    assert type(ee['bb']).__name__ == 'View'
    assert ee['bb'].__class__.__name__ == 'Bend'

    # Vars methods (get, set, eval, get_expr, new_expr, info, get_table)
    assert ee.vars.get('b') == 2 * 2 + 1

    assert str(ee.vars.get_expr('b')) == "((2.0 * vars['a']) + vars['k.1'])"

    assert ee.vars.eval('3*a - sqrt(k.1)') == 5

    ne = ee.vars.new_expr('sqrt(3*a + 3)')
    assert xd.refs.is_ref(ne)
    assert str(ne) == "f.sqrt(((3.0 * vars['a']) + 3.0))"

    ee.vars.info('b')
    ee.vars.info('a')

    ee.vars.set('d', '7*a')
    assert ee.vars.get('d') == 7 * 2

    assert xd.refs.is_ref(ee.vars['b'])

    # View methods get_expr, get_value, get_info, get_table (for now)
    assert xd.refs.is_ref(ee['bb'].get_expr('k0'))
    assert str(ee['bb'].get_expr('k0')) == "(2.0 * vars['b'])"
    assert ee['bb'].get_expr('k0')._value == 2 * (2 * 2 + 1)
    assert ee['bb'].get_value('k0') == 2 * (2 * 2 + 1)

    tt = ee['bb'].get_table()
    assert tt['value', 'k0'] == 2 * (2 * 2 + 1)
    assert tt['expr', 'k0'] == "(2.0 * vars['b'])"


def test_vars_features(tmpdir):
    env = xt.Environment()

    try:
        env['b'] = '3*a'
    except KeyError:
        pass
    else:
        raise ValueError('Variable a should not be present')

    env.vars.default_to_zero = True
    env['b'] = '3*a'

    assert env['a'] == 0
    assert env['b'] == 0

    env['a'] = 3
    assert env['b'] == 9

    # Test compact and to_dict
    tt = env.vars.get_table()
    assert tt['expr', 'b'] == '(3.0 * a)'
    dd = tt.to_dict()
    assert dd['b'] == '(3.0 * a)'
    assert dd['a'] == 3.0

    ee = xt.Environment()
    ee.vars.update(dd)
    assert ee['a'] == 3.0
    assert ee['b'] == 9.0
    assert ee.vars.get_table()['expr', 'b'] == '(3.0 * a)'

    with open(tmpdir / 'env.json', 'w') as fid:
        json.dump(dd, fid)

    ee2 = xt.Environment()
    ee2.vars.load_json(tmpdir / 'env.json')
    assert ee2['a'] == 3.0
    assert ee2['b'] == 9.0
    assert ee2.vars.get_table()['expr', 'b'] == '(3.0 * a)'

    tt1 = env.vars.get_table(compact=False)
    assert tt1['expr', 'b'] ==  "(3.0 * vars['a'])"
    dd1 = tt1.to_dict()
    assert dd1['b'] == "(3.0 * vars['a'])"
    assert dd1['a'] == 3.0


def test_call(tmpdir):
    def _trim(string):
        # Reduce indent level by one
        return '\n'.join(map(lambda line: line[4:], filter(bool, string.split('\n'))))

    preamble = _trim("""
    import xtrack as xt
    env = xt.get_environment()
    env.vars.default_to_zero=True
    """)

    parameters = _trim("""
    env['var1'] = 2
    env['var2'] = 3
    """)

    elements = _trim("""
    env.new('sbend', 'Bend')
    env.new('drift', 'Drift')

    env.new('mb2', 'sbend', length=2)
    env.new('drx', 'drift', length='var1 + var2')
    """)

    lattice = _trim("""
    env.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, energy0=45.6e9)

    env.new_line(
        name='seq',
        components=['mb2', 'drx', 'mb2', 'drx'],
    )
    """)

    parameter_file = tmpdir / 'parameters.py'
    element_file = tmpdir / 'elements.py'
    lattice_file = tmpdir / 'lattice.py'

    with parameter_file.open('w') as f:
        f.write(preamble + parameters)

    with element_file.open('w') as f:
        f.write(preamble + elements)

    with lattice_file.open('w') as f:
        f.write(preamble + lattice)

    env = xt.Environment()
    env.call(parameter_file)
    env.call(element_file)
    env.call(lattice_file)

    line, = env.lines.values()
    assert line.name == 'seq'
    assert line.element_names == ['mb2', 'drx', 'mb2', 'drx']
    assert env['mb2'].length == line['mb2'].length == 2
    assert env['drx'].length == line['drx'].length == 5

    env['var1'] = 10
    assert env['drx'].length == 13


@pytest.mark.parametrize(
    'overwrite_vars,x_value',
    [
        (False, 6),
        (True, 3),
    ]
)
def test_import_line_from_other_env(overwrite_vars, x_value):
    env = xt.Environment()
    env['z'] = 5
    env['x'] = 'z - 2'
    env['y'] = 7

    line = env.new_line(components=[
        env.new('b', xt.Bend, k0='2 * x'),
        env.new('ip', xt.Marker),
        env.new('d', xt.Drift, length='3 * y'),
    ])

    env2 = xt.Environment()
    env2['z'] = 3
    env2['x'] = '2 * z'
    env2.new('b', xt.Bend, k0='x')
    env2.new('ip', xt.Marker)

    env2.import_line(line, line_name='line', overwrite_vars=overwrite_vars)

    assert env2['x'] == x_value
    assert env2['y'] == 7

    assert env2.lines['line'].element_names == ['b/line', 'ip', 'd']
    assert env2['b'].k0 == x_value
    assert env2['b/line'].k0 == 2 * x_value
    assert isinstance(env2['ip'], xt.Marker)
    assert env2['d'].length == 3 * 7


def test_copy_element_from_other_env():
    env1 = xt.Environment()
    env1['var'] = 3
    env1['var2'] = '2 * var'
    env1.new('quad', xt.Quadrupole, length='var', knl=[0, 'var2'])

    env2 = xt.Environment()
    env2['var'] = 4
    env2['var2'] = '2 * var'
    env2.copy_element_from('quad', env1, 'quad/env2')

    assert env2['quad/env2'].length == 4
    assert env2['quad/env2'].knl[0] == 0
    assert env2['quad/env2'].knl[1] == 8

def test_import_line_matrix_attribute():

    apert = xt.LongitudinalLimitRect(
        min_zeta = -1e-3,
        max_zeta = 1e-3,
        min_pzeta = -1e-3,
        max_pzeta = 1e-3)
    tmap = xt.FirstOrderTaylorMap(
        length=0,
        m0=[ 2.3e-3,  3.07e-04,  0,  0,
            -2.06e-5,  0],
        m1=[[ 1, -6.1e-5,  0,
            0,  0, -2.3e-3],
            [ 5.2e-7,  9.9e-1,  0,
            0,  0,  1.0e-7],
            [ 0,  0,  1,
            2.0e-5,  0,  0],
            [ 0,  0,  0,
            1,  0,  0],
            [-1.0e-7, -2.3e-3,  0,
            0,  1,  4.1e-5],
            [ 0,  0,  0,
            0,  0,  1]]
    )
    line = xt.Line(
        elements=[apert, tmap],
        element_names=['apert', 'taylor_map']
    )

    line['a'] = 10
    line['taylor_map'].m1[2, 1]= 'a'

    assert line['taylor_map'].m1[2, 1] == 10
    line['a'] = 15
    assert line['taylor_map'].m1[2, 1] == 15

    env = xt.Environment()
    env.import_line(line, suffix_for_common_elements='', line_name='line_env')
    assert env['a'] == 15
    assert env['taylor_map'].m1[2, 1] == 15

    env['a'] = 20
    assert env['taylor_map'].m1[2, 1] == 20


def test_insert_repeated_elements():

    env = xt.Environment()

    line = env.new_line(
        components=[
            env.new('q0', 'Quadrupole', length=2.0, at=20.0),
            env.new('ql', 'Quadrupole', length=2.0, at=-10.0, from_='q0'),
            env.new('qr', 'Quadrupole', length=2.0, at=10.0, from_='q0'),
            env.new('mk1', 'Marker', at=40),
            env.new('mk2', 'Marker', at=42),
            env.new('end', 'Marker', at=50.),
        ])

    tt0 = line.get_table()
    tt0.show(cols=['name', 's_start', 's_end', 's_center'])

    env.new('ss', 'Sextupole', length='0.1')
    pp_ss = env.place('ss')

    line.insert([
        env.place('q0', at=5.0),
        pp_ss,
        env.place('q0', at=15.0),
        pp_ss,
        env.place('q0', at=41.0),
        pp_ss,
    ])

    tt = line.get_table()
    tt.show(cols=['name', 's_start', 's_end', 's_center'])

    assert np.all(tt.name == np.array(
        ['||drift_4', 'q0::0', 'ss::0', '||drift_6::0', 'ql', '||drift_7',
       'q0::1', 'ss::1', '||drift_6::1', 'q0::2', '||drift_2', 'qr',
       '||drift_1', 'mk1', 'q0::3', 'mk2', 'ss::2', '||drift_8', 'end',
       '_end_point']))
    xo.assert_allclose(tt.s_center, np.array(
        [ 2.  ,  5.  ,  6.05,  7.55, 10.  , 12.5 , 15.  , 16.05, 17.55,
        20.  , 25.  , 30.  , 35.5 , 40.  , 41.  , 42.  , 42.05, 46.05,
        50.  , 50.  ]), rtol=0., atol=1e-14)

def test_insert_with_anchors():

    env = xt.Environment()

    line = env.new_line(
        components=[
            env.new('q0', 'Quadrupole', length=2.0, at=20.0),
            env.new('ql', 'Quadrupole', length=2.0, at=-10.0, from_='q0'),
            env.new('qr', 'Quadrupole', length=2.0, at=10.0, from_='q0'),
            env.new('mk1', 'Marker', at=40),
            env.new('mk2', 'Marker', at=42),
            env.new('end', 'Marker', at=50.),
        ])

    s_tol = 1e-10

    tt0 = line.get_table()
    tt0.show(cols=['name', 's_start', 's_end', 's_center'])

    env.new('ss', 'Sextupole', length='0.1')
    line.insert([
        env.new('q1', 'q0', at=-5.0, from_='ql'),
        env.place('ss'),
        env.new('q2', 'q0', anchor='start', at=15.0 - 1.),
        env.place('ss'),
        env.new('q3', 'q0', anchor='start', at=29, from_='ql@end'),
        env.place('ss'),
    ])

    tt = line.get_table()
    tt.show(cols=['name', 's_start', 's_end', 's_center'])

    assert np.all(tt.name == np.array(
        ['||drift_4', 'q1', 'ss::0', '||drift_6::0', 'ql', '||drift_7',
       'q2', 'ss::1', '||drift_6::1', 'q0', '||drift_2', 'qr',
       '||drift_1', 'mk1', 'q3', 'mk2', 'ss::2', '||drift_8', 'end',
       '_end_point'],))
    xo.assert_allclose(tt.s_center, np.array(
        [ 2.  ,  5.  ,  6.05,  7.55, 10.  , 12.5 , 15.  , 16.05, 17.55,
        20.  , 25.  , 30.  , 35.5 , 40.  , 41.  , 42.  , 42.05, 46.05,
        50.  , 50.  ]), rtol=0., atol=1e-14)

def test_insert_anchors_special_cases():

    env = xt.Environment()

    line = env.new_line(
        components=[
            env.new('q0', 'Quadrupole', length=2.0, at=20.0),
            env.new('ql', 'Quadrupole', length=2.0, at=10.0),
            env.new('qr', 'Quadrupole', length=2.0, at=30),
            env.new('end', 'Marker', at=50.),
        ])


    line.insert([
        env.new('q4', 'q0', anchor='center', at=0, from_='q0@end'), # will replace half of q0
        env.new('q5', 'q0', at=0, from_='ql'), # will replace the full ql
        env.new('m5.0', 'Marker', at='q5@start'),
        env.new('m5.1', 'Marker', at='q5@start'),
        env.new('m5.2', 'Marker', at='q5@end'),
        env.new('m5.3', 'Marker'),
    ])

    line.insert([
        env.new('q6', 'q0', at=0, from_='qr'),
        env.new('mr.0', 'Marker', at='qr@start'),
        env.new('mr.1', 'Marker', at='qr@start'),
        env.new('mr.2', 'Marker', at='qr@end'),
        env.new('mr.3', 'Marker'),
    ])

    tt = line.get_table()
    tt.show(cols=['name', 's_start', 's_end', 's_center'])

    assert np.all(tt.name == np.array(
        ['||drift_1', 'm5.0', 'm5.1', 'q5', 'm5.2', 'm5.3', '||drift_2',
       'q0_entry', 'q0..entry_map', 'q0..0', 'q4', '||drift_5', 'mr.0',
       'mr.1', 'q6', 'mr.2', 'mr.3', '||drift_3', 'end', '_end_point']))
    xo.assert_allclose(tt.s_center, np.array(
        np.array([ 4.5,  9. ,  9. , 10. , 11. , 11. , 15. , 19. , 19. , 19.5, 21. ,
                   25.5, 29. , 29. , 30. , 31. , 31. , 40.5, 50. , 50. ])),
        rtol=0., atol=1e-14)

def test_insert_providing_object():

    env = xt.Environment()

    line = env.new_line(
        components=[
            env.new('q0', 'Quadrupole', length=2.0, at=20.0),
            env.new('ql', 'Quadrupole', length=2.0, at=10.0),
            env.new('qr', 'Quadrupole', length=2.0, at=30),
            env.new('end', 'Marker', at=50.),
        ])

    class MyElement:
        def __init__(self, myparameter):
            self.myparameter = myparameter

        def track(self, particles):
            particles.px += self.myparameter

    myelem = MyElement(0.1)

    line.insert(
        env.place('myname', myelem, at='qr@end'))

    tt = line.get_table()
    tt.show(cols=['name', 's_start', 's_end', 's_center'])

    assert np.all(tt.name == np.array(
        ['||drift_1', 'ql', '||drift_2::0', 'q0', '||drift_2::1', 'qr',
       'myname', '||drift_3', 'end', '_end_point']))
    assert np.all(tt.element_type == np.array(
        ['Drift', 'Quadrupole', 'Drift', 'Quadrupole', 'Drift',
        'Quadrupole', 'MyElement', 'Drift', 'Marker', '']))
    xo.assert_allclose(tt.s, np.array(
        [ 0.,  9., 11., 19., 21., 29., 31., 31., 50., 50.]),
        rtol=0., atol=1e-14)

def test_individual_insertions():

    env = xt.Environment()

    line = env.new_line(
        components=[
            env.new('q0', 'Quadrupole', length=2.0, at=20.0),
            env.new('ql', 'Quadrupole', length=2.0, at=-10.0, from_='q0'),
            env.new('qr', 'Quadrupole', length=2.0, at=10.0, from_='q0'),
            env.new('mk1', 'Marker', at=40),
            env.new('mk2', 'Marker', at=42),
            env.new('end', 'Marker', at=50.),
        ])

    s_tol = 1e-10

    tt0 = line.get_table()
    tt0.show(cols=['name', 's_start', 's_end', 's_center'])

    env.new('q1', 'q0')
    env.new('q2', 'q0')
    env.new('q3', 'q0')

    line.insert('q1', at=5.0)
    line.insert('q2', at=15.0)
    line.insert('q3', at=41.0)

    tt = line.get_table()
    tt.show(cols=['name', 's_start', 's_end', 's_center'])

    assert np.all(tt.name == np.array(
        ['||drift_4', 'q1', '||drift_5::0', 'ql', '||drift_5::1', 'q2',
       '||drift_5::2', 'q0', '||drift_2::0', 'qr', '||drift_1', 'mk1',
       'q3', 'mk2', '||drift_2::1', 'end', '_end_point']))
    xo.assert_allclose(tt.s_center, np.array(
        [ 2. ,  5. ,  7.5, 10. , 12.5, 15. , 17.5, 20. , 25. , 30. , 35.5,
        40. , 41. , 42. , 46. , 50. , 50. ]), rtol=0., atol=1e-14)

def test_individual_insertions_anchors():

    env = xt.Environment()

    line = env.new_line(
        components=[
            env.new('q0', 'Quadrupole', length=2.0, at=20.0),
            env.new('ql', 'Quadrupole', length=2.0, at=10.0),
            env.new('qr', 'Quadrupole', length=2.0, at=30),
            env.new('end', 'Marker', at=50.),
        ])

    env.new('q4', 'q0')
    env.new('q5', 'q0')
    env.new('m5.0', 'Marker')
    env.new('m5.1', 'Marker')
    env.new('m5.2', 'Marker')
    env.new('m5.3', 'Marker')

    line.insert('q4',anchor='center', at=0, from_='q0@end') # will replace half of q0
    line.insert('q5', at=0, from_='ql') # will replace the full ql
    line.insert('m5.0', at='q5@start')
    line.insert('m5.1', at='q5@start')
    line.insert('m5.2', at='q5@end')
    line.insert('m5.3', at='m5.2@end')

    line.insert([
        env.new('q6', 'q0', at=0, from_='qr'),
        env.new('mr.0', 'Marker', at='qr@start'),
        env.new('mr.1', 'Marker', at='qr@start'),
        env.new('mr.2', 'Marker', at='qr@end'),
        env.new('mr.3', 'Marker'),
    ])

    tt = line.get_table()
    tt.show(cols=['name', 's_start', 's_end', 's_center'])

    assert np.all(tt.name == np.array(
        ['||drift_1', 'm5.0', 'm5.1', 'q5', 'm5.2', 'm5.3', '||drift_2',
       'q0_entry', 'q0..entry_map', 'q0..0', 'q4', '||drift_5', 'mr.0',
       'mr.1', 'q6', 'mr.2', 'mr.3', '||drift_3', 'end', '_end_point']))
    xo.assert_allclose(tt.s_center, np.array(
        np.array([ 4.5,  9. ,  9. , 10. , 11. , 11. , 15. , 19., 19. , 19.5, 21. , 25.5,
        29. , 29. , 30. , 31. , 31. , 40.5, 50. , 50.])),
        rtol=0., atol=1e-14)

def test_insert_line():

    env = xt.Environment()

    line = env.new_line(
        components=[
            env.new('q0', 'Quadrupole', length=2.0, at=20.0),
            env.new('ql', 'Quadrupole', length=2.0, at=10.0),
            env.new('qr', 'Quadrupole', length=2.0, at=30),
            env.new('end', 'Marker', at=50.),
        ])

    ln_insert = env.new_line(
        components=[
            env.new('s1', 'Sextupole', length=0.1),
            env.new('s2', 's1', anchor='start', at=0.3, from_='s1@end'),
            env.new('s3', 's1', anchor='start', at=0.3, from_='s2@end')
        ])

    line.insert(ln_insert, anchor='start', at=1, from_='q0@end')

    tt = line.get_table()
    tt.show(cols=['name', 's_start', 's_end', 's_center'])

    assert np.all(tt.name == np.array(
        ['||drift_1', 'ql', '||drift_2', 'q0', '||drift_5', 's1',
       '||drift_4::0', 's2', '||drift_4::1', 's3', '||drift_8', 'qr',
       '||drift_3', 'end', '_end_point']))
    xo.assert_allclose(tt.s_center, np.array(
        [ 4.5 , 10.  , 15.  , 20.  , 21.5 , 22.05, 22.25, 22.45, 22.65,
        22.85, 25.95, 30.  , 40.5 , 50.  , 50.  ]),
        rtol=0., atol=1e-14)

def test_insert_list():
    env = xt.Environment()

    line = env.new_line(
        components=[
            env.new('q0', 'Quadrupole', length=2.0, at=20.0),
            env.new('ql', 'Quadrupole', length=2.0, at=10.0),
            env.new('qr', 'Quadrupole', length=2.0, at=30),
            env.new('end', 'Marker', at=50.),
        ])

    env.new('s1', 'Sextupole', length=0.1)
    env.new('s2', 's1')
    env.new('s3', 's1')

    line.insert(['s1', 's2', 's3'], anchor='start', at=1, from_='q0@end')

    tt = line.get_table()
    tt.show(cols=['name', 's_start', 's_end', 's_center'])

    assert np.all(tt.name == np.array(
        ['||drift_1', 'ql', '||drift_2', 'q0', '||drift_4', 's1', 's2',
       's3', '||drift_6', 'qr', '||drift_3', 'end', '_end_point']))
    xo.assert_allclose(tt.s_center, np.array(
        [ 4.5 , 10.  , 15.  , 20.  , 21.5 , 22.05, 22.15, 22.25, 25.65,
        30.  , 40.5 , 50.  , 50.  ]),
        rtol=0., atol=1e-14)

def test_anchors_in_new_and_place():
    env = xt.Environment()

    components = [
        env.new('q1', 'Quadrupole', length=2.0, anchor='start', at=1.),
        env.new('q2', 'q1', anchor='start', at=10., from_='q1', from_anchor='end'),
        env.new('s2', 'Sextupole', length=0.1, anchor='end', at=-1., from_='q2', from_anchor='start'),

        env.new('q3', 'Quadrupole', length=2.0, at=20.),
        env.new('q4', 'q3', anchor='start', at=0., from_='q3', from_anchor='end'),
        env.new('q5', 'q3'),

        # Sandwich of markers expected [m2.0, m2, m2.1.0, m2.1.1. m2.1]
        env.new('m2_0', 'Marker', at=0., from_='m2', from_anchor='start'),
        env.new('m2', 'Marker', at=0., from_='q2', from_anchor='start'),
        env.new('m2_1', 'Marker', at=0., from_='m2', from_anchor='end'),
        env.new('m2_1_0', 'Marker', at=0., from_='m2_1', from_anchor='start'),
        env.new('m2_1_1', 'Marker'),

        env.new('m1', 'Marker', at=0., from_='q1', from_anchor='start'),

        env.new('m4', 'Marker', at=0., from_='q4', from_anchor='start'),
        env.new('m3', 'Marker', at=0., from_='q3', from_anchor='end'),
    ]

    line = env.new_line(components=components)

    tt = line.get_table()

    tt.show(cols=['name', 's_start', 's_end', 's_center'])

    assert np.all(tt.name == np.array(
        ['||drift_1::0', 'm1', 'q1', '||drift_2', 's2', '||drift_1::1',
       'm2_0', 'm2', 'm2_1_0', 'm2_1_1', 'm2_1', 'q2', '||drift_3', 'q3',
       'm3', 'm4', 'q4', 'q5', '_end_point']))
    xo.assert_allclose(tt.s, np.array(
        [ 0. ,  1. ,  1. ,  3. , 11.9, 12. , 13. , 13. , 13. , 13. , 13. ,
        13. , 15. , 19. , 21. , 21. , 21. , 23. , 25. ]),
        rtol=0., atol=1e-14)
    xo.assert_allclose(tt.s_start, np.array(
        [ 0. ,  1. ,  1. ,  3. , 11.9, 12. , 13. , 13. , 13. , 13. , 13. ,
        13. , 15. , 19. , 21. , 21. , 21. , 23. , 25. ]),
        rtol=0., atol=1e-14)
    xo.assert_allclose(tt.s_end, np.array(
        [ 1. ,  1. ,  3. , 11.9, 12. , 13. , 13. , 13. , 13. , 13. , 13. ,
        15. , 19. , 21. , 21. , 21. , 23. , 25. , 25. ]),
        rtol=0., atol=1e-14)
    xo.assert_allclose(tt.s_center, 0.5*(tt.s_start + tt.s_end),
        rtol=0., atol=1e-14)

def test_anchors_in_new_and_place_compact():

    env = xt.Environment()

    components = [
        env.new('q1', 'Quadrupole', length=2.0, anchor='start', at=1.),
        env.new('q2', 'q1', anchor='start', at=10., from_='q1', from_anchor='end'),
        env.new('s2', 'Sextupole', length=0.1, anchor='end', at=-1., from_='q2', from_anchor='start'),

        env.new('q3', 'Quadrupole', length=2.0, at=20.),
        env.new('q4', 'q3', anchor='start', at=0., from_='q3', from_anchor='end'),
        env.new('q5', 'q3'),

        # Sandwich of markers expected [m2.0, m2, m2.1.0, m2.1.1. m2.1]
        env.new('m2_0', 'Marker', at=0., from_='m2', from_anchor='start'),
        env.new('m2', 'Marker', at=0., from_='q2', from_anchor='start'),
        env.new('m2_1', 'Marker', at=0., from_='m2', from_anchor='end'),
        env.new('m2_1_0', 'Marker', at=0., from_='m2_1', from_anchor='start'),
        env.new('m2_1_1', 'Marker'),

        env.new('m1', 'Marker', at=0., from_='q1', from_anchor='start'),

        env.new('m4', 'Marker', at=0., from_='q4', from_anchor='start'),
        env.new('m3', 'Marker', at=0., from_='q3', from_anchor='end'),
    ]

    line = env.new_line(components=components)

    tt = line.get_table()

    tt.show(cols=['name', 's_start', 's_end', 's_center'])

    assert np.all(tt.name == np.array(
        ['||drift_1::0', 'm1', 'q1', '||drift_2', 's2', '||drift_1::1',
         'm2_0', 'm2', 'm2_1_0', 'm2_1_1', 'm2_1', 'q2', '||drift_3', 'q3',
         'm3', 'm4', 'q4', 'q5', '_end_point']))
    xo.assert_allclose(tt.s, np.array(
        [ 0. ,  1. ,  1. ,  3. , 11.9, 12. , 13. , 13. , 13. , 13. , 13. ,
        13. , 15. , 19. , 21. , 21. , 21. , 23. , 25. ]),
        rtol=0., atol=1e-14)
    xo.assert_allclose(tt.s_start, np.array(
        [ 0. ,  1. ,  1. ,  3. , 11.9, 12. , 13. , 13. , 13. , 13. , 13. ,
        13. , 15. , 19. , 21. , 21. , 21. , 23. , 25. ]),
        rtol=0., atol=1e-14)
    xo.assert_allclose(tt.s_end, np.array(
        [ 1. ,  1. ,  3. , 11.9, 12. , 13. , 13. , 13. , 13. , 13. , 13. ,
        15. , 19. , 21. , 21. , 21. , 23. , 25. , 25. ]),
        rtol=0., atol=1e-14)
    xo.assert_allclose(tt.s_center, 0.5*(tt.s_start + tt.s_end),
        rtol=0., atol=1e-14)

def test_place_lines_with_anchors():

    env = xt.Environment()

    # A simple line made of quadrupoles spaced by 5 m
    env.new_line(name='l1', components=[
        env.new('q1', 'Quadrupole', length=2.0, at=0., anchor='start'),
        env.new('q2', 'Quadrupole', length=2.0, anchor='start', at=5., from_='q1@end'),
        env.new('q3', 'Quadrupole', length=2.0, anchor='start', at=5., from_='q2@end'),
        env.new('q4', 'Quadrupole', length=2.0, anchor='start', at=5., from_='q3@end'),
        env.new('q5', 'Quadrupole', length=2.0, anchor='start', at=5., from_='q4@end'),
    ])

    # Test absolute anchor of start 'l1'
    env.new_line(name='lstart', components=[
        env.place('l1', anchor='start', at=10.),
    ])

    # Test absolute anchor of end 'l1'
    env.new_line(name='lend', components=[
        env.place('l1', anchor='end', at=40.),
    ])

    # Test absolute anchor of center 'l1'
    env.new_line(name='lcenter', components=[
        env.place('l1', anchor='center', at=25.),
    ])

    # Test relative anchor of start 'l1' to start of another element
    env.new_line(name='lstcnt', components=[
        env.new('q0', 'Quadrupole', length=2.0, at=5.),
        env.place('l1', anchor='start', at=5., from_='q0@center'),
    ])

    # Test relative anchor of start 'l1' to end of another element
    env.new_line(name='lstst', components=[
        env.place('q0', at=5.),
        env.place('l1', anchor='start', at=5. + 1., from_='q0@end'),
    ])


    # Test relative anchor of start 'l1' to end of another element
    env.new_line(name='lstend', components=[
        env.place('q0', at=5.),
        env.place('l1', anchor='start', at=5. - 1., from_='q0@end'),
    ])

    tt_l1 = env['l1'].get_table()
    tt_test = tt_l1
    assert np.all(tt_test.name == np.array(
        ['q1', '||drift_1::0', 'q2', '||drift_1::1', 'q3', '||drift_1::2',
       'q4', '||drift_1::3', 'q5', '_end_point']))
    xo.assert_allclose(tt_test.s, np.array(
        [ 0.,  2.,  7.,  9., 14., 16., 21., 23., 28., 30.]),
        rtol=0., atol=1e-15)
    xo.assert_allclose(tt_test.s_start, np.array(
        [ 0.,  2.,  7.,  9., 14., 16., 21., 23., 28., 30.]),
        rtol=0., atol=1e-15)
    xo.assert_allclose(tt_test.s_center, np.array(
        [ 1. ,  4.5,  8. , 11.5, 15. , 18.5, 22. , 25.5, 29. , 30. ]),
        rtol=0., atol=1e-15)
    xo.assert_allclose(tt_test.s_end, np.array(
        [ 2.,  7.,  9., 14., 16., 21., 23., 28., 30., 30.]),
        rtol=0., atol=1e-15)

    tt_lstart = env['lstart'].get_table()
    tt_test = tt_lstart
    assert np.all(tt_test.name == np.array(
        ['||drift_2', 'q1', '||drift_1::0', 'q2', '||drift_1::1', 'q3',
       '||drift_1::2', 'q4', '||drift_1::3', 'q5', '_end_point']))
    xo.assert_allclose(tt_test.s, np.array(
        [ 0., 10., 12., 17., 19., 24., 26., 31., 33., 38., 40.]),
        rtol=0., atol=1e-15)

    tt_lend = env['lend'].get_table()
    tt_test = tt_lend
    assert np.all(tt_test.name == np.array(
        ['||drift_2', 'q1', '||drift_1::0', 'q2', '||drift_1::1', 'q3',
         '||drift_1::2', 'q4', '||drift_1::3', 'q5', '_end_point']))
    xo.assert_allclose(tt_test.s, np.array(
        [ 0., 10., 12., 17., 19., 24., 26., 31., 33., 38., 40.]),
        rtol=0., atol=1e-15)

    tt_lcenter = env['lcenter'].get_table()
    tt_test = tt_lcenter
    assert np.all(tt_test.name == np.array(
        ['||drift_2', 'q1', '||drift_1::0', 'q2', '||drift_1::1', 'q3',
       '||drift_1::2', 'q4', '||drift_1::3', 'q5', '_end_point']))
    xo.assert_allclose(tt_test.s, np.array(
        [ 0., 10., 12., 17., 19., 24., 26., 31., 33., 38., 40.]),
        rtol=0., atol=1e-15)

    tt_lstcnt = env['lstcnt'].get_table()
    tt_test = tt_lstcnt
    assert np.all(tt_test.name == np.array(
        ['||drift_3::0', 'q0', '||drift_3::1', 'q1', '||drift_1::0', 'q2',
         '||drift_1::1', 'q3', '||drift_1::2', 'q4', '||drift_1::3', 'q5',
         '_end_point']))
    xo.assert_allclose(tt_test.s, np.array(
        [ 0.,  4.,  6., 10., 12., 17., 19., 24., 26., 31., 33., 38., 40.]),
        rtol=0., atol=1e-15)

    tt_lstst = env['lstst'].get_table()
    tt_test = tt_lstst
    assert np.all(tt_test.name == np.array(
        ['||drift_3', 'q0', '||drift_4', 'q1', '||drift_1::0', 'q2',
         '||drift_1::1', 'q3', '||drift_1::2', 'q4', '||drift_1::3', 'q5',
         '_end_point']))

    tt_lstend = env['lstend'].get_table()
    tt_test = tt_lstend
    assert np.all(tt_test.name == np.array(
        ['||drift_3::0', 'q0', '||drift_3::1', 'q1', '||drift_1::0', 'q2',
       '||drift_1::1', 'q3', '||drift_1::2', 'q4', '||drift_1::3', 'q5',
       '_end_point']))

def test_remove_element_from_env():

    env = xt.Environment()
    env['a'] = 10
    env['b'] = 20
    env.new('q1', 'Quadrupole', length='2.0 + b', knl=[0, '3*a'])
    env.new('q2', 'q1', length='2.0 + b', knl=[0, '3*a'])

    assert 'q1' in env.elements
    assert len(env.ref['a']._find_dependant_targets()) == 7
    # is:
    # [vars['a'],
    #  element_refs['q2'].knl[1],
    #  element_refs['q2'],
    #  element_refs['q2'].knl,
    #  element_refs['q1'],
    #  element_refs['q1'].knl[1],
    #  element_refs['q1'].knl]

    env._remove_element('q1')
    assert 'q1' not in env.elements
    assert len(env.ref['a']._find_dependant_targets()) == 4
    # is:
    # [vars['a'],
    #  element_refs['q2'].knl[1],
    #  element_refs['q2'],
    #  element_refs['q2'].knl]

def test_remove_element_from_line():
    env = xt.Environment()

    line0 = env.new_line(
        components=[
            env.new('q0', 'Quadrupole', length=2.0, at=20.0),
            env.new('ql', 'Quadrupole', length=2.0, at=-10.0, from_='q0'),
            env.new('qr', 'Quadrupole', length=2.0, at=10.0, from_='q0'),
            env.new('mk1', 'Marker', at=40),
            env.new('mk2', 'Marker'),
            env.new('mk3', 'Marker'),
            env.place('q0'),
            env.new('end', 'Marker', at=50.),
        ])
    tt0 = line0.get_table()
    tt0.show(cols=['name', 's_start', 's_end', 's_center'])

    line1 = line0.copy()
    line1.remove('q0::1')
    tt1 = line1.get_table()
    tt1.show(cols=['name', 's_start', 's_end', 's_center'])

    assert np.all(tt1.name == np.array(
        ['||drift_1::0', 'ql', '||drift_2::0', 'q0', '||drift_2::1', 'qr',
       '||drift_1::1', 'mk1', 'mk2', 'mk3', '||drift_3', '||drift_2::2',
       'end', '_end_point']))
    xo.assert_allclose(tt1.s_center, np.array(
        [ 4.5, 10. , 15. , 20. , 25. , 30. , 35.5, 40. , 40. , 40. , 41. ,
        46. , 50. , 50. ]), rtol=0., atol=1e-14)

    line2 = line0.copy()
    line2.remove('q0')
    tt2 = line2.get_table()
    tt2.show(cols=['name', 's_start', 's_end', 's_center'])

    assert np.all(tt2.name == np.array(
        ['||drift_1::0', 'ql', '||drift_2::0', '||drift_3', '||drift_2::1',
       'qr', '||drift_1::1', 'mk1', 'mk2', 'mk3', '||drift_4',
       '||drift_2::2', 'end', '_end_point']))
    xo.assert_allclose(tt2.s_center, np.array(
        [ 4.5, 10. , 15. , 20. , 25. , 30. , 35.5, 40. , 40. , 40. , 41. ,
        46. , 50. , 50. ]), rtol=0., atol=1e-14)

    line3 = line0.copy()
    line3.remove('q.*')
    tt3 = line3.get_table()
    tt3.show(cols=['name', 's_start', 's_end', 's_center'])

    assert np.all(tt3.name == np.array(
        ['||drift_1::0', '||drift_3', '||drift_2::0', '||drift_4',
       '||drift_2::1', '||drift_5', '||drift_1::1', 'mk1', 'mk2', 'mk3',
       '||drift_6', '||drift_2::2', 'end', '_end_point']))
    xo.assert_allclose(tt3.s_center, np.array(
        [ 4.5, 10. , 15. , 20. , 25. , 30. , 35.5, 40. , 40. , 40. , 41. ,
        46. , 50. , 50. ]), rtol=0., atol=1e-14)

    line4 = line0.copy()
    line4.remove('mk2')
    tt4 = line4.get_table()
    tt4.show(cols=['name', 's_start', 's_end', 's_center'])

    assert np.all(tt4.name == np.array(
        ['||drift_1::0', 'ql', '||drift_2::0', 'q0::0', '||drift_2::1',
       'qr', '||drift_1::1', 'mk1', 'mk3', 'q0::1', '||drift_2::2', 'end',
       '_end_point']))
    xo.assert_allclose(tt4.s_center, np.array(
        [ 4.5, 10. , 15. , 20. , 25. , 30. , 35.5, 40. , 40. , 41. , 46. ,
        50. , 50. ]), rtol=0., atol=1e-14)

    line5 = line0.copy()
    line5.remove('mk.*')
    tt5 = line5.get_table()
    tt5.show(cols=['name', 's_start', 's_end', 's_center'])

    assert np.all(tt5.name == np.array(
        ['||drift_1::0', 'ql', '||drift_2::0', 'q0::0', '||drift_2::1',
       'qr', '||drift_1::1', 'q0::1', '||drift_2::2', 'end', '_end_point']))
    xo.assert_allclose(tt5.s_center, np.array(
        [ 4.5, 10. , 15. , 20. , 25. , 30. , 35.5, 41. , 46. , 50. , 50. ]),
        rtol=0., atol=1e-14)

    line6 = line0.copy()
    tt_remove = line6.get_table().rows['q.*']
    line6.remove(tt_remove.name)
    tt6 = line3.get_table()
    tt6.show(cols=['name', 's_start', 's_end', 's_center'])

    assert np.all(tt6.name == np.array(
        ['||drift_1::0', '||drift_3', '||drift_2::0', '||drift_4',
       '||drift_2::1', '||drift_5', '||drift_1::1', 'mk1', 'mk2', 'mk3',
       '||drift_6', '||drift_2::2', 'end', '_end_point']))
    xo.assert_allclose(tt6.s_center, np.array(
        [ 4.5, 10. , 15. , 20. , 25. , 30. , 35.5, 40. , 40. , 40. , 41. ,
        46. , 50. , 50. ]), rtol=0., atol=1e-14)

def test_replace_element():

    env = xt.Environment()

    line0 = env.new_line(
        components=[
            env.new('q0', 'Quadrupole', length=2.0, at=20.0),
            env.new('ql', 'Quadrupole', length=2.0, at=-10.0, from_='q0'),
            env.new('qr', 'Quadrupole', length=2.0, at=10.0, from_='q0'),
            env.new('mk1', 'Marker', at=40),
            env.new('mk2', 'Marker'),
            env.new('mk3', 'Marker'),
            env.place('q0'),
            env.new('end', 'Marker', at=50.),
        ])
    env.new('qnew', 'Quadrupole', length=2.0)
    env.new('mnew', 'Marker')
    tt0 = line0.get_table()
    tt0.show(cols=['name', 's_start', 's_end', 's_center'])

    line1 = line0.copy(shallow=True)
    line1.replace('q0::1', 'qnew')
    tt1 = line1.get_table()
    tt1.show(cols=['name', 's_start', 's_end', 's_center'])

    assert np.all(tt1.name == np.array(
        ['||drift_1::0', 'ql', '||drift_2::0', 'q0', '||drift_2::1', 'qr',
       '||drift_1::1', 'mk1', 'mk2', 'mk3', 'qnew', '||drift_2::2', 'end',
       '_end_point']))
    xo.assert_allclose(tt1.s_center, np.array(
        [ 4.5, 10. , 15. , 20. , 25. , 30. , 35.5, 40. , 40. , 40. , 41. ,
        46. , 50. , 50. ]), rtol=0., atol=1e-14)

    line2 = line0.copy(shallow=True)
    line2.replace('q0', 'qnew')
    tt2 = line2.get_table()
    tt2.show(cols=['name', 's_start', 's_end', 's_center'])

    assert np.all(tt2.name == np.array(
        ['||drift_1::0', 'ql', '||drift_2::0', 'qnew::0', '||drift_2::1',
        'qr', '||drift_1::1', 'mk1', 'mk2', 'mk3', 'qnew::1',
        '||drift_2::2', 'end', '_end_point']))
    xo.assert_allclose(tt2.s_center, np.array(
        [ 4.5, 10. , 15. , 20. , 25. , 30. , 35.5, 40. , 40. , 40. , 41. ,
        46. , 50. , 50. ]), rtol=0., atol=1e-14)

    line3 = line0.copy(shallow=True)
    line3.replace('q.*', 'qnew')
    tt3 = line3.get_table()
    tt3.show(cols=['name', 's_start', 's_end', 's_center'])

    assert np.all(tt3.name == np.array([
        '||drift_1::0', 'qnew::0', '||drift_2::0', 'qnew::1',
       '||drift_2::1', 'qnew::2', '||drift_1::1', 'mk1', 'mk2', 'mk3',
       'qnew::3', '||drift_2::2', 'end', '_end_point']))
    xo.assert_allclose(tt3.s_center, np.array(
        [ 4.5, 10. , 15. , 20. , 25. , 30. , 35.5, 40. , 40. , 40. , 41. ,
        46. , 50. , 50. ]), rtol=0., atol=1e-14)

    line4 = line0.copy(shallow=True)
    line4.replace('mk2', 'mnew')
    tt4 = line4.get_table()
    tt4.show(cols=['name', 's_start', 's_end', 's_center'])

    assert np.all(tt4.name == np.array([
        '||drift_1::0', 'ql', '||drift_2::0', 'q0::0', '||drift_2::1',
       'qr', '||drift_1::1', 'mk1', 'mnew', 'mk3', 'q0::1',
       '||drift_2::2', 'end', '_end_point']))
    xo.assert_allclose(tt4.s_center, np.array(
        [ 4.5, 10. , 15. , 20. , 25. , 30. , 35.5, 40. , 40. , 40., 41. , 46. ,
        50. , 50. ]), rtol=0., atol=1e-14)

    line5 = line0.copy(shallow=True)
    line5.replace('mk.*', 'mnew')
    tt5 = line5.get_table()
    tt5.show(cols=['name', 's_start', 's_end', 's_center'])

    assert np.all(tt5.name == np.array(
        ['||drift_1::0', 'ql', '||drift_2::0', 'q0::0', '||drift_2::1',
       'qr', '||drift_1::1', 'mnew::0', 'mnew::1', 'mnew::2', 'q0::1',
       '||drift_2::2', 'end', '_end_point']))
    xo.assert_allclose(tt5.s_center, np.array(
        [ 4.5, 10. , 15. , 20. , 25. , 30. , 35.5, 40. , 40. , 40., 41. , 46. ,
        50. , 50. ]), rtol=0., atol=1e-14)

def test_append_to_line():

    env = xt.Environment(
        particle_ref=xt.Particles(p0c=7000e9, x=1e-3, px=1e-3, y=1e-3, py=1e-3))

    line0 = env.new_line(
        components=[
            env.new('b0', 'Bend', length=1.0, anchor='start', at=5.0),
            env.new('q0', 'Quadrupole', length=2.0, at=20.0),
            env.new('ql', 'Quadrupole', length=2.0, at=-10.0, from_='q0'),
            env.new('qr', 'Quadrupole', length=2.0, at=10.0, from_='q0'),
            env.new('mk1', 'Marker', at=40),
            env.new('mk2', 'Marker'),
            env.new('mk3', 'Marker'),
            env.place('q0'),
            env.place('b0'),
            env.new('end', 'Marker', at=50.),
        ])

    line = line0.copy()
    line.env.new('qnew', 'Quadrupole', length=2.0)
    line.append('qnew')

    tt = line.get_table()
    tt.show(cols=['name', 's_start', 's_end', 's_center'])

    assert np.all(tt.name == np.array(
        ['||drift_1', 'b0::0', '||drift_2', 'ql', '||drift_3::0', 'q0::0',
       '||drift_3::1', 'qr', '||drift_4', 'mk1', 'mk2', 'mk3', 'q0::1',
       'b0::1', '||drift_5', 'end', 'qnew', '_end_point']))
    xo.assert_allclose(tt.s_center, np.array(
        [ 2.5,  5.5,  7.5, 10. , 15. , 20. , 25. , 30. , 35.5, 40. , 40. ,
        40. , 41. , 42.5, 46.5, 50. , 51. , 52. ]),
        rtol=0., atol=1e-14)

    class MyElement:
        def __init__(self, myparameter):
            self.myparameter = myparameter

        def track(self, particles):
            particles.px += self.myparameter

    myelem = MyElement(0.1)

    line = line0.copy()
    line.append('myname', myelem)

    tt = line.get_table()
    tt.show(cols=['name', 'element_type', 's_start', 's_end', 's_center'])

    assert np.all(tt.name == np.array(
        ['||drift_1', 'b0::0', '||drift_2', 'ql', '||drift_3::0', 'q0::0',
       '||drift_3::1', 'qr', '||drift_4', 'mk1', 'mk2', 'mk3', 'q0::1',
       'b0::1', '||drift_5', 'end', 'myname', '_end_point']))

    assert np.all(tt.element_type == np.array(
        ['Drift', 'Bend', 'Drift', 'Quadrupole', 'Drift', 'Quadrupole',
        'Drift', 'Quadrupole', 'Drift', 'Marker', 'Marker', 'Marker',
        'Quadrupole', 'Bend', 'Drift', 'Marker', 'MyElement', '']))

    xo.assert_allclose(tt.s_center, np.array(
        [ 2.5,  5.5,  7.5, 10. , 15. , 20. , 25. , 30. , 35.5, 40. , 40. ,
        40. , 41. , 42.5, 46.5, 50. , 50. , 50. ]),
        rtol=0., atol=1e-14)

    line = line0.copy()
    line.env.new('qnew1', 'Quadrupole', length=2.0)
    line.env.new('qnew2', 'Quadrupole', length=2.0)
    line.append(['qnew1', 'qnew2'])

    tt = line.get_table()
    tt.show(cols=['name', 's_start', 's_end', 's_center'])

    assert np.all(tt.name == np.array(
        ['||drift_1', 'b0::0', '||drift_2', 'ql', '||drift_3::0', 'q0::0',
       '||drift_3::1', 'qr', '||drift_4', 'mk1', 'mk2', 'mk3', 'q0::1',
       'b0::1', '||drift_5', 'end', 'qnew1', 'qnew2', '_end_point']))
    xo.assert_allclose(tt.s_center, np.array(
        [ 2.5,  5.5,  7.5, 10. , 15. , 20. , 25. , 30. , 35.5, 40. , 40. ,
        40. , 41. , 42.5, 46.5, 50. , 51. , 53. , 54. ]),
        rtol=0., atol=1e-14)

    line = line0.copy()
    line.env.new('qnew1', 'Quadrupole', length=2.0)
    line.env.new('qnew2', 'Quadrupole', length=2.0)

    l2 = line.env.new_line(components=['qnew1', 'qnew2', 'ql'])
    line.append(l2)

    tt = line.get_table()
    tt.show(cols=['name', 's_start', 's_end', 's_center'])

    assert np.all(tt.name == np.array(
        ['||drift_1', 'b0::0', '||drift_2', 'ql::0', '||drift_3::0',
         'q0::0', '||drift_3::1', 'qr', '||drift_4', 'mk1', 'mk2', 'mk3',
         'q0::1', 'b0::1', '||drift_5', 'end', 'qnew1', 'qnew2', 'ql::1',
         '_end_point'],))

    xo.assert_allclose(tt.s_center, np.array(
        [ 2.5,  5.5,  7.5, 10. , 15. , 20. , 25. , 30. , 35.5, 40. , 40. ,
        40. , 41. , 42.5, 46.5, 50. , 51. , 53. , 55. , 56. ]),
        rtol=0., atol=1e-14)

    line = line0.copy()
    line.env.new('qnew1', 'Quadrupole', length=2.0)
    line.env.new('qnew2', 'Quadrupole', length=2.0)

    l2 = line.env.new_line(components=['qnew1', 'qnew2', 'ql'])
    line.append([l2, 'qr'])

    tt = line.get_table()
    tt.show(cols=['name', 's_start', 's_end', 's_center'])

    assert np.all(tt.name == np.array(
        ['||drift_1', 'b0::0', '||drift_2', 'ql::0', '||drift_3::0',
       'q0::0', '||drift_3::1', 'qr::0', '||drift_4', 'mk1', 'mk2', 'mk3',
       'q0::1', 'b0::1', '||drift_5', 'end', 'qnew1', 'qnew2', 'ql::1',
       'qr::1', '_end_point']))
    xo.assert_allclose(tt.s_center, np.array(
        [ 2.5,  5.5,  7.5, 10. , 15. , 20. , 25. , 30. , 35.5, 40. , 40. ,
        40. , 41. , 42.5, 46.5, 50. , 51. , 53. , 55. , 57. , 58. ]),
        rtol=0., atol=1e-14)

def test_nested_lists():

    env = xt.Environment()

    env.new('q1', 'Quadrupole', length=2.0)
    env.new('q2', 'Quadrupole', length=2.0)

    line = env.new_line(components=[2*['q1'], 'q2'])

    tt = line.get_table()
    tt.show(cols=['name', 's_start', 's_end', 's_center'])
    assert np.all(tt.name == np.array(
        ['q1::0', 'q1::1', 'q2', '_end_point']))


def test_relative_error_definition():

    env = xt.Environment()
    env.vars.default_to_zero = True
    line = env.new_line(components=[
        env.new('mq', 'Quadrupole', length=0.5, k1='kq'),
        env.new('mqs', 'Quadrupole', length=0.5, k1s='kqs'),
        env.new('mb', 'Bend', length=0.5, angle='ang', k0_from_h=True),
    ])

    env.set_multipolar_errors({
        'mq': {'rel_knl': [1e-6, 1e-5, 1e-4], 'rel_ksl': [-1e-6, -1e-5, -1e-4]},
        'mqs': {'rel_knl': [2e-6, 2e-5, 2e-4], 'rel_ksl': [3e-6, 3e-5, 3e-4], 'refer': 'k1s'},
        'mb': {'rel_knl': [2e-6, 3e-5, 4e-4], 'rel_ksl': [5e-6, 6e-5, 7e-4]},
    })

    # Errors respond when variables are changed
    env['kq'] = 0.1
    env['kqs'] = 0.2
    env['ang'] = 0.3


    xo.assert_allclose(env.get('mq').knl[:3], 0.5 * 0.1 * np.array([1e-6, 1e-5, 1e-4]), rtol=1e-7, atol=0)
    xo.assert_allclose(env.get('mq').ksl[:3], 0.5 * 0.1 * np.array([-1e-6, -1e-5, -1e-4]), rtol=1e-7, atol=0)
    xo.assert_allclose(env.get('mqs').knl[:3], 0.5 * 0.2 * np.array([2e-6, 2e-5, 2e-4]), rtol=1e-7, atol=0)
    xo.assert_allclose(env.get('mqs').ksl[:3], 0.5 * 0.2 * np.array([3e-6, 3e-5, 3e-4]), rtol=1e-7, atol=0)
    xo.assert_allclose(env.get('mb').knl[:3], 0.3 * np.array([2e-6, 3e-5, 4e-4]), rtol=1e-7, atol=0)
    xo.assert_allclose(env.get('mb').ksl[:3], 0.3 * np.array([5e-6, 6e-5, 7e-4]), rtol=1e-7, atol=0)

def test_builder_length():
    env = xt.Environment()

    env.new('mq', 'Quadrupole', length=1)

    env['ll'] = 20.

    env.new_line(name='l1', length='ll', components=[
        env.place('mq', at=10)])

    env['l1'].get_table().cols['s_start s_center s_end']

    tt = env['l1'].get_table()

    assert np.all(tt.name == np.array(['||drift_1::0', 'mq', '||drift_1::1', '_end_point']))
    xo.assert_allclose(tt.s_center, np.array([ 4.75, 10.  , 15.25, 20.  ]),
                    rtol=0, atol=1e-10)

def test_enviroment_from_two_lines():

    env1 = xt.Environment()
    env1.vars.default_to_zero  = True
    line1 = env1.new_line(components=[
        env1.new('qq1_thick', xt.Quadrupole, length=1., k1='kk', at=10),
        env1.new('qq1_thin', xt.Quadrupole, length=1., k1='kk', at=20),
        env1.new('qq_shared_thick', xt.Quadrupole, length=1., k1='kk', at=30),
        env1.new('qq_shared_thin', xt.Quadrupole, length=1., k1='kk', at=40),
    ])
    line1.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(slicing=xt.Teapot(2, mode='thick'), name='qq1_thick'),
            xt.Strategy(slicing=xt.Teapot(2, mode='thin'), name='qq1_thin'),
            xt.Strategy(slicing=xt.Teapot(2, mode='thick'), name='qq_shared_thick'),
            xt.Strategy(slicing=xt.Teapot(2, mode='thin'), name='qq_shared_thin'),
        ])

    env2 = xt.Environment()
    env2.vars.default_to_zero  = True
    line2 = env2.new_line(components=[
        env2.new('qq2_thick', xt.Quadrupole, length=1., k1='kk', at=10),
        env2.new('qq2_thin', xt.Quadrupole, length=1., k1='kk', at=20),
        env2.new('qq_shared_thick', xt.Quadrupole, length=1., k1='kk', at=30),
        env2.new('qq_shared_thin', xt.Quadrupole, length=1., k1='kk', at=40),
    ])
    line2.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(slicing=xt.Teapot(2, mode='thick'), name='qq2_thick'),
            xt.Strategy(slicing=xt.Teapot(2, mode='thin'), name='qq2_thin'),
            xt.Strategy(slicing=xt.Teapot(2, mode='thick'), name='qq_shared_thick'),
            xt.Strategy(slicing=xt.Teapot(2, mode='thin'), name='qq_shared_thin'),
        ])

    # Merge the line in one environment
    env = xt.Environment(lines={'line1': line1, 'line2': line2})

    tt1 = env.line1.get_table()
    tt2 = env.line2.get_table()

    tt1.show(cols='name element_type parent_name')
    # name                          element_type         parent_name
    # drift_1                       Drift                None
    # qq1_thick_entry               Marker               None
    # qq1_thick..0                  ThickSliceQuadrupole qq1_thick
    # qq1_thick..1                  ThickSliceQuadrupole qq1_thick
    # qq1_thick_exit                Marker               None
    # drift_2                       Drift                None
    # qq1_thin_entry                Marker               None
    # drift_qq1_thin..0             DriftSliceQuadrupole qq1_thin
    # qq1_thin..0                   ThinSliceQuadrupole  qq1_thin
    # drift_qq1_thin..1             DriftSliceQuadrupole qq1_thin
    # qq1_thin..1                   ThinSliceQuadrupole  qq1_thin
    # drift_qq1_thin..2             DriftSliceQuadrupole qq1_thin
    # qq1_thin_exit                 Marker               None
    # drift_3                       Drift                None
    # qq_shared_thick_entry         Marker               None
    # qq_shared_thick..0/line1      ThickSliceQuadrupole qq_shared_thick/line1
    # qq_shared_thick..1/line1      ThickSliceQuadrupole qq_shared_thick/line1
    # qq_shared_thick_exit          Marker               None
    # drift_4                       Drift                None
    # qq_shared_thin_entry          Marker               None
    # drift_qq_shared_thin..0/line1 DriftSliceQuadrupole qq_shared_thin/line1
    # qq_shared_thin..0/line1       ThinSliceQuadrupole  qq_shared_thin/line1
    # drift_qq_shared_thin..1/line1 DriftSliceQuadrupole qq_shared_thin/line1
    # qq_shared_thin..1/line1       ThinSliceQuadrupole  qq_shared_thin/line1
    # drift_qq_shared_thin..2/line1 DriftSliceQuadrupole qq_shared_thin/line1
    # qq_shared_thin_exit           Marker               None

    tt2.show(cols='name element_type parent_name')
    # name                          element_type         parent_name
    # drift_5                       Drift                None
    # qq2_thick_entry               Marker               None
    # qq2_thick..0                  ThickSliceQuadrupole qq2_thick
    # qq2_thick..1                  ThickSliceQuadrupole qq2_thick
    # qq2_thick_exit                Marker               None
    # drift_6                       Drift                None
    # qq2_thin_entry                Marker               None
    # drift_qq2_thin..0             DriftSliceQuadrupole qq2_thin
    # qq2_thin..0                   ThinSliceQuadrupole  qq2_thin
    # drift_qq2_thin..1             DriftSliceQuadrupole qq2_thin
    # qq2_thin..1                   ThinSliceQuadrupole  qq2_thin
    # drift_qq2_thin..2             DriftSliceQuadrupole qq2_thin
    # qq2_thin_exit                 Marker               None
    # drift_7                       Drift                None
    # qq_shared_thick_entry         Marker               None
    # qq_shared_thick..0/line2      ThickSliceQuadrupole qq_shared_thick/line2
    # qq_shared_thick..1/line2      ThickSliceQuadrupole qq_shared_thick/line2
    # qq_shared_thick_exit          Marker               None
    # drift_8                       Drift                None
    # qq_shared_thin_entry          Marker               None
    # drift_qq_shared_thin..0/line2 DriftSliceQuadrupole qq_shared_thin/line2
    # qq_shared_thin..0/line2       ThinSliceQuadrupole  qq_shared_thin/line2
    # drift_qq_shared_thin..1/line2 DriftSliceQuadrupole qq_shared_thin/line2
    # qq_shared_thin..1/line2       ThinSliceQuadrupole  qq_shared_thin/line2
    # drift_qq_shared_thin..2/line2 DriftSliceQuadrupole qq_shared_thin/line2
    # qq_shared_thin_exit           Marker               None
    # _end_point                                         None

    assert np.all(tt1.name == np.array([
       '||drift_1', 'qq1_thick_entry', 'qq1_thick..entry_map',
       'qq1_thick..0', 'qq1_thick..1', 'qq1_thick..exit_map',
       'qq1_thick_exit', '||drift_2', 'qq1_thin_entry',
       'qq1_thin..entry_map', 'drift_qq1_thin..0', 'qq1_thin..0',
       'drift_qq1_thin..1', 'qq1_thin..1', 'drift_qq1_thin..2',
       'qq1_thin..exit_map', 'qq1_thin_exit', '||drift_3',
       'qq_shared_thick_entry', 'qq_shared_thick..entry_map/line1',
       'qq_shared_thick..0/line1', 'qq_shared_thick..1/line1',
       'qq_shared_thick..exit_map/line1', 'qq_shared_thick_exit',
       '||drift_4', 'qq_shared_thin_entry',
       'qq_shared_thin..entry_map/line1', 'drift_qq_shared_thin..0/line1',
       'qq_shared_thin..0/line1', 'drift_qq_shared_thin..1/line1',
       'qq_shared_thin..1/line1', 'drift_qq_shared_thin..2/line1',
       'qq_shared_thin..exit_map/line1', 'qq_shared_thin_exit',
       '_end_point']))

    assert np.all(tt1.element_type == np.array([
       'Drift', 'Marker', 'ThinSliceQuadrupoleEntry',
       'ThickSliceQuadrupole', 'ThickSliceQuadrupole',
       'ThinSliceQuadrupoleExit', 'Marker', 'Drift', 'Marker',
       'ThinSliceQuadrupoleEntry', 'DriftSliceQuadrupole',
       'ThinSliceQuadrupole', 'DriftSliceQuadrupole',
       'ThinSliceQuadrupole', 'DriftSliceQuadrupole',
       'ThinSliceQuadrupoleExit', 'Marker', 'Drift', 'Marker',
       'ThinSliceQuadrupoleEntry', 'ThickSliceQuadrupole',
       'ThickSliceQuadrupole', 'ThinSliceQuadrupoleExit', 'Marker',
       'Drift', 'Marker', 'ThinSliceQuadrupoleEntry',
       'DriftSliceQuadrupole', 'ThinSliceQuadrupole',
       'DriftSliceQuadrupole', 'ThinSliceQuadrupole',
       'DriftSliceQuadrupole', 'ThinSliceQuadrupoleExit', 'Marker', '']))

    assert np.all(tt1.parent_name == np.array([
       None, None, 'qq1_thick', 'qq1_thick', 'qq1_thick', 'qq1_thick',
       None, None, None, 'qq1_thin', 'qq1_thin', 'qq1_thin', 'qq1_thin',
       'qq1_thin', 'qq1_thin', 'qq1_thin', None, None, None,
       'qq_shared_thick/line1', 'qq_shared_thick/line1',
       'qq_shared_thick/line1', 'qq_shared_thick/line1', None, None, None,
       'qq_shared_thin/line1', 'qq_shared_thin/line1',
       'qq_shared_thin/line1', 'qq_shared_thin/line1',
       'qq_shared_thin/line1', 'qq_shared_thin/line1',
       'qq_shared_thin/line1', None, None]))

    assert np.all(tt2.name == np.array(['||drift_5', 'qq2_thick_entry', 'qq2_thick..entry_map',
       'qq2_thick..0', 'qq2_thick..1', 'qq2_thick..exit_map',
       'qq2_thick_exit', '||drift_6', 'qq2_thin_entry',
       'qq2_thin..entry_map', 'drift_qq2_thin..0', 'qq2_thin..0',
       'drift_qq2_thin..1', 'qq2_thin..1', 'drift_qq2_thin..2',
       'qq2_thin..exit_map', 'qq2_thin_exit', '||drift_7',
       'qq_shared_thick_entry', 'qq_shared_thick..entry_map/line2',
       'qq_shared_thick..0/line2', 'qq_shared_thick..1/line2',
       'qq_shared_thick..exit_map/line2', 'qq_shared_thick_exit',
       '||drift_8', 'qq_shared_thin_entry',
       'qq_shared_thin..entry_map/line2', 'drift_qq_shared_thin..0/line2',
       'qq_shared_thin..0/line2', 'drift_qq_shared_thin..1/line2',
       'qq_shared_thin..1/line2', 'drift_qq_shared_thin..2/line2',
       'qq_shared_thin..exit_map/line2', 'qq_shared_thin_exit',
       '_end_point']))

    assert np.all(tt2.element_type == np.array([
        'Drift', 'Marker', 'ThinSliceQuadrupoleEntry',
       'ThickSliceQuadrupole', 'ThickSliceQuadrupole',
       'ThinSliceQuadrupoleExit', 'Marker', 'Drift', 'Marker',
       'ThinSliceQuadrupoleEntry', 'DriftSliceQuadrupole',
       'ThinSliceQuadrupole', 'DriftSliceQuadrupole',
       'ThinSliceQuadrupole', 'DriftSliceQuadrupole',
       'ThinSliceQuadrupoleExit', 'Marker', 'Drift', 'Marker',
       'ThinSliceQuadrupoleEntry', 'ThickSliceQuadrupole',
       'ThickSliceQuadrupole', 'ThinSliceQuadrupoleExit', 'Marker',
       'Drift', 'Marker', 'ThinSliceQuadrupoleEntry',
       'DriftSliceQuadrupole', 'ThinSliceQuadrupole',
       'DriftSliceQuadrupole', 'ThinSliceQuadrupole',
       'DriftSliceQuadrupole', 'ThinSliceQuadrupoleExit', 'Marker', '']))

    assert np.all(tt2.parent_name == np.array([
       None, None, 'qq2_thick', 'qq2_thick', 'qq2_thick', 'qq2_thick',
       None, None, None, 'qq2_thin', 'qq2_thin', 'qq2_thin', 'qq2_thin',
       'qq2_thin', 'qq2_thin', 'qq2_thin', None, None, None,
       'qq_shared_thick/line2', 'qq_shared_thick/line2',
       'qq_shared_thick/line2', 'qq_shared_thick/line2', None, None, None,
       'qq_shared_thin/line2', 'qq_shared_thin/line2',
       'qq_shared_thin/line2', 'qq_shared_thin/line2',
       'qq_shared_thin/line2', 'qq_shared_thin/line2',
       'qq_shared_thin/line2', None, None]))

    assert 'qq1_thick' in env.elements
    assert 'qq1_thin' in env.elements
    assert 'qq_shared_thick/line1' in env.elements
    assert 'qq_shared_thin/line1' in env.elements
    assert 'qq2_thick' in env.elements
    assert 'qq2_thin' in env.elements
    assert 'qq_shared_thick/line2' in env.elements
    assert 'qq_shared_thin/line2' in env.elements

    line1['kk'] = 1e-1
    line2['kk'] = 1e-1
    env['kk'] = 1e-1

    particle_ref = xt.Particles(p0c=7e12)
    line1.particle_ref = particle_ref
    line2.particle_ref = particle_ref
    env.line1.particle_ref = particle_ref
    env.line2.particle_ref = particle_ref

    tw1 = line1.twiss(betx=1, bety=2)
    tw2 = line2.twiss(betx=1, bety=2)
    tw1i = env.line1.twiss(betx=1, bety=2)
    tw2i = env.line2.twiss(betx=1, bety=2)

    assert np.allclose(tw1.s, tw1i.s, atol=0, rtol=1e-15)
    assert np.allclose(tw1.betx, tw1i.betx, atol=0, rtol=1e-15)
    assert np.allclose(tw1.bety, tw1i.bety, atol=0, rtol=1e-15)

    assert np.allclose(tw2.s, tw2i.s, atol=0, rtol=1e-15)
    assert np.allclose(tw2.betx, tw2i.betx, atol=0, rtol=1e-15)
    assert np.allclose(tw2.bety, tw2i.bety, atol=0, rtol=1e-15)

def test_particle_ref_from_particles_container():

    env = xt.Environment()
    env['a'] = 4.

    env.new_particle('my_particle', p0c=['1e12 * a'])
    assert 'my_particle' in env.particles
    xo.assert_allclose(env['my_particle'].p0c, 4e12, rtol=0, atol=1e-9)
    env['a'] = 5.
    xo.assert_allclose(env['my_particle'].p0c, 5e12, rtol=0, atol=1e-9)

    env.particle_ref = 'my_particle'

    xo.assert_allclose(env.particle_ref.p0c, 5e12, rtol=0, atol=1e-9)
    assert env.particle_ref.__class__.__name__ == 'EnvParticleRef'
    assert env._particle_ref == 'my_particle'
    assert env.ref['my_particle']._value is env.get('my_particle')
    env.particle_ref.p0c = '2e12 * a'
    xo.assert_allclose(env.particle_ref.p0c, 10e12, rtol=0, atol=1e-9)
    env['my_particle'].p0c = '1e12 * a'
    xo.assert_allclose(env.particle_ref.p0c, 5e12, rtol=0, atol=1e-9)

    env2 = xt.Environment.from_dict(env.to_dict())
    assert 'my_particle' in env2.particles
    assert isinstance(env2.get('my_particle'), xt.Particles)
    assert env2.get('my_particle') is not env.get('my_particle')
    assert env2._particle_ref == "my_particle"
    assert env2.ref['my_particle']._value is env2.get('my_particle')
    xo.assert_allclose(env2['my_particle'].p0c, 5e12, rtol=0, atol=1e-9)
    env2['a'] = 6.
    xo.assert_allclose(env2['my_particle'].p0c, 6e12, rtol=0, atol=1e-9)
    env2['a'] = 5.

    assert env2.particle_ref.__class__.__name__ == 'EnvParticleRef'
    env2.particle_ref.p0c = '2e12 * a'
    xo.assert_allclose(env2.particle_ref.p0c, 10e12, rtol=0, atol=1e-9)
    env2['my_particle'].p0c = '1e12 * a'
    xo.assert_allclose(env2.particle_ref.p0c, 5e12, rtol=0, atol=1e-9)

    env2 = env.copy()
    assert 'my_particle' in env2.particles
    assert env2._particle_ref == "my_particle"
    assert env.ref['my_particle']._value is env.get('my_particle')
    assert isinstance(env2.get('my_particle'), xt.Particles)
    assert env2.get('my_particle') is not env.get('my_particle')
    xo.assert_allclose(env2['my_particle'].p0c, 5e12, rtol=0, atol=1e-9)
    env2['a'] = 6.
    xo.assert_allclose(env2['my_particle'].p0c, 6e12, rtol=0, atol=1e-9)
    env2['a'] = 5.

    assert env2.particle_ref.__class__.__name__ == 'EnvParticleRef'
    env2.particle_ref.p0c = '2e12 * a'
    xo.assert_allclose(env2.particle_ref.p0c, 10e12, rtol=0, atol=1e-9)
    env2['my_particle'].p0c = '1e12 * a'
    xo.assert_allclose(env2.particle_ref.p0c, 5e12, rtol=0, atol=1e-9)

    ll = env.new_line(name='my_line', components=[])
    assert ll._particle_ref == 'my_particle'

    xo.assert_allclose(ll.particle_ref.p0c, 5e12, rtol=0, atol=1e-9)
    assert ll.particle_ref.__class__.__name__ == 'LineParticleRef'
    ll.particle_ref.p0c = '2e12 * a'
    xo.assert_allclose(env.particle_ref.p0c, 10e12, rtol=0, atol=1e-9)
    env['my_particle'].p0c = '1e12 * a'
    xo.assert_allclose(ll.particle_ref.p0c, 5e12, rtol=0, atol=1e-9)

    ll2 = xt.Line.from_dict(ll.to_dict())
    assert 'my_particle' in ll2.env.particles
    assert ll2.env.particle_ref is None
    assert ll2.particle_ref.__class__.__name__ == 'LineParticleRef'
    assert ll2._particle_ref == 'my_particle'
    xo.assert_allclose(ll2.env['my_particle'].p0c, 5e12, rtol=0, atol=1e-9)
    ll2['a'] = 7.
    xo.assert_allclose(ll2.env['my_particle'].p0c, 7e12, rtol=0, atol=1e-9)
    ll2['a'] = 5.

    ll2.particle_ref.p0c = '2e12 * a'
    xo.assert_allclose(ll2.particle_ref.p0c, 10e12, rtol=0, atol=1e-9)
    ll2.env['my_particle'].p0c = '1e12 * a'
    xo.assert_allclose(ll2.particle_ref.p0c, 5e12, rtol=0, atol=1e-9)


    ll2 = ll.copy()
    assert 'my_particle' in ll2.env.particles
    assert ll2.env.particle_ref is None
    assert ll2.particle_ref.__class__.__name__ == 'LineParticleRef'
    assert ll2._particle_ref == 'my_particle'
    xo.assert_allclose(ll2.env['my_particle'].p0c, 5e12, rtol=0, atol=1e-9)
    ll2['a'] = 7.
    xo.assert_allclose(ll2.env['my_particle'].p0c, 7e12, rtol=0, atol=1e-9)
    ll2['a'] = 5.

    ll2.particle_ref.p0c = '2e12 * a'
    xo.assert_allclose(ll2.particle_ref.p0c, 10e12, rtol=0, atol=1e-9)
    ll2.env['my_particle'].p0c = '1e12 * a'
    xo.assert_allclose(ll2.particle_ref.p0c, 5e12, rtol=0, atol=1e-9)

def test_particle_ref_as_object():

    env = xt.Environment()
    env['a'] = 4.

    env.new_particle('my_particle', p0c=['1e12 * a'])
    assert 'my_particle' in env.particles
    xo.assert_allclose(env['my_particle'].p0c, 4e12, rtol=0, atol=1e-9)
    env['a'] = 5.
    xo.assert_allclose(env['my_particle'].p0c, 5e12, rtol=0, atol=1e-9)

    part = env['my_particle'].copy()
    env.particle_ref = part

    xo.assert_allclose(env.particle_ref.p0c, 5e12, rtol=0, atol=1e-9)
    assert env.particle_ref.__class__.__name__ == 'EnvParticleRef'
    assert env._particle_ref is part
    env['my_particle'].p0c = '2e12 * a'
    xo.assert_allclose(env.eval('2e12 * a'), 10e12, rtol=0, atol=1e-9)
    xo.assert_allclose(env.particle_ref.p0c, 5e12, rtol=0, atol=1e-9)

    env2 = xt.Environment.from_dict(env.to_dict())
    assert 'my_particle' in env2.particles
    assert isinstance(env2.get('my_particle'), xt.Particles)
    assert env2.get('my_particle') is not env.get('my_particle')
    assert isinstance(env2._particle_ref, xt.Particles)
    assert env2.particle_ref.__class__.__name__ == 'EnvParticleRef'

    env2 = env.copy()
    assert 'my_particle' in env2.particles
    assert isinstance(env2._particle_ref, xt.Particles)
    assert isinstance(env2.get('my_particle'), xt.Particles)
    assert env2.get('my_particle') is not env.get('my_particle')
    assert env2.particle_ref.__class__.__name__ == 'EnvParticleRef'

    ll = env.new_line(name='my_line', components=[])
    assert isinstance(ll._particle_ref, xt.Particles)
    assert ll._particle_ref is env._particle_ref
    xo.assert_allclose(ll.particle_ref.p0c, 5e12, rtol=0, atol=1e-9)
    assert ll.particle_ref.__class__.__name__ == 'LineParticleRef'

    ll2 = xt.Line.from_dict(ll.to_dict())
    assert ll2.env.particle_ref is None
    assert ll2.particle_ref.__class__.__name__ == 'LineParticleRef'
    assert isinstance(ll2._particle_ref, xt.Particles)
    assert ll2._particle_ref is not ll._particle_ref


    ll2 = ll.copy()
    assert ll2.env.particle_ref is None
    assert ll2.particle_ref.__class__.__name__ == 'LineParticleRef'
    assert isinstance(ll2._particle_ref, xt.Particles)
    assert ll2._particle_ref is not ll._particle_ref

def test_line_set_particle_ref():

    line = xt.Line()
    line.set_particle_ref('electron', beta0=0.9)

    xo.assert_allclose(line.particle_ref.q0 , -1, rtol=0, atol=1e-14)
    xo.assert_allclose(line.particle_ref.mass0 , xt.ELECTRON_MASS_EV)
    xo.assert_allclose(line.particle_ref.beta0 , 0.9, rtol=0, atol=1e-14)

    env = xt.Environment()
    env['my_beta0'] = 0.1
    env.new_particle('my_part', 'proton', beta0='my_beta0')

    line = env.new_line()
    line.set_particle_ref('my_part')
    line['my_beta0'] = 0.6

    xo.assert_allclose(line.particle_ref.q0 , 1, rtol=0, atol=1e-14)
    xo.assert_allclose(line.particle_ref.mass0 , xt.PROTON_MASS_EV)
    xo.assert_allclose(line.particle_ref.beta0 , 0.6, rtol=0, atol=1e-14)

    p_ref = xt.Particles('Pb208', p0c=7e12/82)
    line = xt.Line()
    line.set_particle_ref(p_ref)

    xo.assert_allclose(line.particle_ref.q0 , 82, rtol=0, atol=1e-14)
    xo.assert_allclose(line.particle_ref.mass0, xt.particles.masses.Pb208_MASS_EV)
    xo.assert_allclose(line.particle_ref.p0c , 7e12/82, rtol=0, atol=1e-14)


def test_env_set_particle_ref():

    env = xt.Environment()
    env.set_particle_ref('electron', beta0=0.9)

    xo.assert_allclose(env.particle_ref.q0 , -1, rtol=0, atol=1e-14)
    xo.assert_allclose(env.particle_ref.mass0 , xt.ELECTRON_MASS_EV)
    xo.assert_allclose(env.particle_ref.beta0 , 0.9, rtol=0, atol=1e-14)

    env.new_line(name='line1', components=[env.new('d1', 'Drift', length=1)])
    env.line1.set_particle_ref('electron', beta0=0.9)

    xo.assert_allclose(env.line1.particle_ref.q0 , -1, rtol=0, atol=1e-14)
    xo.assert_allclose(env.line1.particle_ref.mass0 , xt.ELECTRON_MASS_EV)
    xo.assert_allclose(env.line1.particle_ref.beta0 , 0.9, rtol=0, atol=1e-14)

    env.set_particle_ref('proton', beta0=0.8)
    xo.assert_allclose(env.particle_ref.q0 , 1, rtol=0, atol=1e-14)
    xo.assert_allclose(env.particle_ref.mass0 , xt.PROTON_MASS_EV)
    xo.assert_allclose(env.particle_ref.beta0 , 0.8, rtol=0, atol=1e-14)
    xo.assert_allclose(env.line1.particle_ref.q0 , 1, rtol=0, atol=1e-14)
    xo.assert_allclose(env.line1.particle_ref.mass0 , xt.PROTON_MASS_EV)
    xo.assert_allclose(env.line1.particle_ref.beta0 , 0.8, rtol=0, atol=1e-14)

    env['my_beta0'] = 0.1
    env.new_particle('my_part', 'proton', beta0='my_beta0')
    env.set_particle_ref('my_part')

    env['my_beta0'] = 0.6
    xo.assert_allclose(env.line1.particle_ref.q0 , 1, rtol=0, atol=1e-14)
    xo.assert_allclose(env.line1.particle_ref.mass0 , xt.PROTON_MASS_EV)
    xo.assert_allclose(env.line1.particle_ref.beta0 , 0.6, rtol=0, atol=1e-14)

    p_ref = xt.Particles('Pb208', p0c=7e12/82)
    env.set_particle_ref(p_ref)

    xo.assert_allclose(env.particle_ref.q0 , 82, rtol=0, atol=1e-14)
    xo.assert_allclose(env.particle_ref.mass0, xt.particles.masses.Pb208_MASS_EV)
    xo.assert_allclose(env.particle_ref.p0c , 7e12/82, rtol=0, atol=1e-14)
    xo.assert_allclose(env.line1.particle_ref.q0 , 82, rtol=0, atol=1e-14)
    xo.assert_allclose(env.line1.particle_ref.mass0, xt.particles.masses.Pb208_MASS_EV)
    xo.assert_allclose(env.line1.particle_ref.p0c , 7e12/82, rtol=0, atol=1e-14)

    env.set_particle_ref('positron', beta0=0.7, lines=None)
    xo.assert_allclose(env.particle_ref.q0 , 1, rtol=0, atol=1e-14)
    xo.assert_allclose(env.particle_ref.mass0 , xt.ELECTRON_MASS_EV)
    xo.assert_allclose(env.particle_ref.beta0 , 0.7, rtol=0, atol=1e-14)
    # line unchanged
    xo.assert_allclose(env.line1.particle_ref.q0 , 82, rtol=0, atol=1e-14)
    xo.assert_allclose(env.line1.particle_ref.mass0, xt.particles.masses.Pb208_MASS_EV)
    xo.assert_allclose(env.line1.particle_ref.p0c , 7e12/82, rtol=0, atol=1e-14)

def test_compose_parametric_lines():

    env = xt.Environment()

    env['a'] = 1.

    env.new_line(name='l1', compose=True)
    env['l1'].new('q1', 'Quadrupole', length='a', at='0.5*a')
    env['l1'].new('q2', 'q1', at='4*a', from_='q1@center')

    l2 = env.new_line(compose=True,
                        components=[
                        env.place('l1', at='7.5*a'),
                        env.place(-env['l1'], at='17.5*a'),
                    ])
    tt1 = l2.get_table()
    # tt1.cols['s', 'name', 'element_type', 'env_name'] is:
    # Table: 9 rows, 4 cols
    # name                   s element_type env_name
    # drift_3                0 Drift        drift_3
    # q1::0                  5 Quadrupole   q1
    # drift_1                6 Drift        drift_1
    # q2::0                  9 Quadrupole   q2
    # drift_4               10 Drift        drift_4
    # q2::1                 15 Quadrupole   q2
    # drift_2               16 Drift        drift_2
    # q1::1                 19 Quadrupole   q1
    # _end_point            20              _end_point

    env['a'] = 2.
    l2.regenerate_from_composer()
    tt2 = l2.get_table()
    # tt2.cols['s', 'name', 'element_type', 'env_name'] is:
    # Table: 9 rows, 4 cols
    # name                   s element_type env_name
    # drift_7                0 Drift        drift_7
    # q1::0                 10 Quadrupole   q1
    # drift_5               12 Drift        drift_5
    # q2::0                 18 Quadrupole   q2
    # drift_8               20 Drift        drift_8
    # q2::1                 30 Quadrupole   q2
    # drift_6               32 Drift        drift_6
    # q1::1                 38 Quadrupole   q1
    # _end_point            40              _end_point

    # Check tt1
    assert np.all(tt1.name ==
        ['||drift_2::0', 'q1::0', '||drift_1::0', 'q2::0', '||drift_2::1',
       'q2::1', '||drift_1::1', 'q1::1', '_end_point'])
    xo.assert_allclose(tt1.s,
            [ 0.,  5.,  6.,  9., 10., 15., 16., 19., 20.],
            rtol=0, atol=1e-12)
    assert np.all(tt1.element_type ==
        ['Drift', 'Quadrupole', 'Drift', 'Quadrupole', 'Drift',
        'Quadrupole', 'Drift', 'Quadrupole', ''])
    assert np.all(tt1.env_name ==
        ['||drift_2', 'q1', '||drift_1', 'q2', '||drift_2', 'q2',
       '||drift_1', 'q1', '_end_point'])

    # Check tt2
    assert np.all(tt2.name ==
        ['||drift_4::0', 'q1::0', '||drift_3::0', 'q2::0', '||drift_4::1',
         'q2::1', '||drift_3::1', 'q1::1', '_end_point'])
    xo.assert_allclose(tt2.s, 2 * tt1.s, rtol=0, atol=1e-12)
    assert np.all(tt2.element_type ==
        ['Drift', 'Quadrupole', 'Drift', 'Quadrupole', 'Drift',
        'Quadrupole', 'Drift', 'Quadrupole', ''])
    assert np.all(tt2.env_name ==
        ['||drift_4', 'q1', '||drift_3', 'q2', '||drift_4', 'q2',
         '||drift_3', 'q1', '_end_point'])



    # Same in MAD-X

    mad_src = """
    a = 1;
    q1: quadrupole, L:=a;
    q2: q1;
    d1: drift, L:=3*a;

    d5: drift, L:=5*a;

    l1: line=(q1,d1,q2);
    l2: line=(d5, l1, d5, -l1);

    s1: sequence, refer=centre, l:=5*a;
        q1, at:=0.5*a;
        q2, at:=4.5*a;
    endsequence;
    sm1: sequence, refer=centre, l:=5*a;
        q2, at:=0.5*a;
        q1, at:=4.5*a;
    endsequence;
    s2: sequence, refer=centre, l:=20*a;
        s1, at:=7.5*a;
        sm1, at:=17.5*a;
    endsequence;

    a=2;

    """
    from cpymad.madx import Madx
    madx = Madx()
    madx.input(mad_src)
    madx.beam()
    madx.use('l2')
    tt_mad_l2 = xt.Table(madx.twiss(betx=1, bety=1), _copy_cols=True)
    madx.use('s2')
    tt_mad_s2 = xt.Table(madx.twiss(betx=1, bety=1), _copy_cols=True)

    env_mad = xt.load(string=mad_src, format='madx')
    tt_xs_mad_l2 = env_mad['l2'].get_table()
    # tt_xs_mad_l2.cols['s', 'name', 'element_type', 'env_name'] is:
    # Table: 9 rows, 4 cols
    # name                   s element_type env_name
    # d5::0                  0 Drift        d5
    # q1::0                 10 Quadrupole   q1
    # d1::0                 12 Drift        d1
    # q2::0                 18 Quadrupole   q2
    # d5::1                 20 Drift        d5
    # q2::1                 30 Quadrupole   q2
    # d1::1                 32 Drift        d1
    # q1::1                 38 Quadrupole   q1
    # _end_point            40              _end_point

    tt_xs_mad_s2 = env_mad['s2'].get_table()
    # tt_xs_mad_s2.cols['s', 'name', 'element_type', 'env_name'] is:
    # Table: 9 rows, 4 cols
    # Table: 9 rows, 4 cols
    # name                   s element_type env_name
    # drift_3                0 Drift        drift_3
    # q1::0                 10 Quadrupole   q1
    # drift_1               12 Drift        drift_1
    # q2::0                 18 Quadrupole   q2
    # drift_4               20 Drift        drift_4
    # q2::1                 30 Quadrupole   q2
    # drift_2               32 Drift        drift_2
    # q1::1                 38 Quadrupole   q1
    # _end_point            40              _end_point

    assert np.all(tt_xs_mad_l2.name ==
        ['d5::0', 'q1::0', 'd1::0', 'q2::0', 'd5::1', 'q2::1',
        'd1::1', 'q1::1', '_end_point'])
    xo.assert_allclose(tt_xs_mad_l2.s,
            2 * tt1.s,
            rtol=0, atol=1e-12)
    assert np.all(tt_xs_mad_l2.element_type ==
        ['Drift', 'Quadrupole', 'Drift', 'Quadrupole', 'Drift',
        'Quadrupole', 'Drift', 'Quadrupole', ''])
    assert np.all(tt_xs_mad_l2.env_name ==
        ['d5', 'q1', 'd1', 'q2', 'd5',
        'q2', 'd1', 'q1', '_end_point'])
    xo.assert_allclose(tt_mad_l2.s[:-1], tt_xs_mad_l2.s, atol=1e-12, rtol=0)

    assert np.all(tt_xs_mad_s2.name ==
        ['||drift_2::0', 'q1::0', '||drift_1::0', 'q2::0', '||drift_2::1',
         'q2::1', '||drift_1::1', 'q1::1', '_end_point'])
    xo.assert_allclose(tt_xs_mad_s2.s,
            2 * tt1.s,
            rtol=0, atol=1e-12)
    assert np.all(tt_xs_mad_s2.element_type ==
        ['Drift', 'Quadrupole', 'Drift', 'Quadrupole', 'Drift',
        'Quadrupole', 'Drift', 'Quadrupole', ''])
    assert np.all(tt_xs_mad_s2.env_name ==
        ['||drift_2', 'q1', '||drift_1', 'q2', '||drift_2', 'q2',
       '||drift_1', 'q1', '_end_point'])

def test_expr_in_builder():

    env = xt.Environment()

    env['a'] = 1.0

    b1 = env.new_builder(name='b1', length='3*a')
    b1.new('q1', 'Quadrupole', length='a', at='1.5*a')

    b2 = env.new_builder(name='b2', length=3*env.ref['a'])
    b2.new('q2', 'Quadrupole', length=env.ref['a'], at=1.5*env.ref['a'])

    env['a'] = 2.0
    b1.build()
    b2.build()

    assert isinstance(env['b1'], xt.Line)
    assert isinstance(env['b2'], xt.Line)

    tt1 = env['b1'].get_table()
    tt2 = env['b2'].get_table()

    # tt1.cols['s', 'name', 'element_type'] is:
    # Table: 4 rows, 3 cols
    # name                   s element_type
    # drift_1                0 Drift
    # q1                     2 Quadrupole
    # drift_2                4 Drift
    # _end_point             6

    # tt2.cols['s', 'name', 'element_type'] is:
    # Table: 4 rows, 3 cols
    # name                   s element_type
    # drift_1                0 Drift
    # q2                     2 Quadrupole
    # drift_2                4 Drift
    # _end_point             6

    assert np.all(tt1.name == np.array(
        ['||drift_1::0', 'q1', '||drift_1::1', '_end_point']))
    xo.assert_allclose(tt1.s,
            np.array([0., 2., 4., 6.]),
            rtol=0, atol=1e-12)
    assert np.all(tt1.element_type == np.array([
        'Drift', 'Quadrupole', 'Drift', '']))
    assert np.all(tt2.name == np.array([
        '||drift_1::0', 'q2', '||drift_1::1', '_end_point']))
    xo.assert_allclose(tt2.s,
            np.array([0., 2., 4., 6.]),
            rtol=0, atol=1e-12)
    assert np.all(tt2.element_type == np.array([
        'Drift', 'Quadrupole', 'Drift', '']))

def test_contains_and_name_clash():

    env = xt.Environment()

    env.vars['a'] = 2.
    env.elements['e1'] = xt.Quadrupole(length=1.)
    env.particles['p1'] = xt.Particles(p0c=1e12)
    env.lines['l1'] = env.new_line(length=3)

    assert 'a' in env
    assert 'a' in env.vars
    assert 'a' not in env.elements
    assert 'a' not in env.particles
    assert 'a' not in env.lines

    assert 'e1' in env
    assert 'e1' in env.elements
    assert 'e1' not in env.vars
    assert 'e1' not in env.particles
    assert 'e1' not in env.lines

    assert 'p1' in env
    assert 'p1' in env.particles
    assert 'p1' not in env.vars
    assert 'p1' not in env.elements
    assert 'p1' not in env.lines

    assert 'l1' in env
    assert 'l1' in env.lines
    assert 'l1' not in env.vars
    assert 'l1' not in env.elements
    assert 'l1' not in env.particles

    assert 'zz' not in env
    assert 'zz' not in env.vars
    assert 'zz' not in env.elements
    assert 'zz' not in env.particles
    assert 'zz' not in env.lines

    with pytest.raises(KeyError):
        _ = env['zz']

    with pytest.raises(KeyError):
        _ = env.vars['zz']

    with pytest.raises(KeyError):
        _ = env.elements['zz']

    with pytest.raises(KeyError):
        _ = env.particles['zz']

    with pytest.raises(KeyError):
        _ = env.lines['zz']

    # ----- Check behavior of vars container -----

    # Updating the variable should be possible
    env.vars['a'] = 3.
    assert env['a'] == 3.

    with pytest.raises(ValueError):
        env.vars['e1'] = 5.  # Clash with element name

    with pytest.raises(ValueError):
        env.vars['p1'] = 5.  # Clash with particle name

    with pytest.raises(ValueError):
        env.vars['l1'] = 5.  # Clash with line name

    with pytest.raises(ValueError):
        env.elements['a'] = xt.Marker()

    # ----- Check behavior of elements container -----

    with pytest.raises(ValueError):
        env.elements['a'] = xt.Marker()  # Clash with var name

    with pytest.raises(ValueError):
        env.elements['e1'] = xt.Marker()  # Clash with existing element name

    with pytest.raises(ValueError):
        env.elements['p1'] = xt.Marker()  # Clash with particle name

    with pytest.raises(ValueError):
        env.elements['l1'] = xt.Marker()  # Clash with line name

    # ----- Check behavior of particles container -----
    with pytest.raises(ValueError):
        env.particles['a'] = xt.Particles()  # Clash with var name

    with pytest.raises(ValueError):
        env.particles['e1'] = xt.Particles()  # Clash with element name

    with pytest.raises(ValueError):
        env.particles['p1'] = xt.Particles()  # Clash with existing particle name

    with pytest.raises(ValueError):
        env.particles['l1'] = xt.Particles()  # Clash with line name

    # ----- Check behavior of lines container -----
    with pytest.raises(ValueError):
        env.lines['a'] = env.new_line()  # Clash with var name

    with pytest.raises(ValueError):
        env.lines['e1'] = env.new_line()  # Clash with element name

    with pytest.raises(ValueError):
        env.lines['p1'] = env.new_line()  # Clash with particle name

    with pytest.raises(ValueError):
        env.lines['l1'] = env.new_line()  # Clash with existing line name

def test_remove():
    env = xt.Environment()

    env.vars['a1'] = 2.
    env.vars['a2'] = 3.
    env.vars['a3'] = 4.
    env.vars['a4'] = 5.
    env.elements['e1'] = xt.Quadrupole(length=1.)
    env.elements['e2'] = xt.Quadrupole(length=2.)
    env.elements['e3'] = xt.Quadrupole(length=3.)
    env.elements['e4'] = xt.Quadrupole(length=4.)
    env.particles['p1'] = xt.Particles(p0c=1e12)
    env.particles['p2'] = xt.Particles(p0c=2e12)
    env.particles['p3'] = xt.Particles(p0c=3e12)
    env.particles['p4'] = xt.Particles(p0c=4e12)
    env.lines['l1'] = env.new_line(length=3)
    env.lines['l2'] = env.new_line(length=4)
    env.lines['l3'] = env.new_line(length=5)
    env.lines['l4'] = env.new_line(length=6)

    with pytest.raises(KeyError):
        env.vars.remove('zz')

    with pytest.raises(KeyError):
        env.elements.remove('zz')

    with pytest.raises(KeyError):
        env.particles.remove('zz')

    with pytest.raises(KeyError):
        env.lines.remove('zz')

    with pytest.raises(KeyError):
        env.remove('zz')

    assert 'a1' in env.vars
    assert 'a1' in env
    env.vars.remove('a1')
    assert 'a1' not in env.vars
    assert 'a1' not in env

    assert 'a2' in env.vars
    assert 'a2' in env
    env.remove('a2')
    assert 'a2' not in env.vars
    assert 'a2' not in env

    assert 'a3' in env.vars
    assert 'a3' in env
    del env.vars['a3']
    assert 'a3' not in env.vars
    assert 'a3' not in env

    assert 'a4' in env.vars
    assert 'a4' in env
    del env['a4']
    assert 'a4' not in env.vars
    assert 'a4' not in env

    assert 'e1' in env.elements
    assert 'e1' in env
    env.elements.remove('e1')
    assert 'e1' not in env.elements
    assert 'e1' not in env

    assert 'e2' in env.elements
    assert 'e2' in env
    env.remove('e2')
    assert 'e2' not in env.elements
    assert 'e2' not in env

    assert 'e3' in env.elements
    assert 'e3' in env
    del env.elements['e3']
    assert 'e3' not in env.elements
    assert 'e3' not in env

    assert 'e4' in env.elements
    assert 'e4' in env
    del env['e4']
    assert 'e4' not in env.elements
    assert 'e4' not in env

    assert 'p1' in env.particles
    assert 'p1' in env
    env.particles.remove('p1')
    assert 'p1' not in env.particles
    assert 'p1' not in env

    assert 'p2' in env.particles
    assert 'p2' in env
    env.remove('p2')
    assert 'p2' not in env.particles
    assert 'p2' not in env

    assert 'p3' in env.particles
    assert 'p3' in env
    del env.particles['p3']
    assert 'p3' not in env.particles
    assert 'p3' not in env

    assert 'p4' in env.particles
    assert 'p4' in env
    del env['p4']
    assert 'p4' not in env.particles
    assert 'p4' not in env

    assert 'l1' in env.lines
    assert 'l1' in env
    env.lines.remove('l1')
    assert 'l1' not in env.lines
    assert 'l1' not in env

    assert 'l2' in env.lines
    assert 'l2' in env
    env.remove('l2')
    assert 'l2' not in env.lines
    assert 'l2' not in env

    assert 'l3' in env.lines
    assert 'l3' in env
    del env.lines['l3']
    assert 'l3' not in env.lines
    assert 'l3' not in env

    assert 'l4' in env.lines
    assert 'l4' in env
    del env['l4']
    assert 'l4' not in env.lines
    assert 'l4' not in env

    env['a'] = 5.
    env.new('e', 'Quadrupole', length='3*a')
    with pytest.raises(RuntimeError):
        env.remove('a') # a is used by element e
    env.remove('e')
    env.remove('a')
    assert 'e' not in env.elements
    assert 'a' not in env.vars

    env.new('e', 'Quadrupole', length=2)
    env['a'] = 3 * env.ref['e'].length
    assert env['a'] == 6.
    assert str(env.ref['a']._expr) == "(3 * element_refs['e'].length)"
    with pytest.raises(RuntimeError):
        env.remove('e') # e is used by variable a
    env.remove('a')
    env.remove('e')

    env['a'] = 4.
    env.new_particle('p', p0c='2*a*1e12')
    assert env['p'].p0c == 8e12
    with pytest.raises(RuntimeError):
        env.remove('a') # a is used by particle p
    env.remove('p')
    env.remove('a')

    env.new_particle('p', p0c=1e12)
    env['a'] = env.ref['p'].p0c
    assert env['a'] == 1e12
    assert str(env.ref['a']._expr) == "particles['p'].p0c"
    with pytest.raises(RuntimeError):
        env.remove('p') # p is used by variable a
    env.remove('a')
    env.remove('p')

def test_line_algebra():

    env = xt.Environment()

    env['a'] = 1.

    l1 = env.new_line(compose=True)
    l1.new('q1', 'Quadrupole', length='a', at='0.5*a')
    l1.new('q2', 'q1', at='4*a', from_='q1@center')

    l2 = env.new_line(compose=True)
    l2.new('s1', 'Sextupole', length='a', at='1.5*a')
    l2.new('s2', 's1', at='5*a', from_='s1@center')

    assert l1.mode == 'compose'
    assert l2.mode == 'compose'

    ss = l1 + l2
    assert ss.mode == 'compose'
    assert len(ss.composer.components) == 2
    tss = ss.get_table()
    tss.cols['s element_type env_name']
    # Table: 8 rows, 4 cols
    # name                   s element_type env_name
    # q1                     0 Quadrupole   q1
    # drift_1                1 Drift        drift_1
    # q2                     4 Quadrupole   q2
    # drift_2                5 Drift        drift_2
    # s1                     6 Sextupole    s1
    # drift_3                7 Drift        drift_3
    # s2                    11 Sextupole    s2
    # _end_point            12              _end_point

    assert np.all(tss.name == np.array(
        ['q1', '||drift_1', 'q2', '||drift_2', 's1', '||drift_3', 's2',
       '_end_point']))
    xo.assert_allclose(tss.s,
            [ 0.,  1.,  4.,  5.,  6.,  7., 11., 12.],
            rtol=0, atol=1e-12)
    assert np.all(tss.element_type ==
        ['Quadrupole', 'Drift', 'Quadrupole', 'Drift', 'Sextupole',
        'Drift', 'Sextupole', ''])

    ss.end_compose()
    assert ss.mode == 'normal'
    tss2 = ss.get_table()
    # is the same as tss apart from different drift names
    assert np.all(tss2.name == np.array(
        ['q1', '||drift_1', 'q2', '||drift_2', 's1', '||drift_3', 's2',
       '_end_point']))
    xo.assert_allclose(tss2.s,
            [ 0.,  1.,  4.,  5.,  6.,  7., 11., 12.],
            rtol=0, atol=1e-12)
    assert np.all(tss2.element_type ==
        ['Quadrupole', 'Drift', 'Quadrupole', 'Drift', 'Sextupole',
        'Drift', 'Sextupole', ''])

    l1.end_compose()
    l2.end_compose()
    ss2 = l1 + l2
    assert ss2.mode == 'normal'
    tss3 = ss2.get_table()
    # is the same as tss apart from different drift names
    assert np.all(tss3.name == np.array(
        ['q1', '||drift_1', 'q2', '||drift_2', 's1', '||drift_3', 's2',
       '_end_point']))
    xo.assert_allclose(tss3.s,
            [ 0.,  1.,  4.,  5.,  6.,  7., 11., 12.],
            rtol=0, atol=1e-12)
    assert np.all(tss3.element_type ==
        ['Quadrupole', 'Drift', 'Quadrupole', 'Drift', 'Sextupole',
        'Drift', 'Sextupole', ''])

    l1.regenerate_from_composer()
    l2.regenerate_from_composer()

    mm = 3*l1
    assert len(mm.composer.components) == 3
    tmm = mm.get_table()
    # tmm.cols['s element_type env_name'] is:
    # Table: 10 rows, 4 cols
    # name                   s element_type env_name
    # q1::0                  0 Quadrupole   q1
    # drift_10               1 Drift        drift_10
    # q2::0                  4 Quadrupole   q2
    # q1::1                  5 Quadrupole   q1
    # drift_11               6 Drift        drift_11
    # q2::1                  9 Quadrupole   q2
    # q1::2                 10 Quadrupole   q1
    # drift_12              11 Drift        drift_12
    # q2::2                 14 Quadrupole   q2
    # _end_point            15              _end_point
    assert np.all(tmm.name == np.array(
        ['q1::0', '||drift_1::0', 'q2::0', 'q1::1', '||drift_1::1', 'q2::1',
       'q1::2', '||drift_1::2', 'q2::2', '_end_point']))
    xo.assert_allclose(tmm.s,
            [ 0.,  1.,  4.,  5.,  6.,  9., 10., 11., 14., 15.],
            rtol=0, atol=1e-12)
    assert np.all(tmm.element_type ==
        ['Quadrupole', 'Drift', 'Quadrupole',
        'Quadrupole', 'Drift', 'Quadrupole',
        'Quadrupole', 'Drift', 'Quadrupole',
        ''])

    l1.end_compose()
    mm2 = 3*l1
    assert mm2.mode == 'normal'
    tmm2 = mm2.get_table()
    # tmm2.cols['s element_type env_name'] is:
    # Table: 10 rows, 4 cols
    # name                    s element_type env_name
    # q1::0                   0 Quadrupole   q1
    # drift_13::0             1 Drift        drift_13
    # q2::0                   4 Quadrupole   q2
    # q1::1                   5 Quadrupole   q1
    # drift_13::1             6 Drift        drift_13
    # q2::1                   9 Quadrupole   q2
    # q1::2                  10 Quadrupole   q1
    # drift_13::2            11 Drift        drift_13
    # q2::2                  14 Quadrupole   q2
    # _end_point             15              _end_point
    assert np.all(tmm2.name == np.array([
        'q1::0', '||drift_1::0', 'q2::0', 'q1::1', '||drift_1::1', 'q2::1',
       'q1::2', '||drift_1::2', 'q2::2', '_end_point']))
    xo.assert_allclose(tmm2.s,
            [ 0.,  1.,  4.,  5.,  6.,  9., 10., 11., 14., 15.],
            rtol=0, atol=1e-12)
    assert np.all(tmm2.element_type ==
        ['Quadrupole', 'Drift', 'Quadrupole',
        'Quadrupole', 'Drift', 'Quadrupole',
        'Quadrupole', 'Drift', 'Quadrupole',
        ''])

    l1.regenerate_from_composer()
    assert l1.mode == 'compose'
    assert l1.composer.mirror == False

    ml1 = -l1
    assert ml1.mode == 'compose'
    assert ml1.composer.mirror == True
    tml1 = ml1.get_table()
    # tml1.cols['s element_type env_name'] is:
    # Table: 4 rows, 4 cols
    # Table: 4 rows, 4 cols
    # name                   s element_type env_name
    # q2                     0 Quadrupole   q2
    # drift_14               1 Drift        drift_14
    # q1                     4 Quadrupole   q1
    # _end_point             5              _end_point
    assert np.all(tml1.name == np.array(
        ['q2', '||drift_1', 'q1', '_end_point']))
    xo.assert_allclose(tml1.s,
            [0., 1., 4., 5.],
            rtol=0, atol=1e-12)
    assert np.all(tml1.element_type ==
        ['Quadrupole', 'Drift', 'Quadrupole', ''])

    l1.end_compose()
    ml2 = -l1
    assert ml2.mode == 'normal'
    tml2 = ml2.get_table()
    # tml2.cols['s element_type env_name'] is:
    # Table: 4 rows, 4 cols
    # name                   s element_type env_name
    # q2                     0 Quadrupole   q2
    # drift_15               1 Drift        drift_15
    # q1                     4 Quadrupole   q1
    # _end_point             5              _end_point
    assert np.all(tml2.name == np.array(
        ['q2', '||drift_1', 'q1', '_end_point']))
    xo.assert_allclose(tml2.s,
            [0., 1., 4., 5.],
            rtol=0, atol=1e-12)
    assert np.all(tml2.element_type ==
        ['Quadrupole', 'Drift', 'Quadrupole', ''])

    l1.regenerate_from_composer()

    m3l1 = -(3*l1)
    assert m3l1.mode == 'compose'
    assert len(m3l1.composer.components) == 3
    assert m3l1.composer.mirror == True
    tm3l1 = m3l1.get_table()
    # tm3l1.cols['s element_type env_name'] is:
    # Table: 10 rows, 4 cols
    # name                   s element_type env_name
    # q2::0                  0 Quadrupole   q2
    # drift_18               1 Drift        drift_18
    # q1::0                  4 Quadrupole   q1
    # q2::1                  5 Quadrupole   q2
    # drift_17               6 Drift        drift_17
    # q1::1                  9 Quadrupole   q1
    # q2::2                 10 Quadrupole   q2
    # drift_16              11 Drift        drift_16
    # q1::2                 14 Quadrupole   q1
    # _end_point            15              _end_point
    assert np.all(tm3l1.name == np.array(
        ['q2::0', '||drift_1::0', 'q1::0', 'q2::1', '||drift_1::1', 'q1::1',
         'q2::2', '||drift_1::2', 'q1::2', '_end_point']))
    xo.assert_allclose(tm3l1.s,
            [ 0.,  1.,  4.,  5.,  6.,  9., 10., 11., 14., 15.],
            rtol=0, atol=1e-12)
    assert np.all(tm3l1.element_type ==
        ['Quadrupole', 'Drift', 'Quadrupole',
        'Quadrupole', 'Drift', 'Quadrupole',
        'Quadrupole', 'Drift', 'Quadrupole',
        ''])

    l1.end_compose()
    m3l1 = -(3*l1)
    assert m3l1.mode == 'normal'
    tm3l1 = m3l1.get_table()
    # tm3l1.cols['s element_type env_name'] is:
    # Table: 10 rows, 4 cols
    # name                    s element_type env_name
    # q2::0                   0 Quadrupole   q2
    # drift_19::0             1 Drift        drift_19
    # q1::0                   4 Quadrupole   q1
    # q2::1                   5 Quadrupole   q2
    # drift_19::1             6 Drift        drift_19
    # q1::1                   9 Quadrupole   q1
    # q2::2                  10 Quadrupole   q2
    # drift_19::2            11 Drift        drift_19
    # q1::2                  14 Quadrupole   q1
    # _end_point             15              _end_point
    assert np.all(tm3l1.name == np.array([
        'q2::0', '||drift_1::0', 'q1::0', 'q2::1', '||drift_1::1', 'q1::1',
       'q2::2', '||drift_1::2', 'q1::2', '_end_point']))
    xo.assert_allclose(tm3l1.s,
            [ 0.,  1.,  4.,  5.,  6.,  9., 10., 11., 14., 15.],
            rtol=0, atol=1e-12)
    assert np.all(tm3l1.element_type ==
        ['Quadrupole', 'Drift', 'Quadrupole',
        'Quadrupole', 'Drift', 'Quadrupole',
        'Quadrupole', 'Drift', 'Quadrupole',
        ''])

def test_place_rbend():
    env = xt.Environment()
    env['angle'] = 0.5

    l = env.new_line(compose=True, length=10.0)
    l.new('rbend1', 'RBend', length_straight=1, angle='angle', at=5.0)

    t1a = l.get_table()
    # t1a.cols['s s_center s_start s_end'] is:
    # Table: 4 rows, 5 cols
    # name                   s      s_center       s_start         s_end
    # drift_1                0       2.24738             0       4.49475
    # rbend1           4.49475             5       4.49475       5.50525
    # drift_2          5.50525       7.75262       5.50525            10
    # _end_point            10            10            10            10
    xo.assert_allclose(t1a['s_center', 'rbend1'], 5, rtol=0, atol=1e-12)
    xo.assert_allclose(l.get_length(), 10, rtol=0, atol=1e-12)
    xo.assert_allclose(t1a.s, [0, 4.49475287, 5.50524713, 10],
            rtol=0, atol=1e-5)
    xo.assert_allclose(t1a.s_center, [2.24737644, 5, 7.75262356, 10],
            rtol=0, atol=1e-5)
    xo.assert_allclose(t1a.s_start, [0, 4.49475287, 5.50524713, 10],
            rtol=0, atol=1e-5)
    xo.assert_allclose(t1a.s_end, [4.49475287, 5.50524713, 10, 10],
            rtol=0, atol=1e-5)

    env['angle'] = 0.4
    l.regenerate_from_composer()

    t1b = l.get_table()
    # t1b.cols['s s_center s_start s_end'] is:
    # Table: 4 rows, 5 cols
    # name                   s      s_center       s_start         s_end
    # drift_3                0       2.24833             0       4.49665
    # rbend1           4.49665             5       4.49665       5.50335
    # drift_4          5.50335       7.75167       5.50335            10
    # _end_point            10            10            10            10
    xo.assert_allclose(t1b['s_center', 'rbend1'], 5, rtol=0, atol=1e-12)
    xo.assert_allclose(l.get_length(), 10, rtol=0, atol=1e-12)
    xo.assert_allclose(t1b.s, [0, 4.49665277, 5.50334723, 10],
            rtol=0, atol=1e-5)
    xo.assert_allclose(t1b.s_center, [2.24832639, 5, 7.75167361, 10],
            rtol=0, atol=1e-5)
    xo.assert_allclose(t1b.s_start, [0, 4.49665277, 5.50334723, 10],
            rtol=0, atol=1e-5)
    xo.assert_allclose(t1b.s_end, [4.49665277, 5.50334723, 10, 10],
            rtol=0, atol=1e-5)

    l2 = env.new_line(compose=True, length=10.0)
    l2.new('rbend2', 'RBend', length_straight=1, angle='angle', anchor='end', at=5.0)
    t2a = l2.get_table()
    # t2a.cols['s s_center s_start s_end'] is:
    # Table: 4 rows, 5 cols
    # name                   s      s_center       s_start         s_end
    # drift_5                0       1.99665             0        3.9933
    # rbend2            3.9933       4.49665        3.9933             5
    # drift_6                5           7.5             5            10
    # _end_point            10            10            10            10
    xo.assert_allclose(t2a['s_end', 'rbend2'], 5, rtol=0, atol=1e-12)
    xo.assert_allclose(l2.get_length(), 10, rtol=0, atol=1e-12)
    xo.assert_allclose(t2a.s, [ 0.,  3.99330209,  5., 10.],
            rtol=0, atol=1e-5)
    xo.assert_allclose(t2a.s_center, [ 1.99665105, 4.49665105, 7.5, 10.],
            rtol=0, atol=1e-5)
    xo.assert_allclose(t2a.s_start, [0. , 3.993302, 5., 10.],
            rtol=0, atol=1e-5)
    xo.assert_allclose(t2a.s_end, [3.99330209, 5, 10, 10],
            rtol=0, atol=1e-5)

def test_rename_var():
    env = xt.Environment()
    env['aa'] = 5
    env['bb'] = '2 * aa'

    env.new('mb', 'Bend', length=1.0, angle='bb * 1e-3', knl=[0.0, '3*bb'])

    assert str(env.ref['bb']._expr) == "(2.0 * vars['aa'])"
    assert str(env.ref['mb'].angle._expr) == "(vars['bb'] * 0.001)"
    assert str(env.ref['mb'].knl[1]._expr) == "(3.0 * vars['bb'])"

    env.vars.rename('bb', 'cc', verbose=True)

    assert str(env.ref['cc']._expr) == "(2.0 * vars['aa'])"
    assert str(env.ref['mb'].angle._expr) == "(vars['cc'] * 0.001)"
    assert str(env.ref['mb'].knl[1]._expr) == "(3.0 * vars['cc'])"

def test_parametric_line_update():

    env = xt.Environment()
    env.particle_ref = xt.Particles(kinetic_energy0=2.86e9, mass0 = xt.ELECTRON_MASS_EV)

    env['l_cell']  = 5.00  # cell length is 5 m
    env['l_bend']  = 1.50  # length of bend (along arc using sector bends)
    env['l_quad']  = 0.40  # length of quads
    env['l_drift'] = '(l_cell - 2*l_bend - 2*l_quad)/4.'  # remaining length for drifts
    print( f"Initial value of env['l_drift'] ={env['l_drift']:6.3f}" )

    env['alfB'] =  np.pi/20 # 20 cells each with two bends in 100 m ring
    env['kQf']  =  0.7
    env['kQd']  = -0.7

    # Definition of elements and two ways to define a FODO cell
    env.new('drift', xt.Drift, length='l_drift')
    env.new('mb',    xt.Bend, length='l_bend', angle='alfB', k0_from_h=True,
            edge_entry_angle='alfB/2', edge_exit_angle='alfB/2') # shoild be kind of RBend

    env.new('mQf', xt.Quadrupole, length='l_quad', k1='kQf')
    env.new('mQd', xt.Quadrupole, length='l_quad', k1='kQd')

    cell_line = env.new_line( components =[  # analogeous to MAD LINE
        env.place('mQf'), env.place('drift'), env.place('mb'), env.place('drift'),
        env.place('mQd'), env.place('drift'), env.place('mb'), env.place('drift'),
        ])

    cell_sequ1 = env.new_line( length='l_cell', components =[  # analogeous to MAD Sequence
        env.place('mQf', at='0*l_drift + 0.5*l_quad + 0.0*l_bend'),
        env.place('mb',  at='1*l_drift + 1.0*l_quad + 0.5*l_bend'),
        env.place('mQd', at='2*l_drift + 1.5*l_quad + 1.0*l_bend'),
        env.place('mb',  at='3*l_drift + 2.0*l_quad + 1.5*l_bend'),
        ])

    cell_sequ2 = env.new_line( length='l_cell', components =[  # analogeous to MAD Sequence
        env.place('mQf', at='0*(l_cell - 2*l_bend - 2*l_quad)/4. + 0.5*l_quad + 0.0*l_bend'),
        env.place('mb',  at='1*(l_cell - 2*l_bend - 2*l_quad)/4. + 1.0*l_quad + 0.5*l_bend'),
        env.place('mQd', at='2*(l_cell - 2*l_bend - 2*l_quad)/4. + 1.5*l_quad + 1.0*l_bend'),
        env.place('mb',  at='3*(l_cell - 2*l_bend - 2*l_quad)/4. + 2.0*l_quad + 1.5*l_bend'),
        ])

    t_cell_line = cell_line.get_table()
    t_cell_sequ1 = cell_sequ1.get_table()
    t_cell_sequ2 = cell_sequ2.get_table()
    for tt in [t_cell_line, t_cell_sequ1, t_cell_sequ2]:
        xo.assert_allclose(tt.s, [0. , 0.4, 0.7, 2.2, 2.5, 2.9, 3.2, 4.7, 5. ],
                           atol=1e-14)

    env['l_quad'] = 0.30
    cell_sequ1.regenerate_from_composer()
    cell_sequ2.regenerate_from_composer()
    t_cell_line = cell_line.get_table()
    t_cell_sequ1 = cell_sequ1.get_table()
    t_cell_sequ2 = cell_sequ2.get_table()
    for tt in [t_cell_line, t_cell_sequ1, t_cell_sequ2]:
        xo.assert_allclose(tt.s, [0.  , 0.3 , 0.65, 2.15, 2.5 , 2.8 , 3.15, 4.65, 5.],
                           atol=1e-14)

    # back to original
    env['l_quad'] = 0.40
    cell_sequ1.regenerate_from_composer()
    cell_sequ2.regenerate_from_composer()
    t_cell_line = cell_line.get_table()
    t_cell_sequ1 = cell_sequ1.get_table()
    t_cell_sequ2 = cell_sequ2.get_table()
    for tt in [t_cell_line, t_cell_sequ1, t_cell_sequ2]:
        xo.assert_allclose(tt.s, [0. , 0.4, 0.7, 2.2, 2.5, 2.9, 3.2, 4.7, 5. ],
                           atol=1e-14)

    # increased cell length
    env['l_cell'] = 5.50
    cell_sequ1.regenerate_from_composer()
    cell_sequ2.regenerate_from_composer()
    t_cell_line = cell_line.get_table()
    t_cell_sequ1 = cell_sequ1.get_table()
    t_cell_sequ2 = cell_sequ2.get_table()
    for tt in [t_cell_line, t_cell_sequ1, t_cell_sequ2]:
        xo.assert_allclose(tt.s,
                [0.   , 0.4  , 0.825, 2.325, 2.75 , 3.15 , 3.575, 5.075, 5.5  ],
                atol=1e-14)

def test_str_in_composer_to_dict_from_dict():
    env = xt.Environment()

    line = env.new_line(components=[
        env.new('q1', 'Quadrupole', length=1.0, at=2),
        'q1',
        'q1']
    )
    assert isinstance(line.composer.components[0], xt.Place)
    assert line.composer.components[0].name == 'q1'
    assert line.composer.components[0].at == 2
    assert line.composer.components[1] == 'q1'
    assert line.composer.components[2] == 'q1'

    line2 = xt.Line.from_dict(line.to_dict())
    assert isinstance(line2.composer.components[0], xt.Place)
    assert line2.composer.components[0].name == 'q1'
    assert line2.composer.components[0].at == 2
    assert line2.composer.components[1] == 'q1'
    assert line2.composer.components[2] == 'q1'

def test_sandwitch_thin_elements():

    # Create an environment
    env = xt.Environment()

    env.new('m0', xt.Marker)
    env.new('m1', xt.Marker)
    env.new('m2', xt.Marker)
    env.new('m3', xt.Marker)
    env.new('m4', xt.Marker)
    env.new('m5', xt.Marker)
    env.new('m6', xt.Marker)
    env.new('m7', xt.Marker)
    env.new('m8', xt.Marker)
    env.new('m9', xt.Marker)
    env.new('m10', xt.Marker)

    env.new_line(name='myline', compose=True)
    composer = env['myline'].composer

    composer.components.extend([
        env.place('m0'),
        env.place('m3', at=10.),
        env.place(['m1', 'm2']),
        env.place(['m6', 'm7'], at='m3@end'),
        env.place(['m4', 'm5'], at='m3@start'),
        env.place('m8', at=10, from_='m0'),
        env.place('m9', at=20.),
        env.place('m10', at=-10, from_='m9'),
    ])

    tt_unsorted = composer.resolve_s_positions(sort=False)
    tt_unsorted.cols['s from_ from_anchor'].show()
    # prints:
    # name             s from_ from_anchor
    # m0               0 None  None
    # m3              10 None  None
    # m1              10 m3    end
    # m2              10 m1    end
    # m6              10 m3    end
    # m7              10 m6    end
    # m4              10 m3    start
    # m5              10 m4    end
    # m8              10 m0    None
    # m9              20 None  None
    # m10             10 m9    None

    assert np.all(tt_unsorted.name == [
        'm0', 'm3', 'm1', 'm2', 'm6', 'm7', 'm4', 'm5', 'm8', 'm9', 'm10'
    ])
    xo.assert_allclose(tt_unsorted.s, [
        0., 10., 10., 10., 10., 10., 10., 10., 10., 20., 10.])
    assert np.all(tt_unsorted.from_ == [
        None, None, 'm3', 'm1', 'm3', 'm6', 'm3', 'm4', 'm0', None, 'm9'
    ])
    assert np.all(tt_unsorted.from_anchor == [
        None, None, 'end', 'end', 'end', 'end', 'start', 'end', None, None, None
    ])

    tt_sorted = composer.resolve_s_positions(sort=True)
    tt_sorted.cols['s from_ from_anchor'].show()
    # prints:
    # name             s from_ from_anchor
    # m0               0 None  None
    # m8              10 m0    None
    # m4              10 m3    start
    # m5              10 m4    end
    # m3              10 None  None
    # m1              10 m3    end
    # m2              10 m1    end
    # m6              10 m3    end
    # m7              10 m6    end
    # m10             10 m9    None
    # m9              20 None  None

    assert np.all(tt_sorted.name == [
        'm0', 'm8', 'm4', 'm5', 'm3', 'm1', 'm2', 'm6', 'm7', 'm10', 'm9'
    ])
    xo.assert_allclose(tt_sorted.s, [
        0., 10., 10., 10., 10., 10., 10., 10., 10., 10., 20.])
    assert np.all(tt_sorted.from_ == [
        None, 'm0', 'm3', 'm4', None, 'm3', 'm1', 'm3', 'm6', 'm9', None
    ])
    assert np.all(tt_sorted.from_anchor == [
        None, None, 'start', 'end', None, 'end', 'end', 'end', 'end', None, None
    ])

def test_sandwitch_thin_elements_insert():

    env = xt.Environment()

    # Create a line with two quadrupoles and a marker
    line = env.new_line(name='myline', components=[
        env.new('q0', xt.Quadrupole, length=2.0, at=10.),
        env.new('q1', xt.Quadrupole, length=2.0, at=20.),
        env.new('m0', xt.Marker, at=40.),
        ])

    # Create a set of new elements to be placed
    env.new('s1', xt.Sextupole, length=0.1, k2=0.2)
    env.new('s2', xt.Sextupole, length=0.1, k2=-0.2)
    env.new('m1', xt.Marker)
    env.new('m2', xt.Marker)
    env.new('m3', xt.Marker)

    # Insert the new elements in the line
    line.insert([
        env.place('s1', at=5.),
        env.place('s2', anchor='end', at=-5., from_='q1@start'),
        env.place(['m1', 'm2'], at='m0@start'),
        env.place('m3', at='m0@end'),
        ])

    tt = line.get_table()
    tt.show(cols=['s_start', 's_center', 's_end'])
    # is:
    # name             s_start      s_center         s_end
    # ||drift_4              0         2.475          4.95
    # s1                  4.95             5          5.05
    # ||drift_6           5.05         7.025             9
    # q0                     9            10            11
    # ||drift_7             11         12.45          13.9
    # s2                  13.9         13.95            14
    # ||drift_8             14          16.5            19
    # q1                    19            20            21
    # ||drift_3             21          30.5            40
    # m1                    40            40            40
    # m2                    40            40            40
    # m0                    40            40            40
    # m3                    40            40            40
    # _end_point            40            40            40

    assert np.all(tt.name == [
        '||drift_4', 's1', '||drift_6', 'q0', '||drift_7', 's2',
        '||drift_8', 'q1', '||drift_3', 'm1', 'm2', 'm0', 'm3', '_end_point'
    ])
    xo.assert_allclose(tt.s_start, [
        0., 4.95, 5.05, 9., 11., 13.9, 14., 19., 21., 40., 40., 40., 40., 40.],
        atol=1e-14)
    xo.assert_allclose(tt.s_center, [
        2.475, 5., 7.025, 10., 12.45, 13.95, 16.5, 20., 30.5, 40., 40., 40., 40., 40.],
        atol=1e-14)
    xo.assert_allclose(tt.s_end, [
        4.95, 5.05, 9., 11., 13.9, 14., 19., 21., 40., 40., 40., 40., 40., 40.],
        atol=1e-14)
