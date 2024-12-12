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
            h=5.)
    assert env['bb'].k0 == 2 * (2 * 4 + 5)
    assert env['bb'].length == 3 + 4 + 2 * 4 + 5
    assert env['bb'].h == 5.

    env.vars['a'] = 2.
    assert env['bb'].k0 == 2 * (2 * 2 + 5)
    assert env['bb'].length == 3 + 2 + 2 * 2 + 5
    assert env['bb'].h == 5.

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
    assert line['bb1'].h == 5.

    assert line['bb'].k0 == 2 * (2 * a + 5)
    assert line['bb'].length == 3 + a + 2 * a + 5
    assert line['bb'].h == 5.

    tt = line.get_table(attr=True)
    tt['s_center'] = tt['s'] + tt['length']/2

    assert np.all(tt.name ==  np.array(['drift_1', 'bb1', 'drift_2', 'bb', '_end_point']))

    assert tt['s_center', 'bb1'] == 2*a
    assert tt['s_center', 'bb'] - tt['s_center', 'bb1'] == 10*a

    old_a = a
    line.vars['a'] = 3.
    a = line.vv['a']
    assert line['bb1'].length == 3 * a
    assert line['bb1'].k0 == 2 * (2 * a + 5)
    assert line['bb1'].h == 5.

    assert line['bb'].k0 == 2 * (2 * a + 5)
    assert line['bb'].length == 3 + a + 2 * a + 5
    assert line['bb'].h == 5.

    tt_new = line.get_table(attr=True)

    # Drifts are not changed:
    tt_new['length', 'drift_1'] == tt['length', 'drift_1']
    tt_new['length', 'drift_2'] == tt['length', 'drift_2']

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
        (
            env.new('before_before_right', xt.Marker),
            env.new('before_right', xt.Sextupole, length=1),
            env.new('right',xt.Quadrupole, length=0.8, at='s.right', from_='ip'),
            env.new('after_right', xt.Marker),
            env.new('after_right2', xt.Marker),
        ),
        env.new('left', xt.Quadrupole, length=1, at='s.left', from_='ip'),
        env.new('after_left', xt.Marker),
        env.new('after_left2', xt.Bend, length='l.after_left2'),
    ])

    tt = line.get_table(attr=True)
    tt['s_center'] = tt['s'] + tt['length']/2
    assert np.all(tt.name == np.array([
        'b1', 'q1', 'drift_1', 'left', 'after_left', 'after_left2',
        'drift_2', 'ip', 'drift_3', 'before_before_right', 'before_right',
        'right', 'after_right', 'after_right2', '_end_point']))

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

    env.new('mb', xt.Bend, length='l.mb', k0='k0.mb', h='k0.mb')
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
        ['drift_1', 'corrector', 'drift_2', 'mq', 'drift_3', 'ms', '_end_point']))
    tt_girder['s_center'] = tt_girder['s'] + \
        tt_girder['length']/2 * np.float64(tt_girder['isthick'])
    xo.assert_allclose(tt_girder['s_center', 'mq'], 1., atol=1e-14, rtol=0)
    xo.assert_allclose(tt_girder['s_center', 'ms'] - tt_girder['s_center', 'mq'], 0.8,
                    atol=1e-14, rtol=0)
    xo.assert_allclose(
        tt_girder['s_center', 'corrector'] - tt_girder['s_center', 'mq'], -0.8,
        atol=1e-14, rtol=0)


    girder_f = girder.clone(name='f')
    girder_d = girder.clone(name='d', mirror=True)
    env.set('mq.f', k1='kqf')
    env.set('mq.d', k1='kqd')

    # Check clone
    tt_girder_f = girder_f.get_table(attr=True)
    assert (~(tt_girder_f.isreplica)).all()
    assert np.all(tt_girder_f.name == np.array(
        ['drift_1.f', 'corrector.f', 'drift_2.f', 'mq.f', 'drift_3.f', 'ms.f', '_end_point']))
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
        ['ms.d', 'drift_3.d', 'mq.d', 'drift_2.d', 'corrector.d', 'drift_1.d', '_end_point']))
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
        ['drift_4', 'ms.d', 'drift_3.d', 'mq.d', 'drift_2.d', 'corrector.d',
        'drift_1.d', 'drift_5', 'mb.1', 'drift_6', 'mb.2', 'drift_7',
        'mb.3', 'drift_8', 'drift_1.f', 'corrector.f', 'drift_2.f', 'mq.f',
        'drift_3.f', 'ms.f', 'drift_9', 'mid', '_end_point']))
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


    hcell_left = halfcell.replicate(name='l', mirror=True)
    hcell_right = halfcell.replicate(name='r')

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
        ['start', 'mid.l', 'drift_9.l', 'ms.f.l', 'drift_3.f.l', 'mq.f.l',
        'drift_2.f.l', 'corrector.f.l', 'drift_1.f.l', 'drift_8.l',
        'mb.3.l', 'drift_7.l', 'mb.2.l', 'drift_6.l', 'mb.1.l',
        'drift_5.l', 'drift_1.d.l', 'corrector.d.l', 'drift_2.d.l',
        'mq.d.l', 'drift_3.d.l', 'ms.d.l', 'drift_4.l', 'drift_4.r',
        'ms.d.r', 'drift_3.d.r', 'mq.d.r', 'drift_2.d.r', 'corrector.d.r',
        'drift_1.d.r', 'drift_5.r', 'mb.1.r', 'drift_6.r', 'mb.2.r',
        'drift_7.r', 'mb.3.r', 'drift_8.r', 'drift_1.f.r', 'corrector.f.r',
        'drift_2.f.r', 'mq.f.r', 'drift_3.f.r', 'ms.f.r', 'drift_9.r',
        'mid.r', 'end', '_end_point']))
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
        [False,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
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

        env.new('mid', xt.Marker, at='l.halfcell'),

        env.new('mq.ss.d', 'mq', k1='kqd.ss', at = '0.5 + l.mq / 2'),
        env.new('mq.ss.f', 'mq', k1='kqf.ss', at = 'l.halfcell - l.mq / 2 - 0.5'),

        env.new('corrector.ss.v', 'corrector', at=0.75, from_='mq.ss.d'),
        env.new('corrector.ss.h', 'corrector', at=-0.75, from_='mq.ss.f')
    ])

    hcell_left_ss = halfcell_ss.replicate(name='l', mirror=True)
    hcell_right_ss = halfcell_ss.replicate(name='r')
    cell_ss = env.new_line(components=[
        env.new('start.ss', xt.Marker),
        hcell_left_ss,
        hcell_right_ss,
        env.new('end.ss', xt.Marker),
    ])


    arc = env.new_line(components=[
        cell.replicate(name='cell.1'),
        cell.replicate(name='cell.2'),
        cell.replicate(name='cell.3'),
    ])

    assert 'cell.2' in env.lines
    tt_cell2 = env.lines['cell.2'].get_table(attr=True)
    assert np.all(tt_cell2.name[:-1] == np.array([
        nn+'.cell.2' for nn in tt_cell.name[:-1]]))
    assert np.all(tt_cell2.s == tt_cell.s)
    assert tt_cell2.isreplica[:-1].all()
    assert tt_cell2['parent_name', 'mq.d.l.cell.2'] == 'mq.d.l'
    assert tt_cell2['parent_name', 'mq.f.l.cell.2'] == 'mq.f.l'
    assert tt_cell['parent_name', 'mq.d.l'] == 'mq.d'
    assert tt_cell['parent_name', 'mq.f.l'] == 'mq.f'

    tt_arc = arc.get_table(attr=True)
    assert len(tt_arc) == 3 * (len(tt_cell)-1) + 1
    n_cell = len(tt_cell) - 1
    assert np.all(tt_arc.name[n_cell:2*n_cell] == tt_cell2.name[:-1])
    for nn in tt_cell2.name[:-1]:
        assert arc.get(nn) is env.get(nn)
        assert arc.get(nn) is env['cell.2'].get(nn)

    ss = env.new_line(components=[
        cell_ss.replicate('cell.1'),
        cell_ss.replicate('cell.2'),
    ])

    ring = env.new_line(components=[
        arc.replicate(name='arc.1'),
        ss.replicate(name='ss.1'),
        arc.replicate(name='arc.2'),
        ss.replicate(name='ss.2'),
        arc.replicate(name='arc.3'),
        ss.replicate(name='ss.3'),
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
    assert env.lines['cell3_copy'] is cell3_select
    assert cell3_select._element_dict is env.element_dict
    assert cell3_select.element_names[0] == 'start.cell.3.arc.2'
    assert cell3_select.element_names[-1] == 'end.cell.3.arc.2'
    assert (np.array(cell3_select.element_names) == np.array(
        tw.rows['start.cell.3.arc.2':'end.cell.3.arc.2'].name)).all()

    # Check that they share the _element_dict
    assert cell._element_dict is env.element_dict
    assert halfcell._element_dict is env.element_dict
    assert halfcell_ss._element_dict is env.element_dict
    assert cell_ss._element_dict is env.element_dict
    assert insertion._element_dict is env.element_dict
    assert ring2._element_dict is env.element_dict

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

    env.new('mb', xt.Bend, length='l.mb', k0='k0.mb', h='k0.mb')
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
        ['drift_1', 'corrector', 'drift_2', 'mq', 'drift_3', 'ms', '_end_point']))
    tt_girder['s_center'] = tt_girder['s'] + \
        tt_girder['length']/2 * np.float64(tt_girder['isthick'])
    xo.assert_allclose(tt_girder['s_center', 'mq'], 1., atol=1e-14, rtol=0)
    xo.assert_allclose(tt_girder['s_center', 'ms'] - tt_girder['s_center', 'mq'], 0.8,
                    atol=1e-14, rtol=0)
    xo.assert_allclose(
        tt_girder['s_center', 'corrector'] - tt_girder['s_center', 'mq'], -0.8,
        atol=1e-14, rtol=0)


    girder_f = girder.clone(name='f')
    girder_d = girder.clone(name='d', mirror=True)
    env.set('mq.f', k1='kqf')
    env.set('mq.d', k1='kqd')

    # Check clone
    tt_girder_f = girder_f.get_table(attr=True)
    assert (~(tt_girder_f.isreplica)).all()
    assert np.all(tt_girder_f.name == np.array(
        ['drift_1.f', 'corrector.f', 'drift_2.f', 'mq.f', 'drift_3.f', 'ms.f', '_end_point']))
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
        ['ms.d', 'drift_3.d', 'mq.d', 'drift_2.d', 'corrector.d', 'drift_1.d', '_end_point']))
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
        ['drift_4', 'ms.d', 'drift_3.d', 'mq.d', 'drift_2.d', 'corrector.d',
        'drift_1.d', 'drift_5', 'mb.1', 'drift_6', 'mb.2', 'drift_7',
        'mb.3', 'drift_8', 'drift_1.f', 'corrector.f', 'drift_2.f', 'mq.f',
        'drift_3.f', 'ms.f', 'drift_9', 'mid', '_end_point']))
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


    hcell_left = halfcell.replicate(name='l', mirror=True)
    hcell_right = halfcell.replicate(name='r')

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
        ['start', 'mid.l', 'drift_9.l', 'ms.f.l', 'drift_3.f.l', 'mq.f.l',
        'drift_2.f.l', 'corrector.f.l', 'drift_1.f.l', 'drift_8.l',
        'mb.3.l', 'drift_7.l', 'mb.2.l', 'drift_6.l', 'mb.1.l',
        'drift_5.l', 'drift_1.d.l', 'corrector.d.l', 'drift_2.d.l',
        'mq.d.l', 'drift_3.d.l', 'ms.d.l', 'drift_4.l', 'drift_4.r',
        'ms.d.r', 'drift_3.d.r', 'mq.d.r', 'drift_2.d.r', 'corrector.d.r',
        'drift_1.d.r', 'drift_5.r', 'mb.1.r', 'drift_6.r', 'mb.2.r',
        'drift_7.r', 'mb.3.r', 'drift_8.r', 'drift_1.f.r', 'corrector.f.r',
        'drift_2.f.r', 'mq.f.r', 'drift_3.f.r', 'ms.f.r', 'drift_9.r',
        'mid.r', 'end', '_end_point']))
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
        [False,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,
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

        env.new('mid', xt.Marker, at='l.halfcell'),

        env.new('mq.ss.d', 'mq', k1='kqd.ss', at = '0.5 + l.mq / 2'),
        env.new('mq.ss.f', 'mq', k1='kqf.ss', at = 'l.halfcell - l.mq / 2 - 0.5'),

        env.new('corrector.ss.v', 'corrector', at=0.75, from_='mq.ss.d'),
        env.new('corrector.ss.h', 'corrector', at=-0.75, from_='mq.ss.f')
    ])

    hcell_left_ss = halfcell_ss.replicate(name='l', mirror=True)
    hcell_right_ss = halfcell_ss.replicate(name='r')
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
    assert np.all(tt_cell2.name[:-1] == np.array([
        nn+'.cell.2' for nn in tt_cell.name[:-1]]))
    assert np.all(tt_cell2.s == tt_cell.s)
    assert tt_cell2.isreplica[:-1].all()
    assert tt_cell2['parent_name', 'mq.d.l.cell.2'] == 'mq.d.l'
    assert tt_cell2['parent_name', 'mq.f.l.cell.2'] == 'mq.f.l'
    assert tt_cell['parent_name', 'mq.d.l'] == 'mq.d'
    assert tt_cell['parent_name', 'mq.f.l'] == 'mq.f'

    tt_arc = arc.get_table(attr=True)
    assert len(tt_arc) == 3 * (len(tt_cell)-1) + 1
    n_cell = len(tt_cell) - 1
    assert np.all(tt_arc.name[n_cell:2*n_cell] == tt_cell2.name[:-1])
    for nn in tt_cell2.name[:-1]:
        assert arc.get(nn) is env.get(nn)
        assert arc.get(nn) is env['cell.2'].get(nn)

    ss = env.new_builder()
    ss.new('cell.1', cell_ss, mode='replica')
    ss.new('cell.2', cell_ss, mode='replica')
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
    assert cell3_select._element_dict is env.element_dict
    assert cell3_select.element_names[0] == 'start.cell.3.arc.2'
    assert cell3_select.element_names[-1] == 'end.cell.3.arc.2'
    assert (np.array(cell3_select.element_names) == np.array(
        tw.rows['start.cell.3.arc.2':'end.cell.3.arc.2'].name)).all()

    # Check that they share the _element_dict
    assert cell._element_dict is env.element_dict
    assert halfcell._element_dict is env.element_dict
    assert halfcell_ss._element_dict is env.element_dict
    assert cell_ss._element_dict is env.element_dict
    assert insertion._element_dict is env.element_dict
    assert ring2._element_dict is env.element_dict

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

    env.new('mb', xt.Bend, length='l.mb', k0='k0.mb', h='k0.mb')
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
        ['drift_1', 'corrector', 'drift_2', 'mq', 'drift_3', 'ms', '_end_point']))
    tt_girder['s_center'] = tt_girder['s'] + \
        tt_girder['length']/2 * np.float64(tt_girder['isthick'])
    xo.assert_allclose(tt_girder['s_center', 'mq'], 1., atol=1e-14, rtol=0)
    xo.assert_allclose(tt_girder['s_center', 'ms'] - tt_girder['s_center', 'mq'], 0.8,
                    atol=1e-14, rtol=0)
    xo.assert_allclose(
        tt_girder['s_center', 'corrector'] - tt_girder['s_center', 'mq'], -0.8,
        atol=1e-14, rtol=0)


    girder_f = girder.clone(name='f')
    girder_d = girder.clone(name='d', mirror=True)
    env.set('mq.f', k1='kqf')
    env.set('mq.d', k1='kqd')

    # Check clone
    tt_girder_f = girder_f.get_table(attr=True)
    assert (~(tt_girder_f.isreplica)).all()
    assert np.all(tt_girder_f.name == np.array(
        ['drift_1.f', 'corrector.f', 'drift_2.f', 'mq.f', 'drift_3.f', 'ms.f', '_end_point']))
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
        ['ms.d', 'drift_3.d', 'mq.d', 'drift_2.d', 'corrector.d', 'drift_1.d', '_end_point']))
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
        ['drift_4', 'ms.d', 'drift_3.d', 'mq.d', 'drift_2.d', 'corrector.d',
        'drift_1.d', 'drift_5', 'mb.1', 'drift_6', 'mb.2', 'drift_7',
        'mb.3', 'drift_8', 'drift_1.f', 'corrector.f', 'drift_2.f', 'mq.f',
        'drift_3.f', 'ms.f', 'drift_9', 'mid', '_end_point']))
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
        ['mid', 'drift_9', 'ms.f', 'drift_3.f', 'mq.f', 'drift_2.f',
       'corrector.f', 'drift_1.f', 'drift_8', 'mb.3', 'drift_7', 'mb.2',
       'drift_6', 'mb.1', 'drift_5', 'drift_1.d', 'corrector.d',
       'drift_2.d', 'mq.d', 'drift_3.d', 'ms.d', 'drift_4', 'drift_4',
       'ms.d', 'drift_3.d', 'mq.d', 'drift_2.d', 'corrector.d',
       'drift_1.d', 'drift_5', 'mb.1', 'drift_6', 'mb.2', 'drift_7',
       'mb.3', 'drift_8', 'drift_1.f', 'corrector.f', 'drift_2.f', 'mq.f',
       'drift_3.f', 'ms.f', 'drift_9', 'mid', '_end_point']))
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

        env.new('mid', xt.Marker, at='l.halfcell'),

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
    assert cell._element_dict is env.element_dict
    assert halfcell._element_dict is env.element_dict
    assert halfcell_ss._element_dict is env.element_dict
    assert cell_ss._element_dict is env.element_dict
    assert insertion._element_dict is env.element_dict
    assert ring2._element_dict is env.element_dict

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
    xo.assert_allclose(twarc_start_end.betx, twarc.betx, atol=1e-12, rtol=0)

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

    env.new('mb', 'Bend', extra={'description': 'Hello Riccarco'},
            k1='3*a', h=4*ee.ref['a'], knl=[0, '5*a', 6*ee.ref['a']])
    assert isinstance(ee['mb'].k1, float)
    assert isinstance(ee['mb'].h, float)
    assert isinstance(ee['mb'].knl[0], float)
    assert ee['mb'].k1 == 9
    assert ee['mb'].h == 12
    assert ee['mb'].knl[0] == 0
    assert ee['mb'].knl[1] == 15
    assert ee['mb'].knl[2] == 18

    ee['a'] = 4
    assert ee['a'] == 4
    assert ee['b1'] == 9
    assert ee['b2'] == 12
    assert ee['c'] == 16
    assert ee['mb'].k1 == 12
    assert ee['mb'].h == 16
    assert ee['mb'].knl[0] == 0
    assert ee['mb'].knl[1] == 20
    assert ee['mb'].knl[2] == 24

    ee['mb'].k1 = '30*a'
    ee['mb'].h = 40 * ee.ref['a']
    ee['mb'].knl[1] = '50*a'
    ee['mb'].knl[2] = 60 * ee.ref['a']
    assert ee['mb'].k1 == 120
    assert ee['mb'].h == 160
    assert ee['mb'].knl[0] == 0
    assert ee['mb'].knl[1] == 200
    assert ee['mb'].knl[2] == 240

    assert isinstance(ee['mb'].k1, float)
    assert isinstance(ee['mb'].h, float)
    assert isinstance(ee['mb'].knl[0], float)

    assert ee.ref['mb'].k1._value == 120
    assert ee.ref['mb'].h._value == 160
    assert ee.ref['mb'].knl[0]._value == 0
    assert ee.ref['mb'].knl[1]._value == 200
    assert ee.ref['mb'].knl[2]._value == 240

    assert ee.get('mb').k1 == 120
    assert ee.get('mb').h == 160
    assert ee.get('mb').knl[0] == 0
    assert ee.get('mb').knl[1] == 200
    assert ee.get('mb').knl[2] == 240

    # Some interesting behavior
    assert type(ee['mb']) is xd.madxutils.View
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
    assert isinstance(ret, xt.environment.Place)
    assert ret.name == 'mm'
    assert ret.at == '4*a'
    assert ret.from_ == 'm'
    assert isinstance(env['mm'], xt.Bend)

    ret = env.new('mm1', 'mm', mode='replica', at='5*a', from_='m1')
    assert isinstance(ret, xt.environment.Place)
    assert isinstance(env['mm1'], xt.Replica)
    assert ret.name == 'mm1'
    assert ret.at == '5*a'
    assert ret.from_ == 'm1'
    assert env['mm1'].parent_name == 'mm'

    ret = env.new('mm2', 'mm', mode='clone', at='6*a', from_='m2')
    assert isinstance(ret, xt.environment.Place)
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
    assert isinstance(ret, xt.environment.Place)
    assert ret.at == '5*a'
    assert ret.from_ == 'm'
    assert isinstance(env['ll4'], xt.Line)
    assert env['ll4'].element_names == ['e1.ll4', 'e2.ll4']
    assert isinstance(env['e1.ll4'], xt.Bend)
    assert isinstance(env['e2.ll4'], xt.Bend)

def test_builder_new():

    env = xt.Environment()
    bdr = env.new_builder()
    bdr['a'] = 3.

    bdr.new('m', xt.Bend, k0='3*a')
    assert isinstance(bdr['m'], xt.Bend)
    assert len(bdr.components) == 1

    bdr.new('m1', 'm', mode='replica')
    assert isinstance(bdr['m1'], xt.Replica)
    assert bdr.get('m1').parent_name == 'm'
    assert len(bdr.components) == 2

    bdr.new('m2', 'm', mode='clone')
    assert isinstance(bdr['m2'], xt.Bend)
    str(bdr.ref['m2'].k0._expr) == "(3.0 * vars['a'])"
    assert len(bdr.components) == 3

    ret = bdr.new('mm', xt.Bend, k0='3*a', at='4*a', from_='m')
    assert isinstance(ret, xt.environment.Place)
    assert ret.name == 'mm'
    assert ret.at == '4*a'
    assert ret.from_ == 'm'
    assert isinstance(bdr['mm'], xt.Bend)
    assert len(bdr.components) == 4
    assert bdr.components[-1] is ret

    ret = bdr.new('mm1', 'mm', mode='replica', at='5*a', from_='m1')
    assert isinstance(ret, xt.environment.Place)
    assert isinstance(bdr['mm1'], xt.Replica)
    assert ret.name == 'mm1'
    assert ret.at == '5*a'
    assert ret.from_ == 'm1'
    assert bdr['mm1'].parent_name == 'mm'
    assert len(bdr.components) == 5
    assert bdr.components[-1] is ret

    ret = bdr.new('mm2', 'mm', mode='clone', at='6*a', from_='m2')
    assert isinstance(ret, xt.environment.Place)
    assert isinstance(bdr['mm2'], xt.Bend)
    assert ret.name == 'mm2'
    assert ret.at == '6*a'
    assert ret.from_ == 'm2'
    assert str(bdr.ref['mm2'].k0._expr) == "(3.0 * vars['a'])"
    assert len(bdr.components) == 6
    assert bdr.components[-1] is ret

    env.new('e1', xt.Bend, k0='3*a')
    env.new('e2', xt.Bend)
    line = bdr.new('ll', xt.Line, components=['e1', 'e2'])
    assert isinstance(line, xt.Line)
    assert line.element_names == ['e1', 'e2']
    assert len(bdr.components) == 7
    assert bdr.components[-1] is line

    line = bdr.new('ll1', 'Line', components=['e1', 'e2'])
    assert isinstance(line, xt.Line)
    assert line.element_names == ['e1', 'e2']
    assert len(bdr.components) == 8
    assert bdr.components[-1] is line

    bdr.new('ll2', 'll') # Should be a clone
    assert isinstance(bdr['ll2'], xt.Line)
    assert bdr['ll2'].element_names == ['e1.ll2', 'e2.ll2']
    assert isinstance(bdr['e1.ll2'], xt.Bend)
    assert isinstance(bdr['e2.ll2'], xt.Bend)
    assert bdr.ref['e1.ll2'].k0._expr == "(3.0 * vars['a'])"
    assert len(bdr.components) == 9
    assert bdr.components[-1] is bdr['ll2']

    bdr.new('ll3', 'll', mode='replica')
    assert isinstance(bdr['ll3'], xt.Line)
    assert bdr['ll3'].element_names == ['e1.ll3', 'e2.ll3']
    assert isinstance(bdr['e1.ll3'], xt.Replica)
    assert isinstance(bdr['e2.ll3'], xt.Replica)
    assert bdr['e1.ll3'].parent_name == 'e1'
    assert bdr['e2.ll3'].parent_name == 'e2'
    assert len(bdr.components) == 10
    assert bdr.components[-1] is bdr['ll3']

    ret = bdr.new('ll4', 'll', at='5*a', from_='m')
    assert isinstance(ret, xt.environment.Place)
    assert ret.at == '5*a'
    assert ret.from_ == 'm'
    assert isinstance(bdr['ll4'], xt.Line)
    assert bdr['ll4'].element_names == ['e1.ll4', 'e2.ll4']
    assert isinstance(bdr['e1.ll4'], xt.Bend)
    assert isinstance(bdr['e2.ll4'], xt.Bend)
    assert len(bdr.components) == 11
    assert bdr.components[-1] is ret

def test_neg_line():

    line = xt.Line(elements=[xt.Bend(k0=0.5), xt.Quadrupole(k1=0.1)])

    line_neg = -line

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
        (
            'mb',
            env.new('ip2', 'Marker', at=20),
            'mb',
        ),
        pp
    ])

    tt = line.get_table()
    assert np.all(tt.env_name == np.array(['mb', 'mb', 'drift_1', 'ip1', 'mb',
                                    'mb', 'drift_2', 'mb', 'ip2', 'mb',
                                    'mb', '_end_point']))
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
        ['drift_3', 'mb', 'mb', 'mid', 'mb', 'mb', 'drift_4', 'ip',
        'drift_5', 'mb', 'mb', 'mid', 'mb', 'mb', '_end_point']))
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
    collider= xt.Multiline.from_json(collider_file)
    collider.vars['test_vars'] = 3.1416
    line   = collider[seq]
    line_sel    = line.select(s_marker,e_marker)

    assert line_sel.element_dict is line.element_dict
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
            h=5., ksl=[0, '3*b']),
    ])

    ee = {'env': env, 'line': line}[container_type]

    # Line/Env methods (get, set, eval, get_expr, new_expr, info)
    assert ee.get('b') == 2 * 2 + 1
    assert ee.get('bb') is env.element_dict['bb']

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
