import xtrack as xt
import numpy as np

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
       'drift_1', 'qq1_thick_entry', 'qq1_thick..0', 'qq1_thick..1',
       'qq1_thick_exit', 'drift_2', 'qq1_thin_entry', 'drift_qq1_thin..0',
       'qq1_thin..0', 'drift_qq1_thin..1', 'qq1_thin..1',
       'drift_qq1_thin..2', 'qq1_thin_exit', 'drift_3',
       'qq_shared_thick_entry', 'qq_shared_thick..0/line1',
       'qq_shared_thick..1/line1', 'qq_shared_thick_exit', 'drift_4',
       'qq_shared_thin_entry', 'drift_qq_shared_thin..0/line1',
       'qq_shared_thin..0/line1', 'drift_qq_shared_thin..1/line1',
       'qq_shared_thin..1/line1', 'drift_qq_shared_thin..2/line1',
       'qq_shared_thin_exit', '_end_point']))

assert np.all(tt1.element_type == np.array([
       'Drift', 'Marker', 'ThickSliceQuadrupole', 'ThickSliceQuadrupole',
       'Marker', 'Drift', 'Marker', 'DriftSliceQuadrupole',
       'ThinSliceQuadrupole', 'DriftSliceQuadrupole',
       'ThinSliceQuadrupole', 'DriftSliceQuadrupole', 'Marker', 'Drift',
       'Marker', 'ThickSliceQuadrupole', 'ThickSliceQuadrupole', 'Marker',
       'Drift', 'Marker', 'DriftSliceQuadrupole', 'ThinSliceQuadrupole',
       'DriftSliceQuadrupole', 'ThinSliceQuadrupole',
       'DriftSliceQuadrupole', 'Marker', '']))

assert np.all(tt1.parent_name == np.array([
       None, None, 'qq1_thick', 'qq1_thick', None, None, None, 'qq1_thin',
       'qq1_thin', 'qq1_thin', 'qq1_thin', 'qq1_thin', None, None, None,
       'qq_shared_thick/line1', 'qq_shared_thick/line1', None, None, None,
       'qq_shared_thin/line1', 'qq_shared_thin/line1',
       'qq_shared_thin/line1', 'qq_shared_thin/line1',
       'qq_shared_thin/line1', None, None]))

assert np.all(tt2.name == np.array([
       'drift_5', 'qq2_thick_entry', 'qq2_thick..0', 'qq2_thick..1',
       'qq2_thick_exit', 'drift_6', 'qq2_thin_entry', 'drift_qq2_thin..0',
       'qq2_thin..0', 'drift_qq2_thin..1', 'qq2_thin..1',
       'drift_qq2_thin..2', 'qq2_thin_exit', 'drift_7',
       'qq_shared_thick_entry', 'qq_shared_thick..0/line2',
       'qq_shared_thick..1/line2', 'qq_shared_thick_exit', 'drift_8',
       'qq_shared_thin_entry', 'drift_qq_shared_thin..0/line2',
       'qq_shared_thin..0/line2', 'drift_qq_shared_thin..1/line2',
       'qq_shared_thin..1/line2', 'drift_qq_shared_thin..2/line2',
       'qq_shared_thin_exit', '_end_point']))

assert np.all(tt2.element_type == np.array([
       'Drift', 'Marker', 'ThickSliceQuadrupole', 'ThickSliceQuadrupole',
       'Marker', 'Drift', 'Marker', 'DriftSliceQuadrupole',
       'ThinSliceQuadrupole', 'DriftSliceQuadrupole',
       'ThinSliceQuadrupole', 'DriftSliceQuadrupole', 'Marker', 'Drift',
       'Marker', 'ThickSliceQuadrupole', 'ThickSliceQuadrupole', 'Marker',
       'Drift', 'Marker', 'DriftSliceQuadrupole', 'ThinSliceQuadrupole',
       'DriftSliceQuadrupole', 'ThinSliceQuadrupole',
       'DriftSliceQuadrupole', 'Marker', '']))

assert np.all(tt2.parent_name == np.array([
       None, None, 'qq2_thick', 'qq2_thick', None, None, None, 'qq2_thin',
       'qq2_thin', 'qq2_thin', 'qq2_thin', 'qq2_thin', None, None, None,
       'qq_shared_thick/line2', 'qq_shared_thick/line2', None, None, None,
       'qq_shared_thin/line2', 'qq_shared_thin/line2',
       'qq_shared_thin/line2', 'qq_shared_thin/line2',
       'qq_shared_thin/line2', None, None]))

assert 'qq1_thick' in env.element_dict
assert 'qq1_thin' in env.element_dict
assert 'qq_shared_thick/line1' in env.element_dict
assert 'qq_shared_thin/line1' in env.element_dict
assert 'qq2_thick' in env.element_dict
assert 'qq2_thin' in env.element_dict
assert 'qq_shared_thick/line2' in env.element_dict
assert 'qq_shared_thin/line2' in env.element_dict

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
