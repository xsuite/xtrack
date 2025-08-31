import xtrack as xt
import xobjects as xo
import numpy as np

env = xt.Environment()
env.particle_ref = xt.Particles(p0c=1e9)

line = env.new_line(length=10, components=[
    env.new('q1', xt.Quadrupole, length=1, k1=0.3, at=4),
    env.new('q2', xt.Quadrupole, length=1, k1=-0.3, at=6)
])

line.insert('m', xt.Marker(), at=2)

tt = line.get_table()
# is:
# Table: 8 rows, 11 cols
# name                   s element_type isthick isreplica parent_name ...
# drift_1..0             0 DriftSlice      True     False drift_1
# m                      2 Marker         False     False None
# drift_1..1             2 DriftSlice      True     False drift_1
# q1                   3.5 Quadrupole      True     False None
# drift_2              4.5 Drift           True     False None
# q2                   5.5 Quadrupole      True     False None
# drift_3              6.5 Drift           True     False None
# _end_point            10                False     False None

assert np.all(tt.name == [
    'drift_1..0', 'm', 'drift_1..1', 'q1', 'drift_2', 'q2', 'drift_3', '_end_point'])
xo.assert_allclose(tt.s, [0, 2, 2, 3.5, 4.5, 5.5, 6.5, 10], atol=1e-10)
assert np.all(tt.element_type == [
    'DriftSlice', 'Marker', 'DriftSlice', 'Quadrupole', 'Drift', 'Quadrupole', 'Drift', ''])

tw = line.twiss4d()
tw_ng = line.madng_twiss(normal_form=False)

xo.assert_allclose(tw_ng.beta11_ng, tw.betx, rtol=1e-8)
xo.assert_allclose(tw_ng.beta22_ng, tw.bety, rtol=1e-8)