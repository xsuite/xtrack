import xtrack as xt
import xobjects as xo
import numpy as np

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
    ['drift_1', 'ql', 'drift_2', 'q0', 'drift_3', 'qr', 'drift_4',
       'mk1', 'mk2', 'mk3', 'drift_6', 'drift_5', 'end', '_end_point']))
xo.assert_allclose(tt1.s_center, np.array(
    [ 4.5, 10. , 15. , 20. , 25. , 30. , 35.5, 40. , 40. , 40. , 41. ,
       46. , 50. , 50. ]), rtol=0., atol=1e-14)

line2 = line0.copy()
line2.remove('q0')
tt2 = line2.get_table()
tt2.show(cols=['name', 's_start', 's_end', 's_center'])

assert np.all(tt2.name == np.array(
    ['drift_1', 'ql', 'drift_2', 'drift_6', 'drift_3', 'qr', 'drift_4',
       'mk1', 'mk2', 'mk3', 'drift_7', 'drift_5', 'end', '_end_point']))
xo.assert_allclose(tt2.s_center, np.array(
    [ 4.5, 10. , 15. , 20. , 25. , 30. , 35.5, 40. , 40. , 40. , 41. ,
       46. , 50. , 50. ]), rtol=0., atol=1e-14)

line3 = line0.copy()
line3.remove('q.*')
tt3 = line3.get_table()
tt3.show(cols=['name', 's_start', 's_end', 's_center'])

assert np.all(tt3.name == np.array(
    ['drift_1', 'drift_6', 'drift_2', 'drift_7', 'drift_3', 'drift_8',
     'drift_4', 'mk1', 'mk2', 'mk3', 'drift_9', 'drift_5', 'end',
     '_end_point']))
xo.assert_allclose(tt3.s_center, np.array(
    [ 4.5, 10. , 15. , 20. , 25. , 30. , 35.5, 40. , 40. , 40. , 41. ,
       46. , 50. , 50. ]), rtol=0., atol=1e-14)

line4 = line0.copy()
line4.remove('mk2')
tt4 = line4.get_table()
tt4.show(cols=['name', 's_start', 's_end', 's_center'])

assert np.all(tt4.name == np.array(
    ['drift_1', 'ql', 'drift_2', 'q0::0', 'drift_3', 'qr', 'drift_4',
       'mk1', 'mk3', 'q0::1', 'drift_5', 'end', '_end_point']))
xo.assert_allclose(tt4.s_center, np.array(
    [ 4.5, 10. , 15. , 20. , 25. , 30. , 35.5, 40. , 40. , 41. , 46. ,
       50. , 50. ]), rtol=0., atol=1e-14)

line5 = line0.copy()
line5.remove('mk.*')
tt5 = line5.get_table()
tt5.show(cols=['name', 's_start', 's_end', 's_center'])

assert np.all(tt5.name == np.array(
    ['drift_1', 'ql', 'drift_2', 'q0::0', 'drift_3', 'qr', 'drift_4',
       'q0::1', 'drift_5', 'end', '_end_point']))
xo.assert_allclose(tt5.s_center, np.array(
    [ 4.5, 10. , 15. , 20. , 25. , 30. , 35.5, 41. , 46. , 50. , 50. ]),
    rtol=0., atol=1e-14)