import xtrack as xt
import xobjects as xo
import numpy as np

# TODO:
# - one/multiple thin elements
# - one/multiple thick elements
# - absolute s
# - relative s
# - at start or at end
# - specify length of the line by place(xt.END, at=...)
# - Archors, transform `refer` into `anchor_default`
# - Check on a sliced line
# - Sort out center/centre
# - What happens with repeated elements


env = xt.Environment()

line = env.new_line(
    components=[
        env.new('q0', 'Quadrupole', length=2.0, at=20.0),
        env.new('ql', 'Quadrupole', length=2.0, at=10.0),
        env.new('qr', 'Quadrupole', length=2.0, at=30),
        env.new('end', 'Marker', at=50.),
    ])


line.insert([
    env.new('q4', 'q0', anchor='center', at=0, from_='end@q0'), # will replace half of q0
    env.new('q5', 'q0', at=0, from_='ql'), # will replace the full ql
    env.new('m5.0', 'Marker', at='start@q5'),
    env.new('m5.1', 'Marker', at='start@q5'),
    env.new('m5.2', 'Marker', at='end@q5'),
    env.new('m5.3', 'Marker'),
])

line.insert([
    env.new('q6', 'q0', at=0, from_='qr'),
    env.new('mr.0', 'Marker', at='start@qr'),
    env.new('mr.1', 'Marker', at='start@qr'),
    env.new('mr.2', 'Marker', at='end@qr'),
    env.new('mr.3', 'Marker'),
])

tt = line.get_table()
tt.show(cols=['name', 's_start', 's_end', 's_center'])

assert np.all(tt.name == np.array(
    ['drift_1', 'm5.0', 'm5.1', 'q5', 'm5.2', 'm5.3', 'drift_2',
       'q0_entry', 'q0..0', 'q4', 'drift_3..1', 'mr.0', 'mr.1', 'q6',
       'mr.2', 'mr.3', 'drift_4', 'end', '_end_point']))
xo.assert_allclose(tt.s_center, np.array(
    np.array([ 4.5,  9. ,  9. , 10. , 11. , 11. , 15. , 19. , 19.5, 21. , 25.5,
       29. , 29. , 30. , 31. , 31. , 40.5, 50. , 50.])),
    rtol=0., atol=1e-14)