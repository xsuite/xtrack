import xtrack as xt
import xobjects as xo
import numpy as np

env = xt.Environment()

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
        env.new('s2', 's1', anchor='start', at=0.3, from_='end@s1'),
        env.new('s3', 's1', anchor='start', at=0.3, from_='end@s2')
    ])

line.insert(ln_insert, anchor='start', at=1, from_='end@q0')

tt = line.get_table()
tt.show(cols=['name', 's_start', 's_end', 's_center'])