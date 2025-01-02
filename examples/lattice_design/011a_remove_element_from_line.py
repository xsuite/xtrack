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
        env.new('mk2', 'Marker', at=42),
        env.new('q0', 'Quadrupole'),
        env.new('end', 'Marker', at=50.),
    ])

line1 = line0.copy()
line1.remove('q0::1')
tt1 = line1.get_table()
tt1.show(cols=['name', 's_start', 's_end', 's_center'])