import xtrack as xt
import xobjects as xo
import numpy as np

env = xt.Environment()

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
tt0 = line0.get_table()
tt0.show(cols=['name', 's_start', 's_end', 's_center'])

line = line0.copy()
line.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(None),
        xt.Strategy(slicing=xt.Teapot(2), name=r'q0.*'),
        xt.Strategy(slicing=xt.Teapot(3), name=r'b0.*'),
    ]
)
tt = line.get_table()
tt.show(cols=['name', 's_start', 's_end', 's_center'])