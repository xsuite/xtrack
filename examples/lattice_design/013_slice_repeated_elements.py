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

line = line0.copy()
line.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(None),
        xt.Strategy(slicing=xt.Teapot(4), name=r'q0.*'),
    ]
)