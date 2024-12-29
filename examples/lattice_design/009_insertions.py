import xtrack as xt
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

# General rule: I want to keep anything I can!

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
