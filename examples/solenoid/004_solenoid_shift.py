import xtrack as xt
import numpy as np

# TODO:
# - field for radiation and spin

env = xt.Environment()
env.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, p0c=20e9)

line = env.new_line(components=[
    env.new('sol',xt.UniformSolenoid, length=3, ks=0.2)
    ])

line.cut_at_s(np.linspace(0, 3, 5))

tw = line.twiss(x=0.1, y=0.2, betx=1, bety=1)

