# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np
import xtrack as xt

# Build a simple line with four bending magnets and three quadrupoles.
env = xt.Environment(particle_ref=xt.Particles(p0c=1e9))

line = env.new_line(length=12, components=[
    env.new('b1', xt.Bend, length=0.5, angle=np.deg2rad(22.5),
            k0_from_h=False, at=1.5),
    env.new('qf1', xt.Quadrupole, length=0.3, k1=0.4, at=2.8),
    env.new('b2', xt.Bend, length=0.5, angle=-np.deg2rad(22.5),
            k0_from_h=False, at=4.0),
    env.new('qd1', xt.Quadrupole, length=0.3, k1=-0.4, at=5.5),
    env.new('b3', xt.Bend, length=0.5, angle=-np.deg2rad(22.5),
            k0_from_h=False, at=8.0),
    env.new('qf2', xt.Quadrupole, length=0.3, k1=0.4, at=9.2),
    env.new('b4', xt.Bend, length=0.5, angle=np.deg2rad(22.5),
            k0_from_h=False, at=10.5),
])

# Compute the survey.
survey = line.survey()

# Inspect selected columns of the survey table.
survey.cols['name s X Y Z theta phi psi']

# Make a floor plot of the reference trajectory in the Z-X plane.
import matplotlib.pyplot as plt
plt.close('all')

survey.plot(
    projection='ZX',
    labels=['b1', 'b2', 'b3', 'b4'],
    element_width=0.12,
    figsize=(6.4, 4.8),
)

fig1 = plt.gcf()
plt.title('Survey floor plot')
fig1.subplots_adjust(left=.13, right=.95, bottom=.13, top=.90)
plt.show()
