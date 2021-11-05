import json

import numpy as np

import sixtracktools

import xobjects as xo
import xtrack as xt
import xpart as xp
import xfields as xf


##############
# Build line #
##############

# Read sixtrack input
sixinput = sixtracktools.SixInput(".")
p0c_eV = sixinput.initialconditions[-3] * 1e6

# Build xline line from sixtrack input
line = xt.Line.from_sixinput(sixinput)
tracker = xt.Tracker(line=line)

part = xp.Particles(
        p0c=p0c_eV,
        mass0= 938272088.1604904,
        x=1e-4,
        px=1e-8,
        y=1e-6,
        py=1e-8,
        zeta=1e-3,
        delta=1e-6)

#tracker.track(part)

partco=part.copy()
xf.configure_orbit_dependent_parameters_for_bb(tracker,
                       particle_on_co=partco, xline=line)

line.elements[2458].d_px


#simple 1 turn
line = xt.Line.from_sixinput(sixinput)
tracker = xt.Tracker(line=line)
part = xp.Particles(
        p0c=p0c_eV,
        mass0= 938272088.1604904,
        x=1e-6,
        px=1e-8,
        y=1e-6,
        py=1e-8,
        zeta=1e-3,
        delta=1e-6)

out=[]
for ii in range(len(line.elements)):
    tracker.track(part,ele_start=ii,num_elements=1)
    out.append(part.copy())


#simple
line = xt.Line.from_sixinput(sixinput)
tracker = xt.Tracker(line=line)
bb=tracker.line.elements[2458]



bb.mean_x=0.04531964292137445
bb.mean_x
bb.fieldmap.mean_x
bb.fieldmap._xobject.mean_x
bb._xobject.fieldmap.mean_x


part = xp.Particles(
        p0c=p0c_eV,
        mass0= 938272088.1604904,
        x=-0.0017891572282001948,
        px=0e-8,
        y=-0.009783090252917367,
        py=0e-8,
        zeta=0e-3,
        delta=0e-6)

bb.track(part)
part.px[0], part.py[0]






