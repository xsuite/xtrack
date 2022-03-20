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

# Build line from sixtrack input
line = xt.Line.from_sixinput(sixinput)

# Info on sixtrack->pyblep conversion
iconv = line.other_info["iconv"]

# Build tracker
tracker = xt.Tracker(line=line)

########################################################
#                  Search closed orbit                 #
# (for comparison purposes we the orbit from sixtrack) #
########################################################

# Load sixtrack tracking data
sixdump_all = sixtracktools.SixDump101("res/dump3.dat")
# Assume first particle to be on the closed orbit
Nele_st = len(iconv)
sixdump_CO = sixdump_all[::2][:Nele_st]

# Get closed-orbit from sixtrack 
part_on_CO = xp.Particles(
        p0c=p0c_eV,
        x=sixdump_CO.x[0],
        px=sixdump_CO.px[0],
        y=sixdump_CO.y[0],
        py=sixdump_CO.py[0],
        zeta=sixdump_CO.zeta[0],
        delta=sixdump_CO.delta[0])

print("Closed orbit at start machine:")
for nn in "x px y py zeta delta".split():
    print(f"    {nn}={getattr(part_on_CO, nn)[0]:.5e}:")


#######################################################
#  Store closed orbit and dipole kicks at BB elements #
#######################################################

xf.configure_orbit_dependent_parameters_for_bb(tracker,
                       particle_on_co=part_on_CO)


################
# Save to json #
################
sixdump = sixdump_all[1::2]  # Particle with deviation from CO
part_dict = xp.Particles.from_dict(
        sixdump[0].get_minimal_beam()).to_dict()
part_dict['state'] = 1

with open('line_and_particle.json', 'w') as fid:
    json.dump({
        'line': line.to_dict(),
        'particle': part_dict},
        fid, cls=xo.JEncoder)

