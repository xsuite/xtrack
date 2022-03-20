import json
import numpy as np

import sixtracktools
import xpart as xp
import xobjects as xo
import xtrack as xt

##################
# Get a sequence #
##################

six = sixtracktools.SixInput(".")
line = xt.Line.from_sixinput(six)


######################
# Get some particles #
######################
sixdump = sixtracktools.SixDump101("res/dump3.dat")
part0= xp.Particles.from_dict(sixdump[0::2][0].get_minimal_beam())

# Force active state
part0.state = 1

with open('line_and_particle.json', 'w') as fid:
    json.dump({
        'line': line.to_dict(),
        'particle': part0.to_dict()},
        fid, cls=xo.JEncoder)

