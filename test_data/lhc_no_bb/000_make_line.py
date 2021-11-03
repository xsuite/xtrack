import json
import numpy as np

import sixtracktools
import xline as xl
import xpart as xp
import xobjects as xo

##################
# Get a sequence #
##################

six = sixtracktools.SixInput(".")
sequence = xl.Line.from_sixinput(six)


######################
# Get some particles #
######################
sixdump = sixtracktools.SixDump101("res/dump3.dat")
part0_pyst = xp.Particles(**sixdump[0::2][0].get_minimal_beam())

# Force active state
part0_pyst.state = 1

with open('line_and_particle.json', 'w') as fid:
    json.dump({
        'line': sequence.to_dict(keepextra=True),
        'particle': part0_pyst.to_dict()},
        fid, cls=xo.JEncoder)

