import pickle

import xtrack as xt
import xpart as xp

from cpymad.madx import Madx

# xt._print.suppress = True

# Load the line
line = xt.Line.from_json(
    '../../test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json')
line.particle_ref = xp.Particles(p0c=7e12, mass=xp.PROTON_MASS_EV)
line.build_tracker()
# collider = xt.Multiline(lines={'lhcb1': line})
# collider.build_trackers()

line.tracker._track_kernel.clear()
lnss = pickle.dumps(line)

ln = pickle.loads(lnss)

