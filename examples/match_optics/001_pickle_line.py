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

lnss = pickle.dumps(line)
ln = pickle.loads(lnss)
tw = line.twiss()

line.discard_tracker()

collider = xt.Multiline(lines={'lhcb1': line})
collider.build_trackers()

# # Let's see what I need to delete to make the roundtrip work
# collider._var_sharing = None
# collider.lhcb1._var_management = None

colliderss = pickle.dumps(collider)
coll = pickle.loads(colliderss)