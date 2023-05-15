import time

import xtrack as xt
import xpart as xp

from cpymad.madx import Madx

# xt._print.suppress = True

# Load the line
line = xt.Line.from_json(
    '../../test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json')
line.particle_ref = xp.Particles(p0c=7e12, mass=xp.PROTON_MASS_EV)
collider = xt.Multiline(lines={'lhcb1': line})
collider.build_trackers()

tw = collider.twiss()

tw_cell = collider.lhcb1.twiss(
    ele_start='s.cell.67.b1',
    ele_stop='e.cell.67.b1',
    twiss_init='preserve')

tw_cell_periodic = collider.lhcb1.twiss(
    method='4d',
    ele_start='s.cell.67.b1',
    ele_stop='e.cell.67.b1',
    twiss_init='periodic')
