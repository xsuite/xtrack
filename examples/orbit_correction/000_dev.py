import json
import numpy as np

import xtrack as xt
import xpart as xp

fname = '../../test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json'

with open(fname) as fid:
    line = xt.Line.from_dict(json.load(fid)['line'])
line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12)
tracker = line.build_tracker()

tw0 = tracker.twiss()

tracker.line['mq.31l2.b1..1'].knl[0] = 1e-6
#tracker.line['mq.31l2.b1..1'].ksl[0] = 1e-6



tw1 = tracker.twiss()

ele_start_range = 'e.ds.r1.b1'
ele_end_range = 's.ds.l2.b1'



tw_init = tw1.get_twiss_init(ele_start_range)

# Force particle on co to zero (assigned initial condition)
tw_init.particle_on_co.x = 0
tw_init.particle_on_co.px = 0
tw_init.particle_on_co.y = 0
tw_init.particle_on_co.py = 0

tw_part_test = tracker.twiss(
    twiss_init=tw_init, ele_start=ele_start_range, ele_stop=ele_end_range)


def _x_target(tw):
    return tw.x[-1]

tracker.match(
    verbose=True,
    vary=[
        xt.Vary('acbh15.l2b1', step=1e-8, limits=[-5e-6, 5e-6]),
        xt.Vary('acbh17.l2b1', step=1e-8, limits=[-5e-6, 5e-6])],
    targets=[
        xt.Target(_x_target, 0, tol=1e-7),
        xt.Target(lambda tw: tw.px[-1], 0, tol=1e-9)],
    twiss_init=tw_init, ele_start=ele_start_range, ele_stop=ele_end_range)

tw_part_test2 = tracker.twiss(
    twiss_init=tw_init, ele_start=ele_start_range, ele_stop=ele_end_range)

tw2 = tracker.twiss()






