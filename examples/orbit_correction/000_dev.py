import json
import numpy as np

import xtrack as xt
import xpart as xp

fname = '../../test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json'

with open(fname) as fid:
    line = xt.Line.from_dict(json.load(fid)['line'])
line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12)
tracker = line.build_tracker()

ele_start_range = 'e.ds.r1.b1'
ele_end_range = 's.ds.l2.b1'

tw0 = tracker.twiss()
tw_init0 = tw0.get_twiss_init(ele_start_range)

tracker.line['mq.31l2.b1..1'].knl[0] = 1e-6
#tracker.line['mq.31l2.b1..1'].ksl[0] = 1e-6

tw1 = tracker.twiss()
tw_init1 = tw1.get_twiss_init(ele_start_range)

# Force particle on co to zero (assigned initial condition)
tw_part_test = tracker.twiss(
    twiss_init=tw_init1, ele_start=ele_start_range, ele_stop=ele_end_range)

tracker.match(
    verbose=True,
    vary=[
        xt.Vary('acbh15.l2b1', step=1e-9, limits=[-5e-6, 5e-6]),
        xt.Vary('acbh17.l2b1', step=1e-9, limits=[-5e-6, 5e-6])],
    targets=[
        xt.Target('x', at='s.ds.l2.b1', value=0, tol=1e-9),
        xt.Target('px', at='s.ds.l2.b1', value=0, tol=1e-13)],
    twiss_init=xt.OrbitOnly(),
    ele_start=ele_start_range, ele_stop=ele_end_range)


tw2 = tracker.twiss()

import matplotlib.pyplot as plt
plt.close('all')
plt.plot(tw0.s, tw0.x)
#plt.plot(tw1.s, tw1.x)
plt.plot(tw2.s, tw2.x)
plt.axvline(x=tw0[ele_start_range, 's'])
plt.axvline(x=tw0[ele_end_range, 's'])
plt.show()






