import time

import xtrack as xt
import xpart as xp

from cpymad.madx import Madx

# Load the line
line = xt.Line.from_json(
    '../../test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json')
line.particle_ref = xp.Particles(p0c=7e12, mass=xp.PROTON_MASS_EV)
collider = xt.Multiline(lines={'lhcb1': line})
collider.build_trackers()


tw_ref = collider.lhcb1.twiss()

ele_start_match = 's.ds.l7.b1'
ele_end_match = 'e.ds.r7.b1'
tw_init = tw_ref.get_twiss_init(ele_start_match)

ele_index_start = line.element_names.index(ele_start_match)
ele_index_end = line.element_names.index(ele_end_match)

tw = collider.twiss(
    #verbose=True,
    ele_start=[ele_index_start],
    ele_stop=[ele_index_end],
    twiss_init=tw_init,
    )

line._kill_tracking = True

n_repeat = 1000
t1 = time.perf_counter()
for repeat in range(n_repeat):
    tw = collider.twiss(
        #verbose=True,
        ele_start=[ele_index_start],
        ele_stop=[ele_index_end],
        twiss_init=tw_init,
        )
t2 = time.perf_counter()

print(f'Average time: {(t2-t1)/n_repeat*1e3} ms')

