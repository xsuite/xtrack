import time

import xtrack as xt

from cpymad.madx import Madx

# Load the line
line = xt.Line.from_json(
    '../../test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json')
line.particle_ref = xt.Particles(p0c=7e12, mass=xt.PROTON_MASS_EV)
collider = xt.Multiline(lines={'lhcb1': line})
collider.build_trackers()


tw_ref = collider.lhcb1.twiss()

ele_start_match = 's.ds.l7.b1'
ele_end_match = 'e.ds.r7.b1'
tw_init = tw_ref.get_twiss_init(ele_start_match)

ttt = collider.twiss(
    #verbose=True,
    start=[ele_start_match],
    end=[ele_end_match],
    init=tw_init,
    _keep_initial_particles=True,
    _keep_tracking_data=True,
    )

line._kill_twiss = False

n_repeat = 1000
t1 = time.perf_counter()
for repeat in range(n_repeat):
    tw = collider.twiss(
        #verbose=True,
        start=[ele_start_match],
        end=[ele_end_match],
        init=tw_init,
        _ebe_monitor=[ttt.lhcb1.tracking_data],
        _initial_particles=[ttt.lhcb1._initial_particles]
        )
t2 = time.perf_counter()

print(f'Average time: {(t2-t1)/n_repeat*1e3} ms')

line._kill_twiss = True

n_repeat = 1000
t1 = time.perf_counter()
for repeat in range(n_repeat):
    tw = collider.twiss(
        #verbose=True,
        start=[ele_start_match],
        end=[ele_end_match],
        init=tw_init,
        _ebe_monitor=[ttt.lhcb1.tracking_data],
        _initial_particles=[ttt.lhcb1._initial_particles]
        )
t2 = time.perf_counter()

print(f'Average time (with kill): {(t2-t1)/n_repeat*1e3} ms')