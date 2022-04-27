import numpy as np

import xtrack as xt
import xpart as xp

elements = [xt.Drift(length=2.) for _ in range(10)]
tracker = xt.Tracker(
    line=xt.Line(elements=elements),
    reset_s_at_end_turn=False
)

# Standard mode
p = xp.Particles(x=[1e-3, 2e-3, 3e-3], p0c=7e12)
tracker.track(p, num_turns=4, turn_by_turn_monitor=True)
assert np.all(p.s == 4 * 10 * 2.)
assert np.all(p.at_turn == 4)
assert tracker.record_last_track.x.shape == (3, 4)


# ele_start > 0
p = xp.Particles(x=[1e-3, 2e-3, 3e-3], p0c=7e12, s=2, at_element=2)
tracker.track(p, num_turns=3, ele_start=2, turn_by_turn_monitor=True)
assert np.all(p.s == 3 * 10 * 2.)
assert np.all(p.at_turn == 2)
assert tracker.record_last_track.x.shape == (3,2)


# 0 <= ele_start < ele_stop
p = xp.Particles(x=[1e-3, 2e-3, 3e-3], p0c=7e12, s=5 * 2., at_element=5)
tracker.track(p, num_turns=4, ele_start=5, ele_stop=8,
              turn_by_turn_monitor=True)
assert np.all(p.s == 3 * 10 * 2. + 8 * 2)
assert np.all(p.at_turn == 3)
assert tracker.record_last_track.x.shape == (3, 4)

# 0 <= ele_start < ele_stop
p = xp.Particles(x=[1e-3, 2e-3, 3e-3], p0c=7e12, s=5 * 2., at_element=5)
tracker.track(p, num_turns=4, ele_start=5, ele_stop=3,
              turn_by_turn_monitor=True)
assert np.all(p.s == 4 * 10 * 2. + 3 * 2)
assert np.all(p.at_turn == 4)
assert tracker.record_last_track.x.shape == (3, 5)