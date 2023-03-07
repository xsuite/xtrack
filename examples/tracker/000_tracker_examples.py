# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xtrack as xt
import xpart as xp

elements = [xt.Drift(length=2.) for _ in range(10)]
elements[5].iscollective = True

line=xt.Line(elements=elements)
line.build_tracker(reset_s_at_end_turn=False
)

# Standard mode
p = xp.Particles(x=[1e-3, 2e-3, 3e-3], p0c=7e12)
line.track(p, num_turns=4, turn_by_turn_monitor=True)
assert np.all(p.s == 4 * 10 * 2.)
assert np.all(p.at_turn == 4)
assert line.record_last_track.x.shape == (3, 4)


# ele_start > 0
p = xp.Particles(x=[1e-3, 2e-3, 3e-3], p0c=7e12, s=4., at_element=2)
line.track(p, num_turns=3, ele_start=2, turn_by_turn_monitor=True)
assert np.all(p.s == 3 * 10 * 2.)
assert np.all(p.at_turn == 3)
assert line.record_last_track.x.shape == (3,3)

### Behavior:

# For num_turns = 1 (default):
#  - line.track(particles, ele_start=3, ele_stop=5) tracks once from element 3 to
#    element 5 e. particles.at_turn is not incremented as the particles never pass through
#    the end of the beam line.
#  - line.track(particles, ele_start=7, ele_stop=3) tracks once from element 5 to
#    the end of the beamline and then from element 0 to element 3 excluded.
#    particles.at_turn is incremented to one as the particles pass once
#    the end of the beam line.

# When indicating num_turns = N with N > 1, additional (N-1) full turns are added to logic
# above. Therefore:
# - if ele_stop < ele_start: stops at element ele_stop when particles.at_turn = N - 1
# - if ele_stop >= ele_start: stops at element ele_stop when particles.at_turn = N



# 0 <= ele_start < ele_stop
p = xp.Particles(x=[1e-3, 2e-3, 3e-3], p0c=7e12, s=5 * 2., at_element=5)
line.track(p, num_turns=4, ele_start=5, ele_stop=8,
              turn_by_turn_monitor=True)
assert np.all(p.s == 3 * 10 * 2. + 8 * 2)
assert np.all(p.at_turn == 3)
assert line.record_last_track.x.shape == (3, 4)

# 0 <= ele_start < ele_stop -- num_turns = 1
p = xp.Particles(x=[1e-3, 2e-3, 3e-3], p0c=7e12, s=5 * 2., at_element=5)
line.track(p, num_turns=1, ele_start=5, ele_stop=2,
              turn_by_turn_monitor=True)
assert np.all(p.s == 10* 2. + 2*2.)
assert np.all(p.at_turn == 1)
assert line.record_last_track.x.shape == (3, 2)

# 0 <= ele_start < ele_stop -- num_turns > 1
p = xp.Particles(x=[1e-3, 2e-3, 3e-3], p0c=7e12, s=5 * 2., at_element=5)
line.track(p, num_turns=4, ele_start=5, ele_stop=2,
              turn_by_turn_monitor=True)
assert np.all(p.s == 4 * 10 * 2. + 2 * 2.)
assert np.all(p.at_turn == 4)
assert line.record_last_track.x.shape == (3, 5)# 0 <= ele_start < ele_stop -- num_turns > 1

# Use ele_start and num_turns
p = xp.Particles(x=[1e-3, 2e-3, 3e-3], p0c=7e12, s=5 * 2., at_element=5)
line.track(p, ele_start=5, num_elements=7,
              turn_by_turn_monitor=True)
assert np.all(p.s==5*2.+7*2.)
assert np.all(p.at_turn == 1)


