# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xtrack as xt

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

elements = [xt.Drift(length=1.) for _ in range(10)]
line=xt.Line(elements=elements)
line.build_tracker()
line.reset_s_at_end_turn = False

# Standard mode
p = xt.Particles(x=[1e-3, 2e-3, 3e-3], p0c=7e12)
line.track(p, num_turns=4)
p.at_turn # is [4, 4, 4]

# ele_start > 0
p = xt.Particles(x=[1e-3, 2e-3, 3e-3], p0c=7e12, s=4., at_element=2)
line.track(p, num_turns=3, ele_start=2)
p.at_turn # is [3, 3, 3]

# 0 <= ele_start < ele_stop
p = xt.Particles(x=[1e-3, 2e-3, 3e-3], p0c=7e12, s=5 * 2., at_element=5)
line.track(p, num_turns=4, ele_start=5, ele_stop=8)
p.at_turn # is [3, 3, 3]

# 0 <= ele_start < ele_stop -- num_turns = 1
p = xt.Particles(x=[1e-3, 2e-3, 3e-3], p0c=7e12, s=5 * 2., at_element=5)
line.track(p, num_turns=1, ele_start=5, ele_stop=2)
p.at_turn # is [1, 1, 1]

# 0 <= ele_start < ele_stop -- num_turns > 1
p = xt.Particles(x=[1e-3, 2e-3, 3e-3], p0c=7e12, s=5 * 2., at_element=5)
line.track(p, num_turns=4, ele_start=5, ele_stop=2)
p.at_turn # is [4, 4, 4]

# Use ele_start and num_turns
p = xt.Particles(x=[1e-3, 2e-3, 3e-3], p0c=7e12, s=5 * 2., at_element=5)
line.track(p, ele_start=5, num_elements=7)
p.at_turn # is [1, 1, 1]


