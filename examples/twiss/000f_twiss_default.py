# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xtrack as xt

# Load a line
line = xt.load('../../test_data/hllhc15_noerrors_nobb/line_and_particle.json')
line.set_particle_ref('proton', energy0=7e12)

# Twiss (built-in defaults)
tw_a = line.twiss()
tw_a.method # is '6d'
tw_a.reference_frame # is 'proper'

# Inspect twiss defaults
line.twiss_default # is {}

# Set some twiss defaults
line.twiss_default['method'] = '4d'
line.twiss_default['reverse'] = True

# Twiss (defaults redefined)
tw_b = line.twiss()
tw_b.method # is '4d'
tw_b.reference_frame # is 'reverse'

# Inspect twiss defaults
line.twiss_default # is {'method': '4d', 'reverse': True}

# Reset twiss defaults
line.twiss_default.clear()

# Twiss (defaults reset)
tw_c = line.twiss()
tw_c.method # is '6d'
tw_c.reference_frame # is 'proper'
