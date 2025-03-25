import xtrack as xt
import time
import numpy as np

# Load a line and build tracker
line = xt.Line.from_json(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, energy0=7e12)
line.vars['vrf400'] = 16
line.build_tracker()

# To store kernels
tw0_6d = line.twiss()
tw0_4d = line.twiss4d()

# Time twiss 4d
start = time.time()
tw4d = line.twiss4d()
end = time.time()
print(f'Time for twiss4d: {end-start}')

p = line.build_particles(x=np.linspace(-1e-3, 1e-3, 1000))
line.track(p, num_turns=20, with_progress=1, time=True)
line.time_last_track
