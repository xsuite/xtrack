import numpy as np

import xtrack as xt

import matplotlib.pyplot as plt

nemitt_x = 1e-6
nemitt_y = 1e-6


line = xt.Line.from_json(
    '../../test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json')
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=7e12)
line.build_tracker()

# Some octupoles and chromaticity to see the footprint moving
line.vars['i_oct_b1'] = 500
line.match(
    targets=[xt.TargetList(['dqx', 'dqy'], value=10, tol=0.01)],
    vary=[xt.VaryList(['ksf.b1', 'ksd.b1'], step=1e-3)])

plt.close('all')
plt.figure(1)

# Compute and plot footprint on momentum
fp1 = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                         freeze_longitudinal=True)
fp1.plot(color='b', label='delta=0')

# Compute and plot footprint off momentum
fp2 = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                         freeze_longitudinal=True, delta0=1e-4)
fp2.plot(color='r', label='delta=1e-4')

assert np.allclose(line.record_last_track.delta[:], 1e-4, atol=1e-12, rtol=0)

# It is 12, 9 instead of 10, 10 because of non-linear chromaticity
assert np.isclose((np.max(fp2.qx) - np.max(fp1.qx))/1e-4, 12, atol=0.5, rtol=0)
assert np.isclose((np.max(fp2.qy) - np.max(fp1.qy))/1e-4, 9, atol=0.5, rtol=0)

plt.legend()

plt.show()