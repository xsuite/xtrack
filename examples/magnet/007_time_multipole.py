import xtrack as xt
import numpy as np

mm = xt.Multipole(knl=[0.001, 0.001, 0.01, 0.02, 0.04, 0.6],
                  hxl=0.002, length=2.)

lst = [mm.copy() for i in range(10000)]

p = xt.Particles(p0c=7e12, x=np.linspace(-1e-3, 1e-3, 1000))

line = xt.Line(elements=lst)
line.track(p)

line.build_tracker()
line.track(p, num_turns=100, with_progress=True, time=True)
print('Time for tracking: ', line.time_last_track)