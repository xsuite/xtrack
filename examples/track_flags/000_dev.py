import xtrack as xt
import numpy as np

# Todo: track flags in save/reload, or mabybe not
# preserve config needs to be preserve_config_and_track_flags

line = xt.Line(elements=[xt.Cavity(lag=90, voltage=1e6)])
line.particle_ref = xt.Particles(p0c=20e9)
line.build_tracker()

p0 = xt.Particles(p0c=20e9)

p1 = p0.copy()
line.track(p1)

line.tracker.track_flags.XS_FLAG_KILL_CAVITY_KICK = True
p2 = p0.copy()
line.track(p2)

print("p1 delta:", p1.delta)
print("p2 delta:", p2.delta)

line.tracker.track_flags.XS_FLAG_KILL_CAVITY_KICK = False
