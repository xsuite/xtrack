import numpy as np

from xtrack import PipelineStatus

class DummyPipelinedElement:
    def __init__(self, n_hold=0):
        self.n_hold = n_hold
        self.iscollective = True
        self.i_hold = 0

    def track(self, particles):
        self.i_hold += 1
        if self.i_hold < self.n_hold:
            return PipelineStatus(on_hold=True,
                info=f'stopped by internal counter {self.i_hold}/{self.n_hold}')
        else:
            self.i_hold = 0


import xtrack as xt
import xpart as xp

line=xt.Line(elements=[xt.Drift(length=1),
                           DummyPipelinedElement(n_hold=3),
                           xt.Drift(length=1)])
line.build_tracker(enable_pipeline_hold=True)


p = xp.Particles(p0c=7e12, x=[0,0,0])

session_on_hold = line.track(p, num_turns=2)
assert np.all(p.s==1)
assert np.all(p.at_turn==0)
session_on_hold = line.resume(session_on_hold)
assert np.all(p.s==1)
assert np.all(p.at_turn==0)
session_on_hold = line.resume(session_on_hold)
assert np.all(p.s==1)
assert np.all(p.at_turn==1)