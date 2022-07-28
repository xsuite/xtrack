import numpy as np

import xtrack as xt
import xpart as xp

from xtrack import PipelineStatus

class DummyPipelinedElement:
    def __init__(self, n_hold=0):
        self.n_hold = n_hold
        self.iscollective = True
        self.i_hold = 0

    def track(self, particles, **kwargs):
        self.i_hold += 1
        if self.i_hold < self.n_hold:
            return PipelineStatus(on_hold=True)
        else:
            self.i_hold = 0

tracker1 = xt.Tracker(
    line=xt.Line(elements=[xt.Drift(length=1),
                           DummyPipelinedElement(n_hold=3),
                           xt.Drift(length=1)],
                 element_names=['d11', 'pipelnd_el1', 'd12']),
    enable_pipeline_hold=True,
    reset_s_at_end_turn=False)

tracker2 = xt.Tracker(
    line=xt.Line(elements=[xt.Drift(length=1),
                           DummyPipelinedElement(n_hold=5),
                           xt.Drift(length=1)],
                element_names=['d11', 'pipelnd_el1', 'd12']),
    enable_pipeline_hold=True,
    reset_s_at_end_turn=False)


p1 = xp.Particles(p0c=7e12, x=[0,0,0])
p2 = xp.Particles(p0c=7e12, x=[0,0,0])

multitracker = xt.PipelineMultiTracker(
    branches=[xt.PipelineBranch(tracker=tracker1, particles=p1),
              xt.PipelineBranch(tracker=tracker2, particles=p2)],
    enable_debug_log=True)

multitracker.track(num_turns=10)