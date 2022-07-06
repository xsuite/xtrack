
class PipelineStatus:
    def __init__(self, on_hold):
        self.on_hold = on_hold

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
