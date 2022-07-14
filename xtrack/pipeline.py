class PipelineStatus:
    def __init__(self, on_hold, data=None):
        self.on_hold = on_hold
        self.data = data

class PipelineBranch:
    def __init__(self, tracker, particles):
        self.tracker = tracker
        self.particles = particles
        self.pipeline_status = None

        self.tracker.enable_pipeline_hold = True

class PipelineMultiTracker:

    def __init__(self, branches):
        self.branches = branches

    def track(self, **kwargs):
        for branch in self.branches:
            branch.pipeline_status = branch.tracker.track(
                 branch.particles, **kwargs)

        while True:
            prrrrrr

