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

        need_resume = True
        while need_resume:
            need_resume = False
            for branch in self.branches:
                if (branch.pipeline_status is not None
                        and branch.pipeline_status.on_hold):
                    print('resumed')
                    branch.pipeline_status = branch.tracker.resume(
                                                        branch.pipeline_status)
                    need_resume = True



