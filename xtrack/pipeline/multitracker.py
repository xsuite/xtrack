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
            need_to_continue = False

            for branch in self.branches:
                branch.pipeline_status = branch.tracker.track(
                    branch.particles, **kwargs)
                if (branch.pipeline_status is not None
                        and branch.pipeline_status.on_hold):
                    need_to_continue = True

            if not(need_to_continue):
                break


