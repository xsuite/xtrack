class PipelineBranch:
    def __init__(self, tracker, particles):
        self.tracker = tracker
        self.particles = particles
        self.pipeline_status = None

        self.tracker.enable_pipeline_hold = True

class PipelineMultiTracker:

    def __init__(self, branches, enable_debug_log=False):
        self.branches = branches
        self.enable_debug_log = enable_debug_log

        if self.enable_debug_log:
            self.debug_log = []

    def track(self, **kwargs):

        for branch in self.branches:
            branch.pipeline_status = branch.tracker.track(
                 branch.particles, **kwargs)

        need_resume = True
        while need_resume:
            need_resume = False
            for i_branch, branch in enumerate(self.branches):
                if (branch.pipeline_status is not None
                        and branch.pipeline_status.on_hold):
                    if self.enable_debug_log:
                        self.debug_log.append({
                            'branch': i_branch,
                            'track_session_turn':
                                            branch.pipeline_status.data['tt'],
                            'held_by_element': branch.tracker._part_names[
                                            branch.pipeline_status.data['ipp']],
                            'info': branch.pipeline_status.data['status_from_element'].info
                        })

                    branch.pipeline_status = branch.tracker.resume(
                                                        branch.pipeline_status)
                    need_resume = True



