import xtrack as xt
_print = xt.general._print

class PipelineBranch:
    def __init__(self, line=None, particles=None, tracker=None):

        if tracker is not None:
            _print( 'Warning! '
                "The argument tracker is deprecated. Please use line instead.")
            assert line is None
            line = tracker.line

        if isinstance(line, xt.Tracker):
            _print('Warning! '
                "The use of Tracker as argument of `PipelineBranch` is deprecated."
                " Please use Line instead.")
            line = line.line

        self.line = line
        self.particles = particles
        self.pipeline_status = None

        if self.line.tracker is None:
            raise ValueError('Tracker not built for line')

        self.line.tracker.enable_pipeline_hold = True

class PipelineMultiTracker:

    def __init__(self, branches, enable_debug_log=False, verbose=False):
        self.branches = branches
        self.enable_debug_log = enable_debug_log
        self.verbose = verbose

        if self.enable_debug_log:
            self.debug_log = []

    def track(self, **kwargs):

        for branch in self.branches:
            branch.pipeline_status = branch.line.track(
                 branch.particles, **kwargs,
                 _called_by_pipeline=True)

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
                            'held_by_element': branch.line.tracker._part_names[
                                            branch.pipeline_status.data['ipp']],
                            'info': branch.pipeline_status.data['status_from_element'].info
                        })
                    if self.verbose:
                        print(
                            f"Pipeline hold at branch {i_branch} "
                            f"at turn {branch.pipeline_status.data['tt']} "
                            f"by element {branch.line.tracker._part_names[branch.pipeline_status.data['ipp']]} "
                            f"with info: {branch.pipeline_status.data['status_from_element'].info}")

                    branch.pipeline_status = branch.line.tracker.resume(
                                                        branch.pipeline_status)
                    need_resume = True



