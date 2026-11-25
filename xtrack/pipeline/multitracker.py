from collections import Counter
from warnings import warn
import xtrack as xt


class PipelineBranch:
    def __init__(self, line=None, particles=None, tracker=None):
        """Create a new pipeline branch.

        Parameters
        ----------
        line : Line
            Line of the branch.
        particles : Particles
            Particles to be passed to the branch.
        tracker : Tracker
            Deprecated: use `line` instead.
        """
        if tracker is not None:
            warn('The argument tracker is deprecated. Please use line instead.', FutureWarning)
            assert line is None
            line = tracker.line

        if isinstance(line, xt.Tracker):
            warn(
                'The use of Tracker as argument of `PipelineBranch` is deprecated. Please use Line instead.',
                FutureWarning
            )
            line = line.line

        self.line = line
        self.particles = particles
        self.pipeline_status = None
        # Time of the last machine sample for a beam sharing a line.
        self._t_last_update_time_dependent_vars = None

        if self.line.tracker is None:
            raise ValueError('Tracker not built for line')

        self.line.tracker.enable_pipeline_hold = True

class PipelineMultiTracker:

    def __init__(self, branches, enable_debug_log=False, verbose=False):
        # Keep iterator inputs available for counting, tracking and resumption.
        if iter(branches) is branches:
            branches = list(branches)
        self.branches = branches
        self.enable_debug_log = enable_debug_log
        self.verbose = verbose

        if self.enable_debug_log:
            self.debug_log = []

    def track(self, **kwargs):

        line_counts = Counter(branch.line for branch in self.branches)
        for branch in self.branches:
            if (branch.line.enable_time_dependent_vars
                    and line_counts[branch.line] > 1):
                # Independent sample times for beams sharing one line.
                # A single branch uses the line's sampling history.
                branch.line._t_last_update_time_dependent_vars = (
                    branch._t_last_update_time_dependent_vars)
            branch.pipeline_status = branch.line.track(
                 branch.particles, **kwargs,
                 _called_by_pipeline=True)
            if branch.line.enable_time_dependent_vars:
                branch._t_last_update_time_dependent_vars = (
                    branch.line._t_last_update_time_dependent_vars)

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
                        xt._print(
                            f"Pipeline hold at branch {i_branch} "
                            f"at turn {branch.pipeline_status.data['tt']} "
                            f"by element {branch.line.tracker._part_names[branch.pipeline_status.data['ipp']]} "
                            f"with info: {branch.pipeline_status.data['status_from_element'].info}")

                    branch.pipeline_status = branch.line.tracker.resume(branch.pipeline_status)
                    if branch.line.enable_time_dependent_vars:
                        branch._t_last_update_time_dependent_vars = (
                            branch.line._t_last_update_time_dependent_vars)
                    need_resume = True


