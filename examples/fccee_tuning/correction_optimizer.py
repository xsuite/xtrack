"""Common interface for the example's independently defined corrections."""

from functools import partial

from xtrack.match import OptimizeLine


class CorrectionOptimizer(OptimizeLine):
    """An ordinary line optimizer with correction metrics and stopping goals.

    Builders supply ordinary functions from their correction modules. Each
    function receives this optimizer as its first argument; callers only supply
    Twiss and any reporting options. Diagnostics never run actions.
    The workflow owns evaluation, sequencing and whole-machine rollback.
    """

    def __init__(self, line, reference, *, settings, compute_metrics, goal,
                 print_metrics=None, nominal_knob_values=None, **kwargs):
        super().__init__(line=line, action_results=[reference],
                         assert_within_tol=False, show_call_counter=False, **kwargs)
        self.reference = reference
        self.settings = settings
        self.compute_metrics = partial(compute_metrics, self)
        self.goal = partial(goal, self)
        if print_metrics is not None:
            self.print_metrics = partial(print_metrics, self)
        self.nominal_knob_values = dict.fromkeys((v.name for v in self.vary), 0.)
        if nominal_knob_values is not None:
            self.nominal_knob_values.update(nominal_knob_values)

    def run_lstsq(self, *args, tag_outcome=False, **kwargs):
        """Apply a correction without evaluating or tagging its outcome by default."""
        return super().run_lstsq(*args, tag_outcome=tag_outcome, **kwargs)

    def print_metrics(self, tw, *, log=print):
        log(f'{self.name}:', self.compute_metrics(tw))
