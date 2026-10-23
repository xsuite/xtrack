"""Common interface for the example's independently defined corrections."""

from xtrack.match import OptimizeLine


class CorrectionOptimizer(OptimizeLine):
    """An ordinary line optimizer with correction metrics and stopping goals.

    Subclasses define setup, active goal parameters and printing in their own
    modules. Diagnostic methods consume Twiss explicitly and never run actions.
    The workflow owns evaluation, sequencing and whole-machine rollback.
    """

    def __init__(self, line, reference, *, settings, nominal_knob_values=None, **kwargs):
        super().__init__(line=line, action_results=[reference],
                         assert_within_tol=False, show_call_counter=False, **kwargs)
        self.reference = reference
        self.settings = settings
        self.nominal_knob_values = dict.fromkeys((v.name for v in self.vary), 0.)
        if nominal_knob_values is not None:
            self.nominal_knob_values.update(nominal_knob_values)

    def configure_goals(self, *, final_refinement, error_fraction):
        """Refresh active tolerances from module settings when ramp state changes."""
        pass

    def compute_metrics(self, tw):
        raise NotImplementedError

    def goal(self, tw):
        raise NotImplementedError

    def print_metrics(self, tw, *, log=print):
        log(f'{self.name}:', self.compute_metrics(tw))
