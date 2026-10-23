"""Inspectable FCC-ee correction workflow used by 002.

This is an example-local API prototype. The class owns increasing errors,
correction, final refinement, Twiss evaluation, snapshots and accepted state.
Optimizers consume supplied Twiss tables and apply least-squares changes.
"""

from dataclasses import dataclass
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import xtrack as xt

import correction_setup as setup
import corr_optics
import corr_orbit
import corr_coupling
import corr_tune


def _silent_log(*args, **kwargs):
    """Suppress detailed output from the response preparation helpers."""
    pass


@dataclass
class MachineSnapshot:
    """Controls and accepted orbit needed to undo an unaccepted error-increase trial.

    Twiss tables are retained by reference; correction steps treat them as
    read-only, including their closed-orbit particle. Variable values are copied.
    """

    knob_values: dict
    accepted_twiss: object
    final_refinement: bool


class FCCeeCorrection:
    """Own correction state and execution for a line prepared by 001.

    Construction builds the nominal reference and five fixed-response optimizers,
    then starts at zero error/nonlinear strength. The optimizers, iteration limits,
    nominal knob values, reference and accepted_twiss are public.

    Snapshot before a trial; update the accepted state only after success.
    """

    correction_errors = (ValueError, RuntimeError, xt.twiss.ClosedOrbitSearchError)

    def __init__(self, line, *, intermediate_max_iterations=5,
                 final_max_iterations=40, optimizer_settings=None, log=None):
        self._start_time = perf_counter()
        self.line = line
        # This workflow uses 4D optics at the reference momentum, as in 000.
        self.line.twiss_default.update(method='4d', delta0=0.0)
        self.intermediate_max_iterations = intermediate_max_iterations
        self.final_max_iterations = final_max_iterations
        self.log = self._default_log if log is None else log
        self._final_refinement = False
        self.correction_failed = False
        self.failure_reason = None
        self.target_error_fraction = None
        self.error_increase_settings = {}
        self.accepted_twiss = None
        self.optimizers = SimpleNamespace()
        self.optimizer_settings = optimizer_settings or {}
        self.stats = dict(calls=0, seconds=0.0, orbit_calls=0, orbit_seconds=0.0)
        self.steps_r_matrix = dict(dx=1e-8, dpx=1e-9, dy=1e-8, dpy=1e-9, dzeta=1e-7, ddelta=1e-8)
        self.log('Preparing nominal reference and correction responses')
        self._prepare_reference()
        self._build_optimizers()
        self._collect_controls()
        self.line.vars.update(self.nominal_knob_values)
        self.set_error_fraction(0.)
        self.accepted_twiss = self._full_twiss(self.reference)

    def _default_log(self, *args, **kwargs):
        """Print elapsed time since construction, flushing output by default."""
        kwargs.setdefault('flush', True)
        print(f'[+{perf_counter() - self._start_time:8.1f} s]', *args, **kwargs)

    def run(self, *, target_error_fraction=1., initial_increment=0.0125,
            min_increment=1e-5, max_increment=0.10):
        """Increase errors and correct, then refine; return whether both succeeded.

        A successful target below one is supported. The completed property is
        true only at full error strength with final goals met.
        """
        if not self.increase_errors_and_correct(
                target_error_fraction=target_error_fraction,
                initial_increment=initial_increment,
                min_increment=min_increment, max_increment=max_increment):
            return False
        return self.refine()

    def increase_errors_and_correct(self, *, target_error_fraction=1.,
                                    initial_increment=0.0125,
                                    min_increment=1e-5, max_increment=0.10):
        """Increase errors/nonlinear strengths together, correcting each trial.

        Restore rejected trials and halve the increment. Keep the latest accepted
        state. On failure, retain the last accepted machine and expose the
        reason in failure_reason; return False. Invalid settings raise before
        changing the machine or execution state.
        """
        if not np.isfinite(target_error_fraction) or not 0 < target_error_fraction <= 1:
            raise ValueError('target_error_fraction must be in (0, 1]')
        if target_error_fraction < self.error_fraction - 1e-12:
            raise ValueError('target_error_fraction cannot be below the current error fraction')
        if (not np.all(np.isfinite([min_increment, initial_increment, max_increment]))
                or not 0 < min_increment <= initial_increment <= max_increment):
            raise ValueError('Require 0 < min_increment <= initial_increment <= max_increment')

        self.correction_failed = False
        self.failure_reason = None
        self.target_error_fraction = target_error_fraction
        self.error_increase_settings = dict(
            target_error_fraction=target_error_fraction,
            initial_increment=initial_increment, min_increment=min_increment,
            max_increment=max_increment)
        self.final_refinement = False
        increment = initial_increment
        while self.error_fraction < target_error_fraction - 1e-12:
            next_error_fraction = self.error_fraction + increment
            if next_error_fraction > target_error_fraction:
                next_error_fraction = target_error_fraction
            before = self.snapshot()
            self.log(f'Increasing error fraction to {next_error_fraction:g}')
            try:
                self.scale_corrections(next_error_fraction)
                self.set_error_fraction(next_error_fraction)
                tw = self.twiss()
                tw, iterations = self.correct_at_fixed_strengths(tw)
            except self.correction_errors as err:
                self.restore(before)
                increment /= 2
                self.log(f'Rejected: {err}')
                if increment < min_increment:
                    self.correction_failed = True
                    self.failure_reason = f'Minimum error increment reached: {err}'
                    return False
                continue

            self.accept(tw)
            if iterations <= 3:
                increment *= 1.25
                if increment > max_increment:
                    increment = max_increment
        return True

    def refine(self):
        """Refine the accepted state at its current error fraction; return success."""
        self.correction_failed = False
        self.failure_reason = None
        self.final_refinement = True
        self.log('Final refinement')
        before = self.snapshot()
        try:
            if not self.goals_met(self.accepted_twiss):
                tw, _ = self.correct_at_fixed_strengths(self.accepted_twiss)
                self.accept(tw)
        except self.correction_errors as err:
            self.restore(before)
            self.correction_failed = True
            self.failure_reason = f'Refinement failed: {err}'
            self.log(self.failure_reason)
            return False
        return True

    @property
    def completed(self):
        """Whether the accepted machine meets final goals at full error strength."""
        return (not self.correction_failed and self.final_refinement
                and abs(self.error_fraction - 1.) < 1e-12
                and self.goals_met(self.accepted_twiss))

    @property
    def error_fraction(self):
        """Live error fraction, including an unaccepted trial until restoration or acceptance."""
        return float(self.line['on_errors'])

    @property
    def final_refinement(self):
        return self._final_refinement

    @final_refinement.setter
    def final_refinement(self, value):
        """Switch optics/coupling limits only when entering or leaving refinement."""
        value = bool(value)
        if value == self._final_refinement:
            return
        optics = self.optimizers.optics
        coupling = self.optimizers.coupling
        if value:
            optics.dx_err_rms_tol = optics.settings.horizontal_dispersion_rms_tol
            optics.dx_err_ip_tol = optics.settings.horizontal_dispersion_ip_tol
            coupling.rdt_rms_tol = coupling.settings.coupling_rms_tol
            coupling.rdt_ip_tol = coupling.settings.coupling_ip_tol
            coupling.c_minus_tol = coupling.settings.c_minus_tol
            coupling.dy_rms_tol = coupling.settings.dispersion_rms_tol
            coupling.dy_ip_tol = coupling.settings.dispersion_ip_tol
        else:
            # Restore the intermediate limits if rolling back across the mode change.
            optics.dx_err_rms_tol = optics.settings.intermediate_horizontal_dispersion_rms_tol
            optics.dx_err_ip_tol = optics.settings.intermediate_horizontal_dispersion_ip_tol
            coupling.rdt_rms_tol = coupling.settings.intermediate_coupling_rms_tol
            coupling.rdt_ip_tol = coupling.settings.intermediate_coupling_ip_tol
            coupling.c_minus_tol = coupling.settings.intermediate_c_minus_tol
            coupling.dy_rms_tol = coupling.settings.intermediate_dispersion_rms_tol
            coupling.dy_ip_tol = coupling.settings.intermediate_dispersion_ip_tol
        self._final_refinement = value

    def twiss(self):
        """Compute full optics; callers explicitly retain and supply the result."""
        return self._full_twiss(seed=self.accepted_twiss)

    def scale_corrections(self, next_error_fraction):
        """Extrapolate deviations from nominal settings, without changing errors."""
        if not 0 <= next_error_fraction <= 1:
            raise ValueError('Error fraction must be in [0, 1]')
        previous = self.error_fraction
        if previous == 0 or previous == next_error_fraction:
            return
        ratio = next_error_fraction / previous
        knobs = self.correction_knobs
        nominal = np.array([self.nominal_knob_values[name] for name in knobs])
        current = np.array(self.line.get(knobs))
        self.line.set(knobs, nominal + ratio * (current - nominal))

    def set_error_fraction(self, fraction):
        """Set errors and nonlinear strengths together; do not change corrections."""
        if not 0 <= fraction <= 1:
            raise ValueError('Error fraction must be in [0, 1]')
        self.line['on_errors'] = self.line['on_nonlinear'] = fraction

    def snapshot(self):
        """Capture every control that an error-increase trial can change, without tracking."""
        values = self.line.vars.val
        return MachineSnapshot(
            knob_values={name: float(values[name]) for name in self.control_names},
            accepted_twiss=self.accepted_twiss,
            final_refinement=self.final_refinement,
        )

    def restore(self, snapshot):
        """Undo a trial, including error strength and accepted-orbit bookkeeping."""
        values = self.line.vars.val
        self.line.vars.update({
            name: value for name, value in snapshot.knob_values.items()
            if values[name] != value
        })
        self.accepted_twiss = snapshot.accepted_twiss
        self.final_refinement = snapshot.final_refinement

    def correct_at_fixed_strengths(self, tw):
        """Return the accepted Twiss and completed iteration count, or raise.

        The caller owns rollback of the entire error-increase trial. Tune correction is
        explicit; coupling correction includes orbit settling. Only goals and
        iteration limits decide success; intermediate errors may increase.
        """
        if self.final_refinement:
            max_iterations = self.final_max_iterations
        else:
            max_iterations = self.intermediate_max_iterations
        orbit = (self.optimizers.orbit_x, self.optimizers.orbit_y)
        optics = self.optimizers.optics
        coupling = self.optimizers.coupling
        tunes = self.optimizers.tunes
        self.log_iteration(0, tw)
        if self.goals_met(tw):
            return tw, 0
        for iteration in range(1, max_iterations + 1):
            if not all(opt.goal(tw) for opt in orbit):
                for opt in orbit:
                    opt.run_lstsq(action_results=[tw], tag_outcome=False)
                tw = self._full_twiss(tw)
            tw = tunes.correct(tw, full_twiss=self._full_twiss)

            if not optics.goal(tw):
                optics.run_lstsq(
                    action_results=[tw], tag_outcome=False)
                tw = self._full_twiss(tw)
            tw = tunes.correct(tw, full_twiss=self._full_twiss)

            if not coupling.goal(tw):
                tw = self.settle_coupling_orbit(tw)
                if not coupling.goal(tw):
                    coupling.run_lstsq(
                        action_results=[tw],
                        step_scale=coupling.step_scale, tag_outcome=False)
                    tw = self.orbit_twiss(tw)
                    tw = self.settle_coupling_orbit(tw)
            tw = tunes.correct(tw, full_twiss=self._full_twiss)

            self.log_iteration(iteration, tw)
            if self.goals_met(tw):
                return tw, iteration
        raise RuntimeError(f'Correction goals not met after {max_iterations} iterations')

    def log_iteration(self, iteration, tw):
        self.log('Initial state' if iteration == 0 else f'Iteration {iteration}')
        for optimizer in vars(self.optimizers).values():
            optimizer.print_metrics(tw, log=self.log)

    def goals_met(self, tw):
        return all(opt.goal(tw) for opt in vars(self.optimizers).values())

    def accept(self, tw):
        """Keep the accepted Twiss, including its closed-orbit particle for the next trial."""
        self.accepted_twiss = tw
        self.log('Accepted')

    def _full_twiss(self, seed=None, co_guess=None, radiation_integrals=False):
        return self._compute_twiss(seed, co_guess, radiation_integrals)

    def orbit_twiss(self, seed=None, co_guess=None):
        return self._compute_twiss(seed, co_guess, only_orbit=True)

    def _compute_twiss(
        self, seed=None, co_guess=None, radiation_integrals=False, only_orbit=False
    ):
        kwargs = {} if seed is None else dict(co_guess=seed.particle_on_co)
        if co_guess is not None:
            kwargs['co_guess'] = co_guess
        started = perf_counter()
        self.stats['calls'] += 1
        if only_orbit:
            self.stats['orbit_calls'] += 1
        try:
            return self.line.twiss(
                chrom=False,
                steps_R_matrix=self.steps_r_matrix.copy(),
                radiation_integrals=radiation_integrals,
                only_orbit=only_orbit,
                **kwargs,
            )
        finally:
            elapsed = perf_counter() - started
            self.stats['seconds'] += elapsed
            if only_orbit:
                self.stats['orbit_seconds'] += elapsed

    def ensure_full_twiss(self, tw):
        # Call only while the line still has the settings used to compute tw.
        # Use the tracked orbit as a guess; keep the ordinary closed-orbit search.
        if 'betx' in tw._col_names:
            return tw
        return self._full_twiss(tw)

    def compute_metrics(self, tw):
        """Collect named diagnostics from every configured correction."""
        metrics = {}
        for optimizer in vars(self.optimizers).values():
            current = optimizer.compute_metrics(tw)
            duplicates = metrics.keys() & current.keys()
            if duplicates:
                raise ValueError(f'Duplicate correction metrics: {duplicates}')
            metrics.update(current)
        return metrics

    def settle_coupling_orbit(self, tw):
        # Settle both the baseline and trials so orbit feed-down does not mask
        # the effect of the skew correction. Tune correction remains explicit.
        orbit = (self.optimizers.orbit_x, self.optimizers.orbit_y)
        for iteration in range(3):
            errors = [opt.correctable_error(tw) for opt in orbit]
            rms = [np.sqrt(np.mean(error**2)) for error in errors]
            if max(rms) < self.optimizers.coupling.orbit_tol:
                break
            before = self.snapshot()
            try:
                for opt in orbit:
                    opt.run_lstsq(action_results=[tw], tag_outcome=False)
                trial = self.orbit_twiss(tw) if iteration < 2 else self._full_twiss(tw)
            except self.correction_errors:
                # Settling continues from its last accepted orbit; unlike the
                # main stages, a failed trial does not reject the error increment.
                self.restore(before)
                break
            tw = trial
        return self.ensure_full_twiss(tw)

    def _prepare_reference(self):
        self.reference = setup.nominal_reference(
            self.line, full_twiss=self._full_twiss, log=_silent_log)
        # Shared observation policy; each correction keeps its own row indices.
        self.monitor_names = self.line.metadata['correction_knobs']['orbit']['x']['monitors']
        self.ip_names = ['ipa', 'ipd', 'ipg', 'ipj']
        self.obs_names = list(dict.fromkeys(self.monitor_names + self.ip_names))

    def _build_optimizers(self):
        self.optimizers.optics = corr_optics.build_opt_corr_optimizer(
            self.line, self.reference,
            knobs=corr_optics.install_optics_knobs(self.line),
            observation_names=self.obs_names,
            interaction_points=list(self.line.get_table().rows['ip.*'].name),
            ip_names=self.ip_names, settings=self.optimizer_settings.get('optics'))
        self.optimizers.orbit_x = corr_orbit.build_orbit_corr_optimizer(
            self.line, self.reference, plane='x', settings=self.optimizer_settings.get('orbit'))
        self.optimizers.orbit_y = corr_orbit.build_orbit_corr_optimizer(
            self.line, self.reference, plane='y', settings=self.optimizer_settings.get('orbit'))
        self.optimizers.coupling = corr_coupling.build_coupling_corr_optimizer(
            self.line, self.reference, observation_names=self.obs_names,
            ip_names=self.ip_names, full_twiss=self._full_twiss, log=_silent_log,
            settings=self.optimizer_settings.get('coupling'))
        self.optimizers.tunes = corr_tune.build_tune_corr_optimizer(
            self.line, self.reference, full_twiss=self._full_twiss,
            settings=self.optimizer_settings.get('tunes'))
        for optimizer in vars(self.optimizers).values():
            _ = optimizer.response.svd

    def _collect_controls(self):
        """Include every optimizer's knobs in scaling and snapshots."""
        self.nominal_knob_values = {}
        for optimizer in vars(self.optimizers).values():
            for vary in optimizer.vary:
                nominal = optimizer.nominal_knob_values[vary.name]
                if (vary.name in self.nominal_knob_values
                        and self.nominal_knob_values[vary.name] != nominal):
                    raise ValueError(f'Conflicting nominal values for {vary.name}')
                self.nominal_knob_values[vary.name] = nominal
        self.correction_knobs = list(self.nominal_knob_values)
        self.control_names = self.correction_knobs + ['on_errors', 'on_nonlinear']
