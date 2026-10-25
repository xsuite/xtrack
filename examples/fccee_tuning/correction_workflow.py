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

import corr_optics
import corr_orbit
import corr_coupling
import corr_tune
import corr_chromaticity
import corr_second_order_chrom


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
    last_accepted_twiss: object


class FCCeeCorrection:
    """Own correction state and execution for a line prepared by 001.

    Construction builds the nominal reference and seven fixed-response optimizers,
    then starts at zero error/nonlinear strength. The optimizers, iteration limits,
    nominal knob values, reference and last_accepted_twiss are public.

    Snapshot before a trial; update the accepted state only after success.
    """

    correction_errors = (ValueError, RuntimeError, xt.twiss.ClosedOrbitSearchError)

    def correct_at_fixed_strengths(self, tw, *, max_iterations, chrom=False):
        """Return the accepted Twiss and completed iteration count, or raise.

        The caller owns rollback of the entire error-increase trial. Tune correction
        is explicit; coupling correction includes an orbit correction before and
        after it. Only goals and iteration limits decide success; intermediate
        errors may increase. With chrom=True, also evaluate and correct chromaticities.
        """
        orbit_x = self.optimizers.orbit_x
        orbit_y = self.optimizers.orbit_y
        optics = self.optimizers.optics
        coupling = self.optimizers.coupling
        tunes = self.optimizers.tunes
        second_order = getattr(self.optimizers, 'second_order_chrom', None)
        if chrom and ('dqx' not in tw or 'dqy' not in tw or
                      (second_order is not None and ('ddqx' not in tw or 'ddqy' not in tw))):
            tw = self.twiss(co_guess=tw.particle_on_co, chrom=True)
        self.log_iteration(0, tw)
        if self.goals_met(tw, chrom=chrom):
            return tw, 0
        for iteration in range(1, max_iterations + 1):
            if not (orbit_x.goal(tw) and orbit_y.goal(tw)):
                orbit_x.run_lstsq(action_results=[tw])
                orbit_y.run_lstsq(action_results=[tw])
                tw = self.twiss(co_guess=tw.particle_on_co, chrom=chrom)

            if not optics.goal(tw):
                optics.run_lstsq(action_results=[tw])
                tw = self.twiss(co_guess=tw.particle_on_co, chrom=chrom)

            if not coupling.goal(tw):
                # Coupling is very sensitive to orbit errors; need orbit correction
                # before and after any coupling correction.
                orbit_x.run_lstsq(action_results=[tw])
                orbit_y.run_lstsq(action_results=[tw])
                tw = self.twiss(co_guess=tw.particle_on_co, chrom=chrom)
                if not coupling.goal(tw):
                    coupling.run_lstsq(
                        action_results=[tw],
                        step_scale=coupling.step_scale)
                    tw = self.twiss(co_guess=tw.particle_on_co, chrom=chrom)
                    orbit_x.run_lstsq(action_results=[tw])
                    orbit_y.run_lstsq(action_results=[tw])
                    tw = self.twiss(co_guess=tw.particle_on_co, chrom=chrom)

            if not tunes.goal(tw):
                tunes.run_lstsq(action_results=[tw])
                tw = self.twiss(co_guess=tw.particle_on_co, chrom=chrom)

            if chrom and not self.optimizers.chromaticity.goal(tw):
                self.optimizers.chromaticity.run_lstsq(action_results=[tw])
                tw = self.twiss(co_guess=tw.particle_on_co, chrom=True)

            if chrom and second_order is not None and not second_order.goal(tw):
                second_order.run_lstsq(action_results=[tw])
                tw = self.twiss(co_guess=tw.particle_on_co, chrom=True)

            self.log_iteration(iteration, tw)
            if self.goals_met(tw, chrom=chrom):
                return tw, iteration
        raise RuntimeError(f'Correction goals not met after {max_iterations} iterations')

    def __init__(self, line, *, intermediate_max_iterations=5,
                 final_max_iterations=40, optimizer_settings=None, log=None):
        self._start_time = perf_counter()
        self.line = line
        self.intermediate_max_iterations = intermediate_max_iterations
        self.final_max_iterations = final_max_iterations
        self.log = self._default_log if log is None else log
        self.failure_reason = None
        self.target_error_fraction = None
        self.last_accepted_twiss = None
        self.optimizers = SimpleNamespace()
        optimizer_settings = optimizer_settings or {}
        second_order_settings = (optimizer_settings.get('second_order_chrom')
                                 or corr_second_order_chrom.DEFAULT_SECOND_ORDER_CHROM_SETTINGS)
        self.delta_chrom = second_order_settings.delta_chrom
        self.stats = dict(calls=0, seconds=0.0, orbit_calls=0, orbit_seconds=0.0)
        self.steps_r_matrix = dict(dx=1e-8, dpx=1e-9, dy=1e-8, dpy=1e-9, dzeta=1e-7, ddelta=1e-8)
        self.log('Preparing nominal reference and correction responses')
        self._prepare_reference()

        # Shared observation policy; each correction keeps its own row indices.
        self.monitor_names = self.line.metadata['correction_knobs']['orbit']['x']['monitors']
        self.ip_names = ['ipa', 'ipd', 'ipg', 'ipj']
        self.obs_names = list(dict.fromkeys(self.monitor_names + self.ip_names))

        twiss_kwargs = dict(delta_chrom=self.delta_chrom,
                            steps_R_matrix=self.steps_r_matrix.copy())
        self.optimizers.optics = corr_optics.build_opt_corr_optimizer(
            self.line, self.reference,
            knobs=self.line.metadata['correction_knobs']['optics']['knobs'],
            observation_names=self.obs_names,
            interaction_points=list(self.line.get_table().rows['ip.*'].name),
            ip_names=self.ip_names, settings=optimizer_settings.get('optics'), **twiss_kwargs)
        self.optimizers.orbit_x = corr_orbit.build_orbit_corr_optimizer(
            self.line, self.reference, plane='x', settings=optimizer_settings.get('orbit'), **twiss_kwargs)
        self.optimizers.orbit_y = corr_orbit.build_orbit_corr_optimizer(
            self.line, self.reference, plane='y', settings=optimizer_settings.get('orbit'), **twiss_kwargs)
        self.optimizers.coupling = corr_coupling.build_coupling_corr_optimizer(
            self.line, self.reference, observation_names=self.obs_names,
            ip_names=self.ip_names, log=_silent_log,
            settings=optimizer_settings.get('coupling'), **twiss_kwargs)
        self.optimizers.tunes = corr_tune.build_tune_corr_optimizer(
            self.line, self.reference,
            settings=optimizer_settings.get('tunes'), **twiss_kwargs)
        self.optimizers.chromaticity = corr_chromaticity.build_chromaticity_corr_optimizer(
            self.line, self.reference,
            settings=optimizer_settings.get('chromaticity'), **twiss_kwargs)
        self.optimizers.second_order_chrom = corr_second_order_chrom.build_second_order_chrom_optimizer(
            self.line, self.reference,
            settings=second_order_settings, **twiss_kwargs)
        for optimizer in vars(self.optimizers).values():
            _ = optimizer.response.svd

        self._identify_correction_knobs()
        self.line.vars.update(self.nominal_knob_values)
        self.set_error_fraction(0.)
        self.last_accepted_twiss = self.twiss(co_guess=self.reference.particle_on_co)

    def _default_log(self, *args, **kwargs):
        """Print elapsed time since construction, flushing output by default."""
        kwargs.setdefault('flush', True)
        print(f'[+{perf_counter() - self._start_time:8.1f} s]', *args, **kwargs)

    def run(self, *, target_error_fraction=1., initial_increment=0.0125,
            min_increment=1e-5, max_increment=0.10):
        """Increase errors and correct, then refine; return whether both succeeded.

        A successful target below one is supported. The return value reports
        success at the requested fraction, including final refinement.
        """
        if not self.iterative_error_increase_and_correction(
                target_error_fraction=target_error_fraction,
                initial_increment=initial_increment,
                min_increment=min_increment, max_increment=max_increment):
            return False
        return self.refine_correction()

    def iterative_error_increase_and_correction(self, *, target_error_fraction=1.,
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

        self.failure_reason = None
        self.target_error_fraction = target_error_fraction
        optics = self.optimizers.optics
        coupling = self.optimizers.coupling
        optics.dx_err_rms_tol = optics.settings.intermediate_horizontal_dispersion_rms_tol
        optics.dx_err_ip_tol = optics.settings.intermediate_horizontal_dispersion_ip_tol
        coupling.rdt_rms_tol = coupling.settings.intermediate_coupling_rms_tol
        coupling.rdt_ip_tol = coupling.settings.intermediate_coupling_ip_tol
        coupling.c_minus_tol = coupling.settings.intermediate_c_minus_tol
        coupling.dy_rms_tol = coupling.settings.intermediate_dispersion_rms_tol
        coupling.dy_ip_tol = coupling.settings.intermediate_dispersion_ip_tol

        increment = initial_increment
        while self.error_fraction < target_error_fraction - 1e-12:
            next_error_fraction = min(self.error_fraction + increment, target_error_fraction)
            before = self.snapshot()
            self.log(f'Increasing error fraction to {next_error_fraction:g}')
            try:
                self.scale_corrections(next_error_fraction)
                self.set_error_fraction(next_error_fraction)
                tw = self.twiss(co_guess=self.last_accepted_twiss.particle_on_co)
                tw, iterations = self.correct_at_fixed_strengths(
                    tw, max_iterations=self.intermediate_max_iterations)
            except self.correction_errors as err:
                self.restore(before)
                increment /= 2
                self.log(f'Rejected: {err}')
                if increment < min_increment:
                    self.failure_reason = f'Minimum error increment reached: {err}'
                    return False
                continue

            self.last_accepted_twiss = tw
            self.log('Accepted')
            if iterations <= 3:
                increment = min(1.25 * increment, max_increment)
        return True

    def refine_correction(self):
        """Refine the accepted state at its current error fraction; return success."""
        self.failure_reason = None
        optics = self.optimizers.optics
        coupling = self.optimizers.coupling
        optics.dx_err_rms_tol = optics.settings.horizontal_dispersion_rms_tol
        optics.dx_err_ip_tol = optics.settings.horizontal_dispersion_ip_tol
        coupling.rdt_rms_tol = coupling.settings.coupling_rms_tol
        coupling.rdt_ip_tol = coupling.settings.coupling_ip_tol
        coupling.c_minus_tol = coupling.settings.c_minus_tol
        coupling.dy_rms_tol = coupling.settings.dispersion_rms_tol
        coupling.dy_ip_tol = coupling.settings.dispersion_ip_tol

        self.log('Final refinement')
        before = self.snapshot()
        try:
            if not self.goals_met(self.last_accepted_twiss):
                tw, _ = self.correct_at_fixed_strengths(
                    self.last_accepted_twiss, max_iterations=self.final_max_iterations, chrom=True)
                self.last_accepted_twiss = tw
                self.log('Accepted')
        except self.correction_errors as err:
            self.restore(before)
            self.failure_reason = f'Refinement failed: {err}'
            self.log(self.failure_reason)
            return False
        return True

    @property
    def correction_failed(self):
        return self.failure_reason is not None

    @property
    def error_fraction(self):
        """Live error fraction, including an unaccepted trial until restoration or acceptance."""
        return float(self.line['on_errors'])

    def scale_corrections(self, next_error_fraction):
        """Extrapolate deviations from nominal settings, without changing errors."""
        if not 0 <= next_error_fraction <= 1:
            raise ValueError('Error fraction must be in [0, 1]')
        previous = self.error_fraction
        if previous == 0 or previous == next_error_fraction:
            return
        ratio = next_error_fraction / previous
        # Family chromaticity trims already scale through on_nonlinear.
        chromaticity_knobs = self.line.metadata.get('correction_knobs', {}).get(
            'chromaticity', {}).get('knobs', [])
        knobs = [name for name in self.correction_knobs if name not in chromaticity_knobs]
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
            last_accepted_twiss=self.last_accepted_twiss,
        )

    def restore(self, snapshot):
        """Undo a trial, including error strength and accepted-orbit bookkeeping."""
        values = self.line.vars.val
        self.line.vars.update({
            name: value for name, value in snapshot.knob_values.items()
            if values[name] != value
        })
        self.last_accepted_twiss = snapshot.last_accepted_twiss

    def log_iteration(self, iteration, tw):
        self.log('Initial state' if iteration == 0 else f'Iteration {iteration}')
        for optimizer in vars(self.optimizers).values():
            optimizer.print_metrics(tw, log=self.log)

    def goals_met(self, tw, *, chrom=True):
        return all(opt.goal(tw) for name, opt in vars(self.optimizers).items()
                   if chrom or name not in ('chromaticity', 'second_order_chrom'))

    def twiss(self, co_guess=None, *, only_orbit=False, chrom=False):
        """Evaluate 4D optics at the reference momentum, using a zero guess if omitted."""
        if co_guess is None:
            co_guess = dict(x=0, px=0, y=0, py=0, zeta=0, delta=0)
        started = perf_counter()
        self.stats['calls'] += 1
        if only_orbit:
            self.stats['orbit_calls'] += 1
        try:
            return self.line.twiss(
                method='4d', delta0=0.0, co_guess=co_guess,
                chrom=chrom,
                delta_chrom=self.delta_chrom,
                steps_R_matrix=self.steps_r_matrix.copy(),
                radiation_integrals=False,
                only_orbit=only_orbit,
            )
        finally:
            elapsed = perf_counter() - started
            self.stats['seconds'] += elapsed
            if only_orbit:
                self.stats['orbit_seconds'] += elapsed

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

    def _prepare_reference(self):
        """Prepare the shared clean reference with nominal nonlinear strengths and chromaticities."""
        line = self.line
        circuits = line.metadata['correction_knobs']
        orbit_knobs = circuits['orbit']['x']['knobs'] + circuits['orbit']['y']['knobs']
        line.set(orbit_knobs, np.zeros(len(orbit_knobs)))
        line.set(dict.fromkeys(circuits['optics']['knobs'], 0.))
        line.set(dict.fromkeys(circuits['coupling']['knobs'], 0.))
        line['on_errors'] = 0.0
        line['on_nonlinear'] = 1.0
        line.set(circuits['tunes']['nominal_values'])
        line.set(dict.fromkeys(circuits['chromaticity']['knobs'], 0.))
        line.set(dict.fromkeys(circuits['second_order_chrom']['knobs'], 0.))
        self.reference = self.twiss(chrom=True)

    def _identify_correction_knobs(self):
        """Collect every optimizer's knobs and nominal values for control and snapshots."""
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


def install_nonlinear_switch(line):
    """Attach on_nonlinear to magnetic strengths of sextupole order and higher.

    Preserve existing expressions and scale relative errors only when they do
    not already follow a ramped main strength. Record and return the affected
    magnet names. Call once, before installing nonlinear correction trims.
    """
    line['on_nonlinear'] = 1.0
    controlled = []
    magnet_types = (
        xt.Bend,
        xt.RBend,
        xt.Quadrupole,
        xt.Sextupole,
        xt.Octupole,
        xt.Multipole,
    )
    for name in dict.fromkeys(line.element_names):
        element = line.element_dict[name]
        if not isinstance(element, magnet_types):
            continue
        main_is_nonlinear = isinstance(element, (xt.Sextupole, xt.Octupole)) or (
            isinstance(element, xt.Multipole) and element.main_order >= 2
        )
        changed = False
        # Preserve existing family expressions, including currently zero ones.
        for field in ('k2', 'k2s', 'k3', 'k3s'):
            if hasattr(element, field):
                strength = getattr(line.ref[name], field)
                strength *= line.ref['on_nonlinear']
                setattr(line.ref[name], field, strength)
                changed = True
        fields = ['knl', 'ksl']
        # Relative errors in sextupoles/octupoles/etc. already follow their
        # ramped main strength. Multiplying these again would give a squared
        # ramp. Relative nonlinear errors in dipoles/quads need the knob here.
        if not main_is_nonlinear:
            fields += ['knl_rel', 'ksl_rel']
        for field in fields:
            if not hasattr(element, field):
                continue
            for order in range(2, len(getattr(element, field))):
                ref = getattr(line.ref[name], field)[order]
                if getattr(element, field)[order] != 0 or ref._expr is not None:
                    getattr(line.ref[name], field)[order] *= line.ref['on_nonlinear']
                    changed = True
        if changed:
            controlled.append(name)

    circuits = line.metadata.setdefault('correction_knobs', {})
    circuits['nonlinear_magnets'] = controlled
    circuits.setdefault('switches', {})['nonlinear'] = 'on_nonlinear'
    return controlled
