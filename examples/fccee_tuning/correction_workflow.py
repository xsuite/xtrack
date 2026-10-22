"""Inspectable FCC-ee correction workflow used by 004.

This is an example-local API prototype. The ramp remains in the user script;
the class owns optimizers, machine snapshots and accepted history. Numerical
trial steps are shared with the unchanged 003b reference.
"""

from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import xtrack as xt

import correction_observables as obs
import correction_steps as steps
import correction_setup as setup
from correction_io import HISTORY_COLUMNS, history_row, save_result
from correction_steps import orbit_guess
from response_builders import orbit_response


def _silent_log(*args, **kwargs):
    """Suppress detailed output from the shared 003b helpers."""
    pass


@dataclass
class CorrectionSettings:
    """Response weights are applied at construction; goals/cutoffs are read per step."""

    beta_rms_tol: float = 0.03
    beta_max_tol: float = 0.5
    tune_tol: float = 1e-05
    alpha_weight: float = 0.1
    ir_beta_weight: float = 5.0
    ir_half_length: float = 500.0
    horizontal_dispersion_rms_tol: float = 0.001
    horizontal_dispersion_ip_tol: float = 0.0001
    ramp_horizontal_dispersion_rms_tol: float = 0.005
    ramp_horizontal_dispersion_ip_tol: float = 0.0005
    orbit_tol: float = 1e-05
    optics_rcond: float = 1e-05
    orbit_rcond: float = 1e-06
    coupling_rcond: float = 1e-06
    coupling_ip_weight: float = 20.0
    coupling_rms_tol: float = 0.003
    coupling_ip_tol: float = 0.002
    c_minus_tol: float = 0.0003
    dispersion_rms_tol: float = 0.001
    dispersion_ip_tol: float = 2e-05
    ramp_c_minus_tol: float = 0.003
    ramp_coupling_rms_tol: float = 0.03
    ramp_coupling_ip_tol: float = 0.03
    ramp_dispersion_rms_tol: float = 0.005
    ramp_dispersion_ip_tol: float = 0.0001
    dispersion_weight: float = 1.0
    coupling_orbit_tol: float = 5e-08
    coupling_step: float = 0.5
    max_iterations: int = 40

    @property
    def horizontal_dispersion_weight(self):
        return self.beta_rms_tol / self.horizontal_dispersion_rms_tol

    @property
    def horizontal_dispersion_ip_weight(self):
        return (
            self.ir_beta_weight
            * self.horizontal_dispersion_rms_tol
            / self.horizontal_dispersion_ip_tol
        )

    @property
    def dispersion_ip_weight(self):
        return self.coupling_ip_weight * self.dispersion_rms_tol / self.dispersion_ip_tol


@dataclass
class MachineSnapshot:
    """Controls and accepted orbit needed to undo an unaccepted ramp trial.

    Twiss tables are retained by reference; correction steps treat them as
    read-only. Variable values and the closed-orbit guess are independent copies.
    """

    knob_values: dict
    co_guess: dict
    accepted_twiss: object
    final_refinement: bool
    history_length: int


class FCCeeCorrection:
    """Own correction state for a line prepared by 003a, leaving the ramp explicit.

    Construction builds the nominal reference and five fixed-response optimizers,
    then starts at zero error/nonlinear strength. The optimizers, settings,
    nominal knob values, reference, accepted_twiss and history are public.

    Snapshot before a trial; record only after accepting it. History stores
    accepted ramp points, including separate final-refinement entries, rather
    than the intermediate backtracking evaluations inside each correction step.
    """

    correction_errors = (ValueError, RuntimeError, xt.twiss.ClosedOrbitSearchError)

    def __init__(self, line, *, settings=None, log=print):
        self.line = line
        self.settings = settings if settings is not None else CorrectionSettings()
        self.log = log
        self.final_refinement = False
        self.history = []
        self.accepted_twiss = None
        self.optimizers = SimpleNamespace()
        self.stats = dict(calls=0, seconds=0.0, orbit_calls=0, orbit_seconds=0.0)
        self.steps_r_matrix = dict(dx=1e-8, dpx=1e-9, dy=1e-8, dpy=1e-9, dzeta=1e-7, ddelta=1e-8)
        self.log('Preparing nominal reference and correction responses')
        self._prepare_reference()
        self._build_optimizers()
        self._set_values(self.orbit_knobs, np.zeros(len(self.orbit_knobs)))
        self._set_values(self.quad_knobs, np.zeros(len(self.quad_knobs)))
        self.set_error_fraction(0.0)
        for name in self.tune_knobs:
            self.line[name] = self.original[name]
        self.accepted_twiss = self._full_twiss(self.reference)
        self.line.twiss_default['co_guess'] = orbit_guess(self.accepted_twiss)

    @property
    def error_fraction(self):
        """Live error fraction, including an unaccepted trial until restore/record."""
        return float(self.line['on_errors'])

    def twiss(self):
        """Evaluate the current machine, starting from its last accepted orbit."""
        return self._full_twiss(co_guess=orbit_guess(self.accepted_twiss))

    def scale_corrections(self, next_error_fraction):
        """Extrapolate deviations from nominal settings, without changing errors."""
        if not 0 <= next_error_fraction <= 1:
            raise ValueError('Error fraction must be in [0, 1]')
        previous = self.error_fraction
        if previous == 0 or previous == next_error_fraction:
            return
        ratio = next_error_fraction / previous
        for knobs in (self.orbit_knobs, self.quad_knobs, self.skew_knobs, self.tune_knobs):
            nominal = np.array([self.nominal_knob_values[name] for name in knobs])
            self._set_values(knobs, nominal + ratio * (self._values(knobs) - nominal))

    def set_error_fraction(self, fraction):
        """Set errors and nonlinear strengths together; do not change corrections."""
        if not 0 <= fraction <= 1:
            raise ValueError('Error fraction must be in [0, 1]')
        self.line['on_errors'] = self.line['on_nonlinear'] = fraction

    def snapshot(self):
        """Capture every control that a ramp trial can change, without tracking."""
        return MachineSnapshot(
            knob_values={name: float(self.line[name]) for name in self.control_names},
            co_guess=deepcopy(self.line.twiss_default.get('co_guess')),
            accepted_twiss=self.accepted_twiss,
            final_refinement=self.final_refinement,
            history_length=len(self.history),
        )

    def restore(self, snapshot):
        """Undo a trial, including error strength and accepted-orbit bookkeeping."""
        self.line.vars.update(snapshot.knob_values)
        self.line.twiss_default['co_guess'] = deepcopy(snapshot.co_guess)
        self.accepted_twiss = snapshot.accepted_twiss
        self.final_refinement = snapshot.final_refinement
        del self.history[snapshot.history_length :]

    def correct_at_fixed_strengths(self, tw):
        """Return the accepted Twiss and completed iteration count, or raise.

        The caller owns rollback of the entire ramp trial. Each numerical step
        retains 003b's own backtracking and orbit/tune settling procedure.
        """
        self.log_iteration(0, tw)
        for iteration in range(self.settings.max_iterations):
            orbit_ok = self.orbit_goal(tw)
            optics_ok = self.optical_goal(tw)
            coupling_ok = self.coupling_goal(tw)
            if orbit_ok and optics_ok and coupling_ok:
                return tw, iteration
            if not orbit_ok or not optics_ok:
                try:
                    candidate = self.orbit_step(tw)
                except RuntimeError:
                    if not orbit_ok:
                        raise
                else:
                    tw = self.tune_cleanup(candidate)
            if not self.coupling_goal(tw):
                tw = self.coupling_step(tw)
            if not self.optical_goal(tw):
                tw = self.optics_step(tw)
            self.log_iteration(iteration + 1, tw)
        raise RuntimeError('Interleaved stage iteration limit')

    def log_iteration(self, iteration, tw):
        """Summarize existing Twiss residuals without an extra model evaluation.

        RDT is the larger RMS residual of f1001 and f1010; tune error is
        the maximum absolute error across the two planes. Full diagnostics
        remain available through measure() and the accepted history.
        """
        if iteration == 0:
            self.log(
                'iteration | orbit RMS x/y [um] | beta RMS x/y [%] | '
                '|dQ| max | dDx/dDy RMS [mm] | RDT RMS max | |C-|'
            )
        m = self.measure(tw)
        x, y = 1e6 * m['orbit_rms']
        bx, by = 100 * m['beta_rms']
        label = 'initial' if iteration == 0 else str(iteration)
        self.log(
            f"{label:>9} | {x:8.2f} / {y:8.2f} | {bx:6.2f} / {by:6.2f} | "
            f"{max(abs(m['tune_error'])):8.1e} | "
            f"{1e3 * m['dx_rms']:6.3f} / {1e3 * m['dy_rms']:6.3f} | "
            f"{max(m['coupling_rms']):11.2e} | {abs(m['c_minus']):.2e}"
        )

    def orbit_goal(self, tw):
        tolerance = max(0.002e-6, self.error_fraction * self.settings.orbit_tol)
        return bool(np.all(obs.rms_arrays(self.orbit_errors(tw)) <= tolerance))

    def goals_met(self, tw):
        return self.orbit_goal(tw) and self.optical_goal(tw) and self.coupling_goal(tw)

    def record(self, tw, *, iterations):
        """Accept a result and append independent knob/observable values to history.

        Final refinement is a separate entry at the same error fraction. CSV
        diagnostics use the latest entry at each fraction, as in 003b.
        """
        metrics = self.measure(tw)
        self.accepted_twiss = tw
        self.line.twiss_default['co_guess'] = orbit_guess(tw)
        self.history.append(
            dict(
                error_fraction=self.error_fraction,
                final_refinement=self.final_refinement,
                iterations=int(iterations),
                knob_values={name: float(self.line[name]) for name in self.correction_knobs},
                co_guess=deepcopy(self.line.twiss_default['co_guess']),
                observables={
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in metrics.items()
                },
                row=history_row(tw, self.error_fraction, self.line, metrics),
            )
        )
        self.log('Accepted')

    @property
    def history_rows(self):
        """Latest accepted diagnostics per error fraction, in 003b CSV order."""
        rows = []
        for entry in self.history:
            row = entry['row']
            if rows and rows[-1][0] == row[0]:
                rows[-1] = list(row)
            else:
                rows.append(list(row))
        return rows

    def save_result(self, path, *, completed, ramp_settings=None):
        """Save the accepted line and history for independent analysis with 003c."""
        tw = self._full_twiss(self.accepted_twiss)
        if completed:
            assert self.goals_met(tw)
        self.line.twiss_default['co_guess'] = orbit_guess(tw)
        self.line.twiss_default['steps_R_matrix'] = self.steps_r_matrix.copy()
        metrics = self.measure(tw)
        save_result(
            self.line,
            path,
            tw_ref=self.reference,
            progress=self.error_fraction,
            completed=completed,
            metrics=metrics,
            settings=dict(
                correction_settings=asdict(self.settings), ramp_settings=ramp_settings or {}
            ),
        )
        np.savetxt(
            Path(path).parent / 'history.csv',
            np.array(self.history_rows, dtype=float),
            delimiter=',',
            header=HISTORY_COLUMNS,
            comments='',
        )
        self.log(f'Saved {path}')
        return tw

    def _values(self, names):
        return steps.values(self.line, names)

    def _set_values(self, names, values):
        steps.set_values(self.line, names, values)

    def _full_twiss(self, seed=None, co_guess=None, radiation_integrals=False):
        return steps.compute_twiss(
            seed,
            co_guess,
            radiation_integrals,
            line=self.line,
            stats=self.stats,
            steps=self.steps_r_matrix,
        )

    def orbit_twiss(self, seed=None, co_guess=None):
        return steps.compute_twiss(
            seed,
            co_guess,
            only_orbit=True,
            line=self.line,
            stats=self.stats,
            steps=self.steps_r_matrix,
        )

    def ensure_full_twiss(self, tw):
        return steps.ensure_full_twiss(tw, full_twiss=self._full_twiss)

    def monitor_alignment(self):
        # Read live expressions: the BPM offsets include the master error knob.
        return [
            np.array(
                [
                    self.env[self.line.metadata['tuning_errors']['bpm_alignment_variables'][n][a]]
                    for n in self.monitor_names
                ],
                dtype=float,
            )
            for a in ('shift_x', 'shift_y', 'rot_s_rad')
        ]

    def optics_vector(self, tw):
        return obs.optics_vector(
            tw,
            tw_ref=self.reference,
            obs_indices=self.obs_indices,
            alpha_indices=self.alpha_indices,
            dispersion_indices=self.dispersion_indices,
        )

    def coupling_vector(self, tw):
        return obs.coupling_vector(
            tw, tw_ref=self.reference, coupling_indices=self.dispersion_indices
        )

    def orbit_errors(self, tw, fraction=None):
        return obs.orbit_errors(
            tw,
            fraction,
            monitor_indices=self.monitor_indices,
            monitor_alignment=self.monitor_alignment,
        )

    def lab_orbit_errors(self, tw):
        return obs.lab_orbit_errors(
            tw, monitor_indices=self.monitor_indices, monitor_alignment=self.monitor_alignment
        )

    def measure(self, tw, *, orbit_errors=True):
        return obs.measure(
            tw,
            tw_ref=self.reference,
            dispersion_indices=self.dispersion_indices,
            ip_indices=self.ip_indices,
            orbit_errors=self.orbit_errors if orbit_errors else None,
        )

    def coupling_goal(self, tw):
        m = self.measure(tw, orbit_errors=None)
        rdt_rms, rdt_ip, c_minus, dy_rms, dy_ip = (
            (
                self.settings.coupling_rms_tol,
                self.settings.coupling_ip_tol,
                self.settings.c_minus_tol,
                self.settings.dispersion_rms_tol,
                self.settings.dispersion_ip_tol,
            )
            if self.final_refinement
            else (
                self.settings.ramp_coupling_rms_tol,
                self.settings.ramp_coupling_ip_tol,
                self.settings.ramp_c_minus_tol,
                self.settings.ramp_dispersion_rms_tol,
                self.settings.ramp_dispersion_ip_tol,
            )
        )
        return (
            np.all(m['coupling_rms'] < rdt_rms)
            and m['coupling_ip'] < rdt_ip
            and (m['c_minus'] < c_minus)
            and (m['dy_rms'] < dy_rms)
            and (m['dy_ip'] < dy_ip)
        )

    def optical_goal(self, tw):
        m = self.measure(tw, orbit_errors=None)
        dx_rms, dx_ip = (
            (
                self.settings.horizontal_dispersion_rms_tol,
                self.settings.horizontal_dispersion_ip_tol,
            )
            if self.final_refinement
            else (
                self.settings.ramp_horizontal_dispersion_rms_tol,
                self.settings.ramp_horizontal_dispersion_ip_tol,
            )
        )
        return (
            np.all(m['beta_rms'] < self.settings.beta_rms_tol)
            and max(m['beta_max']) < self.settings.beta_max_tol
            and (m['dx_rms'] < dx_rms)
            and (m['dx_ip'] < dx_ip)
            and (max(abs(m['tune_error'])) < self.settings.tune_tol)
        )

    def optical_cost(self, tw):
        return np.linalg.norm((self.optics_weights * self.optics_vector(tw))[:-2])

    def coupling_cost(self, tw):
        return np.linalg.norm(self.coupling_weights * self.coupling_vector(tw))

    def optical_direction(self, tw):
        return self.optimizers.optics.get_correction(
            self.optics_vector(tw), rcond=self.settings.optics_rcond
        )

    def steering_direction(self, tw, fraction):
        return np.concatenate(
            [
                opt.get_correction(error, rcond=self.settings.orbit_rcond)
                for error, opt in zip(
                    self.lab_orbit_errors(tw), [self.optimizers.orbit_x, self.optimizers.orbit_y]
                )
            ]
        )

    def correctable_orbit_errors(self, tw):
        return obs.correctable_orbit_errors(
            tw,
            lab_orbit_errors=self.lab_orbit_errors,
            orbit_optimizers=[self.optimizers.orbit_x, self.optimizers.orbit_y],
            rcond=self.settings.orbit_rcond,
        )

    def tune_cleanup(self, tw):
        return steps.tune_cleanup(
            tw,
            line=self.line,
            tune_knobs=self.tune_knobs,
            tw_ref=self.reference,
            tune_tol=self.settings.tune_tol,
            tune_optimizer=self.optimizers.tunes,
            ensure_full_twiss=self.ensure_full_twiss,
            full_twiss=self._full_twiss,
            log=_silent_log,
        )

    def orbit_step(self, tw, fraction=None, max_backtracks=13, only_orbit=False):
        if fraction is None:
            fraction = self.error_fraction
        return steps.orbit_step(
            tw,
            fraction,
            max_backtracks,
            only_orbit,
            line=self.line,
            orbit_knobs=self.orbit_knobs,
            steering_direction=self.steering_direction,
            orbit_errors=self.orbit_errors,
            orbit_twiss=self.orbit_twiss,
            full_twiss=self._full_twiss,
            log=_silent_log,
        )

    def optics_step(self, tw):
        return steps.optics_step(
            tw,
            line=self.line,
            quad_knobs=self.quad_knobs,
            tune_knobs=self.tune_knobs,
            orbit_knobs=self.orbit_knobs,
            optical_direction=self.optical_direction,
            optical_cost=self.optical_cost,
            strength_scale=self.strength_scale,
            orbit_twiss=self.orbit_twiss,
            orbit_step=self.orbit_step,
            tune_cleanup=self.tune_cleanup,
            measure=self.measure,
            log=_silent_log,
        )

    def settle_coupling_orbit(self, tw):
        return steps.settle_coupling_orbit(
            tw,
            line=self.line,
            correctable_orbit_errors=self.correctable_orbit_errors,
            coupling_orbit_tol=self.settings.coupling_orbit_tol,
            orbit_step=self.orbit_step,
            tune_cleanup=self.tune_cleanup,
        )

    def coupling_step(self, tw):
        return steps.coupling_step(
            tw,
            line=self.line,
            skew_knobs=self.skew_knobs,
            orbit_knobs=self.orbit_knobs,
            tune_knobs=self.tune_knobs,
            coupling_optimizer=self.optimizers.coupling,
            coupling_vector=self.coupling_vector,
            coupling_rcond=self.settings.coupling_rcond,
            step_fraction=self.settings.coupling_step,
            coupling_cost=self.coupling_cost,
            coupling_goal=self.coupling_goal,
            settle_coupling_orbit=self.settle_coupling_orbit,
            orbit_twiss=self.orbit_twiss,
            measure=self.measure,
            log=_silent_log,
        )

    def _prepare_reference(self):
        self.env = self.line.env
        self.circuits = self.line.metadata['correction_knobs']
        self.monitor_names = self.circuits['orbit']['x']['monitors']
        self.corrector_names = [self.circuits['orbit'][p]['correctors'] for p in 'xy']
        self.orbit_knobs = (
            self.circuits['orbit']['x']['knobs'] + self.circuits['orbit']['y']['knobs']
        )
        self.quad_names = self.circuits['optics']['quadrupoles']
        self.quad_knobs = self.circuits['optics']['knobs']
        self.skew_groups = self.circuits['coupling']['magnet_groups']
        self.skew_knobs = self.circuits['coupling']['knobs']
        self.tune_knobs = self.circuits['tunes']['knobs']
        self.original = self.circuits['tunes']['nominal_values']
        self.correction_knobs = (
            self.orbit_knobs + self.quad_knobs + self.skew_knobs + self.tune_knobs
        )
        self.control_names = self.correction_knobs + ['on_errors', 'on_nonlinear']
        self.nominal_knob_values = dict.fromkeys(self.correction_knobs, 0.0)
        self.nominal_knob_values.update(self.original)
        self.reference = setup.nominal_reference(
            self.line, full_twiss=self._full_twiss, log=_silent_log
        )
        self.interaction_points = list(self.line.get_table().rows['ip.*'].name)
        self.obs_names, self.alpha_names = setup.optics_observation_names(
            self.reference, self.quad_names, self.interaction_points, phase_step=1 / 16
        )
        self.obs_indices = self.reference.rows.indices[self.obs_names]
        self.alpha_indices = self.reference.rows.indices[self.alpha_names]
        self.monitor_indices = self.reference.rows.indices[self.monitor_names]
        self.ip_names = ['ipa', 'ipd', 'ipg', 'ipj']
        self.ip_indices = self.reference.rows.indices[self.ip_names]
        self.dispersion_names = list(dict.fromkeys(self.monitor_names + self.ip_names))
        self.dispersion_indices = self.reference.rows.indices[self.dispersion_names]
        self._set_values(self.orbit_knobs, np.zeros(len(self.orbit_knobs)))
        self.line['on_errors'] = 0.0
        for name in self.tune_knobs:
            self.line[name] = self.original[name]

    def _build_optimizers(self):
        nominal_response, self.strength_scale, self.optics_weights, self.optics_labels = (
            setup.prepare_optics_response(
                self.line,
                self.reference,
                self.quad_names,
                self.obs_names,
                self.alpha_names,
                self.dispersion_names,
                interaction_points=self.interaction_points,
                dispersion_ips=self.ip_names,
                alpha_weight=self.settings.alpha_weight,
                ir_beta_weight=self.settings.ir_beta_weight,
                ir_half_length=self.settings.ir_half_length,
                dispersion_weight=self.settings.horizontal_dispersion_weight,
                dispersion_ip_weight=self.settings.horizontal_dispersion_ip_weight,
                knobs=self.quad_knobs,
                full_twiss=self._full_twiss,
                log=_silent_log,
            )
        )
        optics_action = xt.Action(lambda: self.optics_vector(self._full_twiss()))
        self.optimizers.optics = self.line.match(
            solve=False,
            name='normal optics and horizontal dispersion',
            vary=[
                xt.Vary(knob, step=1e-07, weight=scale)
                for knob, scale in zip(self.quad_knobs, self.strength_scale)
            ],
            targets=[
                optics_action.target(i, value=0.0, tol=1e-15, weight=weight, tag=label)
                for i, (label, weight) in enumerate(zip(self.optics_labels, self.optics_weights))
            ],
            assert_within_tol=False,
            show_call_counter=False,
        )
        self.optimizers.optics.set_response(nominal_response)
        self.opt_s = self.optimizers.optics.response.svd.s
        del nominal_response
        orbit_action_x = xt.Action(
            lambda: self._full_twiss()['x'][self.monitor_indices] - self.monitor_alignment()[0]
        )
        self.optimizers.orbit_x = self.line.match(
            solve=False,
            name='orbit x',
            vary=[xt.Vary(knob, step=1e-07) for knob in self.circuits['orbit']['x']['knobs']],
            targets=[
                orbit_action_x.target(i, value=0.0, tol=1e-15, tag=f'x@{name}')
                for i, name in enumerate(self.monitor_names)
            ],
            assert_within_tol=False,
            show_call_counter=False,
        )
        self.optimizers.orbit_x.set_response(
            orbit_response(self.reference, self.monitor_names, self.corrector_names[0], 'x')
        )
        orbit_action_y = xt.Action(
            lambda: self._full_twiss()['y'][self.monitor_indices] - self.monitor_alignment()[1]
        )
        self.optimizers.orbit_y = self.line.match(
            solve=False,
            name='orbit y',
            vary=[xt.Vary(knob, step=1e-07) for knob in self.circuits['orbit']['y']['knobs']],
            targets=[
                orbit_action_y.target(i, value=0.0, tol=1e-15, tag=f'y@{name}')
                for i, name in enumerate(self.monitor_names)
            ],
            assert_within_tol=False,
            show_call_counter=False,
        )
        self.optimizers.orbit_y.set_response(
            orbit_response(self.reference, self.monitor_names, self.corrector_names[1], 'y')
        )
        self.coupling_names = self.dispersion_names
        self.coupling_indices = self.dispersion_indices
        coupling_response, self.skew_scale, self.coupling_weights, self.coupling_labels = (
            setup.prepare_coupling_response(
                self.line,
                self.reference,
                self.skew_groups,
                self.coupling_names,
                ip_names=self.ip_names,
                coupling_ip_weight=self.settings.coupling_ip_weight,
                dispersion_weight=self.settings.dispersion_weight,
                dispersion_ip_weight=self.settings.dispersion_ip_weight,
                knobs=self.skew_knobs,
                full_twiss=self._full_twiss,
                log=_silent_log,
            )
        )
        coupling_action = xt.Action(lambda: self.coupling_vector(self._full_twiss()))
        self.optimizers.coupling = self.line.match(
            solve=False,
            name='coupling and vertical dispersion',
            vary=[
                xt.Vary(knob, step=1e-07, weight=scale)
                for knob, scale in zip(self.skew_knobs, self.skew_scale)
            ],
            targets=[
                coupling_action.target(i, value=0.0, tol=1e-15, weight=weight, tag=label)
                for i, (label, weight) in enumerate(
                    zip(self.coupling_labels, self.coupling_weights)
                )
            ],
            assert_within_tol=False,
            show_call_counter=False,
        )
        self.optimizers.coupling.set_response(coupling_response)
        del coupling_response
        self.tune_response = setup.build_tune_response(
            self.reference, line=self.line, tune_knobs=self.tune_knobs, full_twiss=self._full_twiss
        )

        def tune_residuals():
            tw = self._full_twiss()
            return np.array([tw.qx - self.reference.qx, tw.qy - self.reference.qy])

        tune_action = xt.Action(tune_residuals)
        self.optimizers.tunes = self.line.match(
            solve=False,
            name='tunes',
            vary=[xt.Vary(knob, step=1e-07) for knob in self.tune_knobs],
            targets=[
                tune_action.target(i, value=0.0, tol=self.settings.tune_tol / 2, tag=q)
                for i, q in enumerate(('qx', 'qy'))
            ],
            assert_within_tol=False,
            show_call_counter=False,
        )
        self.optimizers.tunes.set_response(self.tune_response)
