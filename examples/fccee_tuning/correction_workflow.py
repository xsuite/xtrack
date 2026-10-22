"""Inspectable FCC-ee correction workflow used by 004.

This is an example-local API prototype. The ramp remains in the user script;
the class owns Twiss evaluation, trial acceptance, snapshots and history.
Optimizers consume supplied Twiss tables and apply least-squares changes.
003b and its shared trial helpers remain the numerical reference.
"""

from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import xtrack as xt
import xdeps as xd

import correction_observables as obs
import correction_steps as steps
import correction_setup as setup
from correction_io import HISTORY_COLUMNS, history_row, save_result
from correction_steps import orbit_guess
import optics_corr
import orbit_corr
import coupling_corr
import tune_corr


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
        # This workflow uses 4D optics at the reference momentum, as in 000.
        self.line.twiss_default.update(method='4d', delta0=0.0)
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
        self._build_goal_optimizers()
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
        values = self.line.vars.val
        return MachineSnapshot(
            knob_values={name: float(values[name]) for name in self.control_names},
            co_guess=deepcopy(self.line.twiss_default.get('co_guess')),
            accepted_twiss=self.accepted_twiss,
            final_refinement=self.final_refinement,
            history_length=len(self.history),
        )

    def restore(self, snapshot):
        """Undo a trial, including error strength and accepted-orbit bookkeeping."""
        values = self.line.vars.val
        self.line.vars.update({
            name: value for name, value in snapshot.knob_values.items()
            if values[name] != value
        })
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
        metrics = dict(orbit_rms=obs.rms_arrays(self.orbit_errors(tw)))
        # Target tolerance checks are strict; preserve 003b's inclusive orbit goal.
        return self._goals_met('orbit', metrics, [np.nextafter(tolerance, np.inf)] * 2)

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
        tw = self.twiss()
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
            seed, co_guess, radiation_integrals,
            line=self.line, stats=self.stats, steps=self.steps_r_matrix)

    def orbit_twiss(self, seed=None, co_guess=None):
        return steps.compute_twiss(
            seed, co_guess, only_orbit=True,
            line=self.line, stats=self.stats, steps=self.steps_r_matrix)

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
        return self._goals_met('coupling', m,
                               [rdt_rms, rdt_rms, rdt_ip, c_minus, dy_rms, dy_ip])

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
        return self._goals_met('optics', m,
                               [self.settings.beta_rms_tol] * 2
                               + [self.settings.beta_max_tol] * 2
                               + [dx_rms, dx_ip] + [self.settings.tune_tol] * 2)

    def _build_goal_optimizers(self):
        # Aggregate RMS/max goals are distinct from the response rows. These
        # inspection-only optimizers do not calculate or apply a direction.
        self.metrics_action = xt.Action(lambda: self.measure(self.twiss()))
        groups = dict(
            orbit=[('orbit_rms', i) for i in range(2)],
            optics=[('beta_rms', i) for i in range(2)]
                   + [('beta_max', i) for i in range(2)]
                   + [('dx_rms', None), ('dx_ip', None)]
                   + [('tune_error', i) for i in range(2)],
            coupling=[('coupling_rms', i) for i in range(2)]
                     + [(key, None) for key in ('coupling_ip', 'c_minus', 'dy_rms', 'dy_ip')],
        )
        self.goal_optimizers = SimpleNamespace()
        metrics = self.measure(self.reference)
        for name, quantities in groups.items():
            targets = [
                self.metrics_action.target(
                    lambda data, key=key, i=i: data[key] if i is None else data[key][i],
                    value=0.0, tol=1e-15, tag=key if i is None else f'{key}_{i}')
                for key, i in quantities
            ]
            setattr(self.goal_optimizers, name, xd.Optimize(
                vary=[], targets=targets, name=f'{name} goals',
                action_results={self.metrics_action: metrics}, show_call_counter=False))
        # Populate the live tolerances immediately for interactive inspection.
        self.orbit_goal(self.reference)
        self.optical_goal(self.reference)
        self.coupling_goal(self.reference)

    def _goals_met(self, name, metrics, tolerances):
        optimizer = getattr(self.goal_optimizers, name)
        for target, tolerance in zip(optimizer.targets, tolerances):
            target.tol = tolerance
        status = optimizer.target_status(ret=True, action_results={self.metrics_action: metrics})
        return bool(np.all(status.tol_met))

    def optical_cost(self, tw):
        status = self.optimizers.optics.target_status(
            ret=True, action_results={self.twiss_action: tw})
        return np.linalg.norm((self.optics_weights * status.residue)[:-2])

    def coupling_cost(self, tw):
        status = self.optimizers.coupling.target_status(
            ret=True, action_results={self.twiss_action: tw})
        return np.linalg.norm(self.coupling_weights * status.residue)

    def correctable_orbit_errors(self, tw):
        return obs.correctable_orbit_errors(
            tw,
            lab_orbit_errors=self.lab_orbit_errors,
            orbit_optimizers=[self.optimizers.orbit_x, self.optimizers.orbit_y],
            rcond=self.settings.orbit_rcond,
        )

    def tune_cleanup(self, tw):
        opt = self.optimizers.tunes
        for target in opt.targets:
            target.tol = self.settings.tune_tol / 2
        tw = self.ensure_full_twiss(tw)
        for iteration in range(7):
            status = opt.target_status(ret=True, action_results={self.twiss_action: tw})
            if np.all(status.tol_met):
                return tw
            before = self.snapshot()
            for backtrack in range(4):
                self.restore(before)
                opt.run_lstsq(action_results={self.twiss_action: tw},
                              rcond=0.0, step_scale=0.5**backtrack,
                              tag_outcome=False)
                try:
                    trial = self._full_twiss(tw)
                except (ValueError, xt.twiss.ClosedOrbitSearchError):
                    continue
                trial_status = opt.target_status(
                    ret=True, action_results={self.twiss_action: trial})
                if np.linalg.norm(trial_status.residue) < np.linalg.norm(status.residue):
                    tw = trial
                    break
            else:
                self.restore(before)
                raise RuntimeError('Tune cleanup failed with the fixed nominal response')
        status = opt.target_status(ret=True, action_results={self.twiss_action: tw})
        if np.all(status.tol_met):
            return tw
        raise RuntimeError('Tune cleanup iteration limit')

    def orbit_step(self, tw, fraction=None, max_backtracks=13, only_orbit=False):
        before = self.snapshot()
        cost = np.linalg.norm(np.concatenate(self.orbit_errors(tw)))
        for backtrack in range(max_backtracks):
            self.restore(before)
            for opt in (self.optimizers.orbit_x, self.optimizers.orbit_y):
                opt.run_lstsq(action_results={self.twiss_action: tw},
                              rcond=self.settings.orbit_rcond, step_scale=0.5**backtrack,
                              tag_outcome=False)
            try:
                trial = self.orbit_twiss(tw) if only_orbit else self._full_twiss(tw)
            except (ValueError, xt.twiss.ClosedOrbitSearchError):
                continue
            if np.linalg.norm(np.concatenate(self.orbit_errors(trial))) < cost * (1 - 1e-6):
                return trial
        self.restore(before)
        raise RuntimeError('No stable improving orbit step')

    def optics_step(self, tw):
        before = self.snapshot()
        cost = self.optical_cost(tw)
        for backtrack in range(10):
            self.restore(before)
            self.optimizers.optics.run_lstsq(
                action_results={self.twiss_action: tw},
                rcond=self.settings.optics_rcond, step_scale=0.5**backtrack,
                tag_outcome=False)
            try:
                trial = self.orbit_twiss(tw)
                try:
                    trial = self.orbit_step(trial)
                except RuntimeError:
                    pass
                trial = self.tune_cleanup(trial)
            except (ValueError, RuntimeError):
                continue
            if self.optical_cost(trial) < cost * (1 - 1e-6):
                return trial
        self.restore(before)
        raise RuntimeError('No stable improving joint step')

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
        tw = self.settle_coupling_orbit(tw)
        if self.coupling_goal(tw):
            return tw
        before = self.snapshot()
        cost = self.coupling_cost(tw)
        for backtrack in range(10):
            self.restore(before)
            self.optimizers.coupling.run_lstsq(
                action_results={self.twiss_action: tw},
                rcond=self.settings.coupling_rcond,
                step_scale=self.settings.coupling_step * 0.5**backtrack,
                tag_outcome=False)
            try:
                trial = self.orbit_twiss(tw)
                trial = self.settle_coupling_orbit(trial)
            except self.correction_errors:
                continue
            if self.coupling_cost(trial) < cost * (1 - 1e-6):
                return trial
        self.restore(before)
        raise RuntimeError('No stable improving coupling step')

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
        # All optimizers consume the same externally supplied Twiss table.
        self.twiss_action = xt.Action(self.twiss)
        self.optimizers.optics = optics_corr.build_optimizer(
            self.line, self.reference, self.twiss_action,
            beta_at=self.obs_names, alpha_at=self.alpha_names,
            dispersion_at=self.dispersion_names,
            interaction_points=self.interaction_points, ip_names=self.ip_names,
            settings=self.settings, full_twiss=self._full_twiss, log=_silent_log,
        )
        self.opt_s = self.optimizers.optics.response.svd.s
        self.optimizers.orbit_x, self.optimizers.orbit_y = orbit_corr.build_optimizers(
            self.line, self.reference, self.twiss_action)
        self.coupling_names = self.dispersion_names
        self.coupling_indices = self.dispersion_indices
        self.optimizers.coupling = coupling_corr.build_optimizer(
            self.line, self.reference, self.twiss_action,
            observation_names=self.coupling_names, ip_names=self.ip_names,
            settings=self.settings, full_twiss=self._full_twiss, log=_silent_log,
        )
        self.optimizers.tunes = tune_corr.build_optimizer(
            self.line, self.reference, self.twiss_action,
            tune_tol=self.settings.tune_tol, full_twiss=self._full_twiss,
        )
        # Keep the existing inspection attributes tied to optimizer configuration.
        self.strength_scale = np.array([vary.weight for vary in self.optimizers.optics.vary])
        self.optics_weights = np.array([target.weight for target in self.optimizers.optics.targets])
        self.optics_labels = [target.tag for target in self.optimizers.optics.targets]
        self.skew_scale = np.array([vary.weight for vary in self.optimizers.coupling.vary])
        self.coupling_weights = np.array([target.weight for target in self.optimizers.coupling.targets])
        self.coupling_labels = [target.tag for target in self.optimizers.coupling.targets]
        self.tune_response = self.optimizers.tunes.response.matrix
        # As in 003b, pay the SVD construction cost during preparation, not in
        # the first trial. Retries then reuse these decompositions.
        for optimizer in vars(self.optimizers).values():
            _ = optimizer.response.svd
