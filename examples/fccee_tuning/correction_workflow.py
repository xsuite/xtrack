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
    ramp_max_iterations: int = 5
    final_max_iterations: int = 40

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
    than the intermediate evaluations inside each correction step.
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

        The caller owns rollback of the entire ramp trial. Tune correction is
        explicit; coupling correction includes orbit settling. Only goals and
        iteration limits decide success; intermediate errors may increase.
        """
        if self.final_refinement:
            max_iterations = self.settings.final_max_iterations
        else:
            max_iterations = self.settings.ramp_max_iterations
        self.log_iteration(0, tw)
        if self.goals_met(tw):
            return tw, 0
        for iteration in range(1, max_iterations + 1):
            if not self.orbit_goal(tw):
                for opt in (self.optimizers.orbit_x, self.optimizers.orbit_y):
                    opt.run_lstsq(action_results={self.twiss_action: tw},
                                  rcond=self.settings.orbit_rcond, tag_outcome=False)
                tw = self._full_twiss(tw)
            tw = self.correct_tunes(tw)

            if not self.optical_goal(tw):
                self.optimizers.optics.run_lstsq(
                    action_results={self.twiss_action: tw},
                    rcond=self.settings.optics_rcond, tag_outcome=False)
                tw = self._full_twiss(tw)
            tw = self.correct_tunes(tw)

            if not self.coupling_goal(tw):
                tw = self.settle_coupling_orbit(tw)
                if not self.coupling_goal(tw):
                    self.optimizers.coupling.run_lstsq(
                        action_results={self.twiss_action: tw},
                        rcond=self.settings.coupling_rcond,
                        step_scale=self.settings.coupling_step, tag_outcome=False)
                    tw = self.orbit_twiss(tw)
                    tw = self.settle_coupling_orbit(tw)
            tw = self.correct_tunes(tw)

            self.log_iteration(iteration, tw)
            if self.goals_met(tw):
                return tw, iteration
        raise RuntimeError(f'Correction goals not met after {max_iterations} iterations')

    def log_iteration(self, iteration, tw):
        """Summarize existing Twiss residuals without an extra model evaluation.

        RDT is the larger RMS residual of f1001 and f1010; tune error is
        the maximum absolute error across the two planes. Full diagnostics
        remain available through compute_metrics() and the accepted history.
        """
        if iteration == 0:
            self.log(
                'iteration | orbit RMS x/y [um] | beta RMS x/y [%] | '
                '|dQ| max | dDx/dDy RMS [mm] | RDT RMS max | |C-|'
            )
        m = self.compute_metrics(tw)
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
        return orbit_corr.orbit_goal(
            tw, monitor_indices=self.monitor_indices, monitor_alignment=self.monitor_alignment,
            error_fraction=self.error_fraction, settings=self.settings)

    def goals_met(self, tw):
        return self.orbit_goal(tw) and self.optical_goal(tw) and self.coupling_goal(tw)

    def record(self, tw, *, iterations):
        """Accept a result and append independent knob/observable values to history.

        Final refinement is a separate entry at the same error fraction. CSV
        diagnostics use the latest entry at each fraction, as in 003b.
        """
        metrics = self.compute_metrics(tw)
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
        metrics = self.compute_metrics(tw)
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
        """Unweighted target residuals, in the installed response-row order."""
        return self.optimizers.optics.target_status(
            ret=True, action_results={self.twiss_action: tw}).residue

    def coupling_vector(self, tw):
        return coupling_corr.coupling_vector(
            tw, self.reference, coupling_indices=self.dispersion_indices
        )

    def orbit_errors(self, tw, fraction=None):
        return orbit_corr.orbit_errors(
            tw,
            monitor_indices=self.monitor_indices,
            monitor_alignment=self.monitor_alignment,
        )

    def lab_orbit_errors(self, tw):
        return orbit_corr.lab_orbit_errors(
            tw, monitor_indices=self.monitor_indices, monitor_alignment=self.monitor_alignment
        )

    def compute_metrics(self, tw):
        """Combine each correction module's diagnostics for logs and history."""
        return {
            **orbit_corr.compute_metrics(
                tw, monitor_indices=self.monitor_indices, monitor_alignment=self.monitor_alignment),
            **optics_corr.compute_metrics(
                tw, self.reference, dispersion_indices=self.dispersion_indices,
                ip_indices=self.ip_indices),
            **coupling_corr.compute_metrics(
                tw, self.reference, coupling_indices=self.coupling_indices,
                ip_indices=self.ip_indices),
            **tune_corr.compute_metrics(tw, self.reference),
        }

    def coupling_goal(self, tw):
        return coupling_corr.coupling_goal(
            tw, self.reference, coupling_indices=self.coupling_indices,
            ip_indices=self.ip_indices, settings=self.settings,
            final_refinement=self.final_refinement)

    def optical_goal(self, tw):
        return optics_corr.optical_goal(
            tw, self.reference, dispersion_indices=self.dispersion_indices,
            ip_indices=self.ip_indices, settings=self.settings,
            final_refinement=self.final_refinement)

    def correctable_orbit_errors(self, tw):
        return orbit_corr.correctable_orbit_errors(
            tw,
            lab_orbit_errors=self.lab_orbit_errors,
            orbit_optimizers=[self.optimizers.orbit_x, self.optimizers.orbit_y],
            rcond=self.settings.orbit_rcond,
        )

    def correct_tunes(self, tw):
        return tune_corr.correct(
            tw, optimizer=self.optimizers.tunes, action=self.twiss_action,
            tolerance=self.settings.tune_tol / 2, full_twiss=self._full_twiss)

    def settle_coupling_orbit(self, tw):
        # Settle both the baseline and trials so orbit feed-down does not mask
        # the effect of the skew correction. Tune correction remains explicit.
        for iteration in range(3):
            errors = self.correctable_orbit_errors(tw)
            rms = [np.sqrt(np.mean(error**2)) for error in errors]
            if max(rms) < self.settings.coupling_orbit_tol:
                break
            before = self.snapshot()
            try:
                for opt in (self.optimizers.orbit_x, self.optimizers.orbit_y):
                    opt.run_lstsq(action_results={self.twiss_action: tw},
                                  rcond=self.settings.orbit_rcond, tag_outcome=False)
                trial = self.orbit_twiss(tw) if iteration < 2 else self._full_twiss(tw)
            except self.correction_errors:
                # Settling continues from its last accepted orbit; unlike the
                # main stages, a failed trial does not reject the ramp step.
                self.restore(before)
                break
            tw = trial
        return self.ensure_full_twiss(tw)

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
        self.ip_names = ['ipa', 'ipd', 'ipg', 'ipj']
        self.obs_names = list(dict.fromkeys(self.monitor_names + self.ip_names))
        self.alpha_names = self.obs_names
        self.dispersion_names = self.obs_names
        self.obs_indices = self.reference.rows.indices[self.obs_names]
        self.alpha_indices = self.reference.rows.indices[self.alpha_names]
        self.monitor_indices = self.reference.rows.indices[self.monitor_names]
        self.ip_indices = self.reference.rows.indices[self.ip_names]
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
            knobs=self.quad_knobs,
            observation_names=self.obs_names,
            interaction_points=self.interaction_points, ip_names=self.ip_names,
            settings=self.settings,
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
