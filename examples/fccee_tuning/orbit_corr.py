"""Orbit correction, including live BPM alignment and correctable-mode metrics."""

from dataclasses import dataclass
import numpy as np
import xtrack as xt

from correction_optimizer import CorrectionOptimizer
from response_builders import orbit_response


@dataclass
class OrbitSettings:
    orbit_tol: float = 1e-5
    rcond: float = 1e-6


class OrbitCorrection(CorrectionOptimizer):
    def __init__(self, line, reference, *, plane, settings=None):
        if plane not in ('x', 'y'):
            raise ValueError('Orbit plane must be x or y')
        settings = settings if settings is not None else OrbitSettings()
        self.plane = plane
        circuits = line.metadata['correction_knobs']['orbit']
        self.monitor_names = list(circuits['x']['monitors'])
        self.monitor_indices = reference.rows.indices[self.monitor_names]
        alignment = line.metadata['tuning_errors']['bpm_alignment_variables']
        self.alignment_variables = alignment
        monitors = circuits[plane]['monitors']
        indices = reference.rows.indices[monitors]
        action = xt.Action(line.twiss)
        env = line.env
        super().__init__(
            line, reference, settings=settings, name=f'orbit {plane}',
            vary=[xt.Vary(knob, step=1e-7) for knob in circuits[plane]['knobs']],
            targets=[action.target(
                lambda tw, p=plane, i=i, offset=alignment[name]['shift_' + plane]:
                    tw[p][i] - env[offset],
                value=0., tol=1e-15, tag=f'{plane}@{name}')
                for i, name in zip(indices, monitors)])
        self.rcond = settings.rcond
        self.set_response(orbit_response(reference, monitors,
                                         circuits[plane]['correctors'], plane))
        self.configure_goals(final_refinement=False, error_fraction=0.)

    def configure_goals(self, *, final_refinement, error_fraction):
        self.orbit_tol = max(2e-9, error_fraction * self.settings.orbit_tol)

    def monitor_alignment(self):
        return [np.array([self.line.env[self.alignment_variables[name][coordinate]]
                          for name in self.monitor_names], dtype=float)
                for coordinate in ('shift_x', 'shift_y', 'rot_s_rad')]

    def orbit_error(self, tw):
        dx, dy, roll = self.monitor_alignment()
        x = tw.x[self.monitor_indices] - dx
        y = tw.y[self.monitor_indices] - dy
        if self.plane == 'x':
            return x * np.cos(roll) + y * np.sin(roll)
        return -x * np.sin(roll) + y * np.cos(roll)

    def compute_metrics(self, tw):
        error = self.orbit_error(tw)
        return {f'orbit_{self.plane}_rms': float(np.sqrt(np.mean(error**2)))}

    def goal(self, tw):
        return self.compute_metrics(tw)[f'orbit_{self.plane}_rms'] <= self.orbit_tol

    def correctable_error(self, tw):
        """Lab-frame orbit projected onto retained nominal response modes."""
        offset = self.monitor_alignment()[0 if self.plane == 'x' else 1]
        error = tw[self.plane][self.monitor_indices] - offset
        svd = self.response.svd
        rcond = self.rcond if self.rcond is not None else svd.rcond
        u = svd.U[:, svd.s > rcond * svd.s[0]]
        return u @ (u.T @ error)

    def print_metrics(self, tw, *, log=print):
        rms = self.compute_metrics(tw)[f'orbit_{self.plane}_rms']
        log(f'Orbit {self.plane}: RMS {1e6 * rms:.3f} um '
            f'(goal {1e6 * self.orbit_tol:.3f} um)')
