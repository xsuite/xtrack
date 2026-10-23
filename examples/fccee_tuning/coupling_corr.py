"""Coupling and vertical dispersion: setup, metrics, goals and reporting."""

from dataclasses import dataclass
import numpy as np
import xtrack as xt

import correction_setup as setup
from correction_optimizer import CorrectionOptimizer


@dataclass
class CouplingSettings:
    rcond: float = 1e-6
    coupling_ip_weight: float = 20.
    coupling_rms_tol: float = 0.003
    coupling_ip_tol: float = 0.002
    c_minus_tol: float = 0.0003
    dispersion_rms_tol: float = 0.001
    dispersion_ip_tol: float = 0.00002
    ramp_c_minus_tol: float = 0.003
    ramp_coupling_rms_tol: float = 0.03
    ramp_coupling_ip_tol: float = 0.03
    ramp_dispersion_rms_tol: float = 0.005
    ramp_dispersion_ip_tol: float = 0.0001
    dispersion_weight: float = 1.
    orbit_tol: float = 5e-8
    step_scale: float = 0.5

    @property
    def dispersion_ip_weight(self):
        return self.coupling_ip_weight * self.dispersion_rms_tol / self.dispersion_ip_tol


def compute_metrics(tw, reference, *, coupling_indices, ip_indices):
    """RDT and vertical dispersion errors at BPMs/IPs, and global c_minus."""
    f1001 = tw['f1001'] - reference['f1001']
    f1010 = tw['f1010'] - reference['f1010']
    dy = tw.dy - reference.dy
    return dict(
        coupling_rms=np.array([np.sqrt(np.mean(np.abs(f1001[coupling_indices])**2)),
                               np.sqrt(np.mean(np.abs(f1010[coupling_indices])**2))]),
        coupling_ip=max(np.max(np.abs(f1001[ip_indices])),
                        np.max(np.abs(f1010[ip_indices]))),
        c_minus=float(tw.c_minus),
        dy_rms=float(np.sqrt(np.mean(dy[coupling_indices]**2))),
        dy_ip=float(np.max(np.abs(dy[ip_indices]))),
    )


class CouplingCorrection(CorrectionOptimizer):
    def __init__(self, line, reference, *, observation_names, ip_names,
                 full_twiss, settings=None, log=print):
        settings = settings if settings is not None else CouplingSettings()
        action = xt.Action(full_twiss)
        circuits = line.metadata['correction_knobs']['coupling']
        knobs = circuits['knobs']
        response, scales, weights, labels = setup.prepare_coupling_response(
            line, reference, circuits['magnet_groups'], observation_names,
            ip_names=ip_names,
            coupling_ip_weight=settings.coupling_ip_weight,
            dispersion_weight=settings.dispersion_weight,
            dispersion_ip_weight=settings.dispersion_ip_weight,
            knobs=knobs, full_twiss=full_twiss, log=log,
        )
        indices = reference.rows.indices[observation_names]
        self.coupling_indices = indices
        self.ip_indices = reference.rows.indices[ip_names]
        observables = (
            [lambda tw, key=key, part=part, i=i:
                getattr(tw[key][i] - reference[key][i], part)
             for key in ('f1001', 'f1010') for part in ('real', 'imag') for i in indices]
            + [lambda tw, i=i: tw.dy[i] - reference.dy[i] for i in indices]
        )
        super().__init__(
            line, reference, settings=settings,
            name='coupling and vertical dispersion',
            vary=[xt.Vary(knob, step=1e-7, weight=scale) for knob, scale in zip(knobs, scales)],
            targets=[action.target(observable, value=0.0, tol=1e-15,
                                   weight=weight, tag=label)
                     for observable, label, weight in zip(observables, labels, weights)],
        )
        self.rcond = settings.rcond
        self.set_response(response)

        self.step_scale = settings.step_scale
        self.orbit_tol = settings.orbit_tol
        self.configure_goals(final_refinement=False, error_fraction=0.)

    def configure_goals(self, *, final_refinement, error_fraction):
        if final_refinement:
            self.rdt_rms_tol = self.settings.coupling_rms_tol
            self.rdt_ip_tol = self.settings.coupling_ip_tol
            self.c_minus_tol = self.settings.c_minus_tol
            self.dy_rms_tol = self.settings.dispersion_rms_tol
            self.dy_ip_tol = self.settings.dispersion_ip_tol
        else:
            self.rdt_rms_tol = self.settings.ramp_coupling_rms_tol
            self.rdt_ip_tol = self.settings.ramp_coupling_ip_tol
            self.c_minus_tol = self.settings.ramp_c_minus_tol
            self.dy_rms_tol = self.settings.ramp_dispersion_rms_tol
            self.dy_ip_tol = self.settings.ramp_dispersion_ip_tol

    def compute_metrics(self, tw):
        return compute_metrics(tw, self.reference, coupling_indices=self.coupling_indices,
                               ip_indices=self.ip_indices)

    def goal(self, tw):
        m = self.compute_metrics(tw)
        return bool(np.all(m['coupling_rms'] < self.rdt_rms_tol)
                    and m['coupling_ip'] < self.rdt_ip_tol
                    and abs(m['c_minus']) < self.c_minus_tol
                    and m['dy_rms'] < self.dy_rms_tol
                    and m['dy_ip'] < self.dy_ip_tol)

    def print_metrics(self, tw, *, log=print):
        m = self.compute_metrics(tw)
        log('Coupling: RDT RMS/IP', m['coupling_rms'], m['coupling_ip'],
            '|C-|', abs(m['c_minus']), 'Dy RMS/IP [mm]',
            1e3 * m['dy_rms'], 1e3 * m['dy_ip'])
