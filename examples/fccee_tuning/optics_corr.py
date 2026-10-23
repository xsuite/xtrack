"""Normal optics: optimizer, response, metrics, goals and reporting."""

from dataclasses import dataclass
import numpy as np
import xtrack as xt

from correction_optimizer import CorrectionOptimizer
from response_builders import normal_quadrupole_response
import tune_corr


@dataclass
class OpticsSettings:
    beta_rms_tol: float = 0.03
    beta_max_tol: float = 0.5
    tune_tol: float = 1e-5
    alpha_weight: float = 0.1
    ir_beta_weight: float = 5.
    ir_half_length: float = 500.
    horizontal_dispersion_rms_tol: float = 0.001
    horizontal_dispersion_ip_tol: float = 0.0001
    ramp_horizontal_dispersion_rms_tol: float = 0.005
    ramp_horizontal_dispersion_ip_tol: float = 0.0005
    rcond: float = 1e-5

    @property
    def horizontal_dispersion_weight(self):
        return self.beta_rms_tol / self.horizontal_dispersion_rms_tol

    @property
    def horizontal_dispersion_ip_weight(self):
        return (self.ir_beta_weight * self.horizontal_dispersion_rms_tol
                / self.horizontal_dispersion_ip_tol)


def compute_metrics(tw, reference, *, dispersion_indices, ip_indices):
    """Whole-ring beta beating and horizontal dispersion errors at BPMs/IPs."""
    beta_x_error = tw.betx / reference.betx - 1
    beta_y_error = tw.bety / reference.bety - 1
    dx_error = tw.dx - reference.dx
    return dict(
        beta_rms=np.array([np.sqrt(np.mean(beta_x_error**2)),
                           np.sqrt(np.mean(beta_y_error**2))]),
        beta_max=np.array([np.max(np.abs(beta_x_error)), np.max(np.abs(beta_y_error))]),
        dx_rms=float(np.sqrt(np.mean(dx_error[dispersion_indices]**2))),
        dx_ip=float(np.max(np.abs(dx_error[ip_indices]))),
    )


class OpticsCorrection(CorrectionOptimizer):
    def __init__(self, line, reference, *, knobs, observation_names,
                 interaction_points, ip_names, settings=None):
        settings = settings if settings is not None else OpticsSettings()
        action = xt.Action(line.twiss)
        self.observation_names = list(observation_names)
        self.dispersion_indices = reference.rows.indices[observation_names]
        self.ip_indices = reference.rows.indices[ip_names]
        ip_names = set(ip_names)
        ip_s = reference['s', interaction_points]

        targets = []
        for name in observation_names:
            distance = np.abs(reference['s', name] - ip_s)
            near_ip = np.any(np.minimum(distance, reference.line_length - distance)
                             < settings.ir_half_length)
            weight = 1 / np.sqrt(len(observation_names))
            if near_ip:
                weight *= settings.ir_beta_weight

            # Normalize by the reference beta for a relative-beta objective.
            betx = reference['betx', name]
            target = xt.Target('betx', at=name, value=betx, action=action,
                               weight=weight / betx, tol=1e-15 * betx,
                               tag=f'betx@{name}')
            targets.append(target)

            bety = reference['bety', name]
            target = xt.Target('bety', at=name, value=bety, action=action,
                               weight=weight / bety, tol=1e-15 * bety,
                               tag=f'bety@{name}')
            targets.append(target)

            weight = settings.alpha_weight / np.sqrt(len(observation_names))
            target = xt.Target('alfx', at=name, value=reference['alfx', name],
                               action=action, weight=weight, tol=1e-15,
                               tag=f'alfx@{name}')
            targets.append(target)

            target = xt.Target('alfy', at=name, value=reference['alfy', name],
                               action=action, weight=weight, tol=1e-15,
                               tag=f'alfy@{name}')
            targets.append(target)

            weight = settings.horizontal_dispersion_weight / np.sqrt(len(observation_names))
            if name in ip_names:
                weight *= settings.horizontal_dispersion_ip_weight
            target = xt.Target('dx', at=name, value=reference['dx', name],
                               action=action, weight=weight, tol=1e-15,
                               tag=f'dx@{name}')
            targets.append(target)

        target = xt.Target('qx', value=reference.qx, action=action,
                           weight=1000., tol=1e-15, tag='qx')
        targets.append(target)
        target = xt.Target('qy', value=reference.qy, action=action,
                           weight=1000., tol=1e-15, tag='qy')
        targets.append(target)

        super().__init__(
            line, reference, settings=settings,
            name='normal optics and horizontal dispersion',
            vary=[xt.Vary(knob, step=1e-7, max_step=0.5) for knob in knobs],
            targets=targets,
        )
        response = build_normal_quad_optics_response_matrix(self, reference)
        # The tune derivatives give the integrated beta functions used to scale
        # these individual quadrupole trims, as in the reference workflow.
        scales = 1 / (4 * np.pi * np.max(np.abs(response[-2:]), axis=0))
        for vary, scale in zip(self.vary, scales):
            vary.weight = scale
        self.rcond = settings.rcond
        self.set_response(response)

        self.configure_goals(final_refinement=False, error_fraction=0.)

    def configure_goals(self, *, final_refinement, error_fraction):
        self.beta_rms_tol = self.settings.beta_rms_tol
        self.beta_max_tol = self.settings.beta_max_tol
        self.tune_tol = self.settings.tune_tol
        if final_refinement:
            self.dx_rms_tol = self.settings.horizontal_dispersion_rms_tol
            self.dx_ip_tol = self.settings.horizontal_dispersion_ip_tol
        else:
            self.dx_rms_tol = self.settings.ramp_horizontal_dispersion_rms_tol
            self.dx_ip_tol = self.settings.ramp_horizontal_dispersion_ip_tol

    def compute_metrics(self, tw):
        return compute_metrics(tw, self.reference,
                               dispersion_indices=self.dispersion_indices,
                               ip_indices=self.ip_indices)

    def goal(self, tw):
        m = self.compute_metrics(tw)
        return bool(np.all(m['beta_rms'] < self.beta_rms_tol)
                    and np.all(m['beta_max'] < self.beta_max_tol)
                    and m['dx_rms'] < self.dx_rms_tol
                    and m['dx_ip'] < self.dx_ip_tol
                    and tune_corr.tune_goal(tw, self.reference, tolerance=self.tune_tol))

    def print_metrics(self, tw, *, log=print):
        m = self.compute_metrics(tw)
        log('Optics: beta RMS/max [%]', 100 * m['beta_rms'], 100 * m['beta_max'],
            'Dx RMS/IP [mm]', 1e3 * m['dx_rms'], 1e3 * m['dx_ip'])


def build_normal_quad_optics_response_matrix(optimizer, reference):
    """Build an unweighted physical response in the optimizer's target/vary order.

    Supports ordinary betx/bety, alfx/alfy, dx and qx/qy targets on one line.
    The supplied periodic, uncoupled Twiss must describe the current line.
    Varies must be independent line variables controlling normal strengths of
    unsliced thick Quadrupoles, with fixed lengths. Total integrated strength
    changes (including knl[1]) are treated as distributed over each quadrupole.

    Discover affected elements through xdeps, probe their get_total_knl_ksl()
    at +/- Vary.step, and restore each knob even if a probe fails. No actions or
    Twiss are run. Target/Vary weights and activation flags are not applied.
    """
    line = optimizer.line
    observations = {'betx': [], 'bety': [], 'alfx': [], 'alfy': [], 'dx': []}
    for target in optimizer.targets:
        if (target.optimize_log or type(target) is not xt.Target
                or hasattr(target.value, 'auxtarget') or target.line is not None):
            raise ValueError('Only ordinary, single-line optics targets are supported')
        if isinstance(target.tar, tuple) and target.tar[0] in observations:
            observations[target.tar[0]].append(target.tar[1])
        elif not isinstance(target.tar, str) or target.tar not in ('qx', 'qy'):
            raise ValueError(f'Unsupported optics target: {target.tar!r}')

    # Include alpha locations in beta_at for the physical-alpha row conversion.
    beta_at = list(dict.fromkeys(observations['betx'] + observations['bety']
                                + observations['alfx'] + observations['alfy']))
    alpha_at = list(dict.fromkeys(observations['alfx'] + observations['alfy']))
    dispersion_at = list(dict.fromkeys(observations['dx']))
    quad_refs = {}
    for name in line.element_names:
        if isinstance(line.get(name), xt.Quadrupole):
            quad_refs.setdefault(line.ref[name], []).append(name)

    knob_quads = []
    for vary in optimizer.vary:
        if vary.container is not line.vars:
            raise ValueError("Varies must use this line's variable container")
        if vary.step is None or not np.isfinite(vary.step) or vary.step <= 0:
            raise ValueError(f'A positive finite probe step is required for {vary.name}')
        ref = line.ref[vary.name]
        if ref.xdeps.expr is not None:
            raise ValueError(f'Vary {vary.name} must be an independent variable')
        names = []
        for controlled in ref.xdeps.controlled_targets:
            names.extend(quad_refs.get(controlled, []))
        knob_quads.append(names)
    quadrupoles = list(dict.fromkeys(name for names in knob_quads for name in names))
    matrix = np.zeros((len(optimizer.targets), len(optimizer.vary)))
    if not quadrupoles:
        return matrix
    lengths = np.array([line.get(name).length for name in quadrupoles])
    if np.any(lengths <= 0):
        raise ValueError('The thick-quadrupole response requires positive lengths')

    # The shared analytical helper returns d(log beta), d(normalized alpha),
    # d(dx), and d(tune) per distributed dk1. Convert to physical target values.
    response, _ = normal_quadrupole_response(
        line, reference, quadrupoles, beta_at, alpha_at, dispersion_at)
    row_keys = ([(quantity, name) for quantity in ('betx', 'bety') for name in beta_at]
                + [(quantity, name) for quantity in ('alfx', 'alfy') for name in alpha_at]
                + [('dx', name) for name in dispersion_at] + ['qx', 'qy'])
    row_indices = {key: i for i, key in enumerate(row_keys)}
    for plane in 'xy':
        for name in alpha_at:
            response[row_indices['alf' + plane, name]] += (
                reference['alf' + plane, name] * response[row_indices['bet' + plane, name]])
        for name in beta_at:
            response[row_indices['bet' + plane, name]] *= reference['bet' + plane, name]
    response = response[[row_indices[target.tar] for target in optimizer.targets]]
    response /= lengths[None, :]  # Derivatives per integrated normal strength.
    quad_indices = {name: i for i, name in enumerate(quadrupoles)}

    for column, (vary, names) in enumerate(zip(optimizer.vary, knob_quads)):
        elements = [line.get(name) for name in names]
        value = vary.get_value()
        try:
            vary.container[vary.name] = value + vary.step
            plus = np.array([element.get_total_knl_ksl()[0][1] for element in elements])
            vary.container[vary.name] = value - vary.step
            minus = np.array([element.get_total_knl_ksl()[0][1] for element in elements])
        finally:
            vary.container[vary.name] = value
        slopes = (plus - minus) / (2 * vary.step)
        for name, slope in zip(names, slopes):
            matrix[:, column] += response[:, quad_indices[name]] * slope
    return matrix
