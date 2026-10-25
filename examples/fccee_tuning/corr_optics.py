"""Normal optics: optimizer, response, metrics, goals and reporting."""

from copy import copy
from types import SimpleNamespace
import numpy as np
import xtrack as xt

from correction_optimizer import CorrectionOptimizer
from response_builders import normal_quadrupole_response
import corr_tune


DEFAULT_OPTICS_CORR_SETTINGS = SimpleNamespace(
    beta_err_rms_tol=0.03,
    beta_err_max_tol=0.5,
    tune_tol=1e-5,
    alpha_weight=0.1,
    alpha_ip_weight=10.,
    alpha_ip_tol=1e-3,
    ir_beta_weight=5.,
    ir_half_length=500.,
    horizontal_dispersion_rms_tol=0.001,
    horizontal_dispersion_ip_tol=0.0001,
    intermediate_horizontal_dispersion_rms_tol=0.005,
    intermediate_horizontal_dispersion_ip_tol=0.0005,
    rcond=1e-5,
)


def build_opt_corr_optimizer(line, reference, *, knobs, observation_names,
                             interaction_points, ip_names, settings=None, **twiss_kwargs):
    """Build a correction optimizer with local targets, response and diagnostics."""
    if settings is None:
        settings = copy(DEFAULT_OPTICS_CORR_SETTINGS)
    ip_indices = reference.rows.indices[ip_names]
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
        target = xt.Target('betx', at=name, value=betx,
                           weight=weight / betx, tol=1e-15 * betx,
                           tag=f'betx@{name}')
        targets.append(target)

        bety = reference['bety', name]
        target = xt.Target('bety', at=name, value=bety,
                           weight=weight / bety, tol=1e-15 * bety,
                           tag=f'bety@{name}')
        targets.append(target)

        weight = settings.alpha_weight / np.sqrt(len(observation_names))
        if name in ip_names:
            # Keep the four waist constraints significant independently of the
            # number of BPMs; the ordinary alpha targets still constrain shape.
            weight = settings.alpha_ip_weight / np.sqrt(len(ip_names))
        target = xt.Target('alfx', at=name, value=reference['alfx', name],
                           weight=weight, tol=1e-15,
                           tag=f'alfx@{name}')
        targets.append(target)

        target = xt.Target('alfy', at=name, value=reference['alfy', name],
                           weight=weight, tol=1e-15,
                           tag=f'alfy@{name}')
        targets.append(target)

        weight = settings.beta_err_rms_tol / settings.horizontal_dispersion_rms_tol
        weight /= np.sqrt(len(observation_names))
        if name in ip_names:
            weight *= (settings.ir_beta_weight * settings.horizontal_dispersion_rms_tol
                       / settings.horizontal_dispersion_ip_tol)
        target = xt.Target('dx', at=name, value=reference['dx', name],
                           weight=weight, tol=1e-15,
                           tag=f'dx@{name}')
        targets.append(target)

    target = xt.Target('qx', value=reference.qx,
                       weight=1000., tol=1e-15, tag='qx')
    targets.append(target)
    target = xt.Target('qy', value=reference.qy,
                       weight=1000., tol=1e-15, tag='qy')
    targets.append(target)

    optimizer = CorrectionOptimizer(
        line=line, reference_twiss=reference, settings=settings,
        method='4d', delta0=0., chrom=False, radiation_integrals=False,
        allow_twiss_failure=False,
        co_guess=dict(x=0, px=0, y=0, py=0, zeta=0, delta=0),
        **twiss_kwargs,
        compute_metrics=compute_metrics, goal=goal,
        print_metrics=print_metrics,
        name='normal optics and horizontal dispersion',
        vary=[xt.Vary(knob, step=1e-7, max_step=0.5, weight=1.) for knob in knobs],
        targets=targets,
    )
    optimizer.observation_names = list(observation_names)
    optimizer.dispersion_indices = reference.rows.indices[observation_names]
    optimizer.ip_indices = ip_indices
    response = build_normal_quad_optics_response_matrix(optimizer, reference)
    optimizer.rcond = settings.rcond
    optimizer.set_response(response)

    optimizer.beta_err_rms_tol = settings.beta_err_rms_tol
    optimizer.beta_err_max_tol = settings.beta_err_max_tol
    optimizer.tune_tol = settings.tune_tol
    optimizer.alpha_ip_tol = settings.alpha_ip_tol
    optimizer.dx_err_rms_tol = settings.intermediate_horizontal_dispersion_rms_tol
    optimizer.dx_err_ip_tol = settings.intermediate_horizontal_dispersion_ip_tol

    return optimizer


def install_optics_knobs(line):
    """Install new optics trims initialized to zero, preserving k1 family expressions."""
    table = line.get_table()
    quadrupoles = list(dict.fromkeys(table.rows.match(element_type='Quadrupole').env_name))
    knobs = [f'dk1_corr_optimics_{name}' for name in quadrupoles]
    switch = 'on_corr_optics'
    assert all(knob not in line.vars for knob in knobs), 'Optics correction knobs already exist'
    line[switch] = 1.
    for name, knob in zip(quadrupoles, knobs):
        line[knob] = 0.
        line.ref[name].k1 += line.ref[switch] * line.ref[knob]

    line.metadata.setdefault('correction_knobs', {})['optics'] = dict(
        knobs=knobs, quadrupoles=quadrupoles, units='m^-2')
    return knobs


def compute_metrics(optimizer, tw):
    """Beta beating, IP waist errors and horizontal dispersion errors."""
    reference = optimizer.reference
    dispersion_indices = optimizer.dispersion_indices
    ip_indices = optimizer.ip_indices
    beta_x_error = tw.betx / reference.betx - 1
    beta_y_error = tw.bety / reference.bety - 1
    dx_error = tw.dx - reference.dx
    return dict(
        beta_err_rms=np.array([np.sqrt(np.mean(beta_x_error**2)),
                           np.sqrt(np.mean(beta_y_error**2))]),
        beta_err_max=np.array([np.max(np.abs(beta_x_error)), np.max(np.abs(beta_y_error))]),
        alpha_err_ip=np.array([
            np.max(np.abs((tw.alfx - reference.alfx)[ip_indices])),
            np.max(np.abs((tw.alfy - reference.alfy)[ip_indices]))]),
        dx_err_rms=float(np.sqrt(np.mean(dx_error[dispersion_indices]**2))),
        dx_err_ip=float(np.max(np.abs(dx_error[ip_indices]))),
    )


def goal(optimizer, tw):
    m = optimizer.compute_metrics(tw)
    return bool(np.all(m['beta_err_rms'] < optimizer.beta_err_rms_tol)
                and np.all(m['beta_err_max'] < optimizer.beta_err_max_tol)
                and np.all(m['alpha_err_ip'] < optimizer.alpha_ip_tol)
                and m['dx_err_rms'] < optimizer.dx_err_rms_tol
                and m['dx_err_ip'] < optimizer.dx_err_ip_tol
                and corr_tune.tune_goal(tw, optimizer.reference, tolerance=optimizer.tune_tol))


def print_metrics(optimizer, tw, *, log=print):
    m = optimizer.compute_metrics(tw)
    log('Optics: beta error RMS/max [%]', 100 * m['beta_err_rms'], 100 * m['beta_err_max'],
        'Dx error RMS/IP [mm]', 1e3 * m['dx_err_rms'], 1e3 * m['dx_err_ip'],
        'IP alpha max error [x,y]', m['alpha_err_ip'])


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

    beta_at = list(dict.fromkeys(observations['betx'] + observations['bety']))
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

    # Get physical target derivatives per distributed dk1 and order the rows.
    response, _ = normal_quadrupole_response(
        line, reference, quadrupoles, beta_at, alpha_at, dispersion_at)
    row_keys = ([(quantity, name) for quantity in ('betx', 'bety') for name in beta_at]
                + [(quantity, name) for quantity in ('alfx', 'alfy') for name in alpha_at]
                + [('dx', name) for name in dispersion_at] + ['qx', 'qy'])
    row_indices = {key: i for i, key in enumerate(row_keys)}
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
