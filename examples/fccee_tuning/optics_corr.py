"""Build the normal-optics/horizontal-dispersion optimizer and its response."""

import numpy as np
import xtrack as xt

from response_builders import normal_quadrupole_response


def build_optimizer(line, reference, action, *, knobs, beta_at, alpha_at, dispersion_at,
                    interaction_points, ip_names, settings):
    """Return an optimizer with weighted targets and a fixed nominal response.

    Construction uses the supplied reference table without running Twiss.
    The caller owns subsequent evaluations and trial acceptance.
    """
    targets = (
        [xt.Target('bet' + p, at=name, value=reference['bet' + p, name], action=action)
         for p in 'xy' for name in beta_at]
        + [xt.Target('alf' + p, at=name, value=reference['alf' + p, name], action=action)
           for p in 'xy' for name in alpha_at]
        + [xt.Target('dx', at=name, value=reference['dx', name], action=action)
           for name in dispersion_at]
        + [xt.Target(q, value=reference[q], action=action) for q in ('qx', 'qy')]
    )
    beta_indices = reference.rows.indices[beta_at]
    ip_s = reference.s[reference.rows.indices[interaction_points]]
    distance = np.abs(reference.s[beta_indices, None] - ip_s[None, :])
    near_ip = np.any(np.minimum(distance, reference.line_length - distance)
                     < settings.ir_half_length, axis=1)
    beta_weights = np.where(near_ip, settings.ir_beta_weight, 1.) / np.sqrt(len(beta_at))
    dx_weights = settings.horizontal_dispersion_weight * np.array([
        settings.horizontal_dispersion_ip_weight if name in ip_names else 1.
        for name in dispersion_at]) / np.sqrt(len(dispersion_at))
    weights = np.r_[
        beta_weights, beta_weights,
        np.full(2 * len(alpha_at), settings.alpha_weight / np.sqrt(len(alpha_at))),
        dx_weights, 1000., 1000.,
    ]
    for target, weight in zip(targets, weights):
        target.tol = 1e-15
        target.weight = weight
        target.tag = '@'.join(target.tar) if isinstance(target.tar, tuple) else target.tar
    # Keep the relative-beta objective while exposing physical target values.
    for target in targets[:2 * len(beta_at)]:
        target.weight /= target.value
        target.tol *= target.value

    optimizer = line.match(
        solve=False,
        name='normal optics and horizontal dispersion',
        vary=[xt.Vary(knob, step=1e-7, max_step=0.5) for knob in knobs],
        targets=targets,
        action_results={action: reference},
        assert_within_tol=False,
        show_call_counter=False,
    )
    response = build_normal_quad_optics_response_matrix(optimizer, reference)
    # The tune derivatives give the integrated beta functions used to scale
    # these individual quadrupole trims, as in the reference workflow.
    scales = 1 / (4 * np.pi * np.max(np.abs(response[-2:]), axis=0))
    for vary, scale in zip(optimizer.vary, scales):
        vary.weight = scale
    optimizer.set_response(response)
    return optimizer


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
