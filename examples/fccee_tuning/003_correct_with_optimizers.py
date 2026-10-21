"""Flat correction prototype: explicit optimizers and explicit orchestration.

Input is the lattice with errors. Run with the response-API branches of Xtrack
and Xdeps. The existing 001 script remains the reference correction recipe.

python 003_correct_with_optimizers.py

Outputs go to response_api/. There are no workflow functions:
edit the targets, varies, response builders, ordering and acceptance below.
"""

from pathlib import Path
import json

import numpy as np
import xtrack as xt

from error_recipe import get_bpm_alignment
from response_builders import (
    orbit_response,
    normal_quadrupole_response,
    skew_quadrupole_response,
)

HERE = Path(__file__).resolve().parent
INPUT = HERE / 'fccee_z_lcc_errors.json'
OUTPUT_DIR = HERE / 'response_api'
STOP_AT = 1.0  # Set e.g. 0.0125 to inspect just the first ramp stage.
BOOTSTRAP_FRACTION = 0.15
MAX_CYCLES = 40
ORBIT_RCOND = 1e-6
OPTICS_RCOND = 1e-5
COUPLING_RCOND = 1e-6
TUNE_TOL = 1e-5
TUNE_KNOBS = ['kqf2', 'kqd1']

# Prepare the model and independent correction circuits.
env = xt.load(INPUT)
line = env.fccee_p_ring
if 'tuning_errors' not in env.metadata:
    raise ValueError('Run 000_install_errors.py first')
if 'orbit_optics_correction' in env.metadata:
    raise ValueError('Input already corrected: use the output of 000')
# Complete the presentation's corrector-in-every-quadrupole layout. Existing
# correctors are preserved; missing zero-length trims go at the quad entrance.
added_correctors = []
placements = []
layout = line.get_table()
for name in layout.env_name[layout.element_type == 'Quadrupole']:
    for prefix in ('hcor_', 'vcor_'):
        corrector = prefix + name
        if corrector not in line.element_dict:
            placements.append(
                env.new(corrector, xt.Multipole, knl=[0], ksl=[0], at=name + '@start')
            )
            added_correctors.append(corrector)
if placements:
    line.insert(placements)
print(
    'Added', len(added_correctors), 'H/V correctors to complete the quadrupole layout'
)
line.configure_radiation(model=None)
line.configure_drift_model(model='exact')
line.twiss_default.update(method='4d', delta0=0.0)
# Same tracking models as the preparation step used by the original 005.
for name in dict.fromkeys(line.element_names):
    element = line.get(name)
    if isinstance(element, (xt.Bend, xt.RBend)):
        element.model = 'rot-kick-rot'
        element.edge_entry_model = element.edge_exit_model = 'full'
    elif isinstance(
        element, (xt.Quadrupole, xt.Sextupole, xt.Octupole, xt.Multipole, xt.Cavity)
    ):
        element.model = 'drift-kick-drift-exact'
    else:
        continue
    element.integrator = 'yoshida-6'
    if hasattr(element, 'num_multipole_kicks'):
        element.num_multipole_kicks = max(element.num_multipole_kicks, 21)

monitor_names = list(get_bpm_alignment(env))
corrector_names = [
    [n for n in line.element_names if n.startswith(prefix)]
    for prefix in ('hcor_', 'vcor_')
]
line['on_orbit_corr'] = 1.0
orbit_knobs = []
for plane, names, field, sign in zip('xy', corrector_names, ('knl', 'ksl'), (-1, 1)):
    for name in names:
        knob = f'orbit_corr_{name}_{plane}'
        if knob in line.vars:
            raise ValueError(f'Correction knob already exists: {knob}')
        line[knob] = 0.0
        getattr(line.ref[name], field)[0] += (
            sign * line.ref['on_orbit_corr'] * line.ref[knob]
        )
        orbit_knobs.append(knob)
line.steering_monitors_x = line.steering_monitors_y = monitor_names
line.steering_correctors_x, line.steering_correctors_y = corrector_names
print('BPMs:', len(monitor_names), 'H/V correctors:', [len(n) for n in corrector_names])

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

nonlinear_magnets = controlled

# Independent skew-quadrupole circuits in sextupoles (one per physical magnet).
# ksl[1] is absolute integrated strength [1/m], and is deliberately not ramped
# with k2. Main sextupoles can be off while the skew trims remain powered.
skew_groups = [
    record['names']
    for record in env.metadata['tuning_errors']['magnets'].values()
    if isinstance(line.get(record['names'][0]), xt.Sextupole)
]
skew_knobs = [f'dk1sl_coupling_{names[0]}' for names in skew_groups]
line['on_coupling_corr'] = 1.0
for names, knob in zip(skew_groups, skew_knobs):
    line[knob] = 0.0
    length = sum(line.get(name).length for name in names)
    for name in names:
        line.ref[name].ksl[1] += (
            line.ref['on_coupling_corr']
            * line.ref[knob]
            * line.get(name).length
            / length
        )
print('Independent skew-quadrupole trims:', len(skew_knobs))


# Reference optics and observation locations are explicit inputs to the fits.
original = {name: float(line[name]) for name in TUNE_KNOBS}
line['on_errors'] = 0.0
line['on_nonlinear'] = 0.0
twiss_kwargs = dict(
    method='4d',
    delta0=0.0,
    chrom=False,
    steps_R_matrix=dict(dx=1e-8, dpx=1e-9, dy=1e-8, dpy=1e-9, dzeta=1e-7, ddelta=1e-8),
)
line.twiss_default['co_guess'] = dict(x=0.0, px=0.0, y=0.0, py=0.0, zeta=0.0, delta=0.0)
tw_ref = line.twiss(**twiss_kwargs)
line['on_nonlinear'] = 1.0
tw_on = line.twiss(**twiss_kwargs)
line['on_nonlinear'] = 0.0
if max(np.max(np.abs(tw_on['bet' + p] / tw_ref['bet' + p] - 1)) for p in 'xy') > 1e-4:
    raise RuntimeError(
        'Nominal optics change with nonlinear strength; rebuild responses per stage'
    )
tt = line.get_table()
quad_names = list(tt.rows[tt.element_type == 'Quadrupole'].name)
quad_indices = tw_ref.rows.indices[quad_names]
lengths = np.array([line[name].length for name in quad_names])
quad_knobs = [f'dk1_optics_{name}' for name in quad_names]
line['on_optics_corr'] = 1.0
for name, knob in zip(quad_names, quad_knobs):
    line[knob] = 0.0
    line.ref[name].k1 += line.ref['on_optics_corr'] * line.ref[knob]
# Add observations at 1/16-turn phase intervals in both planes. Sampling only
# quadrupole entrances can hide beta beating in the intervening drifts and IPs.
selected = set(quad_indices) | set(tw_ref.rows.indices[list(tt.rows['ip.*'].name)])
for plane in 'xy':
    mu = tw_ref['mu' + plane]
    selected.update(np.searchsorted(mu, np.arange(mu[0], mu[-1], 1 / 16)).tolist())
obs_indices = np.array(sorted(selected))
obs_names = list(tw_ref.name[obs_indices])
alpha_indices = tw_ref.rows.indices[
    list(dict.fromkeys(quad_names + list(tt.rows['ip.*'].name)))
]
monitor_indices = tw_ref.rows.indices[monitor_names]
ip_names = ['ipa', 'ipd', 'ipg', 'ipj']
ip_indices = tw_ref.rows.indices[ip_names]
dispersion_names = list(dict.fromkeys(monitor_names + ip_names))
dispersion_indices = tw_ref.rows.indices[dispersion_names]

# One model evaluation supplies all observations. Targets define what to fit.
action = xt.Action(lambda: line.twiss(**twiss_kwargs))
bpm_variables = env.metadata['tuning_errors']['bpm_alignment_variables']

# Orbit: independent x/y solves, both using physical kick variables.
orbit_optimizer_x = line.match(
    solve=False,
    name='orbit x',
    vary=[xt.Vary(f'orbit_corr_{n}_x', step=1e-7) for n in corrector_names[0]],
    targets=[
        action.target(
            lambda tw, n=n: tw['x', n] - env[bpm_variables[n]['shift_x']],
            value=0.0,
            tol=1e-15,
            tag=f'x@{n}',
        )
        for n in monitor_names
    ],
)
orbit_optimizer_x.set_response(
    orbit_response(tw_ref, monitor_names, corrector_names[0], 'x')
)

orbit_optimizer_y = line.match(
    solve=False,
    name='orbit y',
    vary=[xt.Vary(f'orbit_corr_{n}_y', step=1e-7) for n in corrector_names[1]],
    targets=[
        action.target(
            lambda tw, n=n: tw['y', n] - env[bpm_variables[n]['shift_y']],
            value=0.0,
            tol=1e-15,
            tag=f'y@{n}',
        )
        for n in monitor_names
    ],
)
orbit_optimizer_y.set_response(
    orbit_response(tw_ref, monitor_names, corrector_names[1], 'y')
)

# Normal optics: log beta, normalized alpha, Dx and tunes are fitted jointly.
normal_response, normal_scales = normal_quadrupole_response(
    line,
    tw_ref,
    quad_names,
    obs_names,
    list(tw_ref.name[alpha_indices]),
    dispersion_names,
)
# Target.optimize_log uses log10, while the analytical builder uses log_e.
normal_response[: 2 * len(obs_names)] /= np.log(10.0)
ip_s = tw_ref.s[tw_ref.rows.indices[list(tt.rows['ip.*'].name)]]
distance = np.abs(tw_ref.s[obs_indices, None] - ip_s[None, :])
near_ip = np.min(np.minimum(distance, tw_ref.line_length - distance), axis=1) < 500.0
beta_weights = np.where(near_ip, 5.0, 1.0) / np.sqrt(len(obs_names))

optics_targets = [
    xt.Target(
        'bet' + p,
        at=n,
        value=tw_ref,
        action=action,
        optimize_log=True,
        weight=w * np.log(10.0),
        tol=1e-15,
    )
    for p in 'xy'
    for n, w in zip(obs_names, beta_weights)
]
optics_targets += [
    action.target(
        lambda tw, p=p, n=n: (
            tw['alf' + p, n] * tw_ref['bet' + p, n] / tw['bet' + p, n]
            - tw_ref['alf' + p, n]
        ),
        value=0.0,
        weight=0.1 / np.sqrt(len(alpha_indices)),
        tol=1e-15,
        tag=f'normalized_alpha_{p}@{n}',
    )
    for p in 'xy'
    for n in tw_ref.name[alpha_indices]
]
optics_targets += [
    xt.Target(
        'dx',
        at=n,
        value=tw_ref,
        action=action,
        tol=1e-15,
        weight=30.0 * (50.0 if n in ip_names else 1.0) / np.sqrt(len(dispersion_names)),
    )
    for n in dispersion_names
]
optics_targets += [
    xt.Target(q, value=tw_ref, action=action, weight=1000.0, tol=1e-15)
    for q in ('qx', 'qy')
]
optics_optimizer = line.match(
    solve=False,
    name='normal optics and Dx',
    vary=[xt.Vary(n, step=1e-7, weight=s) for n, s in zip(quad_knobs, normal_scales)],
    targets=optics_targets,
)
optics_optimizer.set_response(normal_response)
del normal_response

# Coupling: real/imaginary RDT parts and Dy share the same skew trims.
skew_response, skew_scales = skew_quadrupole_response(
    line, tw_ref, skew_groups, dispersion_names
)
coupling_targets = [
    action.target(
        lambda tw, rdt=rdt, part=part, n=n: getattr(tw[rdt, n] - tw_ref[rdt, n], part),
        value=0.0,
        tol=1e-15,
        tag=f'{part}_{rdt}@{n}',
        weight=(20.0 if n in ip_names else 1.0) / np.sqrt(len(dispersion_names)),
    )
    for rdt in ('f1001', 'f1010')
    for part in ('real', 'imag')
    for n in dispersion_names
]
coupling_targets += [
    xt.Target(
        'dy',
        at=n,
        value=tw_ref,
        action=action,
        tol=1e-15,
        weight=(1000.0 if n in ip_names else 1.0) / np.sqrt(len(dispersion_names)),
    )
    for n in dispersion_names
]
coupling_optimizer = line.match(
    solve=False,
    name='coupling and Dy',
    vary=[xt.Vary(n, step=1e-7, weight=s) for n, s in zip(skew_knobs, skew_scales)],
    targets=coupling_targets,
)
coupling_optimizer.set_response(skew_response)
del skew_response

# Build the tune response once on the unperturbed machine (errors and nonlinear
# magnets are still off), and keep it fixed throughout correction.
tune_optimizer = line.match(
    solve=False,
    name='tunes',
    vary=[xt.Vary(n, step=1e-7) for n in TUNE_KNOBS],
    targets=[
        xt.Target(q, value=tw_ref, action=action, weight=1.0, tol=TUNE_TOL / 2)
        for q in ('qx', 'qy')
    ],
)
tune_optimizer.set_response(tune_optimizer.build_response())

optimizers = [
    orbit_optimizer_x,
    orbit_optimizer_y,
    tune_optimizer,
    coupling_optimizer,
    optics_optimizer,
]
for optimizer in optimizers:
    optimizer.verbose = False
    print(
        optimizer.name,
        'response',
        optimizer.response.matrix.shape,
        'SVD range',
        optimizer.response.svd.s[[0, -1]],
        flush=True,
    )

# Check each analytical joint response with a central distributed probe.
for optimizer, scales, amplitude in [
    (optics_optimizer, normal_scales, 1e-4),
    (coupling_optimizer, skew_scales, 1e-5),
]:
    before = optimizer.get_knob_values()
    probe = amplitude * scales * np.sin(np.arange(len(scales)) + 0.3)
    merit = optimizer.get_merit_function(return_scalar=False)
    samples = []
    try:
        for sign in (1.0, -1.0):
            env.vars.update(
                {n: v + sign * d for (n, v), d in zip(before.items(), probe)}
            )
            samples.append(merit(merit.get_x()))
    finally:
        env.vars.update(before)
    weights = np.array([t.weight for t in optimizer.targets])
    predicted = weights * (optimizer.response.matrix @ probe)
    measured = (samples[0] - samples[1]) / 2
    discrepancy = np.linalg.norm(measured - predicted) / np.linalg.norm(predicted)
    print(optimizer.name, 'directional response error', discrepancy, flush=True)
    if discrepancy > 0.05:
        raise RuntimeError('Analytical response disagrees with tracking')

# Acceptance is separate from fit weights: full BPM RMS, full-ring beta peaks,
# and tighter collision-point dispersion/RDT limits decide whether to advance.
ramp_limits = dict(
    dx_rms=5e-3,
    dx_ip=500e-6,
    dy_rms=5e-3,
    dy_ip=100e-6,
    rdt_rms=3e-2,
    rdt_ip=3e-2,
    c_minus=3e-3,
)
final_limits = dict(
    dx_rms=1e-3,
    dx_ip=100e-6,
    dy_rms=1e-3,
    dy_ip=20e-6,
    rdt_rms=3e-3,
    rdt_ip=2e-3,
    c_minus=3e-4,
)
progress = 0.0
increment = 0.0125
final_refinement = False
history = []
tw = tw_ref
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

while progress < STOP_AT or (STOP_AT == 1.0 and not final_refinement):
    stage_before = {
        n: v for opt in optimizers for n, v in opt.get_knob_values().items()
    }
    stage_before.update(
        on_errors=float(line['on_errors']), on_nonlinear=float(line['on_nonlinear'])
    )
    tw_before = tw
    final_refinement = progress == 1.0
    limits = final_limits if final_refinement else ramp_limits
    boundary = BOOTSTRAP_FRACTION if progress < BOOTSTRAP_FRACTION - 1e-12 else STOP_AT
    trial_progress = (
        progress if final_refinement else min(progress + increment, boundary, STOP_AT)
    )
    print('TRY', trial_progress, 'final refinement', final_refinement, flush=True)

    try:
        if progress > 0 and trial_progress != progress:
            ratio = trial_progress / progress
            env.vars.update(
                {
                    n: original.get(n, 0.0) + ratio * (v - original.get(n, 0.0))
                    for opt in optimizers
                    for n, v in opt.get_knob_values().items()
                }
            )
        line['on_errors'] = trial_progress
        line['on_nonlinear'] = max(
            0.0, (trial_progress - BOOTSTRAP_FRACTION) / (1 - BOOTSTRAP_FRACTION)
        )
        tw = line.twiss(**twiss_kwargs)

        for cycle in range(MAX_CYCLES):
            alignment = get_bpm_alignment(env)
            offsets = np.array(
                [
                    [alignment[n][a] for n in monitor_names]
                    for a in ('shift_x', 'shift_y', 'rot_s_rad')
                ]
            )
            x, y = (
                tw.x[monitor_indices] - offsets[0],
                tw.y[monitor_indices] - offsets[1],
            )
            measured_orbit = np.array(
                [
                    x * np.cos(offsets[2]) + y * np.sin(offsets[2]),
                    -x * np.sin(offsets[2]) + y * np.cos(offsets[2]),
                ]
            )
            beta = np.array([tw['bet' + p] / tw_ref['bet' + p] - 1 for p in 'xy'])
            dispersion = np.array([tw['d' + p] - tw_ref['d' + p] for p in 'xy'])
            rdts = np.array([tw[r] - tw_ref[r] for r in ('f1001', 'f1010')])
            orbit_rms = np.sqrt(np.mean(measured_orbit**2, axis=1))
            beta_rms = np.sqrt(np.mean(beta**2, axis=1))
            beta_max = np.max(np.abs(beta), axis=1)
            disp_rms = np.sqrt(np.mean(dispersion[:, dispersion_indices] ** 2, axis=1))
            disp_ip = np.max(np.abs(dispersion[:, ip_indices]), axis=1)
            rdt_rms = np.sqrt(np.mean(np.abs(rdts[:, dispersion_indices]) ** 2, axis=1))
            rdt_ip = np.max(np.abs(rdts[:, ip_indices]))
            tune_error = max(abs(tw.qx - tw_ref.qx), abs(tw.qy - tw_ref.qy))
            orbit_ok = np.all(orbit_rms <= max(2e-9, trial_progress * 10e-6))
            optics_ok = (
                np.all(beta_rms < 0.03)
                and np.max(beta_max)
                < (0.1 if trial_progress <= BOOTSTRAP_FRACTION else 0.5)
                and disp_rms[0] < limits['dx_rms']
                and disp_ip[0] < limits['dx_ip']
                and tune_error < TUNE_TOL
            )
            coupling_ok = (
                np.all(rdt_rms < limits['rdt_rms'])
                and rdt_ip < limits['rdt_ip']
                and tw.c_minus < limits['c_minus']
                and disp_rms[1] < limits['dy_rms']
                and disp_ip[1] < limits['dy_ip']
            )
            print(
                ' cycle',
                cycle,
                'orbit/optics/coupling',
                orbit_ok,
                optics_ok,
                coupling_ok,
                'orbit [um]',
                orbit_rms * 1e6,
                'beta RMS',
                beta_rms,
                flush=True,
            )
            if orbit_ok and optics_ok and coupling_ok:
                break

            # This is the workflow: steer, fix tunes, then coupling and optics.
            sequence = []
            if not orbit_ok or not optics_ok:
                sequence += [
                    (orbit_optimizer_x, ORBIT_RCOND),
                    (orbit_optimizer_y, ORBIT_RCOND),
                    (tune_optimizer, 0.0),
                ]
            if not coupling_ok:
                sequence.append((coupling_optimizer, COUPLING_RCOND))
            if not optics_ok:
                sequence.append(
                    (
                        optics_optimizer,
                        1e-7 if trial_progress <= BOOTSTRAP_FRACTION else OPTICS_RCOND,
                    )
                )

            made_progress = False
            for optimizer, rcond in sequence:
                before = {
                    n: v for opt in optimizers for n, v in opt.get_knob_values().items()
                }
                direction = optimizer.get_correction(rcond=rcond)
                largest = max(
                    abs(d / v.weight) for d, v in zip(direction, optimizer.vary)
                )
                initial_step = (
                    0.5
                    if optimizer is coupling_optimizer
                    else min(1.0, 0.5 / max(largest, 1e-30))
                    if optimizer is optics_optimizer
                    else 1.0
                )
                accepted = False
                for backtrack in range(10):
                    env.vars.update(before)
                    for vary in optimizer.vary:
                        vary.max_step = max(
                            largest * initial_step * 0.5**backtrack, 1e-30
                        )
                    info = optimizer.run_jacobian(
                        n_steps=7 if optimizer is tune_optimizer else 1,
                        rcond=rcond,
                        take_best=False,
                        verbose=False,
                    )
                    if not info.improved:
                        env.vars.update(before)
                        break

                    # Evaluate optics/skew trials after orbit/tune settling.
                    # Every nested correction is included in the outer snapshot.
                    if optimizer is optics_optimizer or optimizer is coupling_optimizer:
                        for settle_cycle in range(3):
                            for settle_optimizer, settle_rcond in [
                                (orbit_optimizer_x, ORBIT_RCOND),
                                (orbit_optimizer_y, ORBIT_RCOND),
                                (tune_optimizer, 0.0),
                            ]:
                                settle_before = settle_optimizer.get_knob_values()
                                for vary in settle_optimizer.vary:
                                    vary.max_step = None
                                settled = settle_optimizer.run_jacobian(
                                    n_steps=7
                                    if settle_optimizer is tune_optimizer
                                    else 1,
                                    rcond=settle_rcond,
                                    take_best=False,
                                    verbose=False,
                                )
                                if not settled.improved:
                                    env.vars.update(settle_before)
                        merit = optimizer.get_merit_function(return_scalar=False)
                        after = np.linalg.norm(merit(merit.get_x()))
                    else:
                        after = info.penalty_after
                    if after < info.penalty_before * (1 - 1e-6):
                        accepted = True
                        made_progress = True
                        print(
                            '  accepted',
                            optimizer.name,
                            info.penalty_before,
                            '->',
                            after,
                            flush=True,
                        )
                        break
                if not accepted:
                    env.vars.update(before)

            tw = line.twiss(**twiss_kwargs)
            if not made_progress:
                raise RuntimeError('No improving correction in this cycle')
        else:
            raise RuntimeError('Correction cycle limit')

    except (ValueError, RuntimeError, xt.twiss.ClosedOrbitSearchError) as error:
        env.vars.update(stage_before)
        tw = tw_before
        increment *= 0.5
        print(
            'REJECT',
            str(error),
            'restored',
            progress,
            'next increment',
            increment,
            flush=True,
        )
        if final_refinement or increment < 1e-5:
            raise
        continue

    progress = trial_progress
    line.twiss_default['co_guess'] = {
        c: float(tw[c][0]) for c in ('x', 'px', 'y', 'py', 'zeta', 'delta')
    }
    metrics = dict(
        progress=progress,
        final_refinement=final_refinement,
        orbit_rms_m=orbit_rms.tolist(),
        beta_rms=beta_rms.tolist(),
        beta_max=beta_max.tolist(),
        dispersion_rms_m=disp_rms.tolist(),
        dispersion_ip_max_m=disp_ip.tolist(),
        rdt_rms=rdt_rms.tolist(),
        rdt_ip_max=float(rdt_ip),
        c_minus=float(tw.c_minus),
        tune_error=float(tune_error),
    )
    history.append(metrics)
    checkpoint = dict(
        metrics=metrics,
        knobs={n: v for opt in optimizers for n, v in opt.get_knob_values().items()},
    )
    (OUTPUT_DIR / 'checkpoint.json').write_text(json.dumps(checkpoint, indent=2))
    print('ACCEPT', metrics, flush=True)
    if progress == BOOTSTRAP_FRACTION:
        increment = (1 - BOOTSTRAP_FRACTION) * 0.025
    elif cycle <= (3 if progress > BOOTSTRAP_FRACTION else 5):
        increment = min(
            0.085 if progress > BOOTSTRAP_FRACTION else 0.1, increment * 1.25
        )

env.metadata['response_api_correction'] = dict(
    completed=bool(progress == 1.0 and final_refinement),
    history=history,
    nominal_tune_knobs=original,
)
env.to_json(OUTPUT_DIR / 'machine.json')
(OUTPUT_DIR / 'history.json').write_text(json.dumps(history, indent=2))
print('Saved', OUTPUT_DIR / 'machine.json', flush=True)
