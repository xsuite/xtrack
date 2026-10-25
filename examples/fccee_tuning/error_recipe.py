"""FCC-ee LCC static error recipe, with Xdeps switches as in the LHC examples.

Source (slides 4, 6, 7):
https://indico.cern.ch/event/1704042/contributions/7166876/attachments/3307462/5918024/LCC_tuning02072026.pdf

Field harmonics are in 1e-4 units at 10 mm. The factorial/radius conversion is
the one in xmask/xmask/lhc/load_wise.py. Coefficients here use the lattice frame;
LHC measurement-frame sign factors are not imported. Alternate arc orientations
reverse systematic a2/b3/a4 in dipoles and the quadrupole magnetic-centre dx.

This is an example mapping, not the original pyAT simulation: neighbouring
quad/sextupole groups stand in for girders, with common translations and roll.
BPM centres follow the associated quadrupole magnetic centre (post-BBA).
Contiguous pieces with the same family name share one error draw.
Zero-length embedded multipoles do not split the surrounding magnet.
No tapering, orbit correction, or measurement noise is performed here.
Current-dependent b1/a4 from skew trims and a3 from vertical trims are excluded.
"""
from collections import Counter
from math import factorial

import numpy as np


SOURCE = ('https://indico.cern.ch/event/1704042/contributions/7166876/'
          'attachments/3307462/5918024/LCC_tuning02072026.pdf')
R_REF = 0.010
TRUNCATION = 2.5
AXES = {'dx': 'shift_x', 'dy': 'shift_y', 'ds': 'shift_s', 'roll': 'rot_s_rad'}

# x/y/s in metres, roll in radians, relative main-strength error.
SIGMAS = {
    'arc_quad': {'dx': 50e-6, 'dy': 50e-6, 'ds': 100e-6, 'roll': 50e-6, 'relative_strength': 10e-4},
    'arc_sext': {'dx': 50e-6, 'dy': 50e-6, 'ds': 100e-6, 'roll': 50e-6, 'relative_strength': 2e-4},
    'arc_dipole': {'dx': 1e-3, 'dy': 1e-3, 'ds': 500e-6, 'roll': 1e-3, 'relative_strength': 10e-4},
    'arc_other': {'dx': 100e-6, 'dy': 100e-6, 'ds': 100e-6, 'roll': 100e-6, 'relative_strength': 2e-4},
    'straight_dipole': {'dx': 1e-3, 'dy': 1e-3, 'ds': 100e-6, 'roll': 1e-3, 'relative_strength': 10e-4},
    'ip_to_fd': {'dx': 30e-6, 'dy': 30e-6, 'ds': 100e-6, 'roll': 30e-6, 'relative_strength': 0.5e-4},
    'fd_to_cs': {'dx': 50e-6, 'dy': 50e-6, 'ds': 100e-6, 'roll': 50e-6, 'relative_strength': 1e-4},
    'straight_other': {'dx': 100e-6, 'dy': 100e-6, 'ds': 100e-6, 'roll': 100e-6, 'relative_strength': 2e-4},
}
GIRDER_SIGMAS = {'dx': 150e-6, 'dy': 150e-6, 'ds': 500e-6, 'roll': 150e-6}
BPM_SIGMAS = {'dx': 10e-6, 'dy': 10e-6, 'roll': 1000e-6}
# Main-order entries implement dk/k ONCE.
ARC_DIPOLE = {
    'b1': {'systematic': 0, 'random_sigma': 10}, 'b2': {'systematic': 0, 'random_sigma': 5},
    'a2': {'systematic': 1, 'random_sigma': 2}, 'b3': {'systematic': 0.5, 'random_sigma': 0.5},
    'a3': {'systematic': 0.5, 'random_sigma': 0}, 'b4': {'systematic': 0.5, 'random_sigma': 0.5},
    'a4': {'systematic': 0.1, 'random_sigma': 0.1},
}
ARC_QUAD = {'b2': {'systematic': 0, 'random_sigma': 10}, 'a2': {'systematic': 0, 'random_sigma': 12}}


def truncated_normal(rng):
    """Reject tails, rather than clipping or renormalizing the truncated RMS."""
    while True:
        value = float(rng.normal())
        if abs(value) <= TRUNCATION:
            return value


def relative_coefficient(units, n, main_order):
    """n and main_order are zero based; result multiplies main_strength."""
    return (1e-4 * units * R_REF**(main_order - n)
            * factorial(n) / factorial(main_order))


def _inventory(line):
    """Group touching magnet pieces across thin inserts, keeping inserts separate."""
    tt = line.get_table()
    groups = []
    previous_thick = None
    for name, kind, s, end in zip(tt.env_name, tt.element_type, tt.s_start, tt.s_end):
        if kind not in ('Bend', 'RBend', 'Quadrupole', 'Sextupole', 'Multipole'):
            continue
        if name.startswith(('hcor_', 'vcor_', 'inj_kick')):
            continue
        el = line.get(name)
        if kind == 'Multipole':
            # Include installed higher-order IR magnets, including zero settings.
            if not name.startswith(('oc', 'dec')):
                raise ValueError(f'Unclassified multipole {name}')
            order = 4 if name.startswith('dec') else 3
        else:
            order = {'Bend': 0, 'RBend': 0, 'Quadrupole': 1, 'Sextupole': 2}[kind]
        family = name.rsplit('.', 1)[0]
        strength = float(el.main_strength) / (float(el.length) or 1.0)
        signature = (family, kind, strength)
        # An octupole/decapole at the centre of an IR sextupole is a separate
        # error source, but does not divide the sextupole's alignment or trim
        # circuit. Keep the last finite-length magnet across such thin inserts.
        if (previous_thick is not None and previous_thick['signature'] == signature
                and np.isclose(s, previous_thick['end'], atol=1e-8, rtol=0)
                and end > s):
            previous_thick['names'].append(name)
            previous_thick['end'] = float(end)
        else:
            group = dict(names=[name], kind=kind, order=order,
                         s=float(s), end=float(end), signature=signature)
            groups.append(group)
            if end > s:
                previous_thick = group
    for group in groups:
        group['centre'] = (group['s'] + group['end']) / 2
    return tt, groups


def _classify(tt, groups):
    circumference = float(tt.s[-1])
    positions = dict(zip(tt.name, tt.s))
    starts = sorted((float(s), n.removeprefix('end_ds_start_straight_'))
                    for n, s in positions.items()
                    if n.startswith('end_ds_start_straight_'))
    arc_starts = sorted(float(s) for n, s in positions.items()
                        if n.startswith('end_straight_start_ds_'))
    if len(starts) != 8 or len(arc_starts) != 8:
        raise ValueError('Expected the eight straight/DS boundary pairs in the LCC model')
    for group in groups:
        s = group['centre'] % circumference
        # Arc numbering starts downstream of IPA; DS belongs to the arc.
        group['arc_index'] = int(np.searchsorted(arc_starts, s, side='right') - 1) % 8
        straight = None
        for start, ip in starts:
            end = positions['end_straight_start_ds_' + ip]
            if (s - start) % circumference < (end - start) % circumference:
                straight = ip
                break
        if straight is None:
            group['category'] = {0: 'arc_dipole', 1: 'arc_quad', 2: 'arc_sext'}.get(
                group['order'], 'arc_other')
        elif group['order'] == 0:
            group['category'] = 'straight_dipole'
        else:
            group['category'] = 'straight_other'
            if straight not in ('ipa', 'ipd', 'ipg', 'ipj'):
                continue
            ip_s = positions[straight]
            signed = lambda x: (x - ip_s + circumference / 2) % circumference - circumference / 2
            distance = signed(s)
            side = 1 if distance >= 0 else -1
            fd = [abs(signed(g['centre'])) + (g['end'] - g['s']) / 2
                  for g in groups if g['names'][0].startswith(('qd0', 'qf1'))
                  and 0 < side * signed(g['centre']) < 20]
            cs = [abs(signed(g['centre'])) + (g['end'] - g['s']) / 2
                  for g in groups if g['names'][0].startswith('scrab')
                  and 0 < side * signed(g['centre']) < 2000]
            if not fd or not cs:
                raise ValueError(f'Cannot locate FD/CS boundaries at {straight}')
            if abs(distance) <= max(fd):
                group['category'] = 'ip_to_fd'
            elif abs(distance) <= max(cs):
                group['category'] = 'fd_to_cs'


def install_errors(line, seed=12345):
    env = line.env
    if 'tuning_errors' in line.metadata:
        raise ValueError('Errors already installed: reload the nominal model first')
    rng = np.random.default_rng(seed)
    tt, groups = _inventory(line)
    _classify(tt, groups)
    names = [name for group in groups for name in group['names']]
    line.extend_knl_rel_ksl_rel(order=4, element_names=names)
    knob_names = set()

    def knob(name):
        if name not in knob_names:
            if name in env.vars:
                raise ValueError(f'Error knob already exists: {name}')
            env[name] = 1.0
            knob_names.add(name)
        return env.ref[name]

    master = knob('on_errors')
    random = knob('on_error_random')
    systematic = knob('on_error_systematic')
    align = master * knob('on_error_alignment') * random
    field = master * knob('on_error_field')
    girder = master * knob('on_error_girder') * random
    bpm = master * knob('on_error_bpm') * random
    records = {}
    girder_records = {}
    group_by_name = {name: g for g in groups for name in g['names']}
    arc_quads = [g for g in groups if g['category'] == 'arc_quad']

    # In this example a girder is centred at a quadrupole and includes its
    # neighbouring sextupole(s). The association is saved for inspection.
    for group in groups:
        category = group['category']
        support = None
        if category == 'arc_quad':
            support = group['names'][0]
        elif category == 'arc_sext':
            nearest = min(arc_quads, key=lambda q: abs(q['centre'] - group['centre']))
            gap = max(nearest['s'] - group['end'], group['s'] - nearest['end'], 0)
            if gap > 4.5:
                raise ValueError(f'No neighbouring quadrupole for {group["names"][0]}')
            support = nearest['names'][0]
        group['girder'] = support
        if support is not None and support not in girder_records:
            girder_records[support] = [GIRDER_SIGMAS[label] * truncated_normal(rng)
                                       for label in AXES]

    for group in groups:
        category, order = group['category'], group['order']
        sigmas = SIGMAS[category]
        draws = [sigmas[label] * truncated_normal(rng) for label in AXES]
        support = group['girder']
        record = dict(category=category, names=group['names'], girder=support,
                      arc_index=group['arc_index'], alignment=draws, field_units={})
        records[group['names'][0]] = record
        alignment = []
        for ii, label in enumerate(AXES):
            expr = draws[ii] * align * knob(f'on_error_{category}_{label}')
            if support is not None:
                expr += girder_records[support][ii] * girder * knob(f'on_error_girder_{label}')
            alignment.append(expr)

        # Slide 4: equivalent magnetic-centre shifts, added to mechanical 50 um.
        # These replace quadrupole b1/a1, rather than duplicating their kicks.
        if category == 'arc_quad':
            sign = (-1)**group['arc_index']
            magnetic_centre = [sign * 33e-6, 61e-6 * truncated_normal(rng),
                               22e-6 * truncated_normal(rng)]
            record['magnetic_centre'] = magnetic_centre
            alignment[0] += field * knob('on_error_arc_quad_b1') * knob('on_error_field_k0') * (
                magnetic_centre[0] * systematic + magnetic_centre[1] * random)
            alignment[1] += field * knob('on_error_arc_quad_a1') * knob('on_error_field_k0s') * magnetic_centre[2] * random

        for name in group['names']:
            for attr, expr in zip(AXES.values(), alignment):
                old = getattr(env.ref[name], attr)
                baseline = old._expr if old._expr is not None else old._get_value()
                setattr(env.ref[name], attr, baseline + expr)
            if group['kind'] == 'Multipole':
                env[name].main_order = order

        if category == 'arc_dipole':
            harmonics = ARC_DIPOLE
        elif category == 'arc_quad':
            harmonics = ARC_QUAD
        elif category == 'arc_sext':
            # The linked trim-coil presentation identifies 8 units for SD,
            # 20 for SF (not the ambiguous order of the slide-7 footnote).
            is_sf = group['names'][0].startswith('sf')
            if not group['names'][0].startswith(('sf', 'sd')):
                raise ValueError(f'Unclassified arc sextupole {group["names"][0]}')
            harmonics = {'b1': {'systematic': 0, 'random_sigma': 20 if is_sf else 8},
                         'b3': {'systematic': 0, 'random_sigma': 2}}
        else:
            harmonics = {f'b{order + 1}': {'systematic': 0, 'random_sigma': sigmas['relative_strength'] / 1e-4}}
        for harmonic, settings in harmonics.items():
            mean, sigma = settings['systematic'], settings['random_sigma']
            n = int(harmonic[1:]) - 1
            skew = harmonic[0] == 'a'
            # Mirror-related systematic field terms change sign each octant.
            if category == 'arc_dipole' and harmonic in ('a2', 'b3', 'a4'):
                mean *= (-1)**group['arc_index']
            sample = sigma * truncated_normal(rng) if sigma else 0.0
            record['field_units'][harmonic] = [mean, sample]
            expr = (relative_coefficient(mean, n, order) * systematic
                    + relative_coefficient(sample, n, order) * random)
            expr *= field * knob(f'on_error_{category}_{harmonic}')
            expr *= knob(f'on_error_field_k{n}{"s" if skew else ""}')
            for name in group['names']:
                target = env.ref[name].ksl_rel if skew else env.ref[name].knl_rel
                target[n] += expr

    bpm_variables = {}
    bpm_samples = {}
    for name in tt.env_name:
        if not name.startswith('bpm_'):
            continue
        parent = name.removeprefix('bpm_')
        if parent not in group_by_name or group_by_name[parent]['order'] != 1:
            raise ValueError(f'Cannot identify BPM parent: {name}')
        bpm_variables[name] = {}
        bpm_samples[name] = {}
        for label, sigma in BPM_SIGMAS.items():
            attr = AXES[label]
            sample = sigma * truncated_normal(rng)
            variable = f'err_{name}_{label}'
            # Offsets are residual post-BBA: use the displaced magnetic centre.
            # BPM roll follows the support, not the magnet's independent roll.
            if attr == 'rot_s_rad':
                support = group_by_name[parent]['girder']
                base = (girder_records[support][3] * girder * knob('on_error_girder_roll')
                        if support is not None else 0)
            else:
                base = getattr(env.ref[parent], attr)
            env[variable] = base + sample * bpm * knob(f'on_error_bpm_{label}')
            bpm_variables[name][attr] = variable
            bpm_samples[name][attr] = sample

    line.metadata['tuning_errors'] = dict(
        source=SOURCE, slide=7, seed=seed, reference_radius_m=R_REF,
        gaussian_cut=TRUNCATION, knobs=sorted(knob_names),
        counts=dict(Counter(g['category'] for g in groups)),
        magnets=records, girders=girder_records, bpm_samples=bpm_samples,
        bpm_alignment_variables=bpm_variables,
        measurement_sigmas=dict(orbit_m=1e-6, dispersion_m=10e-6,
                                phase=1e-4, coupling_rdts=1e-4),
        assumptions=[
            'Static slide-7 recipe; current-dependent trim errors excluded.',
            'Lattice-frame harmonics; alternate dipole a2/b3/a4 and quad dx signs.',
            'Contiguous pieces of the same family share errors across zero-length inserts.',
            'Arc quad and nearest sextupole share girder translations and roll.',
            'Auxiliary arc/DS octupoles and decapoles use Other elt. tolerances.',
            'BPM offsets are post-BBA relative to quadrupole magnetic centres.',
            'No radiation/tapering, correction, or measurement noise applied.',
        ])
    return line.metadata['tuning_errors']


def get_bpm_alignment(line):
    """Evaluate current knobs for correct_trajectory(monitor_alignment=...).

    Re-evaluate this mapping (and recreate the correction object) after changing
    alignment knobs. Marker transforms do not model a BPM readout offset.
    """
    return {name: {attr: float(line[vname]) for attr, vname in variables.items()}
            for name, variables in line.metadata['tuning_errors'][
                'bpm_alignment_variables'].items()}
