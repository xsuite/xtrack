# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import numpy as np
import xtrack as xt

################################################################################
# Geometry parameters
################################################################################
QUAD_LENGTHS    = [0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0]  # [m]
DRIFT_LENGTHS   = [0.4, 0.5, 0.9, 1.9, 5.0, 7.0, 8.0, 8.0, 6.5]  # [m]

################################################################################
# Define adjustable circuits and reusable magnetic families
################################################################################

########################################
# Declare dependencies without selecting an optical operating point
########################################
env = xt.get_environment()
for knob in ['kqf', 'kqd', 'dkqf', 'dkqd', 'k2lf', 'k2ld', 'k3lf', 'k3ld',
             'voltage_rf', 'frequency_rf', 'phase_rf']:
    env[knob] = 0.0
for family in ['ff', 'ut']:
    for index in range(8):
        env[f'{family}_k{index}'] = 0.0
for sector in range(1, 7):
    env[f'ks_s{sector}'] = 0.0           # Thick skew k1 [m^-2]

########################################
# Define arc families and their explicit 40 m cell
########################################
env.new('arc.qf', xt.Quadrupole, length = 0.5, k1 = 'kqf+dkqf')
env.new('arc.qd', xt.Quadrupole, length = 1.0, k1 = 'kqd+dkqd')
env.new('arc.bend', xt.Bend,
        length              = 5.0,
        angle               = np.pi / 24,
        model               = 'bend-kick-bend',
        edge_entry_model    = 'linear',
        edge_exit_model     = 'linear')
env.new('arc.sf', xt.Multipole, knl = [0.0, 0.0, 'k2lf'])
env.new('arc.sd', xt.Multipole, knl = [0.0, 0.0, 'k2ld'])
arc_cell    = env.new_line(name = 'arc.cell', components = [
    env.new('cell.qf.in', 'arc.qf'),
    env.new('cell.d0', xt.Drift, length = 0.5),
    env.new('cell.sf', 'arc.sf'),
    env.new('cell.d1', xt.Drift, length = 6.5),
    env.new('cell.b1', 'arc.bend'),
    env.new('cell.d2', xt.Drift, length = 7.0),
    env.new('cell.qd', 'arc.qd'),
    env.new('cell.d3', xt.Drift, length = 0.5),
    env.new('cell.sd', 'arc.sd'),
    env.new('cell.d4', xt.Drift, length = 6.5),
    env.new('cell.b2', 'arc.bend'),
    env.new('cell.d5', xt.Drift, length = 7.0),
    env.new('cell.qf.out', 'arc.qf'),
])

################################################################################
# Instantiate named arcs and ideal observation/steering packages
################################################################################

########################################
# Positive horizontal and vertical controls give positive momentum kicks
########################################
correctors  = [f'cor.s{s}' for s in range(1, 7)]
correctors += ['cor.ip.1', 'cor.ip.2', 'cor.ip.3']
for name in correctors:
    suffix  = name.replace('.', '_')
    env[f'h_{suffix}'] = 0.0
    env[f'v_{suffix}'] = 0.0
    env.new(name, xt.Multipole,
            knl = [f'-h_{suffix}'], ksl = [f'v_{suffix}'])

# Multipole Delta px=-knl[0], Delta py=+ksl[0], normalized to reference p0.
env.new('oct.f', xt.Multipole, knl = [0.0, 0.0, 0.0, 'k3lf'])
env.new('oct.d', xt.Multipole, knl = [0.0, 0.0, 0.0, 'k3ld'])
bpms    = []

########################################
# Copy four cells per arc and add local zero errors and trims
########################################
for sector in range(1, 7):
    components  = []
    for cell in range(1, 5):
        prefix  = f'arc.s{sector}.c{cell}'
        for template in arc_cell.element_names:
            part        = template.removeprefix('cell.')
            name        = f'{prefix}.{part}'
            prototype   = (
                xt.Marker if part in ['sf', 'sd'] and cell % 2 == 0
                else template)
            env.new(name, prototype)
            if part == 'qd':
                trim    = f'dq_s{sector}_c{cell}'
                error   = f'eq_s{sector}_c{cell}'
                skew    = f'ks_s{sector}' if cell == 2 else 0.0
                env[trim], env[error] = 0.0, 0.0
                env.set(name, k1 = f'kqd+dkqd+{trim}+{error}', k1s = skew)
            components.append(name)
            if (part, cell) in [('sf', 1), ('sd', 2)]:
                bpm = f'bpm.s{sector}.' + ('f' if part == 'sf' else 'd')
                env.new(bpm, xt.Device, length = 0.0)
                components.append(bpm)
                bpms.append(bpm)
            if part == 'sf' and cell == 3:
                components.append(f'cor.s{sector}')
            if (sector, cell, part) == (2, 1, 'sf'):
                components.append('oct.f')
            if (sector, cell, part) == (5, 1, 'sd'):
                components.append('oct.d')
    env.new_line(name = f'arc.s{sector}', components = components)

################################################################################
# Build reusable half-straights and their symmetric full sections
################################################################################

########################################
# Define each half outward from its midpoint toward the arc
########################################
for family, half_name in [('ff', 'ff.half'), ('ut', 'utility.half')]:
    components  = []
    for index, length in enumerate(QUAD_LENGTHS):
        components.extend([
            env.new(f'{family}.d{index}', xt.Drift,
                    length  = DRIFT_LENGTHS[index]),
            env.new(f'{family}.q{index}', xt.Quadrupole,
                    length  = length,
                    k1      = f'{family}_k{index}'),
        ])
    components.append(env.new(f'{family}.d8', xt.Drift,
                              length    = DRIFT_LENGTHS[8]))
    env.new_line(name = half_name, components = components)

########################################
# Reverse incoming magnet order without changing gradient signs
########################################
for sector in range(1, 7):
    family      = 'ff' if sector == 1 else 'ut'
    half_name   = 'ff.half' if sector == 1 else 'utility.half'
    components  = []
    for side in ['l', 'r']:
        boundary    = (
            f'straight.s{sector}.' + ('start' if side == 'l' else 'mid'))
        if sector == 1 and side == 'r':
            boundary    = 'ip.1'
        env.new(boundary, xt.Marker)
        components.append(boundary)
        half    = env.lines[half_name].element_names
        order   = list(reversed(half)) if side == 'l' else half
        for template in order:
            part    = template.split('.')[1]
            name    = f'straight.s{sector}.{side}.{part}'
            if sector == 1 and side == 'r' and part == 'q0':
                env.new('bpm.ip.r', xt.Device, length = 0.0)
                components.append('bpm.ip.r')
                bpms.append('bpm.ip.r')
            env.new(name, template)
            if part.startswith('q'):
                index   = int(part[1:])
                trim    = f'dq_{family}_s{sector}_{side}{index}'
                error   = f'eq_{family}_s{sector}_{side}{index}'
                env[trim], env[error] = 0.0, 0.0
                env.set(name, k1 = f'{family}_k{index}+{trim}+{error}')
            components.append(name)
            if sector == 1 and (side, part) in [
                ('l', 'q5'), ('l', 'q1'), ('r', 'q3'),
            ]:
                number  = {('l', 'q5'): 1, ('l', 'q1'): 2, ('r', 'q3'): 3}[
                    side, part]
                components.append(f'cor.ip.{number}')
            if sector == 1 and side == 'l' and part == 'q0':
                env.new('bpm.ip.l', xt.Device, length = 0.0)
                components.append('bpm.ip.l')
                bpms.append('bpm.ip.l')
    components.append(env.new(f'straight.s{sector}.end', xt.Marker))
    if sector == 1:
        components.append(env.new('rf.1', xt.Cavity,
                                  voltage   = 'voltage_rf',
                                  frequency = 'frequency_rf',
                                  phase     = 'phase_rf'))
    name    = 'insertion' if sector == 1 else f'utility.s{sector}'
    env.new_line(name = name, components = components)

# Device marks an ideal passive read plane, without BPM electronics.
# Quadrupole, trim, error, nonlinear and RF circuits remain zero. Bends carry
# their design field through their curvature, using k0='from_h'.
