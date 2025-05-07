from typing import Literal

import numpy as np

import xobjects as xo
import xtrack as xt
from bmad_track_twiss_spin import bmad_run

mode: Literal['generate', 'verify'] = 'generate'
file_name = 'bend_bmad.json'

env = xt.Environment()
line = env.new_line(components=[
    env.new('mykicker', xt.Bend, k0=0.01, h=0.01, length=0.02),
    env.new('mymarker', xt.Marker),
])

test_cases = [
    {'x': 1e-3, 'px': 1e-5, 'y': 2e-3, 'py': 2e-5, 'delta': 1e-3, 'spin_x': 0.1, 'spin_z': 0.2, 'atol': 7e-6},
    # Vary delta in np.linspace(-0.01, 0.01, 5)
    {'x': 1e-3, 'px': 1e-5, 'y': 2e-3, 'py': 2e-5, 'delta': -0.01, 'spin_x': 0.1, 'spin_z': 0.2, 'atol': 7e-6},
    {'x': 1e-3, 'px': 1e-5, 'y': 2e-3, 'py': 2e-5, 'delta': -0.005, 'spin_x': 0.1, 'spin_z': 0.2, 'atol': 7e-6},
    {'x': 1e-3, 'px': 1e-5, 'y': 2e-3, 'py': 2e-5, 'delta': 0, 'spin_x': 0.1, 'spin_z': 0.2, 'atol': 7e-6},
    {'x': 1e-3, 'px': 1e-5, 'y': 2e-3, 'py': 2e-5, 'delta': 0.005, 'spin_x': 0.1, 'spin_z': 0.2, 'atol': 7e-6},
    {'x': 1e-3, 'px': 1e-5, 'y': 2e-3, 'py': 2e-5, 'delta': 0.01, 'spin_x': 0.1, 'spin_z': 0.2, 'atol': 7e-6},
    # Vary px, py in np.linspace(-0.03, 0.03, 5), np.linspace(-0.02, 0.02, 5)
    {'x': 1e-3, 'px': -0.03, 'y': 2e-3, 'py': -0.02, 'delta': 1e-3, 'spin_x': 0.1, 'spin_z': 0.2, 'atol': 7e-3},
    {'x': 1e-3, 'px': -0.015, 'y': 2e-3, 'py': -0.01, 'delta': 1e-3, 'spin_x': 0.1, 'spin_z': 0.2, 'atol': 4e-3},
    {'x': 1e-3, 'px': 0, 'y': 2e-3, 'py': 0, 'delta': 1e-3, 'spin_x': 0.1, 'spin_z': 0.2, 'atol': 8e-6},
    {'x': 1e-3, 'px': 0.015, 'y': 2e-3, 'py': 0.01, 'delta': 1e-3, 'spin_x': 0.1, 'spin_z': 0.2, 'atol': 4e-3},
    {'x': 1e-3, 'px': 0.03, 'y': 2e-3, 'py': 0.02, 'delta': 1e-3, 'spin_x': 0.1, 'spin_z': 0.2, 'atol': 7e-3},
]

print(f'atols = {repr([case["atol"] for case in test_cases])}')

p0 = xt.Particles(p0c=700e9, mass0=xt.ELECTRON_MASS_EV,
                  anomalous_magnetic_moment=0.00115965218128)

out = []

if mode == 'verify':
    with open(file_name, 'r') as f:
        out = xt.json.load(f)

for case in test_cases:
    line.particle_ref = p0.copy()

    p0.x = case['x']
    p0.px = case['px']
    p0.y = case['y']
    p0.py = case['py']
    p0.delta = case['delta']
    p0.spin_x = case['spin_x']
    p0.spin_z = case['spin_z']
    case['spin_y'] = np.sqrt(1 - case['spin_x']**2 - case['spin_z']**2)
    p0.spin_y = case['spin_y']
    atol = case.pop('atol')

    if mode == 'generate':
        print('Running Bmad')
        out_bmad = bmad_run(line, track=case)
        bmad_spin = {key: getattr(out_bmad['spin'], key).values[-1] for key in ['spin_x', 'spin_y', 'spin_z']}
        out.append({
            'in': case,
            'out': bmad_spin,
        })
    elif mode == 'verify':
        bmad_spin = None
        for file_case in out:
            if file_case['in'] == case:
                bmad_spin = file_case['out']
                break
        if not bmad_spin:
            raise ValueError(f'Case {case} not found in file {file_name}. Regenerate?')
    else:
        raise ValueError(f'Invalid mode `{mode}`')

    # line.configure_spin(model=True)
    line.config.XTRACK_MULTIPOLE_NO_SYNRAD = False

    p = p0.copy()
    line.track(p)

    try:
        xo.assert_allclose(p.spin_x[0], bmad_spin['spin_x'], atol=atol, rtol=0)
        xo.assert_allclose(p.spin_y[0], bmad_spin['spin_y'], atol=atol, rtol=0)
        xo.assert_allclose(p.spin_z[0], bmad_spin['spin_z'], atol=atol, rtol=0)
        print(f'Success: {case}')
    except AssertionError:
        max_atol = np.max(np.abs([
            p.spin_x[0] - bmad_spin['spin_x'],
            p.spin_y[0] - bmad_spin['spin_y'],
            p.spin_z[0] - bmad_spin['spin_z'],
        ]))
        print(f'Failure: {case} => ATOL {max_atol}')


if mode == 'generate':
    xt.json.dump(out, file_name)
