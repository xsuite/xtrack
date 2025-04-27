import xtrack as xt
import xpart as xp
import xobjects as xo
import numpy as np

num_turns = 500

line = xt.Line.from_json('lep_sol.json')
line.particle_ref.anomalous_magnetic_moment=0.00115965218128
line.particle_ref.gamma0 = 89207.78287659843 # to have a spin tune of 103.45
spin_tune = line.particle_ref.anomalous_magnetic_moment[0]*line.particle_ref.gamma0[0]


tt = line.get_table(attr=True)

out_lines = []
out_lines += [
    f'beam, energy  = {line.particle_ref.energy0[0]/1e9}',
    'parameter[particle] = electron',
    'parameter[geometry] = closed',
    'bmad_com[spin_tracking_on]=T',
    'bmad_com[radiation_damping_on]=T',
    'bmad_com[radiation_fluctuations_on]=T',
    ''
]

for nn in line.element_names:
    ee = line[nn]
    clssname = ee.__class__.__name__

    if clssname == 'Marker':
        out_lines.append(f'{nn}: marker')
    elif clssname == 'Drift':
        out_lines.append(f'{nn}: drift, l = {ee.length}')
    elif clssname == 'Quadrupole':
        if ee.k1s == 0:
            out_lines.append(f'{nn}: quadrupole, l = {ee.length}, k1 = {ee.k1}')
        else:
            assert ee.k1 == 0
            out_lines.append(f'{nn}: quadrupole, l = {ee.length}, k1 = {ee.k1s}, tilt')
    elif clssname == 'Multipole':
        assert np.linalg.norm(ee.hxl) == 0
        if np.linalg.norm(ee.knl) == 0 and np.linalg.norm(ee.ksl) == 0:
            out_lines.append(f'{nn}: marker') # Temporary
        else:
            assert len(ee.knl) == 1
            assert len(ee.ksl) == 1
            if ee.ksl[0] == 0:
                out_lines.append(f'{nn}: multipole, l = {ee.length}, k0l = {ee.knl[0]}')
            else:
                assert ee.knl[0] == 0
                out_lines.append(f'{nn}: multipole, l = {ee.length}, k0l = {ee.ksl[0]}, tilt={np.pi/2}')
    elif clssname == 'Sextupole':
        out_lines.append(f'{nn}: sextupole, l = {ee.length}, k2 = {ee.k2}')
    elif clssname == 'RBend':
        out_lines.append(f'{nn}: rbend, l = {ee.length_straight}, angle = {ee.angle}')
    elif clssname == 'Cavity':
        out_lines.append(f'{nn}: rfcavity, voltage = {ee.voltage},'
                         f'rf_frequency = {ee.frequency}, lag = 0') # Lag hardcoded for now!
    elif clssname == 'Octupole':
        out_lines.append(f'{nn}: octupole, l = {ee.length}, k3 = {ee.k3}')
    elif clssname == 'DriftSlice':
        ll = tt['length', nn]
        out_lines.append(f'{nn}: drift, l = {ll}')
    elif clssname == 'Solenoid':
        out_lines.append(f'{nn}: solenoid, l = {ee.length}, ks = {ee.ks}')
    else:
        raise ValueError(f'Unknown element type {clssname} for {nn}')