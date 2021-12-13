import pickle
import json
import pathlib
import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp

from make_short_line import make_short_line

short_test = False # Short line (5 elements)

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../../test_data').absolute()

fname_line_particles = test_data_folder.joinpath(
                        './hllhc_14/line_and_particle.json')

####################
# Choose a context #
####################

context = xo.ContextCpu()


with open(fname_line_particles, 'r') as fid:
    input_data = json.load(fid)

line = xt.Line.from_dict(input_data['line'])

for ee in line.elements:
    if ee.__class__.__name__.startswith('BeamBeam'):
        assert hasattr(ee, 'q0')
        ee.q0 = 0

print('Build tracker...')
freeze_vars = xp.particles.part_energy_varnames() + ['zeta']
tracker = xt.Tracker(_context=context,
            line=line,
            local_particle_src=xp.gen_local_particle_api(
                                                freeze_vars=freeze_vars),
            )


part0 = xp.Particles(_context=context, **input_data['particle'])

part_on_co = tracker.find_closed_orbit(part0)
RR = tracker.compute_one_turn_matrix_finite_differences(particle_on_co=part_on_co)
r_sigma = 0.01
nemitt_x = 2.5e-6
nemitt_y = 2.5e-6
n_theta = 1000
part_x = xp.build_particles(
            x_norm=r_sigma*np.cos(np.linspace(0, 2*np.pi, n_theta)),
            px_norm=r_sigma*np.sin(np.linspace(0, 2*np.pi, n_theta)),
            zeta=part_on_co.zeta[0], delta=part_on_co.delta[0],
            particle_on_co=part_on_co,
            scale_with_transverse_norm_emitt=(nemitt_x, nemitt_y),
            R_matrix=RR)
part_y = xp.build_particles(
            y_norm=r_sigma*np.cos(np.linspace(0, 2*np.pi, n_theta)),
            py_norm=r_sigma*np.sin(np.linspace(0, 2*np.pi, n_theta)),
            zeta=part_on_co.zeta[0], delta=part_on_co.delta[0],
            particle_on_co=part_on_co,
            scale_with_transverse_norm_emitt=(nemitt_x, nemitt_y),
            R_matrix=RR)

num_elements = len(tracker.line.elements)
max_x = np.zeros(num_elements, dtype=np.float64)
max_y = np.zeros(num_elements, dtype=np.float64)
min_x = np.zeros(num_elements, dtype=np.float64)
min_y = np.zeros(num_elements, dtype=np.float64)
x_co = np.zeros(num_elements, dtype=np.float64)
y_co = np.zeros(num_elements, dtype=np.float64)

for ii, ee in enumerate(tracker.line.elements):
    print(f'{ii}/{len(tracker.line.elements)}        ', end='\r', flush=True)
    max_x[ii] = np.max(part_x.x)
    max_y[ii] = np.max(part_y.y)

    min_x[ii] = np.min(part_x.x)
    min_y[ii] = np.min(part_y.y)

    x_co[ii] = part_on_co.x[0]
    y_co[ii] = part_on_co.y[0]

    tracker.track(part_on_co, ele_start=ii, num_elements=1)
    tracker.track(part_x, ele_start=ii, num_elements=1)
    tracker.track(part_y, ele_start=ii, num_elements=1)

s = tracker.line.get_s_elements()

sigx_max = (max_x - x_co)/r_sigma
sigy_max = (max_y - y_co)/r_sigma
sigx_min = (x_co - min_x)/r_sigma
sigy_min = (y_co - min_y)/r_sigma
sigx = (sigx_max + sigx_min)/2
sigy = (sigy_max + sigy_min)/2

betx = sigx**2*part0.gamma0[0]*part0.beta0[0]/nemitt_x
bety = sigy**2*part0.gamma0[0]*part0.beta0[0]/nemitt_y

