import pickle
import json
import pathlib
import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../../test_data').absolute()

fname_line_particles = test_data_folder.joinpath(
                        './hllhc_14/line_and_particle.json')

fname_line_particles = './temp_precise_lattice/xtline.json'

####################
# Choose a context #
####################

context = xo.ContextCpu()


with open(fname_line_particles, 'r') as fid:
    input_data = json.load(fid)

line = xt.Line.from_dict(input_data['line'])


print('Build tracker...')
tracker = xt.Tracker(_context=context,
            line=line,
            )


part0 = xp.Particles(_context=context, **input_data['particle'])

part_on_co = tracker.find_closed_orbit(part0)
RR = tracker.compute_one_turn_matrix_finite_differences(particle_on_co=part_on_co)
W, Winv, Rot = xp.compute_linear_normal_form(RR)
r_sigma = 0.01
nemitt_x = 2.5e-6
nemitt_y = 2.5e-6
n_theta = 1000
delta_disp = 1e-5
delta_chrom = 1e-4
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
part_y = xp.build_particles(
            y_norm=r_sigma*np.cos(np.linspace(0, 2*np.pi, n_theta)),
            py_norm=r_sigma*np.sin(np.linspace(0, 2*np.pi, n_theta)),
            zeta=part_on_co.zeta[0], delta=part_on_co.delta[0],
            particle_on_co=part_on_co,
            scale_with_transverse_norm_emitt=(nemitt_x, nemitt_y),
            R_matrix=RR)
part_disp = xp.build_particles(
            x_norm=0,
            zeta=part_on_co.zeta[0], delta=delta_disp,
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
x_disp = np.zeros(num_elements, dtype=np.float64)
y_disp = np.zeros(num_elements, dtype=np.float64)

for ii, ee in enumerate(tracker.line.elements):
    print(f'{ii}/{len(tracker.line.elements)}        ', end='\r', flush=True)
    max_x[ii] = np.max(part_x.x)
    max_y[ii] = np.max(part_y.y)

    min_x[ii] = np.min(part_x.x)
    min_y[ii] = np.min(part_y.y)

    x_co[ii] = part_on_co.x[0]
    y_co[ii] = part_on_co.y[0]

    x_disp[ii] = part_disp.x[0]
    y_disp[ii] = part_disp.y[0]

    tracker.track(part_on_co, ele_start=ii, num_elements=1)
    tracker.track(part_x, ele_start=ii, num_elements=1)
    tracker.track(part_y, ele_start=ii, num_elements=1)
    tracker.track(part_disp, ele_start=ii, num_elements=1)

s = tracker.line.get_s_elements()

sigx_max = (max_x - x_co)/r_sigma
sigy_max = (max_y - y_co)/r_sigma
sigx_min = (x_co - min_x)/r_sigma
sigy_min = (y_co - min_y)/r_sigma
sigx = (sigx_max + sigx_min)/2
sigy = (sigy_max + sigy_min)/2

betx = sigx**2*part0.gamma0[0]*part0.beta0[0]/nemitt_x
bety = sigy**2*part0.gamma0[0]*part0.beta0[0]/nemitt_y

dx = (x_disp-x_co)/delta_disp
dy = (y_disp-y_co)/delta_disp

qx = np.angle(np.linalg.eig(Rot)[0][0])/(2*np.pi)
qy = np.angle(np.linalg.eig(Rot)[0][2])/(2*np.pi)



part_chrom_plus = xp.build_particles(
            x_norm=0,
            zeta=part_on_co.zeta[0], delta=delta_chrom,
            particle_on_co=part_on_co,
            scale_with_transverse_norm_emitt=(nemitt_x, nemitt_y),
            R_matrix=RR)
RR_chrom_plus = tracker.compute_one_turn_matrix_finite_differences(
                                            particle_on_co=part_chrom_plus.copy())
WW_chrom_plus, WWinv_chrom_plus, Rot_chrom_plus = xp.compute_linear_normal_form(RR_chrom_plus)
qx_chrom_plus = np.angle(np.linalg.eig(Rot_chrom_plus)[0][0])/(2*np.pi)
qy_chrom_plus = np.angle(np.linalg.eig(Rot_chrom_plus)[0][2])/(2*np.pi)

part_chrom_minus = xp.build_particles(
            x_norm=0,
            zeta=part_on_co.zeta[0], delta=-delta_chrom,
            particle_on_co=part_on_co,
            scale_with_transverse_norm_emitt=(nemitt_x, nemitt_y),
            R_matrix=RR)
RR_chrom_minus = tracker.compute_one_turn_matrix_finite_differences(
                                            particle_on_co=part_chrom_minus.copy())
WW_chrom_minus, WWinv_chrom_minus, Rot_chrom_minus = xp.compute_linear_normal_form(RR_chrom_minus)
qx_chrom_minus = np.angle(np.linalg.eig(Rot_chrom_minus)[0][0])/(2*np.pi)
qy_chrom_minus = np.angle(np.linalg.eig(Rot_chrom_minus)[0][2])/(2*np.pi)

qpx = (qx_chrom_plus - qx_chrom_minus)/delta_chrom/2
qpy = (qy_chrom_plus - qy_chrom_minus)/delta_chrom/2


