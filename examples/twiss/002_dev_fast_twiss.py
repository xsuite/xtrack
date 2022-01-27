import time
import json
import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp

###############
# Load a line #
###############

fname_line_particles = '../../test_data/hllhc15_noerrors_nobb/line_and_particle.json'
#fname_line_particles = '../../test_data/sps_w_spacecharge/line_no_spacecharge_and_particle.json' #!skip-doc

with open(fname_line_particles, 'r') as fid:
    input_data = json.load(fid)
line = xt.Line.from_dict(input_data['line'])
particle_ref = xp.Particles.from_dict(input_data['particle'])

#################
# Build tracker #
#################

tracker = xt.Tracker(line=line)


particle_co_guess = None
co_search_settings = None
steps_r_matrix = None
nemitt_x = 1e-6
nemitt_y = 1e-6
r_sigma = 0.01
delta_disp = 1e-5
delta_chrom = 1e-5
at_elements = ['ip1', 'ip2', 'ip5', 'ip8']

context = tracker._buffer.context

t1 = time.time()

part_on_co = tracker.find_closed_orbit(particle_co_guess=particle_co_guess,
                                    particle_ref=particle_ref,
                                    co_search_settings=co_search_settings)
RR = tracker.compute_one_turn_matrix_finite_differences(
                                            steps_r_matrix=steps_r_matrix,
                                            particle_on_co=part_on_co)

gemitt_x = nemitt_x/part_on_co.beta0/part_on_co.gamma0
gemitt_y = nemitt_y/part_on_co.beta0/part_on_co.gamma0

W, Winv, Rot = xp.compute_linear_normal_form(RR)


s = np.array(tracker.line.get_s_elements())

scale_transverse_x = np.sqrt(gemitt_x)*r_sigma
scale_transverse_y = np.sqrt(gemitt_y)*r_sigma
part_for_twiss = xp.build_particles(_context=context,
                    particle_ref=part_on_co, mode='shift',
                    x=  list(W[0, :4] * scale_transverse_x) + [0],
                    px= list(W[1, :4] * scale_transverse_x) + [0],
                    y=  list(W[2, :4] * scale_transverse_y) + [0],
                    py= list(W[3, :4] * scale_transverse_y) + [0],
                    zeta = 0,
                    delta = 0,
                    )

part_disp = xp.build_particles(
                _context=context,
                x_norm=0,
                zeta=part_on_co.zeta[0], delta=[-delta_disp, +delta_disp],
                particle_on_co=part_on_co,
                scale_with_transverse_norm_emitt=(nemitt_x, nemitt_y),
                R_matrix=RR)

part_for_twiss = xp.Particles.merge([part_for_twiss, part_disp])

tracker.track(part_for_twiss, turn_by_turn_monitor='ONE_TURN_EBE')

x_co = tracker.record_last_track.x[4, :].copy()
y_co = tracker.record_last_track.y[4, :].copy()
px_co = tracker.record_last_track.px[4, :].copy()
py_co = tracker.record_last_track.py[4, :].copy()

x_disp_minus = tracker.record_last_track.x[5, :].copy()
y_disp_minus = tracker.record_last_track.y[5, :].copy()
px_disp_minus = tracker.record_last_track.px[5, :].copy()
py_disp_minus = tracker.record_last_track.py[5, :].copy()

x_disp_plus = tracker.record_last_track.x[6, :].copy()
y_disp_plus = tracker.record_last_track.y[6, :].copy()
px_disp_plus = tracker.record_last_track.px[6, :].copy()
py_disp_plus = tracker.record_last_track.py[6, :].copy()

dx = (x_disp_plus-x_disp_minus)/delta_disp/2
dy = (y_disp_plus-y_disp_minus)/delta_disp/2
dpx = (px_disp_plus-px_disp_minus)/delta_disp/2
dpy = (py_disp_plus-py_disp_minus)/delta_disp/2

W4 = np.zeros(shape=(4,4,len(s)), dtype=np.float64)
W4[0, :, :] = (tracker.record_last_track.x[:4, :] - x_co) / scale_transverse_x
W4[1, :, :] = (tracker.record_last_track.px[:4, :] - px_co) / scale_transverse_x
W4[2, :, :] = (tracker.record_last_track.y[:4, :]  - y_co) / scale_transverse_y
W4[3, :, :] = (tracker.record_last_track.py[:4, :] - py_co) / scale_transverse_y

betx = W4[0, 0, :]**2 + W4[0, 1, :]**2
bety = W4[2, 2, :]**2 + W4[2, 3, :]**2

gamx = W4[1, 0, :]**2 + W4[1, 1, :]**2
gamy = W4[3, 2, :]**2 + W4[3, 3, :]**2

alfx = - W4[0, 0, :] * W4[1, 0, :] - W4[0, 1, :] * W4[1, 1, :]
alfy = - W4[2, 2, :] * W4[3, 2, :] - W4[2, 3, :] * W4[3, 3, :]

mux = np.unwrap(np.arctan2(W4[0, 1, :], W4[0, 0, :]))/2/np.pi
muy = np.unwrap(np.arctan2(W4[2, 3, :], W4[2, 2, :]))/2/np.pi

eta = -((part_disp._xobject.zeta[0] - part_disp._xobject.zeta[1])
            /(2*delta_disp)/tracker.line.get_length())
alpha = eta + 1/particle_ref.gamma0[0]**2

part_chrom_plus = xp.build_particles(
            _context=context,
            x_norm=0,
            zeta=part_on_co.zeta[0], delta=delta_chrom,
            particle_on_co=part_on_co,
            scale_with_transverse_norm_emitt=(nemitt_x, nemitt_y),
            R_matrix=RR)
RR_chrom_plus = tracker.compute_one_turn_matrix_finite_differences(
                                        particle_on_co=part_chrom_plus.copy(),
                                        steps_r_matrix=steps_r_matrix)
(WW_chrom_plus, WWinv_chrom_plus, Rot_chrom_plus
    ) = xp.compute_linear_normal_form(RR_chrom_plus)
qx_chrom_plus = np.angle(np.linalg.eig(Rot_chrom_plus)[0][0])/(2*np.pi)
qy_chrom_plus = np.angle(np.linalg.eig(Rot_chrom_plus)[0][2])/(2*np.pi)

part_chrom_minus = xp.build_particles(
            _context=context,
            x_norm=0,
            zeta=part_on_co.zeta[0], delta=-delta_chrom,
            particle_on_co=part_on_co,
            scale_with_transverse_norm_emitt=(nemitt_x, nemitt_y),
            R_matrix=RR)
RR_chrom_minus = tracker.compute_one_turn_matrix_finite_differences(
                                    particle_on_co=part_chrom_minus.copy(),
                                    steps_r_matrix=steps_r_matrix)
(WW_chrom_minus, WWinv_chrom_minus, Rot_chrom_minus
    ) = xp.compute_linear_normal_form(RR_chrom_minus)
qx_chrom_minus = np.angle(np.linalg.eig(Rot_chrom_minus)[0][0])/(2*np.pi)
qy_chrom_minus = np.angle(np.linalg.eig(Rot_chrom_minus)[0][2])/(2*np.pi)

dqx = (qx_chrom_plus - qx_chrom_minus)/delta_chrom/2
dqy = (qy_chrom_plus - qy_chrom_minus)/delta_chrom/2

twiss_res = {
    'name': tracker.line.element_names,
    's': s,
    'x': x_co,
    'px': px_co,
    'y': y_co,
    'py': py_co,
    'betx': betx,
    'bety': bety,
    'alfx': alfx,
    'alfy': alfy,
    'gamx': gamx,
    'gamy': gamy,
    'dx': dx,
    'dpx': dpx,
    'dy': dy,
    'dpy': dpy,
    'mux': mux,
    'muy': muy,
    'qx': mux[-1],
    'qy': muy[-1],
    'dqx': dqx,
    'dqy': dqy,
    'slip_factor': eta,
    'momentum_compaction_factor': alpha,
    'R_matrix': RR,
    'particle_on_co':part_on_co.copy(_context=xo.context_default)
}

# Downselect based on at_element
enames = tracker.line.element_names
if at_elements is not None:
    indx_twiss = []
    for nn in at_elements:
        if isinstance(nn, (int, np.integer)):
            indx_twiss.append(int(nn))
        else:
            assert nn in tracker.line.element_names
            indx_twiss.append(enames.index(nn))
    indx_twiss = sorted(indx_twiss)

    for kk, vv in twiss_res.items():
        if hasattr(vv, '__len__') and len(vv) == len(s):
            if isinstance(vv, np.ndarray):
                twiss_res[kk] = vv[indx_twiss]
            else:
                twiss_res[kk] = [vv[ii] for ii in indx_twiss]
t2 = time.time()

print(f'dt={(t2-t1)*1000} ms')