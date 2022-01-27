
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

context = tracker._buffer.context

part_on_co = tracker.find_closed_orbit(particle_co_guess=particle_co_guess,
                                    particle_ref=particle_ref,
                                    co_search_settings=co_search_settings)
RR = tracker.compute_one_turn_matrix_finite_differences(
                                            steps_r_matrix=steps_r_matrix,
                                            particle_on_co=part_on_co)

gemitt_x = nemitt_x/part_on_co.beta0/part_on_co.gamma0
gemitt_y = nemitt_y/part_on_co.beta0/part_on_co.gamma0

W, Winv, Rot = xp.compute_linear_normal_form(RR)

tracker.track(part_on_co.copy(), turn_by_turn_monitor='ONE_TURN_EBE')

x_co = tracker.record_last_track.x[0, :].copy()
y_co = tracker.record_last_track.y[0, :].copy()
px_co = tracker.record_last_track.px[0, :].copy()
py_co = tracker.record_last_track.py[0, :].copy()

s = np.array(tracker.line.get_s_elements())

# r_in_sigmas_for_W = 0.1
# part_for_twiss = xp.build_particles(
#                     particle_on_co=part_on_co, R_matrix=RR,
#                     x_norm =  np.array([1,0,0,0]) * r_in_sigmas_for_W,
#                     px_norm = np.array([0,1,0,0]) * r_in_sigmas_for_W,
#                     y_norm =  np.array([1,0,1,0]) * r_in_sigmas_for_W,
#                     py_norm = np.array([0,0,0,1]) * r_in_sigmas_for_W,
#                     zeta = part_on_co.zeta[0],
#                     delta = part_on_co.delta[0],
#                     scale_with_transverse_norm_emitt=(nemitt_x, nemitt_y),
#                     )

scale_transverse_x = np.sqrt(gemitt_x)*r_sigma
scale_transverse_y = np.sqrt(gemitt_y)*r_sigma
part_for_twiss = xp.build_particles(
                    particle_ref=part_on_co, mode='shift',
                    x=  W[0, :4] * scale_transverse_x,
                    px= W[1, :4] * scale_transverse_x,
                    y=  W[2, :4] * scale_transverse_y,
                    py= W[3, :4] * scale_transverse_y,
                    zeta = 0,
                    delta = 0,
                    )
tracker.track(part_for_twiss, turn_by_turn_monitor='ONE_TURN_EBE')

W4 = np.zeros(shape=(4,4,len(s)), dtype=np.float64)
W4[0, :, :] = (tracker.record_last_track.x - x_co) / scale_transverse_x
W4[1, :, :] = (tracker.record_last_track.px - px_co) / scale_transverse_x
W4[2, :, :] = (tracker.record_last_track.y  - y_co) / scale_transverse_y
W4[3, :, :] = (tracker.record_last_track.py - py_co) / scale_transverse_y

betx = W4[0, 0, :]**2 + W4[0, 1, :]**2
bety = W4[2, 2, :]**2 + W4[2, 3, :]**2

gamx = W4[1, 0, :]**2 + W4[1, 1, :]**2
gamy = W4[3, 2, :]**2 + W4[3, 3, :]**2
