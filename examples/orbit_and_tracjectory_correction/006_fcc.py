import xtrack as xt
import numpy as np
import time

line = xt.Line.from_json('../../test_data/fcc_ee/fccee_h_thin.json')

tt = line.get_table()

tt_entry_quads = tt.rows['q.*\.\.0']

insertions = []
line.env.new('corrector', xt.Multipole, knl=[0])
for ii, name_quad in enumerate(tt_entry_quads.name):
    if ii%20 == 0:
        print(f'Processing {ii}/{len(tt_entry_quads)}')
    insertions.append(line.env.new(f'bpm_{ii}', xt.Marker, at=name_quad+'@start'))
    insertions.append(line.env.new(f'corrector_{ii}', 'corrector')) # after bpm
t1 = time.time()
line.insert(insertions)
t2 = time.time()
print(f't_insert = {t2-t1:.2f} s')

tt = line.get_table()
tt_bpm = tt.rows['bpm.*']

tt_h_correctors = tt.rows['corrector.*']
line.steering_correctors_x = tt_h_correctors.name
tt_v_correctors = tt.rows['corrector.*']
line.steering_correctors_y = tt_v_correctors.name

tt_monitors = tt.rows['bpm.*']
line.steering_monitors_x = tt_monitors.name
line.steering_monitors_y = tt_monitors.name

tw_ref = line.twiss4d()

orbit_correction = line.correct_trajectory(twiss_table=tw_ref, run=False)

# Inspect singular values of the response matrices
x_sv = orbit_correction.x_correction.singular_values
y_sv = orbit_correction.y_correction.singular_values

line['qc1r1.1'].knl[0] = 1e-6
line['qc1r1.1'].num_multipole_kicks = 1

t0 = time.time()
orbit_correction.correct()
t1 = time.time()
print(f'Correction time: {t1-t0} s')

# import matplotlib.pyplot as plt
# plt.close('all')

# fig2 = plt.figure(1)
# plt.semilogy(np.abs(x_sv), '.-', label='x')
# plt.semilogy(np.abs(y_sv), '.-', label='y')
# plt.legend()
# plt.xlabel('mode')
# plt.ylabel('singular value (modulus)')

# plt.show()
