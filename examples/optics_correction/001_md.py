import xtrack as xt
import numpy as np

lhc = xt.Environment.from_json('mddata/step2.json')

line = lhc.b1
line.twiss_default.update({'strengths': False, 'method': '4d'})
tw0 = line.twiss()


tt = line.get_table()

observable_list = ['betx', 'bety', 'mux', 'muy', 'dx']

# obs_points = tt.rows['bpm.*'].name
import tfs
meas_betx = tfs.read_tfs('mddata/beta_phase_x.tfs')
obs_points_import = list(map(lambda ss: ss.lower(), list(meas_betx['NAME'].values)))
# Resort
obs_points = tw0.rows[obs_points_import].name

corr_elements = []
corr_vars = [nn for nn in line.vars.get_table().rows['kq.*.b1$'].name
              if 'from' not in nn]

corr_vars = corr_vars[:10] # DEBUUUUUUUG

# corr_vars = []
# corr_elements = [nn for nn in list(tt.rows[tt.element_type == 'Quadrupole'].name)
#                  if 'mqm' in nn]

correctors = []
for nn in corr_vars:
    correctors.append(('var', nn))
for nn in corr_elements:
    correctors.append(('element', nn))

response = {oo: np.zeros((len(obs_points), len(correctors)))
            for oo in observable_list}

dk = 0.5e-5
for ii, cc in enumerate(correctors):
    nn = cc[1]
    corr_type = cc[0]
    print(f'Processing {ii}/{len(correctors)}')
    if corr_type == 'var':
        line.vars[nn] += dk
    elif corr_type == 'element':
        line[nn].k1 += dk

    twp = line.twiss()

    if corr_type == 'var':
        line.vars[nn] -= dk
    elif corr_type == 'element':
        line[nn].k1 -= dk

    for observable in observable_list:
        response[observable][:, ii] = (
            twp.rows[obs_points][observable] - tw0.rows[obs_points][observable]) / dk


# line.vars['kq7.r5b1'] *= 1.01
# tw = line.twiss()
tw_meas = xt.Table({'name': np.array(obs_points_import), 'betx': meas_betx['BETX']._values})

tw = xt.Table({'name': np.array(obs_points),
               'betx': np.array([tw_meas['betx', nn] for nn in obs_points])})

from xtrack.trajectory_correction import _compute_correction

correct_on_observables = ['betx']

err = None
response_matrix = None
for cc in correct_on_observables:
    ee = tw.rows[obs_points][cc] - tw0.rows[obs_points][cc]
    rr = response[cc]

    if err is None:
        err = ee
        response_matrix = rr
    else:
        err = np.concatenate((err, ee))
        response_matrix = np.concatenate((response_matrix, rr), axis=0)

correction_svd = _compute_correction(err, response_matrix, rcond=1e-2)
correction_micado = _compute_correction(err, response_matrix, n_micado=2)

i_micado = np.argmax(np.abs(correction_micado))
print(f'MICADO correction: {correction_micado[i_micado]:.2e} at {correctors[i_micado]}')

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(correction_svd, '.', label='SVD')
plt.plot(correction_micado, '.', label='MICADO')

plt.figure(2)
plt.plot(tw0.rows[obs_points].s, tw.betx / tw0.rows[obs_points].betx -1)

plt.legend()
plt.show()