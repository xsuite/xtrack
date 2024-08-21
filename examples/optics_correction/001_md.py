import xtrack as xt
import numpy as np

lhc = xt.Multiline.from_json('mddata/step2.json')

line = lhc.b1
line.twiss_default.update({'strengths': False, 'method': '4d'})
tw0 = line.twiss()


tt = line.get_table()

observable_list = ['betx'] #, 'bety', 'mux', 'muy', 'dx']

obs_points = tt.rows['bpm.*'].name
corr_vars = [nn for nn in line.vars.get_table().rows['kq.*.b1$'].name
              if 'from' not in nn]
corr_elements = []

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
        line.elements[nn].k1 += dk

    twp = line.twiss()

    if corr_type == 'var':
        line.vars[nn] -= dk
    elif corr_type == 'element':
        line.elements[nn].k1 -= dk

    for jj, mm in enumerate(obs_points):
        for observable in observable_list:
            response[observable][jj, ii] = (twp[observable, mm] - tw0[observable, mm]) / dk


line.vars['kq7.r5b1'] *= 1.01
tw = line.twiss()

tw['betx0'] = tw0['betx']
tw['bety0'] = tw0['bety']
tw['mux0'] = tw0['mux']
tw['muy0'] = tw0['muy']

from xtrack.trajectory_correction import _compute_correction

corr_on_observable = 'betx'

err = tw.rows[obs_points][corr_on_observable] - tw0.rows[obs_points][corr_on_observable]
response_matrix = response[corr_on_observable]

correction_svd = _compute_correction(err, response_matrix, rcond=1e-2)
correction_micado = _compute_correction(err, response_matrix, n_micado=1)

i_micado = np.argmax(np.abs(correction_micado))
print(f'MICADO correction: {correction_micado[i_micado]:.2e} at {correctors[i_micado]}')

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(correction_svd, '.', label='SVD')
plt.plot(correction_micado, '.', label='MICADO')

plt.legend()
plt.show()