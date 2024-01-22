import json
import numpy as np
from cpymad.madx import Madx

import xtrack as xt

import matplotlib.pyplot as plt

core = 'full'
edge = 'full'

line_thick = xt.Line.from_json('psb_03_with_chicane_corrected.json')
line_thick.build_tracker()
line_thick.configure_bend_model(core='full', edge='full')
line_thick.vars['on_chicane_beta_corr'] = 0
line_thick.vars['on_chicane_tune_corr'] = 0
line_thick.configure_bend_model(core=core, edge=edge)

line_thin = xt.Line.from_json('psb_04_with_chicane_corrected_thin.json')
line_thin.build_tracker()
line_thin.vars['on_chicane_beta_corr'] = 0
line_thin.vars['on_chicane_tune_corr'] = 0

t_test = np.linspace(0, 6e-3, 100)

qx_thick = []
qy_thick = []
dqx_thick = []
dqy_thick = []
bety_at_scraper_thick = []
qx_thin = []
qy_thin = []
dqx_thin = []
dqy_thin = []
bety_at_scraper_thin = []

qx_ptc = []
qy_ptc = []
dqx_ptc = []
dqy_ptc = []
bety_at_scraper_ptc = []
for ii, tt in enumerate(t_test):
    print(f'Twiss at t = {tt*1e3:.2f} ms   ', end='\r', flush=True)
    line_thick.vars['t_turn_s'] = tt
    line_thin.vars['t_turn_s'] = tt

    tw_thick = line_thick.twiss()
    bety_at_scraper_thick.append(tw_thick['bety', 'br.stscrap22'])
    qx_thick.append(tw_thick.qx)
    qy_thick.append(tw_thick.qy)
    dqx_thick.append(tw_thick.dqx)
    dqy_thick.append(tw_thick.dqy)

    tw_thin = line_thin.twiss()
    bety_at_scraper_thin.append(tw_thin['bety', 'br.stscrap22'])
    qx_thin.append(tw_thin.qx)
    qy_thin.append(tw_thin.qy)
    dqx_thin.append(tw_thin.dqx)
    dqy_thin.append(tw_thin.dqy)

qx_thick = np.array(qx_thick)
qy_thick = np.array(qy_thick)
dqx_thick = np.array(dqx_thick)
dqy_thick = np.array(dqy_thick)
bety_at_scraper_thick = np.array(bety_at_scraper_thick)
qx_thin = np.array(qx_thin)
qy_thin = np.array(qy_thin)
dqx_thin = np.array(dqx_thin)
dqy_thin = np.array(dqy_thin)
bety_at_scraper_thin = np.array(bety_at_scraper_thin)

with open('ptc_ref.json', 'r') as fid:
    ptc_data = json.load(fid)

t_test_ptc = np.array(ptc_data['t_test'])
qx_ptc = np.array(ptc_data['qx_ptc'])
qy_ptc = np.array(ptc_data['qy_ptc'])
dqx_ptc = np.array(ptc_data['dqx_ptc'])
dqy_ptc = np.array(ptc_data['dqy_ptc'])
bety_at_scraper_ptc = np.array(ptc_data['bety_at_scraper_ptc'])

import matplotlib.pyplot as plt
plt.close('all')

fig1 = plt.figure(1)
sp1 = plt.subplot(2,1,1)

plt.plot(t_test*1e3, qy_thick, '-', label='xsuite thick')
plt.plot(t_test*1e3, qy_ptc, '--', label='ptc')
plt.ylabel(r'$Q_y$')


plt.legend()

sp2 = plt.subplot(2,1,2, sharex=sp1)
plt.plot(t_test*1e3, bety_at_scraper_thick, '-', label='xsuite thick')
plt.plot(t_test*1e3, bety_at_scraper_ptc, '--',label='ptc')

plt.legend()

plt.xlabel('time [ms]')
plt.ylabel(r'$\beta_y$ at scraper [m]')
plt.suptitle(f'Xsuite bend model: core={core}, edge={edge}')
plt.savefig(f'q_beta_core_{core}_edge_{edge}.png', dpi=300)

plt.figure(2)
sp1 = plt.subplot(2,1,1, sharex=sp1)
plt.plot(t_test*1e3, dqx_thick, '-', label='xsuite thick')
plt.plot(t_test*1e3, dqx_ptc, '--', label='ptc')
plt.ylim(bottom=-4.5, top=-2.5)
plt.ylabel(r"$Q'_x$")
plt.legend()

sp2 = plt.subplot(2,1,2, sharex=sp1)
plt.plot(t_test*1e3, dqy_thick, '-', label='xsuite thick')
plt.plot(t_test*1e3, dqy_ptc, '--', label='ptc')
plt.ylabel(r"$Q'_y$")
plt.ylim(bottom=-9,top=-7)
plt.legend()

plt.xlabel('time [ms]')
plt.suptitle(f'Xsuite bend model: core={core}, edge={edge}')
plt.savefig(f'q_prime_core_{core}_edge_{edge}.png', dpi=300)

plt.show()

