import numpy as np

import numpy as np

import xtrack as xt
import xdeps as xd

import matplotlib.pyplot as plt

line = xt.Line.from_json('psb_02_with_chicane_time_functions.json')
line.insert_element(element=xt.Marker(), name='mker_match', at_s=79.874)
line.build_tracker()

line.vars['on_chicane_k0'] = 0
line.vars['on_chicane_k2'] = 0
tw0 = line.twiss()
line.vars['on_chicane_k0'] = 1
line.vars['on_chicane_k2'] = 1

###############################
# Correct tunes and beta beat #
###############################

t_correct = np.linspace(0, 5.5e-3, 30)

# LInk vars
# line.vars['kbrqd14corr'] = line.vars['kbrqd3corr']

kbrqf_corr_list = []
kbrqd_corr_list = []
kbrqd3corr_list = []
kbrqd14corr_list = []
for ii, tt in enumerate(t_correct):
    print(f'Correct tune at t = {tt * 1e3:.2f} ms   \n')
    line.vars['t_turn_s'] = tt

    line.match(
        #verbose=True,
        vary=[
            xt.Vary('kbrqfcorr', step=1e-4),
            xt.Vary('kbrqdcorr', step=1e-4),
            xt.Vary('kbrqd3corr', step=1e-4),
            xt.Vary('kbrqd14corr', step=1e-4),
        ],
        targets = [
            xt.Target('qx', value=tw0.qx, tol=1e-5, scale=1),
            xt.Target('qy', value=tw0.qy, tol=1e-5, scale=1),
            xt.Target('bety', at='mker_match',
                      value=tw0['bety', 'mker_match'], tol=1e-4, scale=100),
            xt.Target('alfy', at='mker_match',
                       value=tw0['alfy', 'mker_match'], tol=1e-4, scale=100)
        ]
    )

    kbrqf_corr_list.append(line.vars['kbrqfcorr']._value)
    kbrqd_corr_list.append(line.vars['kbrqdcorr']._value)
    kbrqd3corr_list.append(line.vars['kbrqd3corr']._value)
    kbrqd14corr_list.append(line.vars['kbrqd14corr']._value)

line.functions['fun_kqf_corr'] = xd.FunctionPieceWiseLinear(
    x=t_correct, y=kbrqf_corr_list)
line.functions['fun_kqd_corr'] = xd.FunctionPieceWiseLinear(
    x=t_correct, y=kbrqd_corr_list)
line.functions['fun_qd3_corr'] = xd.FunctionPieceWiseLinear(
    x=t_correct, y=kbrqd3corr_list)
line.functions['fun_qd14_corr'] = xd.FunctionPieceWiseLinear(
    x=t_correct, y=kbrqd14corr_list)

line.vars['on_chicane_tune_corr'] = 1
line.vars['kbrqfcorr'] = (line.vars['on_chicane_tune_corr']
                            * line.functions.fun_kqf_corr(line.vars['t_turn_s']))
line.vars['kbrqdcorr'] = (line.vars['on_chicane_tune_corr']
                            * line.functions.fun_kqd_corr(line.vars['t_turn_s']))

line.vars['on_chicane_beta_corr'] = 1
line.vars['kbrqd3corr'] = (line.vars['on_chicane_beta_corr']
                         * line.functions.fun_qd3_corr(line.vars['t_turn_s']))
line.vars['kbrqd14corr'] = (line.vars['on_chicane_beta_corr']
                        * line.functions.fun_qd14_corr(line.vars['t_turn_s']))

# Save
line.vars['t_turn_s'] = 0
line.to_json('psb_03_with_chicane_corrected.json')


t_test = np.linspace(0, 6e-3, 100)

k0_bsw1 = []
k2l_bsw1 = []
k0_bsw2 = []
k2l_bsw2 = []
qx = []
qy = []
bety_at_mker = []
bety_at_mker_uncorrected = []
qy_uncorrected = []
for ii, tt in enumerate(t_test):
    print(f'Twiss at t = {tt*1e3:.2f} ms   ', end='\r', flush=True)
    line.vars['t_turn_s'] = tt

    line.vars['on_chicane_beta_corr'] = 1
    line.vars['on_chicane_tune_corr'] = 1
    tw = line.twiss()

    qx.append(tw.qx)
    qy.append(tw.qy)
    bety_at_mker.append(tw['bety', 'mker_match'])
    k0_bsw1.append(line['bi1.bsw1l1.1'].k0)
    k2l_bsw1.append(line['bi1.bsw1l1.1'].knl[2])
    k0_bsw2.append(line['bi1.bsw1l1.2'].k0)
    k2l_bsw2.append(line['bi1.bsw1l1.2'].knl[2])

    line.vars['on_chicane_beta_corr'] = 0
    line.vars['on_chicane_tune_corr'] = 0
    tw_uncorr = line.twiss()
    bety_at_mker_uncorrected.append(tw_uncorr['bety', 'mker_match'])
    qy_uncorrected.append(tw_uncorr.qy)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
sp1 = plt.subplot(2,1,1)
plt.plot(t_test*1e3, k0_bsw1, label='k0 bsw1')
plt.plot(t_test*1e3, k0_bsw2, label='k0 bsw2')
plt.legend()
plt.subplot(2,1,2, sharex=sp1)
plt.plot(t_test*1e3, k2l_bsw1, label='k2l bsw1')
plt.plot(t_test*1e3, k2l_bsw2, label='k2l bsw2')
plt.legend()
plt.xlabel('time [ms]')

plt.figure(2)
sp1 = plt.subplot(2,1,1, sharex=sp1)
plt.plot(t_test*1e3, qy, label='qy')
plt.plot(t_test*1e3, qy_uncorrected, label='qy (uncorrected)')
plt.legend()
sp2 = plt.subplot(2,1,2, sharex=sp1)
plt.plot(t_test*1e3, bety_at_mker , label='bety at marker')
plt.plot(t_test*1e3, bety_at_mker_uncorrected, label='bety at marker (uncorrected)')
plt.legend()
plt.xlabel('time [ms]')


plt.figure(3)
sp1 = plt.subplot(2,1,1)
plt.plot(t_correct*1e3, kbrqf_corr_list, label='kbrqfcorr')
plt.plot(t_correct*1e3, kbrqd_corr_list, label='kbrqdcorr')
plt.legend()
plt.subplot(2,1,2, sharex=sp1)
plt.plot(t_correct*1e3, kbrqd3corr_list, label='kbrqd3corr')
plt.plot(t_correct*1e3, kbrqd14corr_list, label='kbrqd14corr')
plt.legend()
plt.xlabel('time [ms]')

plt.show()
