# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xtrack as xt


line = xt.Line.from_json(
    '../../test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json')
line.particle_ref = xt.Particles(
                    mass0=xt.PROTON_MASS_EV, q0=1, energy0=7e12)
line.build_tracker()
line.vars['on_disp'] = 1

tw = line.twiss()

init_at = 'e.cell.45.b1'

x = tw['x', init_at]
y = tw['y', init_at]
px = tw['px', init_at]
py = tw['py', init_at]
zeta = tw['zeta', init_at]
delta = tw['delta', init_at]
betx = tw['betx', init_at]
bety = tw['bety', init_at]
alfx = tw['alfx', init_at]
alfy = tw['alfy', init_at]
dx = tw['dx', init_at]
dy = tw['dy', init_at]
dpx = tw['dpx', init_at]
dpy = tw['dpy', init_at]
mux = tw['mux', init_at]
muy = tw['muy', init_at]
muzeta = tw['muzeta', init_at]
dzeta = tw['dzeta', init_at]
bets = tw.bets0
reference_frame = 'proper'

tw_init = xt.TwissInit(element_name=init_at,
    x=x, px=px, y=y, py=py, zeta=zeta, delta=delta,
    betx=betx, bety=bety, alfx=alfx, alfy=alfy,
    dx=dx, dy=dy, dpx=dpx, dpy=dpy,
    mux=mux, muy=muy, muzeta=muzeta, dzeta=dzeta,
    bets=bets, reference_frame=reference_frame)
tw_test = line.twiss(start=init_at, end='ip6', init=tw_init)


import matplotlib.pyplot as plt

fig1 = plt.figure(1, figsize=(6.4, 4.8*1.5))
spbet = plt.subplot(3,1,1)
spco = plt.subplot(3,1,2, sharex=spbet)
spdisp = plt.subplot(3,1,3, sharex=spbet)

spbet.plot(tw['s'], tw['betx'])
spbet.plot(tw['s'], tw['bety'])
spbet.plot(tw_test['s'], tw_test['betx'], '--')
spbet.plot(tw_test['s'], tw_test['bety'], '--')

spco.plot(tw['s'], tw['x'])
spco.plot(tw['s'], tw['y'])
spco.plot(tw_test['s'], tw_test['x'], '--')
spco.plot(tw_test['s'], tw_test['y'], '--')

spdisp.plot(tw['s'], tw['dx'])
spdisp.plot(tw['s'], tw['dy'])
spdisp.plot(tw_test['s'], tw_test['dx'], '--')
spdisp.plot(tw_test['s'], tw_test['dy'], '--')

spbet.set_ylabel(r'$\beta_{x,y}$ [m]')
spco.set_ylabel(r'(Closed orbit)$_{x,y}$ [m]')
spdisp.set_ylabel(r'$D_{x,y}$ [m]')
spdisp.set_xlabel('s [m]')

assert tw_test.name[-1] == '_end_point'
tw_part = tw.rows['e.cell.45.b1':'ip6']

tw_test = tw_test.rows[:-1]
assert np.all(tw_test.name == tw_part.name)

atols = dict(
    alfx=1e-8, alfy=1e-8,
    dzeta=1e-3, dx=1e-4, dy=1e-4, dpx=1e-5, dpy=1e-5,
    nuzeta=1e-5, dx_zeta=1e-4, dy_zeta=1e-4, betx2=1e-3, bety1=1e-3,
    muzeta=1e-7,
)

rtols = dict(
    alfx=5e-9, alfy=5e-8,
    betx=1e-8, bety=1e-8, betx1=1e-8, bety2=1e-8,
    gamx=1e-8, gamy=1e-8,
)

atol_default = 1e-11
rtol_default = 1e-9


for kk in tw_test._data.keys():
    if kk in ['name', 'W_matrix', 'particle_on_co', 'values_at', 'method',
            'radiation_method', 'reference_frame', 'orientation']:
        continue # tested separately
    atol = atols.get(kk, atol_default)
    rtol = rtols.get(kk, rtol_default)
    assert np.allclose(
        tw_test._data[kk], tw_part._data[kk], rtol=rtol, atol=atol)

assert tw_test.values_at == tw_part.values_at == 'entry'
assert tw_test.radiation_method == tw_part.radiation_method == 'full'
assert tw_test.reference_frame == tw_part.reference_frame == 'proper'

W_matrix_part = tw_part.W_matrix
W_matrix_test = tw_test.W_matrix

for ss in range(W_matrix_part.shape[0]):
    this_part = W_matrix_part[ss, :, :]
    this_test = W_matrix_test[ss, :, :]

    for ii in range(4):
        assert np.isclose((np.linalg.norm(this_part[ii, :] - this_test[ii, :])
                        /np.linalg.norm(this_part[ii, :])), 0, atol=3e-4)

plt.show()

