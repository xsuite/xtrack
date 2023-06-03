# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xtrack as xt
import xpart as xp


line = xt.Line.from_json(
    '../../test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json')
line.particle_ref = xp.Particles(
                    mass0=xp.PROTON_MASS_EV, q0=1, energy0=7e12)
line.build_tracker()

tw = line.twiss()

ele_init = 'e.ds.l4.b1'

x = tw['x', ele_init]
y = tw['y', ele_init]
px = tw['px', ele_init]
py = tw['py', ele_init]
zeta = tw['zeta', ele_init]
delta = tw['delta', ele_init]
betx = tw['betx', ele_init]
bety = tw['bety', ele_init]
alfx = tw['alfx', ele_init]
alfy = tw['alfy', ele_init]
dx = tw['dx', ele_init]
dy = tw['dy', ele_init]
dpx = tw['dpx', ele_init]
dpy = tw['dpy', ele_init]
mux = tw['mux', ele_init]
muy = tw['muy', ele_init]
muzeta = tw['muzeta', ele_init]
dzeta = tw['dzeta', ele_init]
bets = tw.betz0
reference_frame = 'proper'

tw_init = line.build_twiss_init(element_name=ele_init,
    x=x, px=px, y=y, py=py, zeta=zeta, delta=delta,
    betx=betx, bety=bety, alfx=alfx, alfy=alfy,
    dx=dx, dy=dy, dpx=dpx, dpy=dpy,
    mux=mux, muy=muy, muzeta=muzeta, dzeta=dzeta,
    bets=bets, reference_frame=reference_frame)

tw_check = line.twiss(ele_start=ele_init, ele_stop='ip6', twiss_init=tw_init)

