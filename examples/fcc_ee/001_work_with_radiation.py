import numpy as np
import xtrack as xt

line = xt.Line.from_json('fccee_p_ring_thin.json')

line.build_tracker()

line.vars['on_wiggler_h'] = 0
line.vars['on_wiggler_v'] = 0.

for ee in line.elements:
    if isinstance(ee, xt.Multipole):
        ee.hyl = 0
        ee.ksl[:] = 0

# line.configure_bend_model(edge='suppressed')

tw_before = line.twiss()
tob = line.twiss(
                method='4d',
                ele_start='fccee_p_ring$start', ele_stop=len(line) - 1,
                twiss_init=xt.TwissInit(betx=tw_before.betx[0],
                                        bety=tw_before.bety[0],
                                        alfx=tw_before.alfx[0],
                                        alfy=tw_before.alfy[0],
                                        dx=tw_before.dx[0],
                                        dpx=tw_before.dpx[0],
                                        dy=tw_before.dy[0],
                                        dpy=tw_before.dpy[0]))

for ee in line.elements:
    if isinstance(ee, xt.Multipole):
        knl = ee.knl.copy()
        ksl = ee.ksl.copy()
        hxl = ee.hxl
        hyl = ee.hyl
        # knl[1:2:] *= -1
        # ksl[0:2:] *= -1
        ee.hxl = 0
        ee.hyl = hxl

        ee.knl[0] = 0
        ee.ksl[0] = knl[0]
        if len(knl) > 1:
            ee.knl[1] = -knl[1]
            ee.ksl[1] = 0
        if len(knl) > 2:
            ee.knl[2] = 0
            ee.ksl[2] = -knl[2]

    if isinstance(ee, xt.DipoleEdge):
        ee._r21 = ee._r43
        ee._r43 = -ee._r21

# line['qrf.2..0'].ksl[0] = 1e-6

to = line.twiss(
                method='4d',
                ele_start='fccee_p_ring$start', ele_stop=len(line) - 1,
                twiss_init=xt.TwissInit(betx=tw_before.bety[0],
                                        bety=tw_before.betx[0],
                                        alfx=tw_before.alfy[0],
                                        alfy=tw_before.alfx[0],
                                        dx=tw_before.dy[0],
                                        dpx=tw_before.dpy[0],
                                        dy=tw_before.dx[0],
                                        dpy=tw_before.dpx[0]),
                _continue_if_lost=True)

tw_no_rad = line.twiss(method='4d')

line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()

tw_rad = line.twiss(eneloss_and_damping=True)

line.configure_radiation(model='quantum')
p = line.build_particles(num_particles=30)
line.track(p, num_turns=1000, turn_by_turn_monitor=True, time=True)
print(f'Tracking time: {line.time_last_track}')

ex = tw_rad.nemitt_x_rad / (tw_rad.gamma0 * tw_rad.beta0)
ey = tw_rad.nemitt_y_rad / (tw_rad.gamma0 * tw_rad.beta0)
ez = tw_rad.nemitt_zeta_rad / (tw_rad.gamma0 * tw_rad.beta0)

dl = line.tracker._tracker_data_base.cache['dl_radiation']
hxl = line.tracker._tracker_data_base.cache['hxl_radiation']
hyl = line.tracker._tracker_data_base.cache['hyl_radiation']
px_co = tw_rad.px
py_co = tw_rad.py

mask = (dl != 0)
hx = np.zeros(shape=(len(dl),), dtype=np.float64)
hy = np.zeros(shape=(len(dl),), dtype=np.float64)
hx[mask] = (np.diff(px_co)[mask] + hxl[mask]) / dl[mask]
hy[mask] = (np.diff(py_co)[mask] + hyl[mask]) / dl[mask]
hh = np.sqrt(hx**2 + hy**2)

twe = tw_rad.rows[:-1]
cur_H_x = twe.gamx * twe.dx**2 + 2 * twe.alfx * twe.dx * twe.dpx + twe.betx * twe.dpx**2
I5_x  = np.sum(cur_H_x * hh**3 * dl)
I2_x = np.sum(hh**2 * dl)
I4_x = np.sum(twe.dx * hh**3 * dl) # to be generalized for combined function magnets

cur_H_y = twe.gamy * twe.dy**2 + 2 * twe.alfy * twe.dy * twe.dpy + twe.bety * twe.dpy**2
I5_y  = np.sum(cur_H_y * hh**3 * dl)
I2_y = np.sum(hh**2 * dl)
I4_y = np.sum(twe.dy * hh**3 * dl) # to be generalized for combined function magnets

lam_comp = 2.436e-12 # [m]
gamma0 = tw_rad.gamma0
ex_hof = 55 * np.sqrt(3) / 96 * lam_comp / 2 / np.pi * gamma0**2 * I5_x / (I2_x - I4_x)
ey_hof = 55 * np.sqrt(3) / 96 * lam_comp / 2 / np.pi * gamma0**2 * I5_y / (I2_y - I4_y)

mon = line.record_last_track

import matplotlib.pyplot as plt
plt.close('all')
fig = plt.figure(1)
spx = fig. add_subplot(3, 1, 1)
spx.plot(np.std(mon.x, axis=0))
spx.axhline(np.sqrt(ex * tw_rad.betx[0] + ey * tw_rad.betx2[0] + (np.std(p.delta) * tw_rad.dx[0])**2), color='red')
# spx.axhline(np.sqrt(ex_hof * tw.betx[0] + (np.std(p.delta) * tw.dx[0])**2), color='green')

spy = fig. add_subplot(3, 1, 2, sharex=spx)
spy.plot(np.std(mon.y, axis=0))
spy.axhline(np.sqrt(ex * tw_rad.bety1[0] + ey * tw_rad.bety[0] + (np.std(p.delta) * tw_rad.dy[0])**2), color='red')
# spy.axhline(np.sqrt(ey_hof * tw.bety[0] + (np.std(p.delta) * tw.dy[0])**2), color='green')

spz = fig. add_subplot(3, 1, 3, sharex=spx)
spz.plot(np.std(mon.zeta, axis=0))
spz.axhline(np.sqrt(ez * tw_rad.betz0), color='red')

plt.show()


# # Compare against madx emit

# from cpymad.madx import Madx
# mad = Madx()
# mad.call('fccee_h.seq')
# mad.beam(particle='positron', pc=120)
# mad.use('fccee_p_ring')
# mad.input('twiss, table=tw_no_wig;')

# mad.call('install_wigglers.madx')
# mad.input("exec, define_wigglers_as_kickers()")
# mad.input("exec, install_wigglers()")
# mad.use('fccee_p_ring')

# mad.globals.on_wiggler_v = line.vars['on_wiggler_v']._value

# mad.sequence.fccee_p_ring.beam.radiate = True

# mad.input('twiss, tapering')
# twm = mad.table.twiss

# mad.emit()


