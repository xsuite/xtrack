import numpy as np
import xtrack as xt

line = xt.Line.from_json('fccee_p_ring_thin.json')

line.build_tracker()

for ee in line.elements:
    if isinstance(ee, xt.Cavity):
        ee.voltage *= 100
        ee.frequency /= 100

line.vars['on_wiggler_h'] = 0
line.vars['on_wiggler_v'] = 0.

tt = line.get_table()

for tag_wig in ['a', 'd', 'g', 'j']:

    s_start_wig = tt['s', f'mwi.a4r{tag_wig}_entry']
    s_end_wig = tt['s', f'mwi.i4r{tag_wig}_exit']

    s_wig_kicks = np.linspace(s_start_wig, s_end_wig, 100)
    s_wig_plus = s_wig_kicks[:-1:2]
    s_wig_minus = s_wig_kicks[1::2]

    line.discard_tracker()

    kk_wig = f'k0l_wig.r{tag_wig}'
    line.vars[kk_wig] = 0
    for ii, (ss_p, ss_m) in enumerate(zip(s_wig_plus, s_wig_minus)):
        print(f'Wig {tag_wig} period {ii}/{len(s_wig_plus)}', end='\r', flush=True)
        wmg_plus = xt.Multipole(knl=[0])
        wmg_minus = xt.Multipole(knl=[0])
        wmg_plus.length = ss_m - ss_p
        wmg_minus.length = ss_m - ss_p
        if ii == 0:
            ss_p += 0.5 * wmg_plus.length
        if ii == len(s_wig_plus) - 1:
            ss_m -= 0.5 * wmg_plus.length
        nn_plus = f'mwg.r{tag_wig}.plus.{ii}'
        nn_minus = f'mwg.r{tag_wig}.minus.{ii}'
        line.insert_element(nn_plus, wmg_plus, at_s=ss_p)
        line.insert_element(nn_minus, wmg_minus, at_s=ss_m)
        line.element_refs[nn_plus].ksl[0] = line.vars[kk_wig]
        line.element_refs[nn_minus].ksl[0] = -line.vars[kk_wig]
        if ii == 0:
            line.element_refs[nn_plus].ksl[0] = line.vars[kk_wig] * 0.5
        if ii == len(s_wig_plus) - 1:
            line.element_refs[nn_minus].ksl[0] = -line.vars[kk_wig] * 0.5

line.vars['k0wig'] = 0
line.vars['k0l_wig.ra'] = line.vars['k0wig']
line.vars['k0l_wig.rd'] = line.vars['k0wig']
line.vars['k0l_wig.rg'] = line.vars['k0wig']
line.vars['k0l_wig.rj'] = line.vars['k0wig']

line.build_tracker()

tw_no_rad = line.twiss(method='4d')

line.to_json('fccee_p_ring_thin_wig.json')
prrrr

line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()

tw_rad = line.twiss(eneloss_and_damping=True)
ex = tw_rad.nemitt_x_rad / (tw_rad.gamma0 * tw_rad.beta0)
ey = tw_rad.nemitt_y_rad / (tw_rad.gamma0 * tw_rad.beta0)
ez = tw_rad.nemitt_zeta_rad / (tw_rad.gamma0 * tw_rad.beta0)


line.configure_radiation(model='quantum')
p = line.build_particles(num_particles=30)
line.track(p, num_turns=400, turn_by_turn_monitor=True, time=True)
print(f'Tracking time: {line.time_last_track}')


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


