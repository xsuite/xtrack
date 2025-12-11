import numpy as np
from scipy.constants import c as clight
from scipy.constants import hbar
from scipy.constants import epsilon_0

from cpymad.madx import Madx
import xtrack as xt
import xobjects as xo

mad = Madx()
mad.call('../../test_data/sps_thick/sps.seq')

# mad.input('beam, particle=proton, pc=26;')
# mad.input('beam, particle=electron, pc=20;')

# # realistic
# mad.input('beam, particle=electron, pc=20;')
# v_mv = 25
# num_turns = 8000

# higher energy
mad.input('beam, particle=electron, pc=50;')
v_mv = 250
num_turns = 600

mad.call('../../test_data/sps_thick/lhc_q20.str')

mad.use(sequence='sps')

# # Some vertical orbit
# mad.sequence.sps.elements['mdv.10107'].kick = 100e-6

mad.input('twiss, table=tw4d;')
twm4d = mad.table.tw4d

mad.sequence.sps.elements['actcse.31632'].volt = v_mv * 10   # To stay in the linear region
mad.sequence.sps.elements['actcse.31632'].freq = 0.3
mad.sequence.sps.elements['actcse.31632'].lag = 0.5


mad.input('twiss, table=tw6d;')
twm6d = mad.table.tw6d

mad.sequence.sps.beam.radiate = True
mad.emit()

line = xt.Line.from_madx_sequence(mad.sequence.sps, allow_thick=True,
                                  deferred_expressions=True)
line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV,
                                    q0=-1, gamma0=mad.sequence.sps.beam.gamma)
line.cycle('bpv.11706', inplace=True)

# line.insert('cav1', obj=line['actcse.31632'].copy(), at='bpv.11706')
# line.insert('cav2', obj=line['actcse.31632'].copy(), at='bpv.21508')
# line.insert('cav4', obj=line['actcse.31632'].copy(), at='bpv.41508')
# line.insert('cav5', obj=line['actcse.31632'].copy(), at='bpv.51508')
# line.insert('cav6', obj=line['actcse.31632'].copy(), at='bpv.61508')

tt = line.get_table()

# Remove edge effects
# for nn in tt.rows[tt.element_type=='DipoleEdge'].name:
#     line[nn].k = 0

tw_thick = line.twiss()

Strategy = xt.slicing.Strategy
Teapot = xt.slicing.Teapot

# line.discard_tracker()
# slicing_strategies = [
#     Strategy(slicing=None),  # Default
#     Strategy(slicing=Teapot(2), element_type=xt.Bend),
#     Strategy(slicing=Teapot(8), element_type=xt.Quadrupole),
# ]

# line.slice_thick_elements(slicing_strategies)
# line.build_tracker()

line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu())
line.configure_radiation(model=None)

# line.vv['vkick'] = 1e-6
# line['mdv.10107'].ksl[0] = line.vars['vkick']
line['qd.10110'].shift_y = 'vkick'

skew_quad_names = ['lqsa.12902', 'lqsa.22902', 'lqsa.32902', 'lqsa.42902',
                   'lqsa.52902', 'lqsa.62902']
knobs_skew_quad = []
for nn in skew_quad_names:
    knob_nn = 'ks.' + nn
    line[knob_nn] = 0.0
    line[nn].k1s = knob_nn
    knobs_skew_quad.append(knob_nn)

match_chrom = True

opt = line.match(
    solve=False,
    vary=[
        xt.VaryList(['kqf', 'kqd'], step=1e-5, tag='tune'),
        xt.Vary('vkick', step=1e-7, tag='orbit'),
        xt.VaryList(['klsda', 'klsdb', 'klsfa', 'klsfb', 'klsfc'], step=1e-4, tag='chrom'),
        xt.VaryList(knobs_skew_quad, step=1e-4, tag='coupling'),
    ],
    targets=[
        xt.TargetSet(qx=20.13, qy=20.18, tol=1e-6, tag='tune'),
        xt.Target(lambda tw: np.std(tw.y), 5e-3, tag='orbit'),
        # xt.TargetSet(dqx=-10, dqy=-10, tol=1e-4, tag='chrom'),
        xt.TargetSet(dqx=0.1, dqy=0.1, tol=1e-4, tag='chrom'),
        xt.TargetSet(c_minus_re=0, c_minus_im=0, at=line.element_names[0],
                     tol=1e-4, tag='coupling'),
        ],
)

opt.step(5)
opt.target_status()

# opt.enable_all_targets()
# opt.enable_all_vary()
# opt.disable_targets(tag='chrom')
# opt.disable_vary(tag='chrom')
# opt.solve()

# if match_chrom:

#     opt.disable_all_targets()
#     opt.disable_all_vary()
#     opt.enable_targets(tag='chrom')
#     opt.enable_vary(tag='chrom')
#     opt.solve()

#     opt.enable_all_targets()
#     opt.enable_all_vary()
#     opt.solve()


tw = line.twiss()
tw4d = line.twiss(method='4d', radiation_integrals=True)

line.configure_radiation(model='mean')

# Tapering!!!
line.compensate_radiation_energy_loss()

tw_rad = line.twiss(eneloss_and_damping=True, method='6d',
                    use_full_inverse=False)
tw_rad2 = line.twiss(eneloss_and_damping=True, method='6d',
                     radiation_method='full')

assert tw_rad.eq_gemitt_x is not None
assert tw_rad.eq_gemitt_y is not None
assert tw_rad.eq_gemitt_zeta is not None

# assert tw_rad2.eq_gemitt_x is None
# assert tw_rad2.eq_gemitt_y is None
# assert tw_rad2.eq_gemitt_zeta is None

ex = tw_rad.eq_nemitt_x / (tw_rad.gamma0 * tw_rad.beta0)
ey = tw_rad.eq_nemitt_y / (tw_rad.gamma0 * tw_rad.beta0)
ez = tw_rad.eq_nemitt_zeta / (tw_rad.gamma0 * tw_rad.beta0)


line.configure_radiation(model='quantum')

p = line.build_particles(num_particles=1000)
p.y = tw_rad.y[0]
p.py = tw_rad.py[0]
p.x = tw_rad.x[0]
p.px = tw_rad.px[0]
line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'), use_prebuilt_kernels=False)
line.track(p, num_turns=num_turns, time=True, turn_by_turn_monitor=True,
           with_progress=2)
print(f'Tracking time: {line.time_last_track}')

# twe = tw.rows[:-1]
# cur_H_x = twe.gamx * twe.dx**2 + 2 * twe.alfx * twe.dx * twe.dpx + twe.betx * twe.dpx**2
# I5_x  = np.sum(cur_H_x * hh**3 * dl)
# I2_x = np.sum(hh**2 * dl)
# I4_x = np.sum(twe.dx * hh**3 * dl) # to be generalized for combined function magnets

# cur_H_y = twe.gamy * twe.dy**2 + 2 * twe.alfy * twe.dy * twe.dpy + twe.bety * twe.dpy**2
# I5_y  = np.sum(cur_H_y * hh**3 * dl)
# I2_y = np.sum(hh**2 * dl)
# I4_y = np.sum(twe.dy * hh**3 * dl) # to be generalized for combined function magnets

# lam_comp = 2.436e-12 # [m]
# ex_hof = 55 * np.sqrt(3) / 96 * lam_comp / 2 / np.pi * gamma0**2 * I5_x / (I2_x - I4_x)
# ey_hof = 55 * np.sqrt(3) / 96 * lam_comp / 2 / np.pi * gamma0**2 * I5_y / (I2_y - I4_y)

mon = line.record_last_track

sigma_tab = tw_rad.get_beam_covariance(gemitt_x=tw_rad.eq_gemitt_x,
                                       gemitt_y=tw_rad.eq_gemitt_y,
                                        gemitt_zeta=tw_rad.eq_gemitt_zeta)
sigma_betatron_tab = tw_rad.get_beam_covariance(gemitt_x=tw_rad.eq_gemitt_x,
                                                gemitt_y=tw_rad.eq_gemitt_y,
                                                gemitt_zeta=0)

import matplotlib.pyplot as plt
plt.close('all')
fig = plt.figure(1, figsize=(4.8*1.3, 6.4))

spo = fig. add_subplot(4, 1, 1)
spo.plot(tw.s, tw.y)
spo.plot(tw_rad.s, tw_rad.y, '--')

plt.xlabel('s [m]')
plt.ylabel('y [m]')

spx = fig. add_subplot(4, 1, 2)
spx.plot(np.std(mon.x, axis=0))
spx.axhline(sigma_tab.sigma_x[0], color='red')
plt.ylabel(r'$\sigma_x$ [m]')
# spx.axhline(np.sqrt(ex_hof * tw.betx[0] + (np.std(p.delta) * tw.dx[0])**2), color='green')

spy = fig. add_subplot(4, 1, 3, sharex=spx)
spy.plot(np.std(mon.y, axis=0))
spy.axhline(sigma_tab.sigma_y[0], color='red')
spy.axhline(sigma_betatron_tab.sigma_y[0], ls='--')
plt.ylabel(r'$\sigma_y$ [m]')
# spy.axhline(np.sqrt(ey_hof * tw.bety[0] + (np.std(p.delta) * tw.dy[0])**2), color='green')

spz = fig. add_subplot(4, 1, 4, sharex=spx)
spz.plot(np.std(mon.zeta, axis=0))
spz.axhline(sigma_tab.sigma_zeta[0], color='red')
plt.ylabel(r'$\sigma_z$ [m]')
plt.xlabel('Turns')

plt.suptitle(f"Qx = {tw.qx:.2f} - Qy = {tw.qy:.2f} - Q'x = {tw.dqx:.2f} - Q'y = {tw.dqy:.2f}")
plt.subplots_adjust(left=.16)

# plot centroid motion
fig2 = plt.figure(2, figsize=(6.4, 4.8))
plt.plot(np.mean(mon.x, axis=0), label='x')
plt.plot(np.mean(mon.y, axis=0), label='y')

plt.show()
