from cpymad.madx import Madx
import numpy as np

import xtrack as xt

test_data_folder = '../../test_data/'
mad = Madx()

mad.call(test_data_folder + 'pimms/PIMM.seq')
mad.call(test_data_folder + 'pimms/betatron.str')
mad.beam(particle='proton', gamma=1.21315778) # 200 MeV
mad.use('pimms')
seq = mad.sequence.pimms
def_expr = True

line = xt.Line.from_madx_sequence(seq, deferred_expressions=def_expr)
line.particle_ref = xt.Particles(gamma0=seq.beam.gamma,
                                 mass0=seq.beam.mass * 1e9,
                                 q0=seq.beam.charge)
line.configure_bend_model(core='full', edge='full')

line.insert_element('mysext', xt.Sextupole(length=0.2), at_s=36.)

line.insert_element(
            'septum',
            xt.LimitRect(min_x=-0.1, max_x=0.1, min_y=-0.1, max_y=0.1),
            index='pimms_start')


line.vars['k2mysext'] = 0
line.element_refs['mysext'].k2 = line.vars['k2mysext']

tw0 = line.twiss(method='4d')

line.vars['k2xrr'] = 0
opt = line.match(
    solve=False,
    method='4d',
    vary=[
        xt.VaryList(['qf1k1', 'qd1k1', 'qf2k1'], step=1e-3, tag='quad'),
        xt.VaryList(['k2xcf', 'k2xcd'], step=1e-3, tag='sext'),
    ],
    targets=[
        xt.TargetSet(qx=1.665, qy=1.72, tol=1e-6, tag='tunes'),
        xt.TargetSet(dqx=-0.1, dqy=-0.1, tol=1e-3, tag="chrom"),
        xt.Target(dx=0, tol=1e-3, at='pimms_start', tag='disp'),
    ]
)
opt.disable_targets(tag='chrom')
opt.disable_vary(tag='sext')
opt.solve()
opt.enable_all_targets()
opt.enable_all_vary()
opt.solve()

tw1 = line.twiss(method='4d')

line.vars['k2xrr_extr'] = 8.65
line.vars['k2xrr'] = line.vars['k2xrr_extr']
tw2 = line.twiss(method='4d')


import matplotlib.pyplot as plt
plt.close('all')
plt.figure(33)
ax1 = plt.subplot(2, 1, 1)
plt.plot(tw0.s, tw0.betx, '.-')
plt.plot(tw1.s, tw1.betx, '.-')
plt.plot(tw0.s, tw0.bety, '.-')
plt.plot(tw1.s, tw1.bety, '.-')
plt.ylabel(r'$\beta$ [m]')

plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(tw0.s, tw0.dx, '.-')
plt.plot(tw1.s, tw1.dx, '.-')

class ActionSeparatrix(xt.Action):

    def __init__(self, line):
        self.line = line

    def run(self):
        line = self.line
        tw = line.twiss(method='4d')

        p_test = line.build_particles(x=5e-3, px=0)
        line.track(p_test, num_turns=10000, turn_by_turn_monitor=True)
        mon_test = line.record_last_track
        norm_coord_test = tw.get_normalized_coordinates(mon_test)

        x_t = mon_test.x[:]
        px_t = mon_test.px[:]
        x_norm_t = norm_coord_test.x_norm[:]
        px_norm_t = norm_coord_test.px_norm[:]

        # Select branch closer to the x-axis
        mask_branch = (x_norm_t > 0) & (px_norm_t > -2 * x_norm_t) & (px_norm_t < 2 * x_norm_t)
        x_branch = x_t[mask_branch]
        px_branch = px_t[mask_branch]
        x_norm_branch = x_norm_t[mask_branch]
        px_norm_branch = px_norm_t[mask_branch]

        mask_fit = (x_branch > 0.01) & (x_branch < 0.02)
        poly_geom = np.polyfit(x_branch[mask_fit], px_branch[mask_fit], 1)
        poly_norm = np.polyfit(x_norm_branch[mask_fit], px_norm_branch[mask_fit], 1)

        r_sep_norm = np.abs(poly_norm[1]) / np.sqrt(poly_norm[0]**2 + 1)

        out = {
            'poly_geom': poly_geom,
            'poly_norm': poly_norm,
            'r_sep_norm': r_sep_norm,
            'slope': poly_geom[0],
        }

        return out

action_sep = ActionSeparatrix(line)
res = action_sep.run()
tw = line.twiss(method='4d')

num_particles = 5000
x_norm = np.random.normal(size=num_particles)
px_norm = np.random.normal(size=num_particles)
y_norm = np.random.normal(size=num_particles)
py_norm = np.random.normal(size=num_particles)
delta = 5e-4 * np.random.normal(size=num_particles)
particles = line.build_particles(
    weight=1e10/num_particles,
    method='4d',
    nemitt_x=1e-6/10, nemitt_y=1e-6/10, # !!!!! 
    x_norm=x_norm, px_norm=px_norm, y_norm=y_norm, py_norm=py_norm,
    delta=delta)
tab = tw.get_normalized_coordinates(particles)

poly_geom = res['poly_geom']
poly_norm = res['poly_norm']
x_fit_geom = np.linspace(-0.2, 0.2, 10)
px_fit_geom = poly_geom[0] * x_fit_geom + poly_geom[1]
x_fit_norm = np.linspace(-0.2, 0.2, 10)
px_fit_norm = poly_norm[0] * x_fit_norm + poly_norm[1]

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(100)
plt.plot(x_fit_geom, px_fit_geom, 'grey')
plt.plot(particles.x, particles.px, '.', markersize=1)
plt.ylabel(r'$p_x$')
plt.xlabel(r'$x$ [m]')

plt.figure(101)
plt.plot(x_fit_norm, px_fit_norm, 'grey')
plt.plot(tab.x_norm, tab.px_norm, '.', markersize=1)
plt.axis('equal')


p = line.build_particles(
    method='4d', x=np.linspace(0, 2e-2, 30), px=0, y=0, py=0)

line.track(p, num_turns=10000, turn_by_turn_monitor=True, time=True)
mon = line.record_last_track
norm_coord = tw.get_normalized_coordinates(mon)

plt.figure(200)
plt.plot(mon.x.T, mon.px.T, '.', markersize=1)
# plt.plot(x_branch, px_branch, '.k', markersize=3)
plt.plot(x_fit_geom, px_fit_geom, 'grey')
plt.ylabel(r'$p_x$')
plt.xlabel(r'$x$ [m]')

plt.figure(201)
plt.plot(norm_coord.x_norm.T, norm_coord.px_norm.T, '.', markersize=1)
# plt.plot(x_norm_branch, px_norm_branch, '.k', markersize=3)
plt.plot(x_fit_norm, px_fit_norm, 'grey')
plt.axis('equal')

line.discard_tracker()

class SpillExcitation:
    def __init__(self):
        self.intensity = []
        self.amplitude = 1e-6
        self.gain = 0.
        self.amplitude_max = 100e-6
        self.target_rate = 0.98e10/ 15000
        self.n_ave = 100
        self._i_turn = 0

        self._amplitude_log = []
        self._rate_log = []
        self._gain_log = []

    def track(self, p):
        self.intensity.append(np.sum(p.weight[p.state > 0]))
        i_turn_0 = self._i_turn - self.n_ave
        if i_turn_0 < 0:
            i_turn_0 = 0
        rate = -(self.intensity[self._i_turn] - self.intensity[i_turn_0]) / (self._i_turn - i_turn_0)
        if self._i_turn > 10:
            self.amplitude -= self.amplitude * self.gain * (rate - self.target_rate)/self.target_rate
        if self.amplitude > self.amplitude_max:
            self.amplitude = self.amplitude_max
        if self.amplitude < 0:
            self.amplitude = 1e-7
        self._amplitude_log.append(self.amplitude)
        self._rate_log.append(rate)
        self._gain_log.append(self.gain)
        p.px[p.state > 0] += self.amplitude * np.random.normal(size=np.sum(p.state > 0))
        self._i_turn += 1

line.insert_element('spill_exc', SpillExcitation(), at_s=0)
import xobjects as xo
line.build_tracker(_context=xo.ContextCpu('auto'))

line.functions['fun_xsext'] = xt.FunctionPieceWiseLinear(x=[0, 0.5e-3], y=[0, 1.])
line.vars['k2xrr'] = line.vars['k2xrr_extr'] * line.functions['fun_xsext'](line.vars['t_turn_s'])

line.functions['fun_gain'] = xt.FunctionPieceWiseLinear(x=[0, 0.25e-3, 0.5e-3], y=[0, 0, .001])
line.vars['gain'] = line.functions['fun_gain'](line.vars['t_turn_s'])
line.element_refs['spill_exc'].gain = line.vars['gain']

line['septum'].max_x = 0.02

line.enable_time_dependent_vars = True
line.track(particles, num_turns=15000, with_progress=True)

plt.figure(1000)
ax1 = plt.subplot(4,1,1)
plt.plot(line['spill_exc']._amplitude_log)

ax2 = plt.subplot(4,1,2, sharex=ax1)
plt.plot(line['spill_exc'].intensity)
plt.ylim(bottom=0)

ax3 = plt.subplot(4,1,3, sharex=ax1)
plt.plot(line['spill_exc']._rate_log)
plt.axhline(line['spill_exc'].target_rate, color='grey')

ax4 = plt.subplot(4,1,4, sharex=ax1)
ax4b = ax4.twinx()
plt.plot(line['spill_exc']._gain_log)

plt.figure(1001)
plt.plot(particles.x, particles.px, '.', markersize=2)
plt.plot(x_fit_geom, px_fit_geom, 'grey')
plt.show()