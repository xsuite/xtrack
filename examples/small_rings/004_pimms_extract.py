from cpymad.madx import Madx
import numpy as np

import xtrack as xt

class ActionSeparatrix(xt.Action):

    def __init__(self, line, n_test=20, range_test=(0, 1e-2),
                             range_fit=(0.015, 0.025),
                             i_part_fit=None,
                             num_turns=2000,
                             x_spiral_meas=None):
        self.line = line
        self.n_test = n_test
        self.range_test = range_test
        self.range_fit = range_fit
        self.i_part_fit = i_part_fit
        self.num_turns = num_turns
        self.x_spiral_meas = x_spiral_meas

    def run(self):
        line = self.line
        tw = line.twiss(method='4d')

        p_test = line.build_particles(x=np.linspace(self.range_test[0],
                                                    self.range_test[1],
                                                    self.n_test), px=0)
        line.track(p_test, num_turns=self.num_turns, turn_by_turn_monitor=True)
        mon_test = line.record_last_track
        norm_coord_test = tw.get_normalized_coordinates(mon_test)

        message = 'all ok'
        try:
            i_first_unstable = np.where(mon_test.x.max(axis=1) > 0.02)[0][0] # first unstable particle
            if self.i_part_fit is None:
                i_part = i_first_unstable # first unstable particle
            else:
                i_part = self.i_part_fit
            # i_part = len(p_test.x) - 1

            x_t = mon_test.x[i_part, :]
            px_t = mon_test.px[i_part, :]
            x_norm_t = norm_coord_test.x_norm[i_part, :]
            px_norm_t = norm_coord_test.px_norm[i_part, :]

            # Select branch closer to the x-axis
            # mask_branch = (x_norm_t > 0) & (px_norm_t > -2 * x_norm_t) & (px_norm_t < 2 * x_norm_t)
            mask_branch = (x_norm_t > 0) & (px_norm_t > 0)
            if not mask_branch.any():
                message = 'no branch'
            x_branch = x_t[mask_branch]
            px_branch = px_t[mask_branch]
            x_norm_branch = x_norm_t[mask_branch]
            px_norm_branch = px_norm_t[mask_branch]

            # for when separatrix makes a closed loop
            i_first_loop = np.where(x_branch>self.range_fit[1])[0][0]

            mask_fit = (x_branch > self.range_fit[0]) & (x_branch < self.range_fit[1])
            mask_fit[i_first_loop:] = False
            if not mask_fit.any():
                message = 'no fit'
            poly_geom = np.polyfit(x_branch[mask_fit], px_branch[mask_fit], 1)
            poly_norm = np.polyfit(x_norm_branch[mask_fit], px_norm_branch[mask_fit], 1)

            r_sep_norm = np.abs(poly_norm[1]) / np.sqrt(poly_norm[0]**2 + 1)

            if self.x_spiral_meas is not None:
                x_first_unstable = mon_test.x[i_first_unstable, :]
                i_closest = np.argmin(np.abs(x_first_unstable - self.x_spiral_meas))
                dx_spiral = (mon_test.x[i_first_unstable, i_closest + 3]
                             - mon_test.x[i_first_unstable, i_closest])
                dpx_spiral = (mon_test.px[i_first_unstable, i_closest + 3]
                             - mon_test.px[i_first_unstable, i_closest])
                djx_spiral = (
                    norm_coord_test.x_norm[i_first_unstable, i_closest + 3]**2
                  + norm_coord_test.px_norm[i_first_unstable, i_closest + 3]**2
                  - norm_coord_test.x_norm[i_first_unstable, i_closest]**2
                  - norm_coord_test.px_norm[i_first_unstable, i_closest]**2)
                slope_norm_spiral = (norm_coord_test.px_norm[i_first_unstable, i_closest]
                                     / norm_coord_test.x_norm[i_first_unstable, i_closest])
            else:
                dx_spiral = 999.
                dpx_spiral = 999.
                djx_spiral = 999.
                slope_norm_spiral = 999.

            out = {
                'poly_geom': poly_geom,
                'poly_norm': poly_norm,
                'r_sep_norm': r_sep_norm,
                'slope_geom': poly_geom[0],
                'slope_norm': poly_norm[0],
                'n_fit': np.sum(mask_fit),
                'mon' : mon_test,
                'norm_coord': norm_coord_test,
                'i_part': i_part,
                'message': message,
                'x_fit': x_branch[mask_fit],
                'px_fit': px_branch[mask_fit],
                'dx_spiral': dx_spiral,
                'dpx_spiral': dpx_spiral,
                'djx_spiral': djx_spiral,
                'slope_norm_spiral': slope_norm_spiral,
            }
        except Exception as err:
            print(err)
            out = {
                'poly_geom': [0, 0],
                'poly_norm': [0, 0],
                'r_sep_norm': 999.,
                'slope_geom': 999.,
                'slope_norm': 999.,
                'n_fit': 0,
                'mon' : mon_test,
                'norm_coord': norm_coord_test,
                'i_part': 0,
                'message': message,
            }

        return out

def plot_res(res, title=None):
    mon = res['mon']
    norm_coord = res['norm_coord']

    plt.figure(figsize=(10, 5))
    ax_geom = plt.subplot(1, 2, 1)
    plt.plot(mon.x.T, mon.px.T, '.', markersize=1)
    # plt.plot(x_branch, px_branch, '.k', markersize=3)

    plt.ylabel(r'$p_x$')
    plt.xlabel(r'$x$ [m]')

    ax_norm = plt.subplot(1, 2, 2)
    plt.plot(norm_coord.x_norm.T, norm_coord.px_norm.T, '.', markersize=1)
    # plt.plot(x_norm_branch, px_norm_branch, '.k', markersize=3)
    plt.axis('equal')

    poly_geom = res['poly_geom']
    poly_norm = res['poly_norm']
    x_fit_geom = np.linspace(-0.2, 0.2, 10)
    px_fit_geom = poly_geom[0] * x_fit_geom + poly_geom[1]
    x_fit_norm = np.linspace(-0.2, 0.2, 10)
    px_fit_norm = poly_norm[0] * x_fit_norm + poly_norm[1]

    ax_geom.plot(x_fit_geom, px_fit_geom, 'grey')
    ax_norm.plot(x_fit_norm, px_fit_norm, 'grey')

    if title is not None:
        plt.suptitle(title)

# -----------------------------------------------------------------------------

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


line.insert_element(
            'septum',
            xt.LimitRect(min_x=-0.1, max_x=0.1, min_y=-0.1, max_y=0.1),
            index='pimms_start')


tw0 = line.twiss(method='4d')

line.vars['k2xrr_a_extr'] = 0
line.vars['k2xrr_b_extr'] = 0
line.vars['k2xrr_a'] = line.vars['k2xrr_a_extr']
line.vars['k2xrr_b'] = line.vars['k2xrr_b_extr']

optq = line.match(
    solve=False,
    method='4d',
    vary=[
        xt.VaryList(['qf1k1', 'qd1k1', 'qf2k1'], step=1e-3, tag='quad'),
        xt.VaryList(['k2xcf', 'k2xcd'], step=1e-3, tag='sext'),
    ],
    targets=[
        xt.TargetSet(qx=1.661, qy=1.72, tol=1e-6, tag='tunes'),
        xt.TargetSet(dqx=-0.1, dqy=-0.1, tol=1e-3, tag="chrom"),
        xt.Target(dx=0, tol=1e-3, at='es', tag='disp'),
    ]
)
optq.disable_targets(tag='chrom')
optq.disable_vary(tag='sext')
optq.solve()
optq.enable_all_targets()
optq.enable_all_vary()
optq.solve()

tw1 = line.twiss(method='4d')

line.vars['k2xrr_a_extr'] = 0
line.vars['k2xrr_b_extr'] = 7.5
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

ax2=plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(tw0.s, tw0.dx, '.-')
plt.plot(tw1.s, tw1.dx, '.-')

act_show= ActionSeparatrix(line, range_test=(0e-3, 2e-2), range_fit=(2e-2, 3e-2),
                                n_test=30, x_spiral_meas=0.02)
res0 = act_show.run()



# Tune closer to resonance for separatrix matching
optq.targets[0].value = 1.667
optq.solve()

# plt.axvline(x=tw2['s', 'xrr'], color='green', linestyle='--')

act_match = ActionSeparatrix(line, range_test=(0e-3, 2e-3), range_fit=(2.5e-3, 3.e-3),
                                n_test=5, i_part_fit=4)
res_m0 = act_match.run()

opt = line.match(
    solve=False,
    method='4d',
    vary=xt.VaryList(['k2xrr_a_extr', 'k2xrr_b_extr'], step=1e-3, tag='resonance'),
    targets=[
        act_match.target('r_sep_norm', res_m0['r_sep_norm'], tol=2e-5, tag='resonance', weight=1e4),
        act_match.target('slope_norm', res_m0['slope_norm'], tol=0.05, tag='resonance'),
    ]
)

while opt.targets[1].value > 0.3:
    opt.targets[1].value -= 0.02
    opt.step(40)
    opt.target_status()
    opt.vary_status()

res_m1 = act_match.run()


# Tune closer to resonance for separatrix matching
optq.targets[0].value = 1.661
optq.solve()

res1 = act_show.run()

plot_res(res0, 'before')
plot_res(res1, 'after')
plot_res(res_m0, 'match - first point')
plot_res(res_m1, 'match - last point')

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
    nemitt_x=1e-6, nemitt_y=1e-6,
    x_norm=x_norm, px_norm=px_norm, y_norm=y_norm, py_norm=py_norm,
    delta=delta)
tab = tw.get_normalized_coordinates(particles)




# plt.figure(100)
# plt.plot(x_fit_geom, px_fit_geom, 'grey')
# plt.plot(particles.x, particles.px, '.', markersize=1)
# plt.ylabel(r'$p_x$')
# plt.xlabel(r'$x$ [m]')

# plt.figure(101)
# plt.plot(x_fit_norm, px_fit_norm, 'grey')
# plt.plot(tab.x_norm, tab.px_norm, '.', markersize=1)
# plt.axis('equal')

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
line.vars['k2xrr_a'] = line.vars['k2xrr_a_extr'] * line.functions['fun_xsext'](line.vars['t_turn_s'])
line.vars['k2xrr_b'] = line.vars['k2xrr_b_extr'] * line.functions['fun_xsext'](line.vars['t_turn_s'])

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
# plt.plot(x_fit_geom, px_fit_geom, 'grey')
plt.show()