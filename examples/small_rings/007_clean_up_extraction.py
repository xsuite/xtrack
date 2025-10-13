from cpymad.madx import Madx
import numpy as np

import xtrack as xt

class ActionSeparatrix(xt.Action):

    def __init__(self, line, n_test=20, range_test=(0, 1e-2),
                             range_fit=(0.015, 0.025),
                             i_part_fit=None,
                             num_turns=5000,
                             x_spiral_meas=None):
        self.line = line
        self.n_test = n_test
        self.range_test = range_test
        self.range_fit = range_fit
        self.i_part_fit = i_part_fit
        self.num_turns = num_turns
        self.x_spiral_meas = x_spiral_meas
        self.x_septum = 0.04

    def run(self):
        line = self.line
        tw = line.twiss(method='4d')
        p0 = line.build_particles(x=0, px=0)
        x_stable = 0
        x_unstable = 2e-2
        while x_unstable - x_stable > 1e-6:
            x_test = (x_stable + x_unstable) / 2
            p = p0.copy()
            p.x = x_test
            line.track(p, num_turns=self.num_turns, turn_by_turn_monitor=True)
            mon = line.record_last_track
            if (mon.x > self.x_septum).any():
                x_unstable = x_test
            else:
                x_stable = x_test
        p = line.build_particles(x=[x_unstable, x_stable], px=0)
        line.track(p, num_turns=self.num_turns, turn_by_turn_monitor=True)
        mon_separatrix = line.record_last_track
        norm_coord_separatrix = tw.get_normalized_coordinates(mon_separatrix)

        j_stable = np.sqrt(norm_coord_separatrix.x_norm[1, :]**2 + norm_coord_separatrix.px_norm[1, :]**2)
        i_fixed_point = np.argmax(j_stable)
        x_fixed_point = mon_separatrix.x[1, i_fixed_point]
        px_fixed_point = mon_separatrix.px[1, i_fixed_point]
        x_norm_fixed_point = norm_coord_separatrix.x_norm[1, i_fixed_point]
        px_norm_fixed_point = norm_coord_separatrix.px_norm[1, i_fixed_point]
        j_fixed_point = j_stable[i_fixed_point]

        i_separatrix = 0
        x_first_unstable = mon_separatrix.x[i_separatrix, :].copy()
        px_first_unstable = mon_separatrix.px[i_separatrix, :].copy()

        # To be generalized
        x_first_unstable[px_first_unstable < 0] = 9999999.

        i_closest = np.argmin(np.abs(x_first_unstable - self.x_spiral_meas))
        slope_norm_spiral = (
            (norm_coord_separatrix.px_norm[i_separatrix, i_closest + 3]
                - norm_coord_separatrix.px_norm[i_separatrix, i_closest])
            / (norm_coord_separatrix.x_norm[i_separatrix, i_closest + 3]
                - norm_coord_separatrix.x_norm[i_separatrix, i_closest]))

        out = {
            'slope_norm_spiral': slope_norm_spiral,
            'mon_separatrix': mon_separatrix,
            'j_fixed_point': j_fixed_point,
            'x_fixed_point': x_fixed_point,
            'px_fixed_point': px_fixed_point,
            'x_norm_fixed_point': x_norm_fixed_point,
            'px_norm_fixed_point': px_norm_fixed_point,
        }

        return out

def plot_phase_space(res, title=None, range_test=(0, 1.5e-2), n_test=20,
                     num_turns=5000, line=None):

    tw = line.twiss(method='4d')
    p_test = line.build_particles(
        x=np.linspace(range_test[0], range_test[1], n_test), px=0)
    line.track(p_test, num_turns=num_turns, turn_by_turn_monitor=True)
    mon = line.record_last_track
    norm_coord  = tw.get_normalized_coordinates(mon)


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

    # poly_geom = res['poly_geom_separatrix']
    # poly_norm = res['poly_norm_separatrix']
    # x_fit_geom = np.linspace(-0.2, 0.2, 10)
    # px_fit_geom = poly_geom[0] * x_fit_geom + poly_geom[1]
    # x_fit_norm = np.linspace(-0.2, 0.2, 10)
    # px_fit_norm = poly_norm[0] * x_fit_norm + poly_norm[1]

    # ax_geom.plot(x_fit_geom, px_fit_geom, 'grey')
    # ax_norm.plot(x_fit_norm, px_fit_norm, 'grey')

    # Fixed point
    ax_geom.plot(res['x_fixed_point'], res['px_fixed_point'], 'o')
    ax_norm.plot(res['x_norm_fixed_point'], res['px_norm_fixed_point'], 'o')

    if 'x0' in res:
        # ax_geom.plot([res['x0'], res['x1']], [res['px0'], res['px1']], 'x-r')
        # ax_norm.plot([res['x_norm0'], res['x_norm1']], [res['px_norm0'], res['px_norm1']], 'x-r')
        ax_geom.plot(res['x_first_unstable'], res['px_first_unstable'], '.k')

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
line.configure_bend_model(core='bend-kick-bend', edge='full')


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

line.vars['k2xrr_a_extr'] = 1
line.vars['k2xrr_b_extr'] = -8
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


act_match = ActionSeparatrix(line, range_test=(0e-3, 2e-2), range_fit=(2e-2, 3e-2),
                                n_test=100, x_spiral_meas=0.03)
res_m0 = act_match.run()

opt = line.match(
    solve=False,
    method='4d',
    vary=xt.VaryList(['k2xrr_a_extr', 'k2xrr_b_extr'], step=0.5, tag='resonance',
                     limits=[-10, 10]),
    targets=[
        act_match.target('j_fixed_point', 7e-3, tol=5e-5, tag='resonance', weight=1e2),
        # act_match.target('slope_norm_spiral', -0.3, tol=0.01)
        act_match.target('slope_norm_spiral', 0.6, tol=0.01)
    ]
)


# k_test = np.linspace(1, 10., 50)
# vv = []
# for kk in k_test:
#     print(f'kk = {kk}')
#     line.vars['k2xrr_a_extr'] = kk
#     res = act_match.run()
#     vv.append(res)
# opt.reload(0)

# prrrr
# plt.figure(100)
# plt.subplot(2, 1, 1)
# plt.plot([res['j_fixed_point'] for res in vv])

# plt.subplot(2, 1, 2)
# plt.plot([res['slo'] for res in vv])

def err_fun(x):
    out = opt._err(x, check_limits=False)
    print(f'x = {x}, out = {out}')
    return out


bounds = np.array([vv.limits for vv in opt._err.vary])
opt._err.return_scalar = True
import pybobyqa
soln = pybobyqa.solve(err_fun, x0=opt.log().vary[0, :], bounds=bounds.T,
            rhobeg=10, rhoend=1e-4, maxfun=80, objfun_has_noise=True,
            seek_global_minimum=True)
err_fun(soln.x) # set it to the best solution
opt.tag('pybobyqa')
opt.target_status()

opt.reload(0)
res_m0 = act_match.run()
plot_phase_space(res_m0, 'match - first point', line=line)

opt.reload(tag='pybobyqa')
res_m1 = act_match.run()
plot_phase_space(res_m1, 'match - last point',  line=line)

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
line['spill_exc'].gain = line.vars['gain']

line['septum'].max_x = 0.04

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