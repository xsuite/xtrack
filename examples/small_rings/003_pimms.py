from cpymad.madx import Madx
import numpy as np

import xtrack as xt

test_data_folder = '../../test_data/'
mad = Madx()

mad.call(test_data_folder + 'pimms/PIMM_orig.seq')
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

line.vars['k2xrr'] = 8.65

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
res0 = action_sep.run()

opt = line.match(
    solve=False,
    method='4d',
    vary=[
        xt.VaryList(['k2xrr', 'k2mysext'], step=1e-2, tag='resonance'),
        xt.VaryList(['k2xcf', 'k2xcd'], step=1e-3, tag='chromaticity'),
    ],
    targets=[
        action_sep.target('r_sep_norm', 1e-3, tol=1e-4, tag='resonance'),
        action_sep.target('slope', 0, tol=1e-5, tag='resonance'),
        xt.TargetSet(dqx=-4, dqy=-1, tol=1e-3, tag='chromaticity'),
    ]
)
# opt.disable_targets(tag='chromaticity')
# opt.disable_vary(tag='chromaticity')
# opt.solve()

# opt.enable_all_targets()
# opt.enable_all_vary()
# opt.solve()

res = action_sep.run()

poly_geom = res['poly_geom']
poly_norm = res['poly_norm']

x_fit_geom = np.linspace(-0.1, 0.1, 10)
px_fit_geom = poly_geom[0] * x_fit_geom + poly_geom[1]
x_fit_norm = np.linspace(-0.1, 0.1, 10)
px_fit_norm = poly_norm[0] * x_fit_norm + poly_norm[1]

tw = line.twiss(method='4d')

p = line.build_particles(
    method='4d', x=np.linspace(0, 1e-2, 20), px=0, y=0, py=0)

line.track(p, num_turns=10000, turn_by_turn_monitor=True, time=True)
mon = line.record_last_track
norm_coord = tw.get_normalized_coordinates(mon)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(100)
plt.plot(mon.x.T, mon.px.T, '.', markersize=1)
# plt.plot(x_branch, px_branch, '.k', markersize=3)
plt.plot(x_fit_geom, px_fit_geom, 'grey')
plt.ylabel(r'$p_x$')
plt.xlabel(r'$x$ [m]')

plt.figure(101)
plt.plot(norm_coord.x_norm.T, norm_coord.px_norm.T, '.', markersize=1)
# plt.plot(x_norm_branch, px_norm_branch, '.k', markersize=3)
plt.plot(x_fit_norm, px_fit_norm, 'grey')
plt.axis('equal')

plt.show()