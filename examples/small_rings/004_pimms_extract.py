from cpymad.madx import Madx
import numpy as np

import xtrack as xt

test_data_folder = '../../test_data/'
mad = Madx()

mad.call(test_data_folder + 'pimms/PIMM.seq')
mad.call(test_data_folder + 'pimms/betatron.str')
mad.beam(particle='proton', gamma=1.05328945) # 50 MeV
mad.use('pimms')
seq = mad.sequence.pimms
def_expr = True

line = xt.Line.from_madx_sequence(seq, deferred_expressions=def_expr)
line.particle_ref = xt.Particles(gamma0=seq.beam.gamma,
                                 mass0=seq.beam.mass * 1e9,
                                 q0=seq.beam.charge)
line.configure_bend_model(core='full', edge='full')

line.insert_element('mysext', xt.Sextupole(length=0.2), at_s=36.)

line.vars['k2mysext'] = 0
line.element_refs['mysext'].k2 = line.vars['k2mysext']

# line.insert_element(
#             'septum',
#             xt.LimitRect(min_x=-0.02, max_x=0.015, min_y=-1, max_y=1),
#             index='pimms_start')

line.vars['k2xrr'] = 0
opt = line.match(
    solve=False,
    method='4d',
    vary=[
        xt.VaryList(['qf1k1', 'qd1k1', 'qf2k1'], step=1e-3),
        xt.VaryList(['k2xcf', 'k2xcd'], step=1e-3),
    ],
    targets=[
        xt.TargetSet(qx=1.665, qy=1.72),
        xt.TargetSet(dqx=-4, dqy=-1, tol=1e-3),
        # xt.Target(dx=0, at='pimms_start'),
    ]
)
opt.solve()

line.vars['k2xrr'] = 1
tw = line.twiss(method='4d')


class ActionSeparatrix(xt.Action):

    def __init__(self, line):
        self.line = line

    def run(self):
        line = self.line
        tw = line.twiss(method='4d')

        p_test = line.build_particles(x=1e-2, px=0)
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

        mask_fit = (x_branch > 0.02) & (x_branch < 0.04)
        if mask_fit.any():
            poly_geom = np.polyfit(x_branch[mask_fit], px_branch[mask_fit], 1)
            poly_norm = np.polyfit(x_norm_branch[mask_fit], px_norm_branch[mask_fit], 1)
            r_sep_norm = np.abs(poly_norm[1]) / np.sqrt(poly_norm[0]**2 + 1)
            slope = poly_geom[0]
        else:
            poly_geom = None
            poly_norm = None
            r_sep_norm = None
            slope = None

        out = {
            'x_branch': x_branch,
            'px_branch': px_branch,
            'poly_geom': poly_geom,
            'poly_norm': poly_norm,
            'r_sep_norm': r_sep_norm,
            'slope': slope,
        }

        return out

action_sep = ActionSeparatrix(line)
res = action_sep.run()

num_particles = 1000
x_norm = np.random.normal(size=num_particles)
px_norm = np.random.normal(size=num_particles)
y_norm = np.random.normal(size=num_particles)
py_norm = np.random.normal(size=num_particles)
delta = 5e-4 * np.random.normal(size=num_particles)
particles = line.build_particles(
    method='4d',
    nemitt_x=1e-6, nemitt_y=1e-6,
    x_norm=x_norm, px_norm=px_norm, y_norm=y_norm, py_norm=py_norm,
    delta=delta)
tab = tw.get_normalized_coordinates(particles)

poly_geom = res['poly_geom']
poly_norm = res['poly_norm']
x_fit_geom = np.linspace(-0.1, 0.1, 10)
px_fit_geom = poly_geom[0] * x_fit_geom + poly_geom[1]
x_fit_norm = np.linspace(-0.1, 0.1, 10)
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

plt.show()