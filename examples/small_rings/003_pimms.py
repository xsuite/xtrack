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
        xt.TargetSet(qx=1.6665, qy=1.72),
        xt.TargetSet(dqx=-4, dqy=-1, tol=1e-3),
        # xt.Target(dx=0, at='pimms_start'),
    ]
)
opt.step(10)

tw = line.twiss(method='4d')

line.vars['k2xrr'] = 10
r0 = np.linspace(0, 2, 50)
p = line.build_particles(
    method='4d',
    x_norm=r0*np.cos(np.pi/20.),
    px_norm=r0*np.sin(np.pi/20.),
    nemitt_x=1e-6, nemitt_y=1e-6)

line.track(p, num_turns=100000, turn_by_turn_monitor=True)
mon = line.record_last_track

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(mon.x.T, mon.px.T, '.', markersize=1)

plt.figure(2)
plt.plot(p.x[p.state<=0], p.px[p.state<=0], '.', markersize=2)
plt.show()

