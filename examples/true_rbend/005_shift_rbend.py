import xtrack as xt
import numpy as np
import xobjects as xo

env = xt.Environment(particle_ref=xt.Particles(p0c=10e9))

edge_model = 'linear'
shift = 0.3  # m

line = env.new_line(length=5, components=[
    env.new('mb', 'RBend', angle=0.3, k0_from_h=True, length_straight=3,
            model='bend-kick-bend',
            rbend_model='straight-body',
            rbend_shift=shift,
            edge_entry_model=edge_model, edge_exit_model=edge_model,
            at=2.5)])
line.insert('start', xt.Marker(), at=0)
line.append('end', xt.Marker())
line.cut_at_s(np.linspace(0, line.get_length(), 11))
line.insert('mid', xt.Marker(), at=2.5)

tw = line.twiss(betx=1, bety=1)
sv = line.survey(element0='mid')

xo.assert_allclose(tw['x', 'mb_entry'], 0, atol=1e-14)
xo.assert_allclose(tw['x', 'mid'], line['mb'].sagitta / 2 - shift, atol=1e-14)
xo.assert_allclose(tw['x', 'mb..0'], -line['mb'].sagitta / 2 - shift, atol=1e-14)
xo.assert_allclose(tw['x', 'mb..exit_map'], -line['mb'].sagitta / 2 - shift, atol=1e-14)
xo.assert_allclose(tw['x', 'mb_exit'], 0, atol=1e-14)
xo.assert_allclose(tw.y, 0, atol=1e-14)

xo.assert_allclose(sv.rows['mb_entry':'mb_exit'].X[2:-1], 0, atol=1e-14)
xo.assert_allclose(sv['X', 'mb_entry'], -line['mb'].sagitta / 2 - shift, atol=1e-14)
xo.assert_allclose(sv['X', 'mb_exit'], -line['mb'].sagitta / 2 - shift, atol=1e-14)
xo.assert_allclose(sv.Y, 0, atol=1e-14)

xo.assert_allclose(sv['theta', 'mb_entry'], 0.15, atol=1e-14)
xo.assert_allclose(sv['theta', 'mb_exit'], -0.15, atol=1e-14)
xo.assert_allclose(sv['theta', 'mid'], 0, atol=1e-14)
xo.assert_allclose(sv.phi, 0, atol=1e-14)
xo.assert_allclose(sv.psi, 0, atol=1e-14)

sv_init_start = line.survey(element0='start',
                            X0=sv['X', 'start'],
                            Y0=sv['Y', 'start'],
                            Z0=sv['Z', 'start'],
                            phi0=sv['phi', 'start'],
                            psi0=sv['psi', 'start'],
                            theta0=sv['theta', 'start'])
sv_iinit_end = line.survey(element0='end',
                          X0=sv['X', 'end'],
                          Y0=sv['Y', 'end'],
                          Z0=sv['Z', 'end'],
                          phi0=sv['phi', 'end'],
                          psi0=sv['psi', 'end'],
                          theta0=sv['theta', 'end'])

for sv_test in [sv, sv_init_start, sv_iinit_end]:
    xo.assert_allclose(sv_test.X, sv.X, atol=1e-14)
    xo.assert_allclose(sv_test.Y, sv.Y, atol=1e-14)
    xo.assert_allclose(sv_test.Z, sv.Z, atol=1e-14)
    xo.assert_allclose(sv_test.phi, sv.phi, atol=1e-14)
    xo.assert_allclose(sv_test.psi, sv.psi, atol=1e-14)
    xo.assert_allclose(sv_test.theta, sv.theta, atol=1e-14)

tw_back = line.twiss(init=tw, init_at='end')

xo.assert_allclose(tw_back.x, tw.x, atol=1e-14)
xo.assert_allclose(tw_back.y, tw.y, atol=1e-14)
xo.assert_allclose(tw_back.s, tw.s, atol=1e-14)
xo.assert_allclose(tw_back.zeta, tw.zeta, atol=1e-14)

line['mb'].rbend_model = 'curved-body'
tw_curved = line.twiss(betx=1, bety=1)
sv_curved = line.survey(element0='mid')

xo.assert_allclose(tw_curved.x, 0, atol=1e-14)
xo.assert_allclose(tw_curved.y, 0, atol=1e-14)
xo.assert_allclose(sv_curved['X', 'mb_entry'], -line['mb'].sagitta, atol=1e-14)
xo.assert_allclose(sv_curved['X', 'mid'], 0, atol=1e-14)
xo.assert_allclose(sv_curved['X', 'mb_exit'], -line['mb'].sagitta, atol=1e-14)

sv_curved_init_start = line.survey(element0='start',
                                   X0=sv_curved['X', 'start'],
                                   Y0=sv_curved['Y', 'start'],
                                   Z0=sv_curved['Z', 'start'],
                                   phi0=sv_curved['phi', 'start'],
                                   psi0=sv_curved['psi', 'start'],
                                   theta0=sv_curved['theta', 'start'])
sv_curved_init_end = line.survey(element0='end',
                                   X0=sv_curved['X', 'end'],
                                   Y0=sv_curved['Y', 'end'],
                                   Z0=sv_curved['Z', 'end'],
                                   phi0=sv_curved['phi', 'end'],
                                   psi0=sv_curved['psi', 'end'],
                                   theta0=sv_curved['theta', 'end'])

for sv_test in [sv_curved_init_start, sv_curved_init_end]:
    xo.assert_allclose(sv_test.X, sv_curved.X, atol=1e-14)
    xo.assert_allclose(sv_test.Y, sv_curved.Y, atol=1e-14)
    xo.assert_allclose(sv_test.Z, sv_curved.Z, atol=1e-14)
    xo.assert_allclose(sv_test.phi, sv_curved.phi, atol=1e-14)
    xo.assert_allclose(sv_test.psi, sv_curved.psi, atol=1e-14)
    xo.assert_allclose(sv_test.theta, sv_curved.theta, atol=1e-14)


import matplotlib.pyplot as plt
plt.close('all')

plt.figure(1)
tw.plot('x')

plt.figure(2)
sv.plot()

plt.show()