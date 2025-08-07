import xtrack as xt
import numpy as np
import xobjects as xo

env = xt.Environment(particle_ref=xt.Particles(p0c=10e9))

edge_model = 'full'
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



import matplotlib.pyplot as plt
plt.close('all')

plt.figure(1)
tw.plot('x')

plt.figure(2)
sv.plot()

plt.show()