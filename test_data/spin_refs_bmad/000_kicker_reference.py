import xtrack as xt
import xobjects as xo
import numpy as np

env = xt.Environment()
line = env.new_line(components=[
    env.new('mykicker', xt.Magnet, length=2.0, knl=[1e-3], ksl=[2e-3]),
    env.new('mymarker', xt.Marker),
    ])

p0 = xt.Particles(p0c=700e9, mass0=xt.ELECTRON_MASS_EV,
                  anomalous_magnetic_moment = 0.00115965218128)

line.particle_ref = p0.copy()

p0.x = 1e-3
p0.px = 1e-5
p0.y = 2e-3
p0.py = 2e-5
p0.delta = 1e-3
p0.spin_x = 0.1
p0.spin_z = 0.2
p0.spin_y = np.sqrt(1 - p0.spin_x**2 - p0.spin_z**2)


from bmad_track_twiss_spin import bmad_run

out_bmad = bmad_run(line, track={'x': p0.x[0], 'px': p0.px[0], 'y': p0.y[0], 'py': p0.py[0],
                        'delta': p0.delta[0], 'spin_x': p0.spin_x[0],
                        'spin_y': p0.spin_y[0], 'spin_z': p0.spin_z[0],
                        })

# line.configure_spin(model=True)
line.config.XTRACK_MULTIPOLE_NO_SYNRAD = False

p = p0.copy()
line.track(p)

xo.assert_allclose(
    p.spin_x[0], out_bmad['spin'].spin_x.values[-1], atol=4e-6, rtol=0)
xo.assert_allclose(
    p.spin_y[0], out_bmad['spin'].spin_y.values[-1], atol=4e-6, rtol=0)
xo.assert_allclose(
    p.spin_z[0], out_bmad['spin'].spin_z.values[-1], atol=4e-6, rtol=0)
