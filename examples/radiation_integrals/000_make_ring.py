import xtrack as xt
import numpy as np

env = xt.load_madx_lattice('../../test_data/sps_thick/sps.seq')
env.vars.load_madx('../../test_data/sps_thick/lhc_q20.str')
line = env.sps

line['actcse.31632'].voltage = 4.2e+08
line['actcse.31632'].frequency = 3e6
line['actcse.31632'].lag = 180.

tt = line.get_table()


line.particle_ref = xt.Particles(energy0=20e9, mass0=xt.ELECTRON_MASS_EV)
env.particle_ref = line.particle_ref

import wiggler as wgl

# Wiggler parameters
k0_wig = 5e-3
tilt_rad = np.pi/2

lenwig = 25
numperiods = 20
lambdawig = lenwig / numperiods

wig = wgl.Wiggler(period=lambdawig, amplitude=k0_wig, num_periods=numperiods,
                  angle_rad=tilt_rad, scheme='121a')

tt = line.get_table()
wig_elems = []
for name, element in wig.wiggler_dict.items():
    env.elements[name] = element['element']
    wig_elems.append(name)

wig_line = env.new_line(components=[
                        env.new('s.wig', 'Marker'),
                        wig_elems,
                        env.new('e.wig', 'Marker'),
])


line.insert(wig_line, anchor='start', at=1, from_='qd.31710@end')

tt = line.get_table()
tw4d_thick = line.twiss4d()
tw6d_thick = line.twiss()

env['sps_thick'] = env.sps.copy(shallow=True)

line.discard_tracker()
slicing_strategies = [
    xt.Strategy(slicing=xt.Teapot(1)),  # Default
    xt.Strategy(slicing=xt.Teapot(2), element_type=xt.Bend),
    xt.Strategy(slicing=xt.Teapot(2), element_type=xt.RBend),
    xt.Strategy(slicing=xt.Teapot(8), element_type=xt.Quadrupole),
    xt.Strategy(slicing=xt.Teapot(20), name='mwp.*'),
]
line.slice_thick_elements(slicing_strategies)

tw4d = line.twiss4d()
tw6d = line.twiss()

line.configure_radiation(model='mean')

tw_rad = line.twiss(eneloss_and_damping=True, strengths=True)

# import matplotlib.pyplot as plt
# plt.close('all')
# pl = tw_rad.plot(yl='y', yr='dy')
# pl.xlim(tw_rad['s', 's.wig'] - 10, tw_rad['s', 'e.wig'] + 10)
# plt.show()

# from synchrotron_integrals import SynchrotronIntegral as synint
# integrals = synint(line)

tw = tw_rad

angle_rad = tw['angle_rad']
rot_s_rad = tw['rot_s_rad']
x = tw['x']
y = tw['y']
kin_px = tw['kin_px']
kin_py = tw['kin_py']
delta = tw['delta']
length = tw['length']

betx = tw['betx']             # Twiss beta function x
alfx = tw['alfx']             # Twiss alpha x
gamx = tw['gamx']             # Twiss gamma x
bety = tw['bety']             # Twiss beta function y
alfy = tw['alfy']             # Twiss alpha y
gamy = tw['gamy']             # Twiss gamma y
dx = tw['dx']                 # Dispersion x
dy = tw['dy']                 # Dispersion y
dpx = tw['dpx']               # Dispersion px
dpy = tw['dpy']               # Dispersion py

dxprime = dpx * (1 - delta) - kin_px
dyprime = dpy * (1 - delta) - kin_py

# Curvature of the reference trajectory
kappa_0xy = np.zeros(shape=(2, len(length)))
mask = length != 0
kappa_0xy[0, :][mask] = angle_rad[mask] * np.cos(rot_s_rad[mask]) / length[mask]
kappa_0xy[1, :][mask] = angle_rad[mask] * np.sin(rot_s_rad[mask]) / length[mask]

# Compute x', y', x'', y''
ps = np.sqrt((1 + delta)**2 - kin_px**2 - kin_py**2)
xp = kin_px / ps
yp = kin_py / ps
xp_ele = xp * 0
yp_ele = yp * 0
xp_ele[:-1] = (xp[:-1] + xp[1:]) / 2
yp_ele[:-1] = (yp[:-1] + yp[1:]) / 2

mask_length = length != 0
xpp_ele = xp_ele * 0
ypp_ele = yp_ele * 0
xpp_ele[mask_length] = np.diff(xp, append=0)[mask_length] / length[mask_length]
ypp_ele[mask_length] = np.diff(yp, append=0)[mask_length] / length[mask_length]

# Curvature of the particle trajectory
h = 1 + kappa_0xy[0, :] * x + kappa_0xy[1, :] * y
hprime = kappa_0xy[0, :] * xp_ele + kappa_0xy[1, :] * yp_ele
mask1 = xpp_ele**2 + h**2 != 0
mask2 = xpp_ele**2 + h**2 != 0
kappa_x = (-(h * (xpp_ele - h * kappa_0xy[0, :]) - 2 * hprime * xp_ele)[mask1]
           / (xp_ele**2 + h**2)[mask1]**(3/2))
kappa_y = (-(h * (ypp_ele - h * kappa_0xy[1, :]) - 2 * hprime * yp_ele)[mask2]
           / (yp_ele**2 + h**2)[mask2]**(3/2))

# Curly H
Hx_rad = gamx * dx**2 + 2*alfx * dx * dxprime + betx * dxprime**2
Hy_rad = gamy * dy**2 + 2*alfy * dy * dyprime + bety * dyprime**2
