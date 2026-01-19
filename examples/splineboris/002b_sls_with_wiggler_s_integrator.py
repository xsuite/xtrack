from examples.wiggler_class_test import start_time
from wiggler_class_old import WigglerFieldFitter
from scipy import signal
import bpmeth as bp
import matplotlib.pyplot as plt
import numpy as np

import xtrack as xt
import time

start_time = time.time()
delta_chrom = 1e-4
# deltas = [-2*delta_chrom, -delta_chrom, 0, delta_chrom, 2*delta_chrom]
deltas = [-delta_chrom, delta_chrom]

# x0 = 0.5e-3
x0 = 0

dz = 0.001  # Step size in the z direction for numerical differentiation.

# Create a Wiggler with parameters:
# file_path: Path to the field map file.
# xy_point: (x,y) coordinates of the axis where the field is evaluated.
# dz: Step size in the z direction for numerical differentiation.
#       In this case, dz=0.001 rescales the distances to m instead of mm.
# x_left_slices: Number of slices to the left in the x direction for fitting.
# x_right_slices: Number of slices to the right in the x direction for fitting.
# y_left_slices: Number of slices to the left in the y direction for fitting.
# y_right_slices: Number of slices to the right in the y direction for fitting.
# n_modes_x: Number of modes in the x direction for fitting the sinusoid.
# n_modes_y: Number of modes in the y direction for fitting the sinusoid.
print("FIELDS:")
test_wiggler = WigglerFieldFitter(file_path='example_data/knot_map_test.txt',
                                  xy_point=(0, 0),
                                  dx=dz,
                                  dy=dz,
                                  ds=dz,
                                  peak_window=(99, 2100),
                                  n_modes=[3, 3, 1],
                                  poly_deg=[[4, 4], [4, 4], [4, 4]],
                                  poly_pieces=[[19, 35], [19, 36], [8, 8]],
                                  der=False
                                  )

test_wiggler.set()
#Test_Wiggler.plot_fields()
#Test_Wiggler.plot_integrated_fields()

print("DERIVATIVES:")
test_wiggler_der = WigglerFieldFitter(file_path='example_data/knot_map_test.txt',
                                      xy_point=(0, 0),
                                      dx=dz,
                                      dy=dz,
                                      ds=dz,
                                      n_modes=[6, 4, 1],
                                      poly_deg=[[4, 4], [4, 4], [4, 4]],
                                      poly_pieces=[[15, 15], [15, 15], [15, 15]],
                                      peak_window=(99, 2100),
                                      der=True,
                                      filter_params=(None, 2090, 7, 11, 3)
                                      )

test_wiggler_der.set()

Bx_string = test_wiggler.export_piecewise_sympy(field="Bx")
Bx_der_string = test_wiggler_der.export_piecewise_sympy(field="Bx")
By_string = test_wiggler.export_piecewise_sympy(field="By")
By_der_string = test_wiggler_der.export_piecewise_sympy(field="By")
Bs_string = test_wiggler.export_piecewise_sympy(field="Bs")

a1 = Bx_string
b1 = By_string
bs = Bs_string

a2 = 0
b2 = 0

a3 = Bx_der_string
b3 = By_der_string

print(f"Type of a1: {type(a1)}")

curv=0
import time
start_time = time.time()
wiggler_map = bp.GeneralVectorPotential(hs=f"{curv}",a=(f"{a1}", f"{a2}", f"{a3}"),b=(f"{b1}", f"{b2}", f"{b3}"), bs=f"{bs}")
end_time = time.time()
print(f"Time to make the segments: {end_time - start_time:.6f} seconds")
start_time = time.time()
Bxfun, Byfun, Bsfun = wiggler_map.get_Bfield()
end_time = time.time()
print(f"Time to create field functions: {end_time - start_time:.6f} seconds")

class MyWiggler:
    def __init__(self, Bx_fun, By_fun, Bs_fun, s0=0, x0=0, y0=0):
        self.Bx_fun = Bx_fun
        self.By_fun = By_fun
        self.Bs_fun = Bs_fun
        self.s0 = s0
        self.x0 = x0
        self.y0 = y0
        self.scale = 1.  # Scale to meters

    def get_field(self, x, y, s):
        Bx = self.scale * self.Bx_fun(x - self.x0, y + self.y0, s + self.s0)
        By = self.scale * self.By_fun(x - self.x0, y + self.y0, s + self.s0)
        Bs = self.scale * self.Bs_fun(x - self.x0, y + self.y0, s + self.s0)
        return Bx, By, Bs

mywig = MyWiggler(Bxfun, Byfun, Bsfun, s0=-1.1, x0=x0)


p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
                  energy0=2.4e9)

p = p0.copy()

n_steps = 1000
l_wig = 2.2
n_slices = 1000

s_cuts = np.linspace(0, l_wig, n_slices + 1)
s_mid = 0.5 * (s_cuts[:-1] + s_cuts[1:])

wig_slices = []
for ii in range(n_slices):
    wig = xt.BorisSpatialIntegrator(fieldmap_callable=mywig.get_field, s_start=s_cuts[ii], s_end=s_cuts[ii + 1],
                                 n_steps=np.round(n_steps / n_slices).astype(int),
                                 verbose=True)
    wig_slices.append(wig)

# NOTE: The class can now basically execute this code until here.

Bx_mid, By_mid, Bs_mid = wig_slices[0].fieldmap_callable(0, 0, s_mid)
print('extracting a1, b1, a3, b3 at mid points')
a1_mid  = [wiggler_map.a[0].subs({wiggler_map.s: ss}) for ss in s_mid + mywig.s0]
b1_mid  = [wiggler_map.b[0].subs({wiggler_map.s: ss}) for ss in s_mid + mywig.s0]
a3_mid  = [wiggler_map.a[2].subs({wiggler_map.s: ss}) for ss in s_mid + mywig.s0]
b3_mid  = [wiggler_map.b[2].subs({wiggler_map.s: ss}) for ss in s_mid + mywig.s0]
print('done')


env = xt.load('example_data/b075_2024.09.25.madx')
line = env.ring
env['ring_no_wiggler'] = line.copy(shallow=True)
line.configure_bend_model(core='mat-kick-mat')
line.particle_ref = p0.copy()

for ii in range(n_slices):
    env.elements[f'wigslice_{ii}'] = wig_slices[ii]
wiggler = env.new_line(components=['wigslice_' + str(ii) for ii in range(n_slices)])

env['k0l_corr1'] = 0.
env['k0l_corr2'] = 0.
env['k0l_corr3'] = 0.
env['k0l_corr4'] = 0.
env['k0sl_corr1'] = 0.
env['k0sl_corr2'] = 0.
env['k0sl_corr3'] = 0.
env['k0sl_corr4'] = 0.
env['on_wig_corr'] = 1.0

env.new('corr1', xt.Multipole, knl=['on_wig_corr * k0l_corr1'], ksl=['on_wig_corr * k0sl_corr1'])
env.new('corr2', xt.Multipole, knl=['on_wig_corr * k0l_corr2'], ksl=['on_wig_corr * k0sl_corr2'])
env.new('corr3', xt.Multipole, knl=['on_wig_corr * k0l_corr3'], ksl=['on_wig_corr * k0sl_corr3'])
env.new('corr4', xt.Multipole, knl=['on_wig_corr * k0l_corr4'], ksl=['on_wig_corr * k0sl_corr4'])

wiggler.insert([
    env.place('corr1', at=0.02),
    env.place('corr2', at=0.1),
    env.place('corr3', at=l_wig - 0.1),
    env.place('corr4', at=l_wig - 0.02),
    ], s_tol=5e-3
)
wiggler.particle_ref = line.particle_ref

# Computed for 1000 slices, 1000 steps
env.vars.update(
{'k0l_corr1': np.float64(-0.0004540792291112204),
 'k0sl_corr1': np.float64(-1.213769189237666e-06),
 'k0l_corr2': np.float64(0.0008135172335552242),
 'k0sl_corr2': np.float64(0.00023470961164860475),
 'k0l_corr3': np.float64(-0.0001955197609031625),
 'k0sl_corr3': np.float64(-0.00021394733008765638),
 'k0l_corr4': np.float64(-0.00015806879956816854),
 'k0sl_corr4': np.float64(3.370506139561265e-05)})

# For x0 = 0.5e-3
# env.vars.update(
# {'k0l_corr1': np.float64(-0.0004640274435485036),
#  'k0sl_corr1': np.float64(-1.2297340793905685e-06),
#  'k0l_corr2': np.float64(0.0008265782650066877),
#  'k0sl_corr2': np.float64(0.0002344711077490433),
#  'k0l_corr3': np.float64(-0.00018319740840498774),
#  'k0sl_corr3': np.float64(-0.00021346101458338208),
#  'k0l_corr4': np.float64(-0.00016749244113701785),
#  'k0sl_corr4': np.float64(3.3646895667713495e-05)})

# # To compute the kicks
# opt = wiggler.match(
#     solve=False,
#     betx=0, bety=0,
#     only_orbit=True,
#     include_collective=True,
#     vary=xt.VaryList(['k0l_corr1', 'k0sl_corr1',
#                       'k0l_corr2', 'k0sl_corr2',
#                       'k0l_corr3', 'k0sl_corr3',
#                       'k0l_corr4', 'k0sl_corr4',
#                       ], step=1e-6),
#     targets=[
#         xt.TargetSet(x=0, px=0, y=0, py=0., at=xt.END),
#         xt.TargetSet(x=0., y=0, at='wigslice_167'),
#         xt.TargetSet(x=0., y=0, at='wigslice_833')
#         ],
# )
# opt.step(2)

print('Twiss wiggler only')
tw_wig_only = wiggler.twiss(include_collective=True, betx=1, bety=1)
end_time = time.time()
print(f"Time to compute twiss wiggler only: {end_time - start_time} seconds")
prrrrrr
wiggler_places = [
    'ars02_uind_0500_1',
    'ars03_uind_0380_1',
    'ars04_uind_0500_1',
    'ars05_uind_0650_1',
    'ars06_uind_0500_1',
    'ars07_uind_0200_1',
    'ars08_uind_0500_1',
    'ars09_uind_0790_1',
    'ars11_uind_0210_1',
    'ars11_uind_0610_1',
    'ars12_uind_0500_1',
]

tt = line.get_table()
for wig_place in wiggler_places:
    line.insert(wiggler, anchor='start', at=tt['s', wig_place])

env['on_wig_corr'] = 0
mywig.scale = 0
tw_no_wig = line.twiss4d(strengths=True)
tw_vs_momentum_no_wig = {}
for dd in deltas:
    tw_vs_momentum_no_wig[dd] = line.twiss4d(delta0=dd,
                                         compute_chromatic_properties=False)

env['on_wig_corr'] = 1.0
mywig.scale = 1.0

print('Twiss full line with wiggler')
p_co = tw_no_wig.particle_on_co.copy()
p_co.at_element=0

tw = line.twiss4d(include_collective=True, particle_on_co=p_co,
                  compute_chromatic_properties=False)



tw_vs_momentum = {}
for delta in deltas:
    print(f'Twiss off momentum, delta = {delta}')
    p_off = p_co.copy()
    p_off.delta += delta
    p_off.x += tw.dx[0] * delta
    p_off.px += tw.dpx[0] * delta
    p_off.y += tw.dy[0] * delta
    p_off.py += tw.dpy[0] * delta
    p_off.at_element=0
    tw_vs_momentum[delta] = line.twiss4d(include_collective=True,
                                         particle_on_co=p_off,
                                         compute_chromatic_properties=False)


cols_chrom, scalars_chrom = xt.twiss._compute_chromatic_functions(line, init=None,
                                      delta_chrom=delta_chrom,
                                      steps_r_matrix=None,
                                      matrix_responsiveness_tol=None,
                                      matrix_stability_tol=None,
                                      symplectify=None,
                                      tw_chrom_res=[tw_vs_momentum[-delta_chrom],
                                                    tw_vs_momentum[delta_chrom]],
                                      on_momentum_twiss_res=tw)

tw._data.update(cols_chrom)
tw._data.update(scalars_chrom)
tw._col_names += list(cols_chrom.keys())

dl = np.diff(s_cuts)
wig_mult_places = []
for ii, (bbx, bby) in enumerate(zip(Bx_mid, By_mid)):
    nn = f'wig_mult_{ii}'
    pp = env.new(nn, xt.Bend,
                 length=dl[ii],
                 knl=[dl[ii] * b1_mid[ii] / p0.rigidity0[0], 0, dl[ii] * b3_mid[ii] / p0.rigidity0[0]],
                 ksl=[dl[ii] * a1_mid[ii] / p0.rigidity0[0], 0, dl[ii] * a3_mid[ii] / p0.rigidity0[0]],
                 shift_x=x0,
                 at=s_mid[ii])
    wig_mult_places.append(pp)

wiggler_mult = wiggler.copy(shallow=True)
tt_slices = wiggler.get_table().rows['wigslic.*']

wiggler_mult.remove(tt_slices.name)
wiggler_mult.insert(wig_mult_places)

tw_wig_mult = wiggler_mult.twiss(betx=1, bety=1)

line_wig_mult = env['ring_no_wiggler'].copy(shallow=True)
line_wig_mult.particle_ref = p0.copy()

for wig_place in wiggler_places:
    line_wig_mult.insert(wiggler_mult, anchor='start', at=tt['s', wig_place])

tw_wig_mult = line_wig_mult.twiss4d()

tt_mult = wiggler_mult.get_table().rows['wig_mult_.*']
for nn in tt_mult.name:
    env[nn].knl = 0
    env[nn].ksl = 0
env['on_wig_corr'] = 0.0
tw_wig_mult_off = line_wig_mult.twiss4d()


plt.close('all')
plt.figure(1, figsize=(6.4, 4.8))
ax = plt.subplot(111)
ax.plot(tw_wig_mult.s, tw_wig_mult.betx/tw_wig_mult_off.betx - 1, label='multipoles')
ax.plot(tw.s, tw.betx/tw_no_wig.betx - 1, '-', label='BPMETH')
ax.set_ylabel(r'$\Delta \beta_x / \beta_x$')
ax.set_xlabel('s [m]')
ax.legend()

plt.figure(2, figsize=(6.4, 4.8))
ax = plt.subplot(111)
ax.plot(tw_wig_mult.s, tw_wig_mult.betx2, label=r'Multipoles, $|C^-|$='+f'{tw_wig_mult.c_minus:.2e}')
ax.plot(tw.s, tw.betx2, '-', label='BPMETH, '+f'$|C^-|$={tw.c_minus:.2e}')
ax.set_ylabel(r'$\beta_{x,2}$ [m]')
ax.set_xlabel('s [m]')
ax.legend()
ax.set_ylim(0, 0.2)

plt.show()