from pymadng import MAD
import numpy as np

# length = 2
# angle = np.pi / 4
# k2l = 0.
# x_test = 1e-4

length = 14
angle = 2 * np.pi / 1000
k2l = 0
x_test = 1e-4

knl = [0, 0, k2l]

xsuite_models = [
    'bend-kick-bend',
]

slices = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000,
          10000, 20000, 50000, 100000, 200000, 500000, 1000000]

madng_model = 'TKT'
madng_method = 6
madng_reference_slices = slices[0]

with MAD() as mad:

    mad.load('MAD', "sequence")
    mad.load('MAD.element', "sbend")

    mad.send(f"""
seq = sequence {{
    l = {length},
    sbend {{ l = {length}, angle = {angle:.17g}, knl = {{{', '.join(str(k) for k in knl)}}} }},
    beam = beam {{}}
}}
""")

    x_list = []
    t_list = []
    for ss in slices:
        mad['tbl', 'flw'] = mad.track(
            sequence='seq', observe=0, save='"atall"', X0={'x': x_test},
            method=madng_method, model=f"'{madng_model}'",
            nslice=ss)
        df = mad.tbl.to_df()
        x_list.append(df['x'].values[-1])
        t_list.append(df['t'].values[-1])

madng_reference_x = x_list[-1]
madng_reference_t = t_list[-1]

import xtrack as xt

env = xt.Environment()
line = env.new_line(components=[
    env.new('b', 'Bend', length=length, angle=angle, knl=knl,
            model=xsuite_models[0],
            integrator='yoshida4', # is actually yoshida6
            num_multipole_kicks=1)
])

line.set_particle_ref('positron', energy0=1e9)
xsuite_x_by_model = {}
xsuite_zeta_by_model = {}
for model in xsuite_models:
    x_list_xt = []
    zeta_list_xt = []
    for ss in slices:
        line.configure_bend_model(
            edge='full', core=model, num_multipole_kicks=7 * ss)
        tw = line.twiss(betx=1, bety=1, x=x_test)
        x_list_xt.append(tw.x[-1])
        zeta_list_xt.append(tw.zeta[-1])
    xsuite_x_by_model[model] = x_list_xt
    xsuite_zeta_by_model[model] = zeta_list_xt

import matplotlib.pyplot as plt
title = f'length={length:.3f} angle={angle:.3g}, k2l={k2l:.3g}, x_in={x_test:.3e}'

plt.close('all')
plt.figure(1)
plt.loglog(slices, np.abs(np.array(x_list) - x_list[0]),
           'x-', label='MAD-NG')
for model, x_list_xt in xsuite_x_by_model.items():
    plt.loglog(slices, np.abs(np.array(x_list_xt) - x_list_xt[0]), '.-',
               label=f'Xsuite {model}')
plt.xlabel('Number of slices')
plt.ylabel('|x - x(1 slice)|')
plt.suptitle(title)
plt.ylim(1e-18, 1e-8)
plt.legend()


plt.show()
