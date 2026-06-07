from pymadng import MAD
import numpy as np

length = 2
angle = np.pi / 4
knl = [0, 0, 0.]
xsuite_models = [
    'rot-kick-rot-low-order',
    'rot-kick-rot',
    'rot-kick-rot-high-order',
]

slices = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

madng_model = 'DKD'
madng_method = 6

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
    for ss in slices:
        mad['tbl', 'flw'] = mad.track(
            sequence='seq', observe=0, save='"atall"', X0={'x': 0e-3},
            method=madng_method, model=f"'{madng_model}'",
            nslice=ss)
        df = mad.tbl.to_df()
        x_list.append(df['x'].values[-1])

import xtrack as xt

env = xt.Environment()
line = env.new_line(components=[
    env.new('b', 'Bend', length=length, angle=angle, knl=knl,
            model=xsuite_models[0],
            integrator='yoshida4', # is actually yoshida6
            num_multipole_kicks=1)
])

line.set_particle_ref('proton', energy0=1e9)
xsuite_x_by_model = {}
for model in xsuite_models:
    x_list_xt = []
    for ss in slices:
        line.configure_bend_model(
            edge='full', core=model, num_multipole_kicks=7 * ss)
        tw = line.twiss(betx=1, bety=1, x=0e-3)
        x_list_xt.append(tw.x[-1])
    xsuite_x_by_model[model] = x_list_xt

import matplotlib.pyplot as plt
plt.close('all')
plt.loglog(slices, np.abs(np.array(x_list)-x_list[-1]), '.-', label='MAD-NG')
for model, x_list_xt in xsuite_x_by_model.items():
    plt.loglog(slices, np.abs(np.array(x_list_xt)-x_list_xt[-1]), '.-',
               label=f'Xsuite {model}')
plt.xlabel('Number of slices')
plt.legend()

plt.show()
