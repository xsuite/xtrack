from pymadng import MAD
import numpy as np

mad = MAD()

mad.load('MAD', "sequence")
mad.load('MAD.element', "sbend")

mad.send("""
seq = sequence { l = 2, sbend { l = 2, angle=math.pi/4 , knl={0, 0, 2.}}, beam = beam {} }
""")
# mad.send("""
# seq = sequence { l = 2, sbend { l = 2, angle=math.pi/4}, beam = beam {} }
# """)
methods = {
    'yoshida2': 2,
    'yoshida4': 4,
    'yoshida6': 6,
    'yoshida8': 8,
    'teapot': 'teapot2',
}

models = {
    'bend-kick-bend': "'TKT'",
    'rot-kick-rot': "'DKD'",
}
slices = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
x_list = []
for ss in slices:
    mad['tbl', 'flw'] = mad.track(
        sequence='seq', observe=0, save='"atall"', X0={'x': 0e-3},
        method=6, model=models['rot-kick-rot'],
        nslice=ss)
    df = mad.tbl.to_df()
    x_list.append(df['x'].values[-1])

import xtrack as xt

env = xt.Environment()
line = env.new_line(components=[
    env.new('b', 'Bend', length=2, angle=np.pi/4, knl=[0, 0, 2.],
            model='rot-kick-rot', 
            integrator='yoshida4', # is actually yoshida6
            num_multipole_kicks=1000)
])

line.set_particle_ref('proton', energy0=1e9)
x_list_xt = []
for ss in slices:
    line.configure_bend_model(edge='full', num_multipole_kicks=7*ss)
    tw = line.twiss(betx=1, bety=1, x=0e-3)
    x_list_xt.append(tw.x[-1])

import matplotlib.pyplot as plt
plt.close('all')
plt.loglog(slices, np.abs(np.array(x_list)-x_list[-1]), '.-', label='MAD-NG')
plt.loglog(slices, np.abs(np.array(x_list_xt)-x_list_xt[-1]), '.-', label='Xsuite')
plt.xlabel('Number of slices')
plt.legend()

plt.show()
