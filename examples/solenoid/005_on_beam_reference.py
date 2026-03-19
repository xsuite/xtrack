from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField
import numpy as np

sf = SolenoidField(L=3., a=0.2, B0=2., z0=0)

theta = -0.015
s = np.linspace(-4, 4, 201)

s_sol = s * np.cos(theta)
x_sol = s * np.sin(theta)
y_sol = 0 * x_sol

bx_sol, by_sol, bz_sol = sf.get_field(x_sol, y_sol, s_sol)

bx = bx_sol * np.cos(theta) - bz_sol * np.sin(theta)
bz = bx_sol * np.sin(theta) + bz_sol * np.cos(theta)
by = by_sol

import xtrack as xt
env = xt.Environment()
env.new_particle('ref_part', 'positron', energy0=45.6e9)
p = env['ref_part']

ks = bz / p.rigidity0[0]
k0s = bx / p.rigidity0[0]
k0 = by / p.rigidity0[0]

env['on_sol'] = 1

ele_names = []
for ii in range(len(s)-1):
    ks_entry = ks[ii]
    ks_exit = ks[ii+1]
    k0s_entry = k0s[ii]
    k0s_exit = k0s[ii+1]
    k0_entry = k0[ii]
    k0_exit = k0[ii+1]
    s_entry = s[ii]
    s_exit = s[ii+1]

    length = s_exit - s_entry
    s_mid = 0.5 * (s_entry + s_exit)
    x0_mid = s_mid * np.tan(theta)

    env.new(f'sol_slice_{ii}', xt.VariableSolenoid,
        length=length,
        ks_profile=[ks_entry * env.ref['on_sol'], ks_exit * env.ref['on_sol']],
        knl=[0.5 * (k0_exit + k0_entry) * length * env.ref['on_sol']],
        ksl=[0.5 * (k0s_exit + k0s_entry) * length * env.ref['on_sol']],
        x0=x0_mid,
        y0=0,
    )
    ele_names.append(f'sol_slice_{ii}')

line_solenoid = env.new_line(components=ele_names)
line_solenoid.set_particle_ref('ref_part')

tw = line_solenoid.twiss4d(betx=1, bety=1)

fcc = env.new_line(length=100, components=[
    env.new('ip', xt.Marker, at=50)
])

fcc.insert(line_solenoid, anchor='center', at=0, from_='ip')

fcc.select('ip', "sol_slice_183").get_table()

import matplotlib.pyplot as plt
plt.close('all')
plt.plot(s, bx, '.-',label='bx')
plt.plot(s, by, '.-', label='by')
plt.plot(s, bz, '.-', label='bz')
plt.legend()
plt.xlabel('s [m]')
plt.ylabel('B [T]')
plt.title('Field on the reference trajectory')
plt.grid()

plt.show()
