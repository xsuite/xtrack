from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField
import numpy as np



class TiltedSolenoid:

    def __init__(self, L, a, B0, theta):

        self._sf = SolenoidField(L=L, a=a, B0=B0, z0=0)
        self.theta = theta
        self._break = False

    def get_field(self, x, y, z):

        if self._break:
            breakpoint()

        theta = self.theta
        stheta = np.sin(theta)
        ctheta = np.cos(theta)
        x_sol = x * ctheta + z * stheta
        z_sol = -x * stheta + z * ctheta
        y_sol = y

        bx_sol, by_sol, bz_sol = self._sf.get_field(x_sol, y_sol, z_sol)

        bx = bx_sol * np.cos(theta) - bz_sol * np.sin(theta)
        bz = bx_sol * np.sin(theta) + bz_sol * np.cos(theta)
        by = by_sol

        return bx, by, bz


theta = -0.015
sf = TiltedSolenoid(L=3., a=0.2, B0=2., theta=theta)

s = np.linspace(-4, 4, 201)
bx, by, bz = sf.get_field(0*s, 0*s, s)


import xtrack as xt
env = xt.Environment()
env.new_particle('ref_part', 'positron', energy0=45.6e9)
p = env['ref_part']

ks = bz / p.rigidity0[0]
k0s = bx / p.rigidity0[0]
k0 = by / p.rigidity0[0]

env['on_sol'] = 1

ele_names = []
boris_ele_names = []
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
        # # x0=x0_mid,
        # y0=0,
    )
    ele_names.append(f'sol_slice_{ii}')

    new_boris = xt.BorisSpatialIntegrator(
        fieldmap_callable=sf.get_field,
        s_start=s_entry,
        s_end=s_exit,
        n_steps=100,
        verbose=False,
    )
    env.elements[f'boris_{ii}'] = new_boris
    boris_ele_names.append(f'boris_{ii}')

line_solenoid = env.new_line(components=ele_names)
line_solenoid.set_particle_ref('ref_part')

line_boris = env.new_line(components=boris_ele_names)
line_boris.set_particle_ref('ref_part')

tw = line_solenoid.twiss4d(betx=1, bety=1)
tw_boris = line_boris.twiss4d(betx=1, bety=1, include_collective=True)

fcc = env.new_line(length=100, components=[
    env.new('ip', xt.Marker, at=50)
])

fcc.insert(line_solenoid, anchor='center', at=0, from_='ip')

fcc.select('ip', "sol_slice_183").get_table()

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(s, bx, '.-',label='bx')
plt.plot(s, by, '.-', label='by')
plt.plot(s, bz, '.-', label='bz')
plt.legend()
plt.xlabel('s [m]')
plt.ylabel('B [T]')
plt.title('Field on the reference trajectory')
plt.grid()

plt.figure(2)
ax1 = plt.subplot(211)
ax1.plot(tw.s, tw.x, label='Xsuite')
ax1.plot(tw_boris.s, tw_boris.x, '--', label='Boris + Fieldmap')
ax1.legend()
ax1.set_xlabel('s [m]')
ax1.set_ylabel('x [m]')
ax1.grid()
ax2 = plt.subplot(212, sharex=ax1)
ax2.plot(tw.s, tw.y, label='Xsuite')
ax2.plot(tw_boris.s, tw_boris.y, '--', label='Boris + Fieldmap')
ax2.legend()
ax2.set_xlabel('s [m]')
ax2.set_ylabel('y [m]')
ax2.grid()

plt.figure(3)
ax31 = plt.subplot(211)
ax31.plot(tw.s, tw.kin_px, label='Xsuite')
ax31.plot(tw_boris.s, tw_boris.kin_px, '--', label='Boris + Fieldmap')
ax31.legend()
ax31.set_xlabel('s [m]')
ax31.set_ylabel(r'$p_{x,kin}$')
ax31.grid()
ax32 = plt.subplot(212, sharex=ax31)
ax32.plot(tw.s, tw.kin_py, label='Xsuite')
ax32.plot(tw_boris.s, tw_boris.kin_py, '--', label='Boris + Fieldmap')
ax32.legend()
ax32.set_xlabel('s [m]')
ax32.set_ylabel(r'$p_{y,kin}$')
ax32.grid()
plt.subplots_adjust(left=.15)

plt.show()
