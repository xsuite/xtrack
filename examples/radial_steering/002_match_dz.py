import xtrack as xt
from scipy.constants import c as clight

env = xt.load(['../../test_data/ps_sftpro/ps.seq',
               '../../test_data/ps_sftpro/ps_hs_sftpro.str'])
line = env.ps
line.set_particle_ref('proton', kinetic_energy0=500e6)

tt = line.get_table()
tt_bend = tt.rows[tt.element_type == 'Bend']

env.set(tt_bend, model='rot-kick-rot', integrator='yoshida4',
        num_multipole_kicks=20)

tw4d = line.twiss4d()

hrf = 8
line['frf'] = hrf / tw4d.circumference * tw4d.beta0 * clight
line['vrf'] = 20e3

line['pa.c40.77'].voltage = 'vrf'
line['pa.c40.77'].frequency = 'frf'

tw6d = line.twiss6d()

env['circumference'] = tw4d.circumference
env['df_hz'] = 0.  # desired shift in RF frequency

env['dzeta'] = 'circumference * df_hz / frf'
env.new('z_shift', xt.ZetaShift, dzeta='dzeta')
line.append('z_shift')

tw0 = line.twiss6d()
env['df_hz'] = 1000.
tw1 = line.twiss6d()

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(tw0.s, tw0.x, label=f'df_hz=0 Hz, delta={tw0.delta[0]:.2e}')
plt.plot(tw1.s, tw1.x, label=f'df_hz=10 Hz, delta={tw1.delta[0]:.2e}')
plt.xlabel('s [m]')
plt.ylabel(r'$\delta$')
plt.legend()
plt.title('PS with radial steering matched via ZetaShift')

tt_bpm = tt.rows['pr\.bpm[0-9].*']

env['df_hz'] = 1234.
tw_meas = line.twiss6d()
tw_meas_bpm = tw_meas.rows[tt_bpm.name]
x_meas = tw_meas_bpm.x

env['df_hz'] = 0.

targets = []
for nn, xx in zip(tt_bpm.name, x_meas):
    targets.append(xt.Target('x', xx, at=nn, tol=1e-6))

opt = line.match(
    solve=False,
    vary=xt.Vary('df_hz', limits=(-5000, 5000), step=10.),
    targets=targets)
print('Targets before matching:')
opt.target_status()

# March
opt.step(10)

print('Targets after matching:')
opt.target_status()
print(f'Matched df_hz = {env["df_hz"]} Hz')
