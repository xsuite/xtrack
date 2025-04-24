import xtrack as xt
import xpart as xp
import xobjects as xo
import numpy as np

num_turns = 10000

line = xt.Line.from_json('lep_sol.json')
line.particle_ref.anomalous_magnetic_moment=0.00115965218128
spin_tune = line.particle_ref.anomalous_magnetic_moment[0]*line.particle_ref.gamma0[0]

tw = line.twiss4d(spin=True, radiation_integrals=True)

# All off
line['on_sol.2'] = 0
line['on_sol.4'] = 0
line['on_sol.6'] = 0
line['on_sol.8'] = 0
line['on_spin_bump.2'] = 0
line['on_spin_bump.4'] = 0
line['on_spin_bump.6'] = 0
line['on_spin_bump.8'] = 0
line['on_coupl_sol.2'] = 0
line['on_coupl_sol.4'] = 0
line['on_coupl_sol.6'] = 0
line['on_coupl_sol.8'] = 0
line['on_coupl_sol_bump.2'] = 0
line['on_coupl_sol_bump.4'] = 0
line['on_coupl_sol_bump.6'] = 0
line['on_coupl_sol_bump.8'] = 0

# RF
line['vrfc231'] = 12.65 # qs=0.6

# Bare machine
tw_bare = line.twiss(spin=True, radiation_integrals=True)

line.configure_radiation('mean')
tw_bare_rad = line.twiss(spin=True, eneloss_and_damping=True)

p_bare = xp.generate_matched_gaussian_bunch(
    line=line,
    nemitt_x=tw_bare_rad.eq_nemitt_x,
    nemitt_y=tw_bare_rad.eq_nemitt_y,
    sigma_z=np.sqrt(tw_bare_rad.eq_gemitt_zeta * tw_bare_rad.bets0),
    num_particles=100,
    engine='linear')
p_bare.zeta += tw_bare_rad.zeta[0]
p_bare.delta += tw_bare_rad.delta[0]
p_bare.spin_x = tw_bare_rad.spin_x[0]
p_bare.spin_y = tw_bare_rad.spin_y[0]
p_bare.spin_z = tw_bare_rad.spin_z[0]

line.configure_radiation(model='quantum')
line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu(omp_num_threads=10))
line.track(p_bare, num_turns=num_turns, turn_by_turn_monitor=True,
           with_progress=10)
mon_bare = line.record_last_track

pol_x_bare = mon_bare.spin_x.sum(axis=0)/mon_bare.state.sum(axis=0)
pol_y_bare = mon_bare.spin_y.sum(axis=0)/mon_bare.state.sum(axis=0)
pol_z_bare = mon_bare.spin_z.sum(axis=0)/mon_bare.state.sum(axis=0)
pol_bare = np.sqrt(pol_x_bare**2 + pol_y_bare**2 + pol_z_bare**2)

line.configure_radiation(model=None)
line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))

# Solenoids without bumps
line['on_sol.2'] = 1
line['on_sol.4'] = 1
line['on_sol.6'] = 1
line['on_sol.8'] = 1
line['on_coupl_sol.2'] = 1
line['on_coupl_sol.4'] = 1
line['on_coupl_sol.6'] = 1
line['on_coupl_sol.8'] = 1

tw_sol = line.twiss(spin=True, radiation_integrals=True)
line.configure_radiation(model='mean')
tw_sol_rad = line.twiss(spin=True, eneloss_and_damping=True)
p_sol = xp.generate_matched_gaussian_bunch(
    line=line,
    nemitt_x=tw_sol_rad.eq_nemitt_x,
    nemitt_y=tw_sol_rad.eq_nemitt_y,
    sigma_z=np.sqrt(tw_sol_rad.eq_gemitt_zeta * tw_sol_rad.bets0),
    num_particles=100,
    engine='linear')
p_sol.zeta += tw_sol_rad.zeta[0]
p_sol.delta += tw_sol_rad.delta[0]
p_sol.spin_x = tw_sol_rad.spin_x[0]
p_sol.spin_y = tw_sol_rad.spin_y[0]
p_sol.spin_z = tw_sol_rad.spin_z[0]

line.configure_radiation(model='quantum')
line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu(omp_num_threads=10))
line.track(p_sol, num_turns=num_turns, turn_by_turn_monitor=True,
           with_progress=10)
mon_sol = line.record_last_track
pol_x_sol = mon_sol.spin_x.sum(axis=0)/mon_sol.state.sum(axis=0)
pol_y_sol = mon_sol.spin_y.sum(axis=0)/mon_sol.state.sum(axis=0)
pol_z_sol = mon_sol.spin_z.sum(axis=0)/mon_sol.state.sum(axis=0)
pol_sol = np.sqrt(pol_x_sol**2 + pol_y_sol**2 + pol_z_sol**2)

line.configure_radiation(model=None)
line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))

# Solenoids with bumps
line['on_spin_bump.2'] = 1
line['on_spin_bump.4'] = 1
line['on_spin_bump.6'] = 1
line['on_spin_bump.8'] = 1
line['on_coupl_sol_bump.2'] = 1
line['on_coupl_sol_bump.4'] = 1
line['on_coupl_sol_bump.6'] = 1
line['on_coupl_sol_bump.8'] = 1

tw_sol_bump = line.twiss(spin=True, radiation_integrals=True)
line.configure_radiation(model='mean')
tw_sol_bump_rad = line.twiss(spin=True, eneloss_and_damping=True)
p_sol_bump = xp.generate_matched_gaussian_bunch(
    line=line,
    nemitt_x=tw_sol_bump_rad.eq_nemitt_x,
    nemitt_y=tw_sol_bump_rad.eq_nemitt_y,
    sigma_z=np.sqrt(tw_sol_bump_rad.eq_gemitt_zeta * tw_sol_bump_rad.bets0),
    num_particles=100,
    engine='linear')
p_sol_bump.zeta += tw_sol_bump_rad.zeta[0]
p_sol_bump.delta += tw_sol_bump_rad.delta[0]
p_sol_bump.spin_x = tw_sol_bump_rad.spin_x[0]
p_sol_bump.spin_y = tw_sol_bump_rad.spin_y[0]
p_sol_bump.spin_z = tw_sol_bump_rad.spin_z[0]
line.configure_radiation(model='quantum')
line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu(omp_num_threads=10))
line.track(p_sol_bump, num_turns=num_turns, turn_by_turn_monitor=True,
           with_progress=10)
mon_sol_bump = line.record_last_track
pol_x_sol_bump = mon_sol_bump.spin_x.sum(axis=0)/mon_sol_bump.state.sum(axis=0)
pol_y_sol_bump = mon_sol_bump.spin_y.sum(axis=0)/mon_sol_bump.state.sum(axis=0)
pol_z_sol_bump = mon_sol_bump.spin_z.sum(axis=0)/mon_sol_bump.state.sum(axis=0)
pol_sol_bump = np.sqrt(pol_x_sol_bump**2 + pol_y_sol_bump**2 + pol_z_sol_bump**2)
line.configure_radiation(model=None)
line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))


import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(pol_bare, label='bare')
plt.plot(pol_sol, label='sol')
plt.plot(pol_sol_bump, label='sol & bump')

plt.xlabel('turns')
plt.ylabel('polarization')
plt.legend()

plt.show()
