import xtrack as xt
import numpy as np

env = xt.Environment()
line = env.new_line(components=[
    env.new('m', xt.Multipole, length=2)])

array = np.array
p = xt.Particles.from_dict(
    {'mass0': np.float64(510998.95),
 'ay': array([0.]),
 'charge_ratio': array([1.]),
 'particle_id': array([0]),
 'spin_z': array([0]),
 'q0': np.float64(1.0),
 # 'spin_y': array([6.26131193e-29]),
 # 'py': array([-8.95174893e-25]),
 'pdg_id': array([0]),
 'state': array([1]),
 'at_turn': array([0]),
 # 'zeta': array([0.13147089]),
 # 'y': array([-3.3910354e-23]),
 'start_tracking_at_element': np.int64(-1),
 'ax': array([0.]),
 'weight': array([1.]),
 'at_element': array([0]),
 'anomalous_magnetic_moment': array([0.00115965]),
 'parent_particle_id': array([0]),
 't_sim': np.float64(8.892442545593747e-05),
 'px': array([7.3740031e-07]),
 'spin_x': array([1.]),
 'chi': array([1.]),
 # 'x': array([1.e-05]),
 's': array([0.]),
 # 'delta': ([0.001]),
 'p0c': array([4.55850834e+10]),
 'beta0': array([1.]),
 'gamma0': array([89207.7828766])})

# Initialize spin along x axis !
p.spin_x = 1
p.spin_y = 0
p.spin_z = 0

line.configure_spin('auto')

line.track(p)