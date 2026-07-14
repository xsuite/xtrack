import xtrack as xt
import numpy as np

env = xt.load(
    '/Users/giadarol/xsuite_packages/xmask/examples/hllhc19/'
    'lhc_thick_03_tuned_and_leveled_bb_off.json')
line = env.b1

num_particles = 1000
num_turns = 5
p0 = line.build_particles(x=np.linspace(-1e-5, 1e-5, num_particles))

line.configure_bend_model(core='rot-kick-rot', num_multipole_kicks=7)
p=p0.copy()
print('Tracking with rot-kick-rot')
line.track(p, num_turns=num_turns, time=True)
t_rot_kick_rot = line.time_last_track

line.configure_bend_model(core='rot-kick-rot-low-order', num_multipole_kicks=10*7)
p=p0.copy()
print('Tracking with rot-kick-rot-low-order')
line.track(p, num_turns=num_turns, time=True)
t_rot_kick_rot_low_order = line.time_last_track

line.configure_bend_model(core='rot-kick-rot-high-order', num_multipole_kicks=7)
p=p0.copy()
print('Tracking with rot-kick-rot-high-order')
line.track(p, num_turns=num_turns, time=True)
t_rot_kick_rot_high_order = line.time_last_track

# Bar plot with the results (number on top of the bars)
import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1, figsize=(6.4*1.7, 4.8))
plt.bar(['rot-kick-rot-low-order (10 slices)', 'rot-kick-rot (1 slice)', 
         'rot-kick-rot-high-order (1 slice)'],
        [t_rot_kick_rot_low_order, t_rot_kick_rot, t_rot_kick_rot_high_order])
plt.ylabel('Tracking time (s)')
plt.title('Tracking time for different bend models\n'
          f'num_particles={num_particles}, num_turns={num_turns}')
for i, v in enumerate([t_rot_kick_rot_low_order, t_rot_kick_rot, t_rot_kick_rot_high_order]):
    plt.text(i, v + 0.01, f'{v:.2f} s', ha='center', va='bottom')
          
plt.show()

