import xtrack as xt

env = xt.Environment()

env.new('q1', 'Quadrupole', length=2.0)
components=[]

for ii in range(1000):
    # components.extend([
    #     env.new(f'c{ii}q1', 'Quadrupole', length=2.0, anchor='start', at=1. + 50 * ii),
    #     env.new(f'c{ii}q2', 'q1', anchor='start', at=10., from_=f'end@c{ii}q1'),
    #     env.new(f'c{ii}s2', 'Sextupole', length=0.1, anchor='end', at=-1., from_=f'start@c{ii}q2'),
    #     env.new(f'c{ii}q3', 'Quadrupole', length=2.0, at=20.+ 50 * ii),
    #     env.new(f'c{ii}q4', f'c{ii}q3', anchor='start', at=f'end@c{ii}q3'),
    #     env.new(f'c{ii}q5', f'c{ii}q3'),
    #     env.new(f'c{ii}m2', 'Marker', at=f'start@c{ii}q2'),
    #     env.new(f'c{ii}m2_0', 'Marker', at=f'start@c{ii}m2'),
    #     env.new(f'c{ii}m2_1', 'Marker', at=f'end@c{ii}m2'),
    #     env.new(f'c{ii}m2_1_0', 'Marker', at=f'start@c{ii}m2_1'),
    #     env.new(f'c{ii}m1', 'Marker', at=f'start@c{ii}q1'),
    #     env.new(f'c{ii}m4', 'Marker', at=f'start@c{ii}q4'),
    #     env.new(f'c{ii}m3', 'Marker', at=f'end@c{ii}q3'),
    # ])

    components.extend([
        env.new(f'c{ii}q1', 'Quadrupole', length=2.0, anchor='start', at=1. + 50 * ii),
        env.new(f'c{ii}q2', 'q1', anchor='start', at=10., from_=f'end@c{ii}q1'),
        env.new(f'c{ii}s2', 'Sextupole', length=0.1, anchor='end', at=-1., from_=f'start@c{ii}q2'),
        env.new(f'c{ii}q3', 'Quadrupole', length=2.0, at=20.+ 50 * ii),
        env.new(f'c{ii}q4', f'c{ii}q3', anchor='start', at=f'end@c{ii}q3'),
        env.new(f'c{ii}q5', f'c{ii}q3'),
        env.new(f'c{ii}m2_0', 'Marker', at=f'start@c{ii}q2'),
        env.new(f'c{ii}m2', 'Marker', at=f'start@c{ii}q2'),
        env.new(f'c{ii}m2_1_0', 'Marker',at=f'start@c{ii}q2'),
        env.new(f'c{ii}m2_1', 'Marker', at=f'start@c{ii}q2'),
        env.new(f'c{ii}m1', 'Marker', at=f'start@c{ii}q1'),
        env.new(f'c{ii}m3', 'Marker', at=f'end@c{ii}q3'),
        env.new(f'c{ii}m4', 'Marker', at=f'start@c{ii}q4'),
    ])

print('Components built')

import time
t0 = time.time()
line = env.new_line(components=components)
t1 = time.time()
print(f'Line built in {t1-t0} s')

# line.get_table().show(cols=['name', 's_start', 's_end', 's_center'])