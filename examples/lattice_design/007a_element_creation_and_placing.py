import xtrack as xt

env = xt.Environment()

# Create some quadrupole elements
env['kquad'] = 0.1
env.new('q1', xt.Quadrupole, length=1.0, k1='kquad')
env.new('q2', xt.Quadrupole, length=1.0, k1='-kquad')
env.new('q3', xt.Quadrupole, length=1.0, k1='kquad')
env.new('q4', xt.Quadrupole, length=1.0, k1='-kquad')
env.new('s4', xt.Sextupole, length=0.1)

myline = env.new_line(name='myline', components=[
    # Place element center at s = 3.0
    env.place('q1', at=3.0),
    # Place element start at s = 5.0
    env.place('q2', anchor='start', at=5.0),
    # Place element start at the end of q2
    env.place('q3', anchor='start', at='q2@end'),
    # Place element center at 5 m from the end of q3
    env.place('q4', anchor='center', at=5.0, from_='q3@start'),
    # Placed right after previous element
    env.place('s4')
    ])

tt = myline.get_table()
tt.show(cols=['s_start', 's_center', 's_end'])
# is:
# name             s_start      s_center         s_end
# drift_1                0          1.25           2.5
# q1                   2.5             3           3.5
# drift_2              3.5          4.25             5
# q2                     5           5.5             6
# q3                     6           6.5             7
# drift_3                7          8.75          10.5
# q4                  10.5            11          11.5
# s4                  11.5         11.55          11.6
# _end_point          11.6          11.6          11.6


# The elements can also be created directly in the line definition:
myline = env.new_line(name='myline', components=[
    env.new('q10', xt.Quadrupole, length=1.0, k1='kquad',
            # Place element center at s = 3.0
            at=3.0),
    env.new('q20', xt.Quadrupole, length=1.0, k1='-kquad',
            # Place element start at s = 5.0
            anchor='start', at=5.0),
    env.new('q30', xt.Quadrupole, length=1.0, k1='kquad',
            # Place element start at the end of q2
            anchor='start', at='q20@end'),
    env.new('q40', xt.Quadrupole, length=1.0, k1='-kquad',
            # Place element center at the end of q3
            anchor='center', at=5.0, from_='q30@start'),
    env.new('s40', xt.Sextupole, length=0.1) # Placed right after previous
    ])

tt = myline.get_table()
tt.show(cols=['s_start', 's_center', 's_end'])
# is:
# name             s_start      s_center         s_end
# drift_4                0          1.25           2.5
# q10                  2.5             3           3.5
# drift_5              3.5          4.25             5
# q20                    5           5.5             6
# q30                    6           6.5             7
# drift_6                7          8.75          10.5
# q40                 10.5            11          11.5
# s40                 11.5         11.55          11.6
# _end_point          11.6          11.6          11.6