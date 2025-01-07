import xtrack as xt

myline = env.new_line(name='myline', components=[
    env.new('q10', xt.Quadrupole, length=1.0, k1='kquad',
            # Place element center at s = 3.0
            at=3.0),
    env.new('q20', xt.Quadrupole, length=1.0, k1='-kquad',
            # Place element start at s = 5.0
            anchor='start', at=5.0),
    env.new('q30', xt.Quadrupole, length=1.0, k1='kquad',
            # Place element start at the end of q2
            anchor='start', at='end@q20'),
    env.new('q40', xt.Quadrupole, length=1.0, k1='-kquad',
            # Place element center at the end of q3
            anchor='center', at=5.0, from_='start@q30'),
    env.new('s40', xt.Sextupole, length=0.1) # Placed right after previous
    ])