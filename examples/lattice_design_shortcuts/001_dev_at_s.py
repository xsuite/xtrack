import xtrack as xt

def _plot_line(line):
    tt = line.get_table(attr=True)
    xt.twiss.TwissTable.plot(tt, yl='', yr='')

class Place:
    def __init__(self, name, s, from_=None):
        self.name = name
        self.s = s
        self.from_ = from_

env = xt.Environment()

seq = [
    Place(env.new_element('ip', xt.Marker), s=10),
    Place(env.new_element('left', xt.Marker), s=-5, from_='ip'),
    env.new_element('after_left', xt.Marker),
    Place(env.new_element('right', xt.Marker),s=+5, from_='ip'),
]

s_dct = {}
n_resolved = 0
n_resolved_prev = -1
while n_resolved != n_resolved_prev:
    for ss in seq:
        if ss.from_ is None:
            s_dct[ss.name] = ss.s
            n_resolved += 1
        else:
            if ss.from_ in s_dct:
                s_dct[ss.name] = s_dct[ss.from_] + ss.s
                n_resolved += 1
    n_resolved_prev = n_resolved



