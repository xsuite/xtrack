import xtrack as xt
import numpy as np


def _plot_line(line):
    tt = line.get_table(attr=True)
    xt.twiss.TwissTable.plot(tt, yl='', yr='')


env = xt.Environment()

env.vars({
    'l.b1': 1.0,
    'l.q1': 0.5,
    's.ip': 10,
    's.left': -5,
    's.right': 5,
    'l.before_right': 1,
    'l.after_left2': 0.5,
})

# names, tab_sorted = handle_s_places(seq)
line = env.new_line(components=[
    env.new_element('b1', xt.Bend, length='l.b1'),
    env.new_element('q1', xt.Quadrupole, length='l.q1'),
    env.new_element('ip', xt.Marker, at='s.ip'),
    (
        env.new_element('before_before_right', xt.Marker),
        env.new_element('before_right', xt.Sextupole, length=1),
        env.new_element('right',xt.Quadrupole, length=1, at='s.right', from_='ip'),
        env.new_element('after_right', xt.Marker),
        env.new_element('after_right2', xt.Marker),
    ),
    env.new_element('left', xt.Quadrupole, length=1, at='s.left', from_='ip'),
    env.new_element('after_left', xt.Marker),
    env.new_element('after_left2', xt.Bend, length='l.after_left2'),
])

import matplotlib.pyplot as plt
plt.close('all')
line.survey().plot()

plt.show()