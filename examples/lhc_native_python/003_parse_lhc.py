import xtrack as xt
from xtrack.environment import _reverse_element

env = xt.load_madx_lattice('../../test_data/lhc_2024/lhc.seq')
env.vars.load_madx('../../test_data/lhc_2024/injection_optics.madx')

reverse_lines = ['lhcb2']

rlines = {}
for nn in reverse_lines:
    ll = env.lines[nn]
    llr = ll.copy()

    for enn in llr.element_names:
        _reverse_element(llr, enn)

    llr.discard_tracker()
    llr.element_names = llr.element_names[::-1]

    rlines[nn] = llr

all_lines = {}
for nn in env.lines.keys():
    if nn in rlines:
        all_lines[nn] = rlines[nn]
    else:
        all_lines[nn] = env.lines[nn]

new_env = xt.Environment(lines=all_lines)

new_env.lhcb1.particle_ref = xt.Particles(p0c=7e12)
new_env.lhcb2.particle_ref = xt.Particles(p0c=7e12)

new_env.lhcb1.twiss4d().plot()
new_env.lhcb2.twiss4d(reverse=True).plot()