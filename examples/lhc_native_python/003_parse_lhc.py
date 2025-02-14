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

# Adapt builders
for nn in env.lines.keys():
    bb = env.lines[nn].builder.__class__(new_env)
    bb.__dict__.update(env.lines[nn].builder.__dict__)
    bb.env = new_env
    this_rename = new_env.lines[nn]._renamed_elements
    for cc in bb.components:
        cc.name = this_rename.get(cc.name, cc.name)
        cc.from_ = this_rename.get(cc.from_, cc.from_)

    if nn in reverse_lines:
        length = env.lines[nn].get_length()
        bb.components = bb.components[::-1]
        for cc in bb.components:
            if cc.at is not None:
                assert isinstance(cc.at, str) # Float still to be handled
                if cc.from_ is not None:
                    cc.at = f'-({cc.at})'
                else:
                    cc.at = f'({length} - {cc.at})'
    new_env.lines[nn].builder = bb


new_env.lhcb1.particle_ref = xt.Particles(p0c=7e12)
new_env.lhcb2.particle_ref = xt.Particles(p0c=7e12)

new_env.lhcb1.twiss4d().plot()
new_env.lhcb2.twiss4d(reverse=True).plot()

# Check builder
new_env.lhcb2.builder.name = None # Not to overwrite the line
lb2 = new_env.lhcb2.builder.build()
lb2.particle_ref = xt.Particles(p0c=7e12)
lb2.twiss4d(reverse=True).plot()