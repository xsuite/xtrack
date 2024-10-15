import numpy as np
import xtrack as xt
from xtrack.mad_parser.loader import MadxLoader
from xtrack.json_utils import to_json
from cpymad.madx import Madx
import matplotlib.pyplot as plt

particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=7000e9)

which_beamline = 'lhcb1'
start_point = 'ip1.l1' if which_beamline == 'lhcb2' else 'ip1'

mad = Madx(stdout=False)
mad.call('lhc.seq')
mad.call('optics.madx')
mad.input(f'beam, sequence=lhcb1, particle=proton, energy=7000;')
mad.input(f'beam, sequence=lhcb2, bv=-1, particle=proton, energy=7000;')
mad.use(which_beamline)
line_ref = xt.Line.from_madx_sequence(mad.sequence[which_beamline], deferred_expressions=True)
line_ref.particle_ref = particle_ref
tw_ref = line_ref.twiss4d()
tw_init = tw_ref.get_twiss_init(start_point)

loader = MadxLoader(reverse_lines=['lhcb2'])
loader.load_file("lhc.seq")
loader.load_file("optics.madx")
env = loader.env
line = env.lines[which_beamline]
line.particle_ref = particle_ref
tw = line.twiss4d()
# For debugging purposes, can try this:
# tw = line.twiss4d(start=start_point, end='_end_point', init=tw_init, _continue_if_lost=True)

plt.subplot(2, 1, 1)
plt.plot(tw_ref.s, tw_ref.betx, label='betx_ref', linestyle='--')
plt.plot(tw.s, tw.betx, label='betx', linestyle='-')
plt.legend()
plt.suptitle(f'cpymad+madloader vs new madloader ({which_beamline})')
text = (
    fr'$\log(\Delta q_x)$ = {np.log10(np.abs(tw_ref.qx - tw.qx)):.2f}, '
    fr'$\log(\Delta q_y)$ = {np.log10(np.abs(tw_ref.qy - tw.qy)):.2f}'
    '\n'
    fr'$\log(\Delta dq_x)$ = {np.log10(np.abs(tw_ref.dqx - tw.dqx)):.2f}, '
    fr'$\log(\Delta dq_y)$ = {np.log10(np.abs(tw_ref.dqy - tw.dqy)):.2f}'
)
plt.title(text)

plt.subplot(2, 1, 2)
plt.plot(tw_ref.s, tw_ref.bety, label='bety_ref', linestyle='--')
plt.plot(tw.s, tw.bety, label='bety', linestyle='-')
plt.legend()

plt.show()

# dump jsons for comparison
line.discard_tracker()
line_ref.discard_tracker()

line.merge_consecutive_drifts()
line_ref.merge_consecutive_drifts()

dct = line.to_dict()
dct_ref = line_ref.to_dict()

def fmt(dct):
    el_list = []
    for name in dct['element_names']:
        el = dct['elements'][name]
        el_list.append(el)
    return el_list

to_json(fmt(dct), f'{which_beamline}.json', 2)
to_json(fmt(dct_ref), f'{which_beamline}_ref.json', 2)
