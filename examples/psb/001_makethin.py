import xtrack as xt
from xtrack.slicing import Teapot, Strategy

line0 = xt.Line.from_json('psb_with_chicane.json')

# for nn in line0.element_dict.keys():
#     if isinstance (line0.element_dict[nn], xt.TrueBend):
#         ee_dct = line0.element_dict[nn].to_dict()
#         new_ele = xt.CombinedFunctionMagnet.from_dict(ee_dct)
#         line0.element_dict[nn] = new_ele

# for ii in [1, 2, 3, 4]:
#     line0[f'bi1.bsw1l1.{ii}'].num_multipole_kicks= 1

line0.build_tracker()
line0.twiss_default['method'] = '4d'
tw0 = line0.twiss()

line = line0.copy()

slicing_strategies = [
    Strategy(slicing=Teapot(1)),  # Default catch-all as in MAD-X
    Strategy(slicing=Teapot(50), element_type=xt.TrueBend),
    Strategy(slicing=Teapot(50), element_type=xt.CombinedFunctionMagnet),
]

print("Slicing thick elements...")
line.slice_in_place(slicing_strategies)
line.build_tracker()
tw = line.twiss()

import matplotlib.pyplot as plt
plt.close('all')
sp1 = plt.subplot(3,1,1)
plt.plot(tw0.s, tw0.betx, label='thick')
plt.plot(tw.s, tw.betx, label='thin')
plt.plot(tw0.s, tw0.bety, label='thick')
plt.plot(tw.s, tw.bety, label='thin')
plt.legend()
plt.ylabel('beta')
plt.subplot(3,1,2, sharex=sp1)
plt.plot(tw0.s, tw0.dx, label='thick')
plt.plot(tw.s, tw.dx, label='thin')
plt.ylabel('dispersion')
plt.subplot(3,1,3, sharex=sp1)
plt.plot(tw0.s, tw0.x, label='thick')
plt.plot(tw.s, tw.x, label='thin')
plt.ylabel('closed orbit')
plt.xlabel('s [m]')
plt.legend()
plt.show()


