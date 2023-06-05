import xtrack as xt
import xpart as xp

from xtrack.slicing import Teapot, Strategy, Slicer

line = xt.Line.from_json('lhc_thick_with_knobs.json')
line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, energy0=7e12)
line.build_tracker()
tw = line.twiss(method='4d')

line.unfreeze()

line0 = line.copy()

slicing_strategies = [
    Strategy(slicing=Teapot(1)),  # Default catch-all as in MAD-X
    Strategy(slicing=Teapot(2), name=r'^mb.*'),
    Strategy(slicing=Teapot(4), name=r'^mq.*'),
    Strategy(slicing=Teapot(16), name=r'^mqx.*'),
    Strategy(
        slicing=Teapot(4),
        name=r'(mbx|mbrb|mbrc|mbrs|mbh|mqwa|mqwb|mqy|mqm|mqmc|mqml)\..*',
    ),
    Strategy(slicing=Teapot(2), name=r'(mqt|mqtli|mqtlh)\..*'),
]



print("Slicing thick elements...")
Slicer(line, slicing_strategies).slice_in_place()

print("Building tracker...")
line.build_tracker()

print("Twiss...")
tw_thin = line.twiss(method='4d')

print(f'qx diff = {tw.qx} - {tw_thin.qx} = {tw.qx - tw_thin.qx}')
print(f'qx diff = {tw.qy} - {tw_thin.qy} = {tw.qy - tw_thin.qy}')
