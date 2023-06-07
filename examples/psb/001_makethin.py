import xtrack as xt
from xtrack.slicing import Teapot, Strategy

line0 = xt.Line.from_json('psb_with_chicane.json')
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