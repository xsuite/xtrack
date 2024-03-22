import xtrack as xt
from cpymad.madx import Madx

import numpy as np

mad = Madx()

tilt_deg = 12
k1 = 0.2
tilt_rad = np.deg2rad(tilt_deg)

mad.input(f"""
k1={k1};

elm: quadrupole,
    k1:=k1,
    l=1,
    tilt={tilt_rad};

seq: sequence, l=1;
elm: elm, at=0.5;
endsequence;

beam;
use, sequence=seq;
""")

elm = xt.Quadrupole(k1=k1, length=1)

elm_tilted = xt.Quadrupole(k1=k1, length=1, tilt=tilt_deg)

lsandwitch = xt.Line(elements=[
    xt.SRotation(angle=tilt_deg),
    elm,
    xt.SRotation(angle=-tilt_deg)
])
lsandwitch.build_tracker()

l_tilted = xt.Line(elements=[elm_tilted])
l_tilted.build_tracker()

lmad = xt.Line.from_madx_sequence(mad.sequence.seq)
lmad.build_tracker()


p0 = xt.Particles(x=1e-3, px=2e-3, y=3e-3, py=4e-3, p0c=1e9)

pmad = p0.copy()
lmad.track(pmad)

psandwitch = p0.copy()
lsandwitch.track(psandwitch)

plinetilted = p0.copy()
l_tilted.track(plinetilted)

peletitled = p0.copy()
elm_tilted.track(peletitled)

pmad.get_table().show()
psandwitch.get_table().show()
plinetilted.get_table().show()
peletitled.get_table().show()

