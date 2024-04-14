import xtrack as xt
from cpymad.madx import Madx

import numpy as np

mad = Madx()

tilt_deg = 12
shift_x = -2e-3
shift_y = 3e-3
k1 = 0.2
tilt_rad = np.deg2rad(tilt_deg)

x_test = 1e-3
px_test = 2e-3
y_test = 3e-3
py_test = 4e-3

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

select,flag=error,clear;
select,flag=error,pattern=elm;
ealign, dx={shift_x}, dy={shift_y};

""")

elm = xt.Quadrupole(k1=k1, length=1)

elm_tilted = xt.Quadrupole(k1=k1, length=1, rot_s_rad=tilt_rad,
                           shift_x=shift_x, shift_y=shift_y)

lsandwitch = xt.Line(elements=[
    xt.XYShift(dx=shift_x, dy=shift_y),
    xt.SRotation(angle=tilt_deg),
    elm,
    xt.SRotation(angle=-tilt_deg),
    xt.XYShift(dx=-shift_x, dy=-shift_y)
])
lsandwitch.build_tracker()

l_tilted = xt.Line(elements=[elm_tilted])
l_tilted.build_tracker()

lmad = xt.Line.from_madx_sequence(mad.sequence.seq, enable_align_errors=True)
lmad.build_tracker()

p0 = xt.Particles(x=x_test, px=px_test, y=y_test, py=py_test, p0c=1e9)

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

assert elm.rot_s_rad == 0
elm.rot_s_rad = tilt_rad
elm.shift_x = shift_x
elm.shift_y = shift_y
pprop = p0.copy()
elm.track(pprop)

pprop.get_table().show()

pref = psandwitch

assert_allclose = np.testing.assert_allclose
for pp in [plinetilted, pmad, peletitled, pprop]:
    assert_allclose(pp.x, pref.x, rtol=0, atol=1e-12)
    assert_allclose(pp.px, pref.px, rtol=0, atol=1e-12)
    assert_allclose(pp.y, pref.y, rtol=0, atol=1e-12)
    assert_allclose(pp.py, pref.py, rtol=0, atol=1e-12)
    assert_allclose(pp.zeta, pref.zeta, rtol=0, atol=1e-12)
    assert_allclose(pp.delta, pref.delta, rtol=0, atol=1e-12)
