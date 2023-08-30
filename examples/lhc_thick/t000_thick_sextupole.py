import numpy as np
import xtrack as xt

k2 = 3.
k2s = 5.
length = 0.4

line_thin = xt.Line(elements=[
    xt.Drift(length=length/2),
    xt.Multipole(knl=[0., 0., k2 * length],
                 ksl=[0., 0., k2s * length],
                 length=length),
    xt.Drift(length=length/2),
])
line_thin.build_tracker()

line_thick = xt.Line(elements=[
    xt.Sextupole(k2=k2, k2s=k2s, length=length),
])
line_thick.build_tracker()

p = xt.Particles(
    p0c=6500e9,
    x=[-3e-2, -2e-3, 0, 1e-3, 2e-3, 3e-2],
    px=[1e-6, 2e-6,  0, 2e-6, 1e-6, 1e-6],
    y=[-2e-2, -5e-3, 0, 5e-3, -4e-3, 2e-2],
    py=[2e-6, 4e-6,  0, 2e-6, 1e-6, 1e-6],
    delta=[1e-3, 2e-3, 0, -2e-3, -1e-3, -1e-3],
    zeta=[-5e-2, -6e-3, 0, 6e-3, 5e-3, 5e-2],
)

p_thin = p.copy()
p_thick = p.copy()

line_thin.track(p_thin)
line_thick.track(p_thick)

assert np.allclose(p_thin.x, p_thick.x, rtol=0, atol=1e-14)
assert np.allclose(p_thin.px, p_thick.px, rtol=0, atol=1e-14)
assert np.allclose(p_thin.y, p_thick.y, rtol=0, atol=1e-14)
assert np.allclose(p_thin.py, p_thick.py, rtol=0, atol=1e-14)
assert np.allclose(p_thin.delta, p_thick.delta, rtol=0, atol=1e-14)
assert np.allclose(p_thin.zeta, p_thick.zeta, rtol=0, atol=1e-14)

# slicing
Teapot = xt.slicing.Teapot
Strategy = xt.slicing.Strategy

line_sliced = line_thick.copy()
line_sliced.slice_thick_elements(
    slicing_strategies=[Strategy(slicing=Teapot(5))])
line_sliced.build_tracker()

p_sliced = p.copy()
line_sliced.track(p_sliced)

assert np.allclose(p_sliced.x, p_thick.x, rtol=0, atol=5e-6)
assert np.allclose(p_sliced.px, p_thick.px, rtol=0.01, atol=1e-10)
assert np.allclose(p_sliced.y, p_thick.y, rtol=0, atol=5e-6)
assert np.allclose(p_sliced.py, p_thick.py, rtol=0.01, atol=1e-10)
assert np.allclose(p_sliced.delta, p_thick.delta, rtol=0, atol=1e-14)
assert np.allclose(p_sliced.zeta, p_thick.zeta, rtol=0, atol=2e-7)

line_thin.track(p_thin, backtrack=True)
line_thick.track(p_thick, backtrack=True)
line_sliced.track(p_sliced, backtrack=True)

assert np.allclose(p_thin.x, p.x, rtol=0, atol=1e-14)
assert np.allclose(p_thin.px, p.px, rtol=0, atol=1e-14)
assert np.allclose(p_thin.y, p.y, rtol=0, atol=1e-14)
assert np.allclose(p_thin.py, p.py, rtol=0, atol=1e-14)
assert np.allclose(p_thin.delta, p.delta, rtol=0, atol=1e-14)

assert np.allclose(p_thick.x, p.x, rtol=0, atol=1e-14)
assert np.allclose(p_thick.px, p.px, rtol=0, atol=1e-14)
assert np.allclose(p_thick.y, p.y, rtol=0, atol=1e-14)
assert np.allclose(p_thick.py, p.py, rtol=0, atol=1e-14)
assert np.allclose(p_thick.delta, p.delta, rtol=0, atol=1e-14)

assert np.allclose(p_sliced.x, p.x, rtol=0, atol=1e-14)
assert np.allclose(p_sliced.px, p.px, rtol=0, atol=1e-14)
assert np.allclose(p_sliced.y, p.y, rtol=0, atol=1e-14)
assert np.allclose(p_sliced.py, p.py, rtol=0, atol=1e-14)
assert np.allclose(p_sliced.delta, p.delta, rtol=0, atol=1e-14)
assert np.allclose(p_sliced.zeta, p.zeta, rtol=0, atol=1e-14)

from cpymad.madx import Madx
mad = Madx()
mad.input(f"""
    knob_a := 1.0;
    knob_b := 2.0;
    knob_l := 0.4;
    ss: sequence, l:=2 * knob_b, refer=entry;
        elem: sextupole, at=0, l:=knob_l, k2:=3*knob_a, k2s:=5*knob_b;
    endsequence;
    """)
mad.beam()
mad.use(sequence='ss')

line_mad = xt.Line.from_madx_sequence(mad.sequence.ss, allow_thick=True,
                                      deferred_expressions=True)
line_mad.build_tracker()

elem = line_mad['elem']
assert isinstance(elem, xt.Sextupole)
assert np.isclose(elem.length, 0.4, rtol=0, atol=1e-14)
assert np.isclose(elem.k2, 3, rtol=0, atol=1e-14)
assert np.isclose(elem.k2s, 10, rtol=0, atol=1e-14)

line_mad.vv['knob_a'] = 0.5
line_mad.vv['knob_b'] = 0.6
line_mad.vv['knob_l'] = 0.7

assert np.isclose(elem.length, 0.7, rtol=0, atol=1e-14)
assert np.isclose(elem.k2, 1.5, rtol=0, atol=1e-14)
assert np.isclose(elem.k2s, 3.0, rtol=0, atol=1e-14)

