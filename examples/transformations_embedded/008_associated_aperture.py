from cpymad.madx import Madx
import xtrack as xt

mad = Madx(stdout=False)

mad.input("""
k1=0.2;
tilt=0.1;

elm: sbend,
    k1:=k1,
    l=1,
    angle=0.1,
    tilt=0.2,
    apertype="rectellipse",
    aperture={0.1,0.2,0.11,0.22},
    aper_tol={0.1,0.2,0.3},
    aper_tilt:=tilt,
    aper_offset={0.2, 0.3};

seq: sequence, l=1;
elm: elm, at=0.5;
endsequence;

beam;
use, sequence=seq;
""")

line = xt.Line.from_madx_sequence(
    sequence=mad.sequence.seq,
    allow_thick=True,
    install_apertures=True,
    deferred_expressions=True,
)

assert line['elm'].name_associated_aperture == 'elm_aper'
assert type(line['elm_aper']) == xt.LimitRectEllipse

line.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(slicing=xt.Teapot(3), element_type=xt.Bend),
    ])