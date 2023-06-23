from cpymad.madx import Madx

import numpy as np
import xpart as xp
import xtrack as xt
import xdeps as xd

x = 1e-3
y = 1e-3
betx = 1
bety = 1
px = 0.01
py = 0.01

mad = Madx()
mad.input("""
    l = 1;
    angle = 0.1;
    !rlen = angle * l / (2 * sin(angle / 2));
    rlen = l;

    elm: sbend, angle=angle, k0=0.12, e1=0.1, e2=0.2, l=l;

    seq: sequence, l=rlen;
    elm: elm, at=rlen / 2;
    mk: marker, at=rlen;
    endsequence;

    beam;
    use,sequence=seq;
""")

mad.input(f"""
    twiss, betx={betx}, bety={bety}, x={x}, px={px}, y={y}, py={py};
""")

mad.input(f"""
    PTC_CREATE_UNIVERSE;
    PTC_CREATE_LAYOUT, model=2;
    PTC_twiss, x={x}, px={px}, y={y}, py={py}, betx={betx}, bety={bety};
    PTC_END;
""")

tw_ptc = xd.Table(mad.table.ptc_twiss)
tw_mad = xd.Table(mad.table.twiss)

line = xt.Line.from_madx_sequence(
    mad.sequence.seq, enable_edges=False, enable_fringes=True, allow_thick=True)

line.particle_ref = xp.Particles(mass0=xp.ELECTRON_MASS_EV, p0c=1e9)
line.build_tracker()
tw = line.twiss(method='4d',
                twiss_init=xt.TwissInit(x=x, y=y, betx=betx, bety=bety, px=px, py=py,
                                        line=line, element_name=line.element_names[0]),
                ele_start=0, ele_stop=len(line) - 1)
