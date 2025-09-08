from cpymad.madx import Madx
import xtrack as xt
import xobjects as xo
import numpy as np

mad = Madx()

mad.input("""

    on_srot = 1;
    pi = 3.14159265358979323846;

    rs2: srotation, angle=-1.04*on_srot;

    r3: yrotation, angle=-0.1;
    r4: yrotation, angle=0.1;

    rx1 : xrotation, angle=0.1;
    rx2 : xrotation, angle=-0.1;

    bh1 : sbend, angle=0.1, k0=1e-22, l=0.1;
    bh2 : sbend, angle=-0.1, k0=1e-22, l=0.1;

    bv1: sbend, tilt=pi/2, angle=0.2, k0=1e-22, l=0.1;
    bv2: sbend, tilt=pi/2, angle=-0.2, k0=1e-22, l=0.1;

    beam;
    ss: sequence,l=20;
        rs2, at=5.5;
        rx1, at=6;
        rx2, at=7;
        r3, at=8;
        r4, at=9;
        bh1, at=10;
        bh2, at=11;
        bv1, at=12;
        bv2, at=14;
        end: marker, at=16;
    endsequence;

    use,sequence=ss;
    twiss,betx=1,bety=1,x=1e-3,y=2e-3;
    survey;

    ptc_create_universe;
    ptc_create_layout, model=1, method=6, exact=True, NST=100;
    ptc_align;
    ptc_twiss, icase=56, betx=1., bety=1., betz=1,x=1e-3, y=2e-3;

    """)

line = xt.Line.from_madx_sequence(mad.sequence.ss)
line.particle_ref = xt.Particles(p0c=1E9)

line['bh1'].k0_from_h = False
line['bh2'].k0_from_h = False
line['bh1'].k0 = 0
line['bh2'].k0 = 0

sv_mad = xt.Table(mad.table.survey)
tw_ptc = xt.Table(mad.table.ptc_twiss)

line.config.XTRACK_GLOBAL_XY_LIMIT = None
line.config.XTRACK_USE_EXACT_DRIFTS = True

sv = line.survey()
tw = line.twiss(betx=1, bety=1, x=1e-3, y=2e-3)

p = tw.x[:, None] * sv.ex + tw.y[:, None] * sv.ey + sv.p0
X = p[:, 0]
Y = p[:, 1]
Z = p[:, 2]

assert (tw.name == np.array([
    'ss$start', 'drift_0', 'rs2', 'drift_1', 'rx1', 'drift_2', 'rx2',
    'drift_3', 'r3', 'drift_4', 'r4', 'drift_5', 'bh1', 'drift_6',
    'bh2', 'drift_7', 'bv1', 'drift_8', 'bv2', 'drift_9', 'end',
    'drift_10', 'ss$end', '_end_point'
], dtype=object)).all()

assert (tw_ptc.name == np.array([
    'ss$start:1', 'drift_0:0', 'rs2:1', 'drift_1:0', 'rx1:1',
    'drift_2:0', 'rx2:1', 'drift_3:0', 'r3:1', 'drift_4:0', 'r4:1',
    'drift_5:0', 'bh1:1', 'drift_6:0', 'bh2:1', 'drift_7:0', 'bv1:1',
    'drift_8:0', 'bv2:1', 'drift_9:0', 'end:1', 'drift_10:0',
    'ss$end:1'
], dtype=object)).all()

# MAD gives results at the end of the element
xo.assert_allclose(tw.x[1:], tw_ptc.x, atol=1e-14, rtol=0)
xo.assert_allclose(tw.y[1:], tw_ptc.y, atol=1e-14, rtol=0)

xo.assert_allclose(sv.X[1:], sv_mad.x, atol=1e-14, rtol=0)
xo.assert_allclose(sv.Y[1:], sv_mad.y, atol=1e-14, rtol=0)
xo.assert_allclose(sv.Z[1:], sv_mad.z, atol=1e-14, rtol=0)

xo.assert_allclose(sv.s[1:], sv_mad.s, atol=1e-14, rtol=0)

xo.assert_allclose(p[:, 0], 1e-3, atol=1e-14)
xo.assert_allclose(p[:, 1], 2e-3, atol=1e-14)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1, figsize=(6.4, 4.8 * 1.5))
plt.subplot(3,1,1)
plt.plot(tw.s, tw.x, label='Twiss x')
plt.plot(sv.s, tw.y, label='Twiss y')
plt.legend()
plt.subplot(3,1,2)
plt.plot(sv.Z, sv.X, label='Survey')
plt.plot(Z, X, label='Part. trajectory')
plt.legend()
plt.xlabel('Z [m]')
plt.ylabel('X [m]')
plt.subplot(3,1,3)
plt.plot(sv.Z, sv.Y, label='Survey')
plt.plot(Z, Y, label='Part. trajectory')
plt.legend()
plt.xlabel('Z [m]')
plt.ylabel('Y [m]')
plt.subplots_adjust(hspace=0.3)
plt.suptitle('Init on the left')

plt.show()