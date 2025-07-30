from cpymad.madx import Madx
import xtrack as xt

mad = Madx()

mad.input("""

    on_srot = 1;

    r3: yrotation, angle=-0.5;
    r4: yrotation, angle=0.5;
    rs2: srotation, angle=-1.04*on_srot;
    beam;
    ss: sequence,l=10;
        rs2, at=5.5;
        r3, at=8;
        r4, at=9;
        end: marker, at=10;
    endsequence;

    use,sequence=ss;
    twiss,betx=1,bety=1;
    survey;
    """)

line = xt.Line.from_madx_sequence(mad.sequence.ss)
line.particle_ref = xt.Particles(p0c=1E9)

sv_mad = xt.Table(mad.table.survey)
tw_mad = xt.Table(mad.table.twiss)

line.config.XTRACK_GLOBAL_XY_LIMIT = None
line.config.XTRACK_USE_EXACT_DRIFTS = True

sv = line.survey()
tw = line.twiss(betx=1, bety=1)

p = tw.x[:, None] * sv.ix + tw.y[:, None] * sv.iy + sv.p0
X = p[:, 0]
Y = p[:, 1]
Z = p[:, 2]

p_end = tw.x[-1] * sv.ix[-1, :] + tw.y[-1] * sv.iy[-1, :] + sv.p0[-1, :]

from _madpoint import MadPoint
mp = MadPoint(name='end:1', mad=mad, use_twiss=True, use_survey=True)

import matplotlib.pyplot as plt
plt.close('all')
plt.plot(sv.Z, sv.X, label='Survey X')
plt.plot(sv_mad['z'], sv_mad['x'], label='MAD-X')