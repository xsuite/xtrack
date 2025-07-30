from cpymad.madx import Madx
import xtrack as xt

mad = Madx()

mad.input("""
    r3: yrotation, angle=-0.5;
    r4: yrotation, angle=0.5;
    rs2: srotation, angle=-1.04;
    beam;
    ss: sequence,l=10;
        rs2, at=5.5;
        r3, at=8;
        r4, at=9;
    endsequence;

    use,sequence=ss;
    twiss,betx=1,bety=1;
    survey;
    """)

line = xt.Line.from_madx_sequence(mad.sequence.ss)
line.particle_ref = xt.Particles(p0c=1E9)

sv_mad = xt.Table(mad.table.survey)

sv = line.survey()