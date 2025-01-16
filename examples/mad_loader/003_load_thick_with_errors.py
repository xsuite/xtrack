from cpymad.madx import Madx
import xtrack as xt

mad = Madx()

mad.input('''

    k1=0.2;
    l = 5.;

    radius = 0.3;

    dknr0=0.02;
    dknr1=-0.04;
    dknr2=0.06;
    dkns3=0.2;

    dksr0=0.03;
    dksr1=-0.05;
    dksr2=0.07;
    dksr3=0.3;

    elm: quadrupole, k1:=k1, l=5.;

    seq: sequence, l=50;
    elm1:elm, at=5;
    elm2:elm, at=10;
    elm3:elm, at=20;
    endsequence;

    beam;
    use,sequence=seq;

    select,pattern=elm,flag=error;
    efcomp, radius=radius,
        dknr={dknr0,dknr1,dknr2,dknr3},
        dksr={dksr0,dksr1,dksr2,dksr3}
        ;

''')

line = xt.Line.from_madx_sequence(mad.sequence.seq,
                                  enable_field_errors=True,
                                  allow_thick=True)

# Remember to set num_multipole_kicks




