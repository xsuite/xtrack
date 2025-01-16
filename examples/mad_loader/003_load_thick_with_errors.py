from cpymad.madx import Madx
import xtrack as xt

mad = Madx()

mad.input('''

    k1=0.2;

    elm: quadrupole, k1:=k1, l=5.;

    seq: sequence, l=50;
    elm1:elm, at=5;
    elm2:elm, at=10;
    elm3:elm, at=20;
    endsequence;

    beam;
    use,sequence=seq;

    select,pattern=elm,flag=error;
    efcomp,order=0,radius=0.3,
    dkn={0.01,-0.02,0.03,0.2},
    dks={-0.01,0.02,-0.03,0.3,5},
    dknr={0.02,-0.04,0.06},
    dksr={-0.03,0.05,-0.07};

''')

line = xt.Line.from_madx_sequence(mad.sequence.seq,
                                  enable_field_errors=True,
                                  allow_thick=True)



