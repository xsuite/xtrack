from cpymad.madx import Madx

mad = Madx()

mad.input('''

    k1=0.2;

    elm: quadrupole, k1:=k1;

    seq: sequence, l=1;
    elm1:elm, at=0;
    mk:marker, at=0.1;
    mk2:marker, at=0.1,aperture={0.1,0.2,0.11,0.22},apertype="rectellipse";
    elm2:elm, at=0.5;
    elm3:elm, at=0.5;
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

