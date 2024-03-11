from cpymad.madx import Madx

mad = Madx()

mad.input('''
ss: sequence, l=1;
mm: multipole,knl={1,1,1},ksl={1,1,1},at=0;
kk: kicker,hkick=1,vkick=1,at=0.;
bb: sbend,angle=1,l=1,at=0.5;
endsequence;
beam,bv=-1;
select,flag=twiss,column=name,angle,hkick,vkick,k0l,k0sl,k1l,k1sl;
use,sequence=ss;
twiss,betx=1,bety=1;
! write,table=twiss;
''')

import xtrack as xt
line = xt.Line.from_madx_sequence(mad.sequence.ss)
line.particle_ref = xt.Particles(p0c=6500e9, q0=1)
line.twiss_default['reverse'] = True

tw = line.twiss(method='4d', start='ss$start', end='ss$end',
                betx=1, bety=1, strengths=True)