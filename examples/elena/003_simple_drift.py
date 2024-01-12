from cpymad.madx import Madx

mad = Madx()
mad.input('''
seq: sequence, l = 15;
m0: marker, at = 0;
m1: marker, at = 1;
m2: marker, at = 2;
m3: marker, at = 3;
m4: marker, at = 4;
m5: marker, at = 5;
m6: marker, at = 6;
m7: marker, at = 7;
m8: marker, at = 8;
m9: marker, at = 9;
m10: marker, at = 10;
b1: sbend, angle=0.1, l=1, at=11;
endsequence;
beam, particle = proton, gamma=1.05;
use, sequence=seq;
''')

import xtrack as xt
line = xt.Line.from_madx_sequence(mad.sequence.seq)
line.particle_ref = xt.Particles(gamma0=mad.sequence.seq.beam.gamma)

tw0 = line.twiss(betx=1, bety=1, start='m0', end='m10')
tw1 = line.twiss(betx=1, bety=1, start='m0', end='m10', delta=1e-3)

mad.input(f'''
twiss, betx=1, bety=1, table=tw0;
twiss, betx=1, bety=1, table=tw1, pt={tw1.ptau[0]}
''')

tm0 = mad.table.tw0
tm1 = mad.table.tw1


