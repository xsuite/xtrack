from cpymad.madx import Madx
import xtrack as xt

madx = Madx()

madx.input('''
b: rbend, l=1.0, angle=0.5;

ss: sequence, l=5.0, refer=center;
    b, at=2.5;
endsequence;
beam;
use, sequence=ss;
twiss, betx=1, bety=1;
survey;
''')
sv_madx = xt.Table(madx.table.survey, _copy_cols=True)

line = xt.Line.from_madx_sequence(madx.sequence.ss)
line.remove('ss$start')
line.remove('ss$end')
line.particle_ref = xt.Particles(p0c=1e9)
sv_xs = line.survey()

mad_str = line.to_madx_sequence(sequence_name='ggg')
madx2 = Madx()
madx2.input(mad_str)
madx2.input('''
beam;
use, sequence=ggg;
twiss, betx=1, bety=1;
survey;
''')
sv_madx2 = xt.Table(madx2.table.survey, _copy_cols=True)

env2 = xt.load(string=mad_str, format='madx')
sv_xs2 = env2.ggg.survey()