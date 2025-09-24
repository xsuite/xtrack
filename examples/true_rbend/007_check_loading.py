from cpymad.madx import Madx
import xtrack as xt
import xobjects as xo
import numpy as np

madx = Madx()

l_straight = 1.0
angle = 0.5
l_arc = l_straight / np.sinc(angle / np.pi / 2) # np.sinc is sin(pi*x)/(pi*x)

madx.input('''
b: rbend, l=1.0, angle=0.5;

ss: sequence, l=5.0, refer=centre;
    b, at=2.5;
endsequence;
beam;
use, sequence=ss;
twiss, betx=1, bety=1;
survey;
''')
sv_madx = xt.Table(madx.table.survey, _copy_cols=True)
xo.assert_allclose(madx.elements['b'].l, l_straight, rtol=1e-12)
xo.assert_allclose(sv_madx['s', 'b:1'], 2.5 + l_arc / 2, rtol=1e-12) # b si at exit

# Check cpymad loader
line = xt.Line.from_madx_sequence(madx.sequence.ss)
line.remove('ss$start')
line.remove('ss$end')
line.particle_ref = xt.Particles(p0c=1e9)
sv_xs = line.survey()
xo.assert_allclose(line['b'].length_straight, l_straight, rtol=1e-12)
xo.assert_allclose(line['b'].length, l_arc, rtol=1e-12)
xo.assert_allclose(sv_xs['s', 'b'], 2.5 - l_arc / 2, rtol=1e-12) # b si at entrance

# Check to madx
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
xo.assert_allclose(madx2.elements['b'].l, l_straight, rtol=1e-12)

# Check native madx loader
env2 = xt.load(string=mad_str, format='madx')
sv_xs2 = env2.ggg.survey()
xo.assert_allclose(env2['b'].length_straight, l_straight, rtol=1e-12)
xo.assert_allclose(env2['b'].length, l_arc, rtol=1e-12)
xo.assert_allclose(sv_xs2['s', 'b'], 2.5 - l_arc / 2, rtol=1e-12) # b si at entrance

# Check madng
sv_ng = line.madng_survey()
xo.assert_allclose(sv_ng['s', 'b'], 2.5 - l_arc / 2, rtol=1e-12) # b si at entrance
