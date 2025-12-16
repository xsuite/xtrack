from cpymad.madx import Madx
import xtrack as xt
import xobjects as xo

madx_src = """
hh = 5;
ff = 100;
cav_harm: rfcavity, harmon:=hh;
cav_freq: rfcavity, freq:=ff;

seq: sequence, l=10;
    cav_harm, at=5;
    cav_freq, at=7;
endsequence;

"""

mad = Madx()
mad.input(madx_src)
mad.input('''
beam;
use, sequence=seq;
''')

lcpymad = xt.Line.from_madx_sequence(mad.sequence.seq,
                                     deferred_expressions=True)

env_native = xt.load(string=madx_src, format='madx')
lnative = env_native['seq']

for ll in [lcpymad, lnative]:
    assert str(ll.ref['cav_harm'].harmonic._expr) == "vars['hh']"
    assert ll.ref['cav_harm'].frequency._expr is None
    assert ll.ref['cav_harm'].frequency._value == 0

    assert str(ll.ref['cav_freq'].frequency._expr) == "(vars['ff'] * 1000000.0)"
    assert ll.ref['cav_freq'].harmonic._expr is None
    assert ll.ref['cav_freq'].harmonic._value == 0