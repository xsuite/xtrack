from cpymad.madx import Madx

madx_src = """
cav: rfcavity, harmon=2, freq=10;

seq: sequence, l=10;
    cav, at=5;
endsequence;
beam;
use, sequence=seq;
twiss, betx=1, bety=1;
"""

mad = Madx()
mad.input(madx_src)

print(f'{mad.sequence.seq.elements['cav'].freq}')
# prints:
59.95848377182102