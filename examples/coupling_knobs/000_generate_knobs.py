from cpymad.madx import Madx

import coupling_knobs

mad=Madx(stdout=False)
mad.call('../../test_data/hllhc15_noerrors_nobb/sequence.madx')
mad.use('lhcb1')
t1=mad.twiss(sequence='lhcb1')
knobs=coupling_knobs.get_knob_defs_from_twiss(t1)
for lhs,rhs in knobs:
    print(f"{lhs}:={rhs};")











