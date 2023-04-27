import xdeps
from cpymad.madx import Madx


mad = Madx()
mad.call('lhc_sequence.madx')
mad.beam()
mad.sequence.lhcb1.use()

#mad.sequence.lhcb1.expanded_elements[2].l

from xtrack.mad_loader import MadLoader, MadElem

ml = MadLoader(mad.sequence.lhcb1)
mad_elem = MadElem('name', mad.sequence.lhcb1.expanded_elements[11], mad.sequence.lhcb1)

#implement  convert_quadrupole in MadLoader to emit [Drift(l/2) SimpleThinQuadruple(k1*l) Drift(l/2)]
#similarly for the otherelement



