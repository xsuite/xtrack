import xdeps
from cpymad.madx import Madx

mad = Madx()
mad.call('lhc_sequence.madx')
mad.beam()
mad.sequence.lhcb1.use()

from xtrack.mad_loader import MadLoader, MadElem

ml = MadLoader(mad.sequence.lhcb1)
mad_elem = MadElem('name', mad.sequence.lhcb1.expanded_elements[11], mad.sequence.lhcb1)

