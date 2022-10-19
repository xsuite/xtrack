import xtrack as xt
from cpymad.madx import Madx

mad = Madx()
mad.call('SPS_2021_Pb_ions_thin_test.seq')
mad.use('sps')
mad.twiss()

twmad_4d = mad.table.twiss.dframe()
summad_4d = mad.table.summ.dframe()

V_RF = 1.7e6 # V (control room definition, energy gain per charge)

# I switch on one cavity
charge = mad.sequence.sps.beam.charge
mad.sequence.sps.elements['actcse.31632'].volt = V_RF/1e6 * charge
mad.sequence.sps.elements['actcse.31632'].lag = 0
mad.sequence.sps.elements['actcse.31632'].freq = 200.

twmad_6d = mad.table.twiss.dframe()
summad_6d = mad.table.summ.dframe()

mad.emit()
qs_mad = mad.table.emitsumm.qs[0]

# Make xsuite line
line = xt.Line.from_madx_sequence(mad.sequence.sps, deferred_expressions=True)



