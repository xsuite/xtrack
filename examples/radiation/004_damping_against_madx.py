import numpy as np

from cpymad.madx import Madx

import xtrack as xt
import xpart as xp
import xobjects as xo

mad = Madx()

# Import thick sequence
mad = Madx()
mad.call('../../test_data/clic_dr/sequence.madx')
mad.use('ring')

# Twiss
twthick = mad.twiss().dframe()

# Emit
mad.sequence.ring.beam.radiate = True
mad.emit()
mad_emit_table = mad.table.emit.dframe()
mad_emit_summ = mad.table.emitsumm.dframe()

# Makethin
mad.input(f'''
select, flag=MAKETHIN, SLICE=4, thick=false;
select, flag=MAKETHIN, pattern=wig, slice=1;
MAKETHIN, SEQUENCE=ring, MAKEDIPEDGE=true;
use, sequence=RING;
''')
mad.use('ring')
mad.twiss()

# Build xtrack line
line = xt.Line.from_madx_sequence(mad.sequence['RING'])
line.particle_ref = xp.Particles(
        mass0=xp.ELECTRON_MASS_EV,
        q0=-1,
        gamma0=mad.sequence.ring.beam.gamma)

# Build tracker
tracker = xt.Tracker(line=line)

# Switch on radiation
for ee in line.elements:
    if isinstance(ee, xt.Multipole):
        ee.radiation_flag = 1

# Twiss
tw = tracker.twiss(eneloss_and_damping=True)

# Checks
met = mad_emit_table
assert np.isclose(tw['eneloss_turn'], mad_emit_summ.u0[0]*1e9,
                  rtol=3e-3, atol=0)
assert np.isclose(tw['damping_constants_s'][0],
    met[met.loc[:, 'parameter']=='damping_constant']['mode1'][0],
    rtol=3e-3, atol=0
    )
assert np.isclose(tw['damping_constants_s'][1],
    met[met.loc[:, 'parameter']=='damping_constant']['mode2'][0],
    rtol=1e-3, atol=0
    )
assert np.isclose(tw['damping_constants_s'][2],
    met[met.loc[:, 'parameter']=='damping_constant']['mode3'][0],
    rtol=3e-3, atol=0
    )

assert np.isclose(tw['partition_numbers'][0],
    met[met.loc[:, 'parameter']=='damping_partion']['mode1'][0],
    rtol=3e-3, atol=0
    )
assert np.isclose(tw['partition_numbers'][1],
    met[met.loc[:, 'parameter']=='damping_partion']['mode2'][0],
    rtol=1e-3, atol=0
    )
assert np.isclose(tw['partition_numbers'][2],
    met[met.loc[:, 'parameter']=='damping_partion']['mode3'][0],
    rtol=3e-3, atol=0
    )