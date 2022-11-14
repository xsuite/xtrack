from cpymad.madx import Madx
import xtrack as xt
import xpart as xp

import numpy as np

# Import a thick sequence
mad = Madx()
mad.call('../../test_data/clic_dr/sequence.madx')
mad.use('ring')

# Makethin
mad.input(f'''
select, flag=MAKETHIN, SLICE=4, thick=false;
select, flag=MAKETHIN, pattern=wig, slice=1;
MAKETHIN, SEQUENCE=ring, MAKEDIPEDGE=true;
use, sequence=RING;
''')
mad.use('ring')

# Build xtrack line
print('Build xtrack line...')
line = xt.Line.from_madx_sequence(mad.sequence['RING'])
line.particle_ref = xp.Particles(
        mass0=xp.ELECTRON_MASS_EV,
        q0=-1,
        gamma0=mad.sequence.ring.beam.gamma
        )



line.particle_ref = xp.Particles(
        mass0=xp.ELECTRON_MASS_EV,
        q0=-1,
        gamma0=15000 # I push up the energy loss
        )

#line0 = line.copy()

line['rf'].voltage *= 20 # I push up the voltage

line = line.cycle('qdw1..1:38')

c0 = line['rf']
v0 = c0.voltage
c0.frequency /= 100
s0 = line.get_s_position('rf')


line.insert_element(at_s=41., element=c0.copy(), name='rf1')
line.insert_element(at_s=line.get_length()-s0, element=c0.copy(), name='rf2')
line.insert_element(at_s=line.get_length()-41, element=c0.copy(), name='rf3')
