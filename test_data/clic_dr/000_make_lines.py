from cpymad.madx import Madx
import xtrack as xt
import xpart as xp
import xobjects as xo

import json

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

with open('line.json', 'w') as f:
    json.dump(line.to_dict(), f, cls=xo.JEncoder)

# make a line suitable to test tapering

line.particle_ref = xp.Particles(
        mass0=xp.ELECTRON_MASS_EV,
        q0=-1,
        gamma0=15000 # I push up the energy loss
        )
#line0 = line.copy()

line['rf'].voltage *= 20 # I push up the voltage to be able to compensate the energy loss

# I cycle to the center of the undulators straight section (where I want to impose delta = 0)
line = line.cycle('qdw1..1:38')

c0 = line['rf']
v0 = c0.voltage
s0 = line.get_s_position('rf')

# I install some more cavities to test the tapering
line.insert_element(at_s=41., element=c0.copy(), name='rf1')
line.insert_element(at_s=line.get_length()-s0, element=c0.copy(), name='rf2a')
line.insert_element(at_s=line.get_length()-s0, element=c0.copy(), name='rf2b')
line.insert_element(at_s=line.get_length()-41, element=c0.copy(), name='rf3')
line.insert_element(at_s=line.get_length()-41, element=c0.copy(), name='rf_off')

line['rf2a'].voltage *= 0.6 # I split the voltage unevenly to test the partitioning
line['rf2b'].voltage *= 0.4
line['rf_off'].voltage *= 0.0

with open('line_for_taper.json', 'w') as f:
    json.dump(line.to_dict(), f, cls=xo.JEncoder)