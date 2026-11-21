from pathlib import Path

from cpymad.madx import Madx
import xtrack as xt
import xpart as xp
import xobjects as xo

import json

folder = Path(__file__).parent

# Import a thick sequence
mad = Madx()
mad.call(str(folder / 'sequence.madx'))
mad.use('ring')
line_thick = xt.Line.from_madx_sequence(mad.sequence['RING'], allow_thick=True)

# Makethin
mad.input('''
select, flag=MAKETHIN, SLICE=4, thick=false;
select, flag=MAKETHIN, pattern=wig, slice=1;
MAKETHIN, SEQUENCE=ring, MAKEDIPEDGE=true;
use, sequence=RING;
''')
mad.use('ring')

# Build xtrack line
print('Build xtrack line...')
line_thin = xt.Line.from_madx_sequence(mad.sequence['RING'])
line_thin.particle_ref = xp.Particles(
        mass0=xp.ELECTRON_MASS_EV,
        q0=-1,
        gamma0=mad.sequence.ring.beam.gamma
        )

with open(folder / 'line.json', 'w') as f:
    json.dump(line_thin.to_dict(), f, cls=xo.JEncoder)

# Make both lines suitable to test tapering.
for line, start, filename in [
    (line_thin, 'qdw1..1:38', 'line_for_taper.json'),
    (line_thick, 'qdw1:38', 'line_for_taper_thick.json'),
]:
    line.particle_ref = xp.Particles(
            mass0=xp.ELECTRON_MASS_EV,
            q0=-1,
            gamma0=15000 # I push up the energy loss
            )

    line['rf'].voltage *= 20 # Enough voltage to compensate the energy loss

    # Start at the same quadrupole in the center of the undulator straight.
    line = line.cycle(start)

    c0 = line['rf']
    s0 = line.get_table()['s', 'rf']

    # Install more cavities to test the tapering.
    line.insert_element(at_s=41., element=c0.copy(), name='rf1')
    line.insert_element(at_s=line.get_length()-s0, element=c0.copy(), name='rf2a')
    line.insert_element(at_s=line.get_length()-s0, element=c0.copy(), name='rf2b')
    line.insert_element(at_s=line.get_length()-41, element=c0.copy(), name='rf3')
    line.insert_element(at_s=line.get_length()-41, element=c0.copy(), name='rf_off')

    line['rf2a'].voltage *= 0.6 # Uneven splitting tests the energy partitioning
    line['rf2b'].voltage *= 0.4
    line['rf_off'].voltage *= 0.0

    with open(folder / filename, 'w') as f:
        json.dump(line.to_dict(), f, cls=xo.JEncoder)
