from cpymad.madx import Madx

mad = Madx()

mad.input('''
set, format="19.15f";
option,update_from_parent=true;

radiationandrf = 1;

call,file="fccee_t.seq";

pbeam =   182.5;
exbeam = 1.46e-9;
eybeam = 2.9e-12;
nbun =    48;
npar =   2.3e11;
halfcrossingangle = 0.015;
ebeam = sqrt( pbeam^2 + emass^2 );
beam, particle=positron, npart=npar, kbunch=nbun, energy=ebeam, radiate=false, bv=+1, ex=exbeam, ey=eybeam;

use, sequence = fccee_p_ring;

voltca1save = voltca1;
voltca2save = voltca2;


!rf and radiation off (for 4d twiss)
voltca1 = 0;
voltca2 = 0;
''')

tw_thick = mad.twiss().dframe()

mad.input('''
select, flag=makethin, class=rfcavity, slice = 1, thick = false;
select, flag=makethin, class=rbend, slice = 4, thick = false;
select, flag=makethin, class=quadrupole, slice = 4, thick = false;
select, flag=makethin, class=sextupole, slice = 4;
select, flag=makethin, pattern="qc*", slice=20;
select, flag=makethin, pattern="sy*", slice=20;

makethin, sequence=fccee_p_ring, style=teapot, makedipedge=false;
use, sequence = fccee_p_ring;
''')

tw_thin = mad.twiss().dframe()

# Restore rf_voltage
mad.input('''
    voltca1 = voltca1save;
    voltca2 = voltca2save;
'''
)

import xtrack as xt
import xpart as xp
line = xt.Line.from_madx_sequence(mad.sequence.fccee_p_ring)
line.particle_ref = xp.Particles(mass0=xp.ELECTRON_MASS_EV,
                                gamma0=mad.sequence.fccee_p_ring.beam.gamma)

import json
import xobjects as xo
with open('line_no_radiation.json', 'w') as f:
    json.dump(line.to_dict(), f, cls=xo.JEncoder)