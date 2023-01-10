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

!rf and radiation off
use, sequence = fccee_p_ring;
voltca1save = voltca1;
voltca2save = voltca2;


!rf and radiation off

voltca1 = 0;
voltca2 = 0;
SAVEBETA, LABEL=B.IP, PLACE=#s, SEQUENCE=FCCEE_P_RING;
''')

tw_0 = mad.twiss(chrom=True).dframe()
summ_0 = mad.table.summ.dframe()


mad.input('''
! radiation on

voltca1 = voltca1save;
voltca2 = voltca2save;
beam, radiate=true;

use, sequence = fccee_p_ring;


! tapering and matching
MATCH, sequence=FCCEE_P_RING, BETA0 = B.IP, tapering;
  VARY, NAME=LAGCA1, step=1.0E-7;
  VARY, NAME=LAGCA2, step=1.0E-7;
  CONSTRAINT, SEQUENCE=FCCEE_P_RING, RANGE=#e, PT=0.0;
  JACOBIAN,TOLERANCE=1.0E-14, CALLS=3000;
ENDMATCH;

twiss, BETA0 = B.IP, tolerance=1e-12;

MATCH, sequence=FCCEE_P_RING, BETA0 = B.IP, tapering;
  VARY, NAME=LAGCA1, step=1.0E-7;
  VARY, NAME=LAGCA2, step=1.0E-7;
  CONSTRAINT, SEQUENCE=FCCEE_P_RING, RANGE=#e, PT=0.0;
  JACOBIAN,TOLERANCE=1.0E-14, CALLS=3000;
ENDMATCH;

use, sequence = fccee_p_ring;
''')

tw_1 = mad.twiss(chrom=True).dframe()
summ_1 = mad.table.summ.dframe()

import matplotlib.pyplot as plt
plt.close('all')

# Plot beta beating
plt.figure(1)
ax1 = plt.subplot(211)
plt.plot(tw_0['s'], tw_1['betx']/tw_0['betx']-1)
plt.ylabel(r'$\Delta\beta_x/\beta_x$')
plt.subplot(212, sharex=ax1)
plt.plot(tw_0['s'], tw_1['bety']/tw_0['bety']-1)
plt.ylabel(r'$\Delta\beta_y/\beta_y$')
plt.xlabel('s [m]')

plt.show()