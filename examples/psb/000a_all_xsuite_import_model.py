import numpy as np
from cpymad.madx import Madx

import xtrack as xt

import matplotlib.pyplot as plt

from cpymad.madx import Madx

mad = Madx()

# Load mad model and apply element shifts
mad.input('''
call, file = '../../test_data/psb_chicane/psb.seq';
call, file = '../../test_data/psb_chicane/psb_fb_lhc.str';

beam, particle=PROTON, pc=0.5708301551893517;
use, sequence=psb1;

select,flag=error,clear;
select,flag=error,pattern=bi1.bsw1l1.1*;
ealign, dx=-0.0057;

select,flag=error,clear;
select,flag=error,pattern=bi1.bsw1l1.2*;
select,flag=error,pattern=bi1.bsw1l1.3*;
select,flag=error,pattern=bi1.bsw1l1.4*;
ealign, dx=-0.0442;

twiss;
''')

line = xt.Line.from_madx_sequence(
    sequence=mad.sequence.psb1,
    allow_thick=True,
    enable_align_errors=True,
    deferred_expressions=True,
)
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV,
                            gamma0=mad.sequence.psb1.beam.gamma)
#!end-doc-part

line.configure_bend_model(core='full')
line.to_json('psb_00_from_mad.json')
