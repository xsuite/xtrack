import numpy as np
from cpymad.madx import Madx

import xtrack as xt
import xpart as xp
import xdeps as xd

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

k0bi1bsw1l11 = 1e-2; // To have some non-zero orbit

twiss;
''')

line = xt.Line.from_madx_sequence(mad.sequence.psb1,
                                allow_thick=True,
                                enable_align_errors=True,
                                deferred_expressions=True)
line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV,
                            gamma0=mad.sequence.psb1.beam.gamma)
line.configure_bend_method('full')
line.twiss_default['method'] = '4d'

line.build_tracker()

tw = line.twiss()
tw_comp = line.twiss(group_compound_elements=True)

for nn in tw._col_names:
    assert len(tw[nn]) == len(tw['name'])
    assert len(tw_comp[nn]) == len(tw_comp['name'])

assert 'bi1.bsw1l1.2_entry' in tw.name
assert 'bi1.bsw1l1.2_offset_entry' in tw.name
assert 'bi1.bsw1l1.2_den' in tw.name
assert 'bi1.bsw1l1.2' in tw.name
assert 'bi1.bsw1l1.2_dex' in tw.name
assert 'bi1.bsw1l1.2_offset_exit' in tw.name
assert 'bi1.bsw1l1.2_exit' in tw.name

assert 'bi1.bsw1l1.2_entry' in tw_comp.name
assert 'bi1.bsw1l1.2_offset_entry' not in tw_comp.name
assert 'bi1.bsw1l1.2_den' not in tw_comp.name
assert 'bi1.bsw1l1.2' not in tw_comp.name
assert 'bi1.bsw1l1.2_dex' not in tw_comp.name
assert 'bi1.bsw1l1.2_offset_exit' not in tw_comp.name
assert 'bi1.bsw1l1.2_exit' not in tw_comp.name

assert tw_comp['name', -2] == tw['name', -2] == 'psb1$end'
assert tw_comp['name', -1] == tw['name', -1] == '_end_point'

assert np.isclose(tw_comp['px', 'br1.dhz16l1'],
                  tw['px', 'br1.dhz16l1'], rtol=0, atol=1e-15)

assert np.allclose(tw_comp['W_matrix', 'bi1.bsw1l1.2_entry'],
                   tw['W_matrix', 'bi1.bsw1l1.2_entry'], rtol=0, atol=1e-15)

tw_init = tw.get_twiss_init('bi1.ksw16l1_entry')
tw_init_comp = tw_comp.get_twiss_init('bi1.ksw16l1_entry')

assert np.allclose(tw_init_comp.W_matrix, tw_init.W_matrix,
                    rtol=0, atol=1e-15)
assert np.isclose(tw_init_comp.mux, tw_init.mux, rtol=0, atol=1e-15)
assert np.isclose(tw_init_comp.x, tw_init.x, rtol=0, atol=1e-15)

tw_comp_local = line.twiss(group_compound_elements=True,
                           twiss_init=tw_init_comp,
                           ele_start='bi1.ksw16l1_entry',
                           ele_stop='br.stscrap161')
tw_local = line.twiss(twiss_init=tw_init,
                        ele_start='bi1.ksw16l1_entry',
                        ele_stop='br.stscrap161')

for nn in tw_local._col_names:
    assert len(tw_local[nn]) == len(tw_local['name'])
    assert len(tw_comp_local[nn]) == len(tw_comp_local['name'])

assert 'br.bhz161_entry' in tw_local.name
assert 'br.bhz161_den' in tw_local.name
assert 'br.bhz161' in tw_local.name
assert 'br.bhz161_dex' in tw_local.name
assert 'br.bhz161_exit' in tw_local.name

assert 'br.bhz161_entry' in tw_comp_local.name
assert 'br.bhz161_den' not in tw_comp_local.name
assert 'br.bhz161' not in tw_comp_local.name
assert 'br.bhz161_dex' not in tw_comp_local.name
assert 'br.bhz161_exit' not in tw_comp_local.name

assert tw_comp_local['name', -2] == tw_local['name', -2] == 'br.stscrap161'
assert tw_comp_local['name', -1] == tw_local['name', -1] == '_end_point'

assert np.isclose(tw_comp_local['px', 'br1.dhz16l1'],
                  tw_local['px', 'br1.dhz16l1'], rtol=0, atol=1e-15)


import matplotlib.pyplot as plt
plt.close('all')
plt.plot(tw.s, tw.x)
plt.plot(tw_comp.s, tw_comp.x)
plt.show()