from cpymad.madx import Madx
import xtrack as xt
import xpart as xp

import numpy as np

# hllhc15 can be found at git@github.com:lhcopt/hllhc15.git

mad = Madx()

mad.input(f"""
call,file="lhc.seq";
call,file="hllhc_sequence.madx";
seqedit,sequence=lhcb1;flatten;cycle,start=IP7;flatten;endedit;
seqedit,sequence=lhcb2;flatten;cycle,start=IP7;flatten;endedit;
beam, sequence=lhcb1, particle=proton, pc=7000;
beam, sequence=lhcb2, particle=proton, pc=7000, bv=-1;
call,file="opt_round_150_1500.madx";
use, sequence=lhcb1;
use, sequence=lhcb2;
twiss;
set, format="12d", "-18.12e", "25s";
save, file="temp_lhc_thick.seq";
""")

mad2 = Madx()
mad.call('temp_lhc_thick.seq')
mad.beam()
mad.use('lhcb1') # check no negative drifts in madx


env = xt.load('temp_lhc_thick.seq', s_tol=1e-6, _rbend_correct_k0=True,
              reverse_lines=['lhcb2'])


env.lhcb1.set_particle_ref('proton', p0c=7000e9)
env.lhcb2.set_particle_ref('proton', p0c=7000e9)

# tt_rbend = env.elements.get_table().rows.match(element_type='RBend')
# for nn in tt_rbend.name:
#     ll = env[nn].length_straight
#     env[nn].length_straight = 0
#     env[nn].length = ll
#     env[nn].k0=0
#     env[nn].k0_from_h=True

# env.lhcb1.composer.s_tol = 1e-6
# env.lhcb1.regenerate_from_composer()
# env.lhcb1.end_compose()

tt_cav = env.elements.get_table().rows.match(element_type='Cavity')
for nn in tt_cav.name:
    env[nn].frequency = 400.79e6  # Hz

line = env.lhcb1

line.to_json('lhc_thick_with_knobs.json', include_var_management=True)
