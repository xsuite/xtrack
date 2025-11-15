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
save, file="temp_lhc_thick.seq";
""")

env = xt.load('temp_lhc_thick.seq', s_tol=5e-6, _rbend_correct_k0=True,
              reverse_lines=['lhcb2'])

env.lhcb1.set_particle_ref('proton', p0c=7000e9)
env.lhcb2.set_particle_ref('proton', p0c=7000e9)

tt_cav = env.elements.get_table().rows.match(element_type='Cavity')
for nn in tt_cav.name:
    env[nn].frequency = 400.79e6  # Hz

line = env.lhcb1

line.to_json('lhc_thick_with_knobs.json', include_var_management=True)

env.lhcb1.twiss_default['method'] = '4d'
env.lhcb2.twiss_default['method'] = '4d'
env.lhcb2.twiss_default['reverse'] = True

env.to_json('hllhc15_collider_thick.json')
