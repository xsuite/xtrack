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


env = xt.load('temp_lhc_thick.seq', s_tol=1e-6,
              _rbend_correct_k0=True, # LHC sequences are defined with rbarc=False
              reverse_lines=['lhcb2'])


env.lhcb1.set_particle_ref('proton', p0c=7000e9)
env.lhcb2.set_particle_ref('proton', p0c=7000e9)

line = env.lhcb1

line.to_json('lhc_thick_with_knobs.json', include_var_management=True)
