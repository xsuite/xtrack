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
call,file="opt_round_150_1500.madx";
""")

mad.use(sequence="lhcb1")
seq = mad.sequence.lhcb1
mad.twiss()

line = xt.Line.from_madx_sequence(mad.sequence.lhcb1,
            allow_thick=True, deferred_expressions=True)
line.particle_ref = xp.Particles(mass0=seq.beam.mass*1e9, gamma0=seq.beam.gamma)

tt = line.get_table()
for nn in tt.rows[tt.element_type=='Solenoid'].name:
    ee_elen = line[nn].length
    line.element_dict[nn] = xt.Drift(length=ee_elen)

line.to_json('lhc_thick_with_knobs.json', include_var_management=True)
