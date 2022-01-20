from cpymad.madx import Madx

# hllhc15 can be found at git@github.com:lhcopt/hllhc15.git

mad = Madx()

mad.input("""
call,file="../../../../hllhc15/util/lhc.seq";
call,file="../../../../hllhc15/hllhc_sequence.madx";
call,file="../../../../hllhc15/toolkit/macro.madx";
seqedit,sequence=lhcb1;flatten;cycle,start=IP3;flatten;endedit;
seqedit,sequence=lhcb2;flatten;cycle,start=IP3;flatten;endedit;
exec,mk_beam(7000);
exec,myslice;
call,file="../../../../hllhc15/round/opt_round_150_1500_thin.madx";
on_x1 = 1;
on_x5 = 1;
on_disp = 0;
exec,check_ip(b1);
exec,check_ip(b2);
""")

import xtrack as xt
import xpart as xp
import xobjects as xo

line = xt.Line.from_madx_sequence(sequence=mad.sequence.lhcb1)
for nn, ee in zip(line.element_names, line.elements):
    if nn.startswith('acs'):
        assert ee.__class__.__name__ == 'Cavity'
        ee.voltage = 1e6
        ee.frequency = 400e6
particle = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1,
                        gamma0=mad.sequence.lhcb1.beam.gamma)

import json
with open('xtline.json', 'w') as fid:
    json.dump({'line': line.to_dict(), 'particle': particle.to_dict()}, fid,
              cls=xo.JEncoder)
