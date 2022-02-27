from cpymad.madx import Madx

# hllhc15 can be found at git@github.com:lhcopt/hllhc15.git

mad = Madx()

mad.input("""
call,file="../../../hllhc15/util/lhc.seq";
call,file="../../../hllhc15/hllhc_sequence.madx";
call,file="../../../hllhc15/toolkit/macro.madx";
seqedit,sequence=lhcb1;flatten;cycle,start=IP3;flatten;endedit;
seqedit,sequence=lhcb2;flatten;cycle,start=IP3;flatten;endedit;
exec,mk_beam(7000);
exec,myslice;
call,file="../../../hllhc15/round/opt_round_150_1500_thin.madx";
on_x1 = 1;
on_x5 = 1;
on_disp = 0;
exec,check_ip(b1);
exec,check_ip(b2);
""")
mad.use(sequence="lhcb1")
mad.globals['vrf400'] = 16
mad.globals['lagrf400.b1'] = 0.5
mad.twiss()
mad.save(sequence='lhcb1',beam=True,file="sequence.madx")
