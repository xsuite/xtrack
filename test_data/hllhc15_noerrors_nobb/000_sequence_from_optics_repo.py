# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

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
mad.globals['lagrf400.b2'] = 0.
mad.twiss()
mad.save(sequence=['lhcb1', 'lhcb2'], beam=True, file="sequence.madx")
mad.call('../../../hllhc15/toolkit/enable_crabcavities.madx')
mad.twiss()
mad.save(sequence=['lhcb1'], beam=True, file="sequence_with_crabs.madx")

mad_b4 = Madx()

mad_b4.input("""
mylhcbeam = 4;
call,file="../../../hllhc15/util/lhcb4.seq";
call,file="../../../hllhc15/hllhc_sequence.madx";
call,file="../../../hllhc15/toolkit/macro.madx";
seqedit,sequence=lhcb2;flatten;cycle,start=IP3;flatten;endedit;
exec,mk_beam(7000);
exec,myslice;
call,file="../../../hllhc15/round/opt_round_150_1500_thin.madx";
on_x1 = 1;
on_x5 = 1;
on_disp = 0;
exec,check_ip(b2);
""")
mad_b4.use(sequence="lhcb2")
mad_b4.globals['vrf400'] = 16
mad_b4.globals['lagrf400.b2'] = 0
mad_b4.twiss()
mad_b4.save(sequence=['lhcb2'], beam=True, file="sequence_b4.madx")