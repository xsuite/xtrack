import sys
import numpy as np
import pandas as pd
import os

command_log_file='log.madx'
stdout_file='stdout.madx'
# Madx = pm.Madxp
from cpymad.madx import Madx

mad = Madx()

# Starting from optics file 1 m, r=1
filename0 = 'slhc/ramp/opt_ramp_1000_1500.madx'
# beta squeeze
betas   = [0.99] #np.arange(0.99, 0.63, -0.01)
# beta pre-squeeze
betas_0 = [0.99] #np.arange(0.99, 0.63, -0.01)


kk = 0

betx_IP1  = betas[kk]
betx0_IP1 = betas_0[kk]
betx_IP5  = betas[kk]
betx0_IP5 = betas_0[kk]
bety_IP1  = betas[kk]
bety0_IP1 = betas_0[kk]
bety_IP5  = betas[kk]
bety0_IP5 = betas_0[kk]

madx_to_read = filename0
################## Load initial optics ################## 
mad.input('''
Option,  warn,info;
System,"rm -rf temp";
System,"mkdir temp";

system,"ln -fns ../../../hllhc15 slhc";

call,file="slhc/lhc/lhc.seq";
call,file="slhc/toolkit/macro.madx";
call,file="slhc/hllhc_sequence.madx";
!call,file="slhc/ramp/opt_ramp_1000_1500_thin.madx";
call,file="%s";
!exec,myslice;

exec,mk_beam(7000);
exec,check_ip(b1); exec,check_ip(b2);

use, sequence=lhcb1;
select, flag=twiss, clear;
twiss,chrom,file="last_twiss.tfs";

''' %(madx_to_read))
################## Change betas and betas0 ##################     
mad.globals.betx_IP1 =betx_IP1
mad.globals.betx0_IP1=betx0_IP1
mad.globals.betx_IP5 =betx_IP5
mad.globals.betx0_IP5 =betx0_IP5

mad.globals.bety_IP1 =bety_IP1
mad.globals.bety0_IP1=bety0_IP1
mad.globals.bety_IP5 =bety_IP5
mad.globals.bety0_IP5 =bety0_IP5

if kk==len(betas)-1:
    print("Requested betas: ", mad.globals.betx_IP1, mad.globals.betx0_IP1, mad.globals.betx_IP5, mad.globals.betx0_IP5)


################## Run rematch_hllhc script up to W rematch ################## 
mad.input('''

exec, round_phases;

exec,rematch_arc(1,2,12);
exec,rematch_arc(2,3,23);
exec,rematch_arc(3,4,34);
exec,rematch_arc(4,5,45);
exec,rematch_arc(5,6,56);
exec,rematch_arc(6,7,67);
exec,rematch_arc(7,8,78);
exec,rematch_arc(8,1,81);

scxir1=betx_IP1/betx0_IP1;
scyir1=bety_IP1/bety0_IP1;
scxir5=betx_IP5/betx0_IP5;
scyir5=bety_IP5/bety0_IP5;

value,scxir1,scyir1,scxir5,scyir5;

exec,selectIR15(5,45,56,b1);
exec,selectIR15(5,45,56,b2);
exec,selectIR15(1,81,12,b1);
exec,selectIR15(1,81,12,b2);

on_holdselect=1; jac_calls=   15;jac_tol=1e-20; jac_bisec=3;

if (sc79==0){sc79=0.999;};
if (sch==0) {sch =0.99; };
if (imb==0) {imb=1.50;};
if (scl==0) {scl=0.06;};
grad=132.6; bmaxds=500;
betxip5b1=betx0_IP5; betyip5b1=bety0_IP5;
betxip5b2=betx0_IP5; betyip5b2=bety0_IP5;
betxip1b1=betx0_IP1; betyip1b1=bety0_IP1;
betxip1b2=betx0_IP1; betyip1b2=bety0_IP1;

''')
prrr
mad.input('match_on_triplet=0; call,file="slhc/toolkit/rematch_ir15b12.madx";')

mad.input('call,file="slhc/toolkit/rematch_ir15b12.madx";')


mad.input('''

betxip5b1=betx_IP5; betyip5b1=bety_IP5;
betxip5b2=betx_IP5; betyip5b2=bety_IP5;
betxip1b1=betx_IP1; betyip1b1=bety_IP1;
betxip1b2=betx_IP1; betyip1b2=bety_IP1;

exec,select(3,23,34,b1);
exec,select(3,23,34,b2);
exec,select(7,67,78,b1);
exec,select(7,67,78,b2);

call,file="slhc/toolkit/rematch_ir3b1.madx";
call,file="slhc/toolkit/rematch_ir3b2.madx";
call,file="slhc/toolkit/rematch_ir7b1.madx";
call,file="slhc/toolkit/rematch_ir7b2.madx";


exec,selectIRAUX(7,8,1,2,3,b1,scxir1,scyir1,betx0_ip1,bety0_ip1);
exec,selectIRAUX(7,8,1,2,3,b2,scxir1,scyir1,betx0_ip1,bety0_ip1);
exec,selectIRAUX(3,4,5,6,7,b1,scxir5,scyir5,betx0_ip5,bety0_ip5);
exec,selectIRAUX(3,4,5,6,7,b2,scxir5,scyir5,betx0_ip5,bety0_ip5);

call,file="slhc/toolkit/rematch_ir2b12.madx";
call,file="slhc/toolkit/rematch_ir8b12.madx";
call,file="slhc/toolkit/rematch_ir4b1.madx";
call,file="slhc/toolkit/rematch_ir4b2.madx";
jac_calls=   15;
!dxip6b1=0;dpxip6b1=0; dxip6b2=0;dpxip6b2=0;
!call,file=rematch_ir6b1m.madx;
!call,file=rematch_ir6b2m.madx;
!dxip6b1=0;dpxip6b1=0; dxip6b2=0;dpxip6b2=0;
nomatch_dx=1; nomatch_dpx=1;
call,file="slhc/toolkit/rematch_ir6b1.madx";
call,file="slhc/toolkit/rematch_ir6b2.madx";
nomatch_dx=0; nomatch_dpx=0;
call,file="slhc/toolkit/rematch_ir6b1.madx";
call,file="slhc/toolkit/rematch_ir6b2.madx";
value,tarir2b1,tarir4b1,tarir6b1,tarir8b1;
value,tarir2b2,tarir4b2,tarir6b2,tarir8b2;
value,tarir3b1,tarir7b1,tarir3b2,tarir7b2;
value,tarir1b1,tarir5b1,tarir1b2,tarir5b2;


tarsqueeze=tarir2b1+tarir4b1+tarir6b1+tarir8b1+
            tarir2b2+tarir4b2+tarir6b2+tarir8b2+
            tarir3b1+tarir7b1+tarir3b2+tarir7b2+
            tarir1b1+tarir5b1+tarir1b2+tarir5b2;

exec,check_ip(b1); exec,check_ip(b2);
value,tarsqueeze;

if (tarsqueeze>1e-15 || match_optics_only==1){return;};


call,file="slhc/toolkit/rematch_xing_ir15.madx";
call,file="slhc/toolkit/rematch_xing_ir28.madx";

value,tar_xing_ir15,tar_xing_ir28;

ksf=0.06;ksd=-0.099;
exec,set_sext_all(ksf,ksd,ksf,ksd); k2max=0.38;
''')

################## Save intermediate optics in optics dir##################
filename = 'opt_ramp_%s_1500_set_sext_all' %int(mad.globals.betx_IP1*1000)
mad.input('''
    exec,save_optics_hllhc("%s.madx");
    ''' %filename)