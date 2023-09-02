import xtrack as xt
import xpart as xp

from cpymad.madx import Madx

mad=Madx()
mad.call('../../test_data/hllhc15_thick/lhc.seq')
mad.call('../../test_data/hllhc15_thick/hllhc_sequence.madx')
mad.input('beam, sequence=lhcb1, particle=proton, energy=7000;')
mad.use('lhcb1')
mad.input('beam, sequence=lhcb2, particle=proton, energy=7000, bv=-1;')
mad.use('lhcb2')
mad.call("../../test_data/hllhc15_thick/opt_round_150_1500.madx")
mad.twiss()

mad.call('../../../hllhc15/toolkit/macro.madx')
mad.call('make_one_crossing_knob.madx') # only macro definitions

# Create empty tables
mad.input('''
! reset values
exec,mktable_orbit8h(orbit_ir8h);
exec,mktable_orbit8v(orbit_ir8v);
''')

# Clean up (?)
mad.input('''
setvars_const,table=orbit_ir8h;
setvars_const,table=orbit_ir8v;
''')

mad.input('''
testkqx8=abs(kqx.l8)*7000./0.3;

xip8b1 :=1e-3*(on_o8h   +on_sep8h);
xip8b2 :=1e-3*(on_o8h   -on_sep8h);
yip8b1 :=1e-3*(on_o8v   +on_sep8v);
yip8b2 :=1e-3*(on_o8v   -on_sep8v);
pxip8b1:=1e-6*(on_a8h   +on_x8h  );
pxip8b2:=1e-6*(on_a8h   -on_x8h  );
pyip8b1:=1e-6*(on_a8v   +on_x8v  );
pyip8b2:=1e-6*(on_a8v   -on_x8v  );


if(testkqx8> 210.) {acbxx.ir8= 1.0e-6/170;acbxs.ir8= 18.0e-6/2;};

if(testkqx8< 210.) {acbxx.ir8= 11.0e-6/170;acbxs.ir8= 16.0e-6/2;};


xang=170;psep=2;off=0.5;aoff=30;

if (betxip8b1<5){xang=300;};

exec,set_mcbx8; on_x8h=xang;exec,mkknob(ir8h,on_x8h);
!nn=18;while(tar_on_x8h>1e-10 && nn>2){
!acbxx.ir8= nn*1e-6/170;
!exec,set_mcbx8; on_x8h=xang;exec,mkknob(ir8h,on_x8h);
!nn=nn-2;
!};
exec,set_mcbx8; on_x8v=xang;exec,mkknob(ir8v,on_x8v);


! setting knobs
setvars_const,table=orbit_ir8h;
setvars_const,table=orbit_ir8v;

setvars_knob,table=knob_on_x8h  ,knob=on_x8h;
setvars_knob,table=knob_on_x8v  ,knob=on_x8v;

''')

mad.use('lhcb1')
mad.use('lhcb2')

mad.globals['on_x8h'] = 100
mad.input('twiss, sequence=lhcb1, table=twb1;')
mad.input('twiss, sequence=lhcb2, table=twb2;')

import xdeps as xd
twb1 = xd.Table(mad.table.twb1)
twb2 = xd.Table(mad.table.twb2)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(twb1.s, twb1.x, label='x')
plt.plot(twb1.s, twb1.y, label='y')
plt.plot(twb2.s, twb2.x, label='x')
plt.plot(twb2.s, twb2.y, label='y')


# Inspect knob expressions
all_knobs_ip2ip8 = ['acbxh3.r2', 'acbchs5.r2b1', 'pxip2b1', 'acbxh2.l8',
    'acbyhs4.r8b2', 'pyip2b1', 'acbxv1.l8', 'acbyvs4.l2b1', 'acbxh1.l8',
    'acbxv2.r8', 'pxip8b2', 'yip8b1', 'pxip2b2', 'acbcvs5.r2b1', 'acbyhs4.l8b1',
    'acbyvs4.l8b1', 'acbxh2.l2', 'acbxh3.l2', 'acbxv1.r8', 'acbxv1.r2',
    'acbyvs4.r2b2', 'acbyvs4.l2b2', 'yip8b2', 'xip2b2', 'acbxh2.r2',
    'acbyhs4.l2b2', 'acbxv2.r2', 'acbyhs5.r8b1', 'acbxh2.r8', 'acbxv3.r8',
    'acbyvs5.r8b2', 'acbyvs5.l2b2', 'yip2b1', 'acbxv2.l2', 'acbyhs4.r2b2',
    'acbyhs4.r2b1', 'xip8b2', 'acbyvs5.l2b1', 'acbyvs4.r8b1', 'acbyvs4.r8b2',
    'acbyvs5.r8b1', 'acbxh1.r8', 'acbyvs4.l8b2', 'acbyhs5.l2b1', 'acbyvs4.r2b1',
    'acbcvs5.r2b2', 'acbcvs5.l8b2', 'acbyhs4.r8b1', 'pxip8b1', 'acbxv1.l2',
    'yip2b2', 'acbyhs4.l8b2', 'acbxv3.r2', 'xip8b1', 'acbchs5.r2b2', 'acbxh3.l8',
    'acbxh3.r8', 'acbyhs5.r8b2', 'acbxv2.l8', 'acbxh1.l2', 'pyip8b1', 'pyip8b2',
    'acbxv3.l8', 'xip2b1', 'acbyhs5.l2b2', 'acbchs5.l8b2', 'acbcvs5.l8b1',
    'pyip2b2', 'acbxv3.l2', 'acbchs5.l8b1', 'acbyhs4.l2b1', 'acbxh1.r2']
all_knobs_ip2ip8 = sorted(all_knobs_ip2ip8)
all_knobs_ip2 = [kk for kk in all_knobs_ip2ip8 if '.l2' in kk or '.r2' in kk or 'ip2' in kk]
all_knobs_ip8 = [kk for kk in all_knobs_ip2ip8 if '.l8' in kk or '.r8' in kk or 'ip8' in kk]

for kk in all_knobs_ip2:
    mad.show(kk)
for kk in all_knobs_ip8:
    mad.show(kk)

plt.show()
