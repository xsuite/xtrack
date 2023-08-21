import os
import time

from cpymad.madx import Madx

from cpymad.madx import Madx
mad=Madx()
mad.call('../../../hllhc15/lhc/lhc.seq')
mad.call('../../../hllhc15/hllhc_sequence.madx')
mad.input('beam, sequence=lhcb1, particle=proton, energy=7000;')
mad.input('beam, sequence=lhcb2, particle=proton, energy=7000, bv=-1;')
mad.use('lhcb1')
mad.call("../../../hllhc15/round/opt_round_150_1500.madx")
mad.twiss()

mad.call("../../../hllhc15/toolkit/macro.madx")

# Get the absolute path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
link_path = os.path.join(current_dir, 'slhc')
target_path = os.path.join(current_dir, '../../../hllhc15/')

# Create symlink if it doesn't exist
if not os.path.exists(link_path):
    os.symlink(target_path, link_path)

mad.input('''
    mux_ip15=30.830335;
    muy_ip15=29.869337;
''')

t1 = time.time()
mad.input('''
weakdeltaphasearc(BIM): macro = {
	exec,phasearc(2,3,23,BIM);
	exec,phasearc(3,4,34,BIM);
	exec,phasearc(6,7,67,BIM);
	exec,phasearc(7,8,78,BIM);
	dmux15=mux23BIM+mux34BIM;
	dmuy15=muy23BIM+muy34BIM;
	dmux51=mux67BIM+mux78BIM;
	dmuy51=muy67BIM+muy78BIM;
};

mkdelta_ip15(BIM,tmux,tmuy): macro = {
	exec,weakdeltaphasearc(BIM);
	dmuxtotaux15=dmux15; dmuytotaux15=dmuy15;
	dmuxtotaux51=dmux51; dmuytotaux51=dmuy51;
	match,use_macro;
	vary, name=kqf.a34;
	vary, name=kqf.a78;
	vary, name=kqd.a34;
	vary, name=kqd.a78;
	vary, name=kqf.a23;
	vary, name=kqf.a67;
	vary, name=kqd.a23;
	vary, name=kqd.a67;
	use_macro,name=weakdeltaphasearc(BIM);
	constraint,expr=  dmux15=dmuxtotaux15+tmux;
	constraint,expr=  dmuy15=dmuytotaux15+tmuy;
	constraint,expr=  dmux51=dmuxtotaux51-tmux;
	constraint,expr=  dmuy51=dmuytotaux51-tmuy;
	jacobian,calls=10,tolerance=1.0e-19;
	endmatch;
	!kqf.a23=kqf.a23; kqf.a34=kqf.a34; kqf.a67=kqf.a67; kqf.a78=kqf.a78;
	!kqd.a23=kqd.a23; kqd.a34=kqd.a34; kqd.a67=kqd.a67; kqd.a78=kqd.a78;
};
''')


mad.input('''
	exec,check_ip(b1);
	dphix15=table(twiss,IP1,mux)-table(twiss,IP5,mux);
	dphiy15=table(twiss,IP1,muy)-table(twiss,IP5,muy);
	if (dphix15<0){ dphix15=dphix15+refqxb1;};
	if (dphiy15<0){ dphiy15=dphiy15+refqyb1;};
	value,dphix15,dphiy15;
	dphix15= dphix15-mux_ip15;
	dphiy15= dphiy15-muy_ip15;
	value,dphix15,dphiy15;
	exec,mkdelta_ip15(b1,dphix15,dphiy15);
''')
t2 = time.time()

mad.input('''

	! select all insertion
	jac_calls=15;
	jac_tol=1e-20;
	exec,select(7,67,78,b1);
	exec,select(3,23,34,b1);
	scxir1=betx_IP1/betx0_IP1; scyir1=bety_IP1/bety0_IP1;
	scxir5=betx_IP5/betx0_IP5; scyir5=bety_IP5/bety0_IP5;
	value,scxir1,scyir1,scxir5,scyir5;
	exec,selectIRAUX(7,8,1,2,3,b1,scxir1,scyir1,betx0_IP1,bety0_IP1);
	exec,selectIRAUX(3,4,5,6,7,b1,scxir5,scyir5,betx0_IP5,bety0_IP5);


	! rematch all insertion
	relax_match15=0;
	call,file="../../../hllhc15/toolkit/rematch_ir234678_b1.madx";
	value,tarir2b1,tarir3b1,tarir4b1,tarir6b1,tarir7b1,tarir8b1;
	value,relax_match15;
	while (tarir3b1+tarir7b1+tarir4b1+tarir6b1+tarir2b1+tarir8b1>1e-10){
		if (relax_match15>=3){
	    	print, text="Error in rematching phase change";
	    	tarir2b1=0; tarir3b1=0; tarir4b1=0; tarir6b1=0; tarir7b1=0; tarir8b1=0;
	    	exit;
		};
		if (relax_match15<3){
	    	relax_match15=relax_match15+1;
	    	call,file="../../../hllhc15/toolkit/rematch_ir234678_b1.madx";
	    };
	};

''')
t3 = time.time()

print('\nTime match arcs: ', t2-t1)
print('\nTime match IRs: ', t3-t2)