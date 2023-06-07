from cpymad.madx import Madx
import xtrack as xt
import xpart as xp

from xtrack.slicing import Teapot, Strategy

# hllhc15 can be found at git@github.com:lhcopt/hllhc15.git

thin = False
kill_fringes_and_edges = True

mad = Madx()

mad.input(f"""
call,file="../../../hllhc15/util/lhc.seq";
call,file="../../../hllhc15/hllhc_sequence.madx";
call,file="../../../hllhc15/toolkit/macro.madx";
seqedit,sequence=lhcb1;flatten;cycle,start=IP7;flatten;endedit;
seqedit,sequence=lhcb2;flatten;cycle,start=IP7;flatten;endedit;
exec,mk_beam(7000);
call,file="../../../hllhc15/round/opt_round_150_1500.madx";
{('exec,myslice;' if thin else '')}
exec,check_ip(b1);
exec,check_ip(b2);
// acbv11.r8b1 = 2e-6;
""")

mad.use(sequence="lhcb1")
seq = mad.sequence.lhcb1
mad.twiss()

line0 = xt.Line.from_madx_sequence(mad.sequence.lhcb1, allow_thick=True, deferred_expressions=True)
line0.particle_ref = xp.Particles(mass0=seq.beam.mass*1e9, gamma0=seq.beam.gamma)
line0.twiss_default['method'] = '4d'
line0.twiss_default['matrix_stability_tol'] = 100
line0.build_tracker()
tw0_before = line0.twiss()

line = line0.copy()

n_slice_bends = 4
n_slice_quads = 20
n_slice_mb = 2
n_slice_mq = 20
n_slice_mqx = 60

slicing_strategies = [
    Strategy(slicing=Teapot(1)),  # Default catch-all as in MAD-X
    Strategy(slicing=Teapot(n_slice_bends), element_type=xt.TrueBend),
    Strategy(slicing=Teapot(n_slice_quads), element_type=xt.CombinedFunctionMagnet),
    Strategy(slicing=Teapot(n_slice_mb), name=r'^mb\..*'),
    Strategy(slicing=Teapot(n_slice_mq), name=r'^mq\..*'),
    Strategy(slicing=Teapot(n_slice_mqx), name=r'^mqx.*'),
]

print("Slicing thick elements...")
line.slice_in_place(slicing_strategies)

print("Building tracker...")
line.build_tracker()

tw0_after = line0.twiss()
tw_after = line.twiss()

# Compare tunes
print("Tunes before slicing:")
print(f"Thick: qx = {tw0_before.qx} \tqy = {tw0_before.qy}")
print(f"Thin:  qx = {tw_after.qx} \tqy = {tw_after.qy}")
print(f"Diffs: qx = {tw_after.qx - tw0_before.qx} \tqy = {tw_after.qy - tw0_before.qy}")


# mad.input(
# f'''
#   select, flag=makethin, clear;
#   select, flag=makethin, class=mb, slice={('0' if thick_arc_bends else '2')};
#   select, flag=makethin, class=mq, slice={('0' if thick_arc_quads else '2')};
#   select, flag=makethin, class=mqxa,  slice={n_slice_mqx};  !old triplet
#   select, flag=makethin, class=mqxb,  slice={n_slice_mqx};  !old triplet
#   select, flag=makethin, class=mqxc,  slice={n_slice_mqx};  !new mqxa (q1,q3)
#   select, flag=makethin, class=mqxd,  slice={n_slice_mqx};  !new mqxb (q2a,q2b)
#   select, flag=makethin, class=mqxfa, slice={n_slice_mqx};  !new (q1,q3 v1.1)
#   select, flag=makethin, class=mqxfb, slice={n_slice_mqx};  !new (q2a,q2b v1.1)
#   select, flag=makethin, class=mbxa,  slice={n_slice_mb};   !new d1
#   select, flag=makethin, class=mbxf,  slice={n_slice_mb};   !new d1 (v1.1)
#   select, flag=makethin, class=mbrd,  slice={n_slice_mb};   !new d2 (if needed)
#   select, flag=makethin, class=mqyy,  slice={n_slice_mq};   !new q4
#   select, flag=makethin, class=mqyl,  slice={n_slice_mq};   !new q5
#   select, flag=makethin, pattern=mbx\.,    slice={n_slice_mb};
#   select, flag=makethin, pattern=mbrb\.,   slice={n_slice_mb};
#   select, flag=makethin, pattern=mbrc\.,   slice={n_slice_mb};
#   select, flag=makethin, pattern=mbrs\.,   slice={n_slice_mb};
#   select, flag=makethin, pattern=mbh\.,    slice={n_slice_mb};
#   select, flag=makethin, pattern=mqwa\.,   slice=4;
#   select, flag=makethin, pattern=mqwb\.,   slice=4;
#   select, flag=makethin, pattern=mqy\.,    slice=4;
#   select, flag=makethin, pattern=mqm\.,    slice=4;
#   select, flag=makethin, pattern=mqmc\.,   slice=4;
#   select, flag=makethin, pattern=mqml\.,   slice=4;
#   select, flag=makethin, pattern=mqtlh\.,  slice=2;
#   select, flag=makethin, pattern=mqtli\.,  slice=2;
#   select, flag=makethin, pattern=mqt\.  ,  slice=2;

#   beam;
#   use,sequence=lhcb1; makethin,sequence=lhcb1,makedipedge=false,style=teapot;
#   use,sequence=lhcb2; makethin,sequence=lhcb2,makedipedge=false,style=teapot;
# ''')