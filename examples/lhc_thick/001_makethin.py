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

slicing_strategies = [
    Strategy(slicing=Teapot(1)),  # Default catch-all as in MAD-X
    Strategy(slicing=Teapot(2), name=r'^mb.*'),
    Strategy(slicing=Teapot(20), name=r'^mq.*'),
    Strategy(slicing=Teapot(60), name=r'^mqx.*'),
    Strategy(
        slicing=Teapot(4),
        name=r'(mbx|mbrb|mbrc|mbrs|mbh|mqwa|mqwb|mqy|mqm|mqmc|mqml)\..*',
    ),
    Strategy(slicing=Teapot(2), name=r'(mqt|mqtli|mqtlh)\..*'),
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
