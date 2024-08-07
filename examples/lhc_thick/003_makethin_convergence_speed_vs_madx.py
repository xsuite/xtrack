import numpy as np
from cpymad.madx import Madx
import xtrack as xt

from xtrack.slicing import Teapot, Strategy

# hllhc15 can be found at git@github.com:lhcopt/hllhc15.git

thin = False
kill_fringes_and_edges = True

mq_slice_list = np.arange(2, 12, 2)

diff_qx_xsuite_list = []
diff_qy_xsuite_list = []
diff_qx_mad_list = []
diff_qy_mad_list = []


for ii, mq_sl in enumerate(mq_slice_list):
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

    if ii == 0:
        line0 = xt.Line.from_madx_sequence(
            mad.sequence.lhcb1, allow_thick=True, deferred_expressions=True)
        line0.particle_ref = xt.Particles(mass0=seq.beam.mass*1e9, gamma0=seq.beam.gamma)
        line0.twiss_default['method'] = '4d'
        line0.twiss_default['matrix_stability_tol'] = 100
        line0.build_tracker()
        tw0_before = line0.twiss()

    line = line0.copy()

    n_slice_bends = 4
    n_slice_quads = 20
    n_slice_mb = 2
    # n_slice_mq = 10
    n_slice_mq = mq_sl
    n_slice_mqt = 5
    n_slice_mqx = 60

    slicing_strategies = [
        Strategy(slicing=Teapot(1)),  # Default catch-all as in MAD-X
        Strategy(slicing=None, element_type=xt.Solenoid),
        Strategy(slicing=Teapot(n_slice_bends), element_type=xt.Bend),
        Strategy(slicing=Teapot(n_slice_quads), element_type=xt.Quadrupole),
        Strategy(slicing=Teapot(n_slice_mb), name=r'^mb\..*'),
        Strategy(slicing=Teapot(n_slice_mq), name=r'^mq\..*'),
        Strategy(slicing=Teapot(n_slice_mq), name=r'^mqt.*'),
        Strategy(slicing=Teapot(n_slice_mqx), name=r'^mqx.*'),
    ]

    print("Slicing thick elements...")
    line.slice_thick_elements(slicing_strategies)

    print("Building tracker...")
    line.build_tracker()

    tw0_after = line0.twiss()
    tw_after = line.twiss()

    # Compare tunes
    print("Tunes before slicing:")
    print(f"Thick: qx = {tw0_before.qx} \tqy = {tw0_before.qy}")
    print(f"Thin:  qx = {tw_after.qx} \tqy = {tw_after.qy}")
    print(f"Diffs: qx = {tw_after.qx - tw0_before.qx} \tqy = {tw_after.qy - tw0_before.qy}")

    thick_arc_bends = False
    thick_arc_quads = False

    twmad_thick = mad.twiss().dframe()

    mad.input(
    f'''
    select, flag=makethin, clear;
    select, flag=makethin, class=mb, slice={('0' if thick_arc_bends else n_slice_mb)};
    select, flag=makethin, class=mq, slice={('0' if thick_arc_quads else n_slice_mq)};
    select, flag=makethin, class=mqxa,       slice={n_slice_mqx};  !old triplet
    select, flag=makethin, class=mqxb,       slice={n_slice_mqx};  !old triplet
    select, flag=makethin, class=mqxc,       slice={n_slice_mqx};  !new mqxa (q1,q3)
    select, flag=makethin, class=mqxd,       slice={n_slice_mqx};  !new mqxb (q2a,q2b)
    select, flag=makethin, class=mqxfa,      slice={n_slice_mqx};  !new (q1,q3 v1.1)
    select, flag=makethin, class=mqxfb,      slice={n_slice_mqx};  !new (q2a,q2b v1.1)
    select, flag=makethin, class=mbxa,       slice={n_slice_bends};   !new d1
    select, flag=makethin, class=mbxf,       slice={n_slice_bends};   !new d1 (v1.1)
    select, flag=makethin, class=mbrd,       slice={n_slice_bends};   !new d2 (if needed)
    select, flag=makethin, class=mqyy,       slice={n_slice_mq};   !new q4
    select, flag=makethin, class=mqyl,       slice={n_slice_mq};   !new q5
    select, flag=makethin, pattern=mbx\.,    slice={n_slice_bends};
    select, flag=makethin, pattern=mbrb\.,   slice={n_slice_bends};
    select, flag=makethin, pattern=mbrc\.,   slice={n_slice_bends};
    select, flag=makethin, pattern=mbrs\.,   slice={n_slice_bends};
    select, flag=makethin, pattern=mbh\.,    slice={n_slice_bends};
    select, flag=makethin, pattern=mqwa\.,   slice={n_slice_quads};
    select, flag=makethin, pattern=mqwb\.,   slice={n_slice_quads};
    select, flag=makethin, pattern=mqy\.,    slice={n_slice_quads};
    select, flag=makethin, pattern=mqm\.,    slice={n_slice_quads};
    select, flag=makethin, pattern=mqmc\.,   slice={n_slice_quads};
    select, flag=makethin, pattern=mqml\.,   slice={n_slice_quads};
    select, flag=makethin, pattern=mqtlh\.,  slice={n_slice_mqt};
    select, flag=makethin, pattern=mqtli\.,  slice={n_slice_mqt};
    select, flag=makethin, pattern=mqt\.  ,  slice={n_slice_mqt};

    beam;
    use,sequence=lhcb1; makethin,sequence=lhcb1,makedipedge=true,style=teapot;
    use,sequence=lhcb2; makethin,sequence=lhcb2,makedipedge=true,style=teapot;
    use, sequence=lhcb2;
    use, sequence=lhcb1;

    ''')

    twmad_thin = mad.twiss().dframe()

    # Compare tunes
    print(f"Xsuite thick: qx = {tw0_before.qx} \tqy = {tw0_before.qy}")
    print(f"Xsuite thin:  qx = {tw_after.qx} \tqy = {tw_after.qy}")

    qx_mad_before = twmad_thick.mux[-1]
    qy_mad_before = twmad_thick.muy[-1]
    qx_mad_after = twmad_thin.mux[-1]
    qy_mad_after = twmad_thin.muy[-1]

    print(f"MAD-X thick: qx = {qx_mad_before} \tqy = {qy_mad_before}")
    print(f"MAD-X thin:  qx = {qx_mad_after} \tqy = {qy_mad_after}")

    print(f"Xsuite diffs: qx = {tw_after.qx - tw0_before.qx} \tqy = {tw_after.qy - tw0_before.qy}")
    print(f"MAD-X diffs: qx = {qx_mad_after - qx_mad_before} \tqy = {qy_mad_after - qy_mad_before}")

    diff_qx_xsuite_list.append(tw_after.qx - tw0_before.qx)
    diff_qy_xsuite_list.append(tw_after.qy - tw0_before.qy)
    diff_qx_mad_list.append(qx_mad_after - qx_mad_before)
    diff_qy_mad_list.append(qy_mad_after - qy_mad_before)

import matplotlib.pyplot as plt
plt.close('all')
plt.plot(mq_slice_list, diff_qx_xsuite_list, label='qx xsuite', color='g')
plt.plot(mq_slice_list, diff_qy_xsuite_list, label='qy xsuite', color='m')
plt.plot(mq_slice_list, diff_qx_mad_list, '--', label='qx mad', color='b')
plt.plot(mq_slice_list, diff_qy_mad_list, '--', label='qy mad', color='r')
plt.legend()
plt.xlabel('n_slice_mq')
plt.ylabel(r'$Q_{thin} - Q_{thick}$')
plt.grid()
plt.show()