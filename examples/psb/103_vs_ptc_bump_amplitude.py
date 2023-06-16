import numpy as np
from cpymad.madx import Madx

import xtrack as xt
import xpart as xp
import xdeps as xd

import matplotlib.pyplot as plt

plt.close('all')
bump_amplitude = np.linspace(0, 1, 11)
qx_mad =[]
qy_mad =[]
dqx_mad2 =[]
dqy_mad2 =[]
qx_xsuite =[]
qy_xsuite =[]
dqx_xsuite =[]
dqy_xsuite =[]
qx_ptc =[]
qy_ptc =[]
dqx_ptc =[]
dqy_ptc =[]
betx_sel_xsuite = []
betx_sel_ptc = []
bety_sel_xsuite = []
bety_sel_ptc = []
for ii, bb in enumerate(bump_amplitude):

    bumper_names = ['bi1.bsw1l1.1', 'bi1.bsw1l1.2', 'bi1.bsw1l1.3', 'bi1.bsw1l1.4']
    thick_bumpers = {
    'bi1.bsw1l1.1' : {'k0_name': 'k0BI1BSW1L11'},
    'bi1.bsw1l1.2' : {'k0_name': 'k0BI1BSW1L12'},
    'bi1.bsw1l1.3' : {'k0_name': 'k0BI1BSW1L13'},
    'bi1.bsw1l1.4' : {'k0_name': 'k0BI1BSW1L14'},
    }

    mad = Madx()

    # Load model
    mad.input('''
    call, file = 'psb.seq';
    call, file = 'psb_fb_lhc.str';
    beam, particle=PROTON, pc=0.5708301551893517;
    use, sequence=psb1;
    twiss;
    ''')

    # Add shifts
    mad.input('''
    SELECT,FLAG=ERROR,CLEAR;
    SELECT,FLAG=ERROR,PATTERN=BI1.BSW1L1.1*;
    EALIGN, DX=-0.0057;

    SELECT,FLAG=ERROR,CLEAR;
    SELECT,FLAG=ERROR,PATTERN=BI1.BSW1L1.2*;
    SELECT,FLAG=ERROR,PATTERN=BI1.BSW1L1.3*;
    SELECT,FLAG=ERROR,PATTERN=BI1.BSW1L1.4*;
    EALIGN, DX=-0.0442;
    ''')

    # Match tunes
    mad.input(f'''
    QH = 4.4;
    QV = 4.45;

    MATCH, Sequence=psb1;
        VARY, NAME = kbrqf, STEP = 1e-3;
        VARY, NAME = kbrqd, STEP = 1e-3;
        GLOBAL, Q1 = QH;
        GLOBAL, Q2 = QV;
        JACOBIAN,CALLS=1000,TOLERANCE=1.0E-18,STRATEGY=3;
    ENDMATCH;
    ''')

    # # Handle bumpers (angle = 0)
    # seq = mad.sequence.psb1
    # seq.expanded_elements['bi1.bsw1l1.1'].angle = 1e-20
    # seq.expanded_elements['bi1.bsw1l1.2'].angle = 1e-20
    # seq.expanded_elements['bi1.bsw1l1.3'].angle = 1e-20
    # seq.expanded_elements['bi1.bsw1l1.4'].angle = 1e-20

    # Store bumpers length
    for nn in bumper_names:
        thick_bumpers[nn]['length'] = mad.sequence.psb1.expanded_elements[nn].l

    # Set K0 and K2 for thick bumpers
    mad.input(f'''

    l_bsw1l1.1 = {thick_bumpers['bi1.bsw1l1.1']['length']};
    l_bsw1l1.2 = {thick_bumpers['bi1.bsw1l1.2']['length']};
    l_bsw1l1.3 = {thick_bumpers['bi1.bsw1l1.3']['length']};
    l_bsw1l1.4 = {thick_bumpers['bi1.bsw1l1.4']['length']};

    BSW_K0L = 6.6E-2 * {bb};
    BSW_K2L = 0;                    // -9.7429e-02;
    '''
    )

    mad.input('''
    SELECT,FLAG=ERROR,CLEAR;
    SELECT,FLAG=ERROR,PATTERN=BI1.BSW1L1.1;
    SELECT,FLAG=ERROR,PATTERN=BI1.BSW1L1.4;
    EFCOMP, DKN:={+BSW_K0L, 0, +BSW_K2L};
    !EFCOMP, DKN:={0, 0, +BSW_K2L};

    SELECT,FLAG=ERROR,CLEAR;
    SELECT,FLAG=ERROR,PATTERN=BI1.BSW1L1.2;
    SELECT,FLAG=ERROR,PATTERN=BI1.BSW1L1.3;
    EFCOMP, DKN:={-BSW_K0L, 0, -BSW_K2L};

    ''')

    # Twiss and ptc twiss with errors
    twmad = mad.twiss()

    beta0 = mad.sequence.psb1.beam.beta
    qx_mad.append(twmad.summary.q1)
    qy_mad.append(twmad.summary.q2)

    twmad = twmad.dframe()

    mad.input('''
    ptc_create_universe;
    ptc_create_layout, model=2, method=2, nst=5, exact=true;
    ptc_setswitch, debuglevel=0, nocavity=false, fringe=true,
                exact_mis=true, time=true, totalpath=true;
    PTC_ALIGN;
    ptc_twiss, closed_orbit, table=ptc_twiss, icase=4, no=4,
                summary_table=ptc_twiss_summary, DELTAP_DEPENDENCY=true;
    ptc_end;
    ''')
    twptc = mad.table.ptc_twiss.dframe()
    qx_ptc.append(mad.table.ptc_twiss.mu1[-1])
    qy_ptc.append(mad.table.ptc_twiss.mu2[-1])
    t_ptc = xd.Table(mad.table.ptc_twiss)
    betx_sel_ptc.append(t_ptc['betx', 'br.stscrap22:1'])
    bety_sel_ptc.append(t_ptc['bety', 'br.stscrap22:1'])

    mad.use(sequence='psb1') # wipes out the errors

    # Put back misalignments
    mad.input('''
    SELECT,FLAG=ERROR,CLEAR;
    SELECT,FLAG=ERROR,PATTERN=BI1.BSW1L1.1*;
    EALIGN, DX=-0.0057;

    SELECT,FLAG=ERROR,CLEAR;
    SELECT,FLAG=ERROR,PATTERN=BI1.BSW1L1.2*;
    SELECT,FLAG=ERROR,PATTERN=BI1.BSW1L1.3*;
    SELECT,FLAG=ERROR,PATTERN=BI1.BSW1L1.4*;
    EALIGN, DX=-0.0442;
    ''')

    # K0 and K2 as strengths
    mad.input('''

    k0BI1BSW1L11 := BSW_K0L/l_bsw1l1.1;
    k0BI1BSW1L12 := -BSW_K0L/l_bsw1l1.2;
    k0BI1BSW1L13 := -BSW_K0L/l_bsw1l1.3;
    k0BI1BSW1L14 := BSW_K0L/l_bsw1l1.4;

    bi1.bsw1l1.1, k2 := BSW_K2L/l_bsw1l1.1;
    bi1.bsw1l1.2, k2 := -BSW_K2L/l_bsw1l1.2;
    bi1.bsw1l1.3, k2 := -BSW_K2L/l_bsw1l1.3;
    bi1.bsw1l1.4, k2 := BSW_K2L/l_bsw1l1.4;

    ''')

    twmad2 = mad.twiss()
    dqx_mad2.append(twmad2.summary.dq1 * beta0)
    dqy_mad2.append(twmad2.summary.dq2 * beta0)


    line = xt.Line.from_madx_sequence(mad.sequence.psb1,
                                    allow_thick=True,
                                    apply_madx_errors=True,
                                    deferred_expressions=True)
    line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV,
                                gamma0=mad.sequence.psb1.beam.gamma)
    line.build_tracker()
    line.to_json('psb_with_chicane.json')

    tw = line.twiss(method='4d')
    qx_xsuite.append(tw.qx)
    qy_xsuite.append(tw.qy)
    dqx_xsuite.append(tw.dqx)
    dqy_xsuite.append(tw.dqy)
    betx_sel_xsuite.append(tw['betx', 'br.stscrap22'])
    bety_sel_xsuite.append(tw['bety', 'br.stscrap22'])


    beta0 = line.particle_ref.beta0[0]

    plt.figure(1+ii)
    sp1 = plt.subplot(2,1,1)
    plt.plot(tw.s, tw.betx, label='xtrack')
    # plt.plot(twmad.s, twmad.betx, label='madx')
    plt.plot(twptc.s, twptc.betx, label='ptc')
    plt.plot(tw.s, tw.bety, label='xtrack')
    # plt.plot(twmad.s, twmad.bety, label='madx')
    plt.plot(twptc.s, twptc.bety, label='ptc')
    plt.legend()
    plt.ylabel('betx/bety [m]')

    # plt.subplot(3,1,2, sharex=sp1)
    # plt.plot(tw.s, tw.dx, label='xtrack')
    # plt.plot(twptc.s, twptc.dx * beta0, label='ptc')
    # plt.plot(tw.s, tw.dy, label='xtrack')
    # plt.plot(twptc.s, twptc.dy * beta0, label='ptc')

    plt.subplot(2,1,2, sharex=sp1)
    plt.plot(tw.s, tw.x, label='xtrack')
    plt.plot(twptc.s, twptc.x, label='ptc')
    plt.plot(tw.s, tw.y, label='xtrack')
    plt.plot(twptc.s, twptc.y, label='ptc')
    plt.legend()
    plt.ylabel('x/y [m]')

    plt.suptitle(f'bump amplitude = {bb*100:.1f} %')

    plt.legend()

plt.figure(100)
sp1 = plt.subplot(2,1,1)
# plt.plot(bump_amplitude, qx_mad, label='madx')
plt.plot(bump_amplitude, qx_xsuite, label='xtrack')
plt.plot(bump_amplitude, qx_ptc, label='ptc')
plt.legend()
plt.ylabel('qx')

plt.subplot(2,1,2, sharex=sp1, sharey=sp1)
# plt.plot(bump_amplitude, qy_mad, label='madx')
plt.plot(bump_amplitude, qy_xsuite, label='xtrack')
plt.plot(bump_amplitude, qy_ptc, label='ptc')
plt.legend()
plt.ylabel('qy')
plt.xlabel('bump amplitude')

plt.figure(101)
sp1 = plt.subplot(2,1,1)
plt.plot(bump_amplitude, dqx_mad2, label='madx')
plt.plot(bump_amplitude, dqx_xsuite, label='xtrack')
plt.legend()
plt.ylabel('dqx')


plt.subplot(2,1,2, sharex=sp1, sharey=sp1)
plt.plot(bump_amplitude, dqy_mad2, label='madx')
plt.plot(bump_amplitude, dqy_xsuite, label='xtrack')
plt.legend()
plt.ylabel('dqy')
plt.xlabel('bump amplitude')

plt.figure(102)
sp1 = plt.subplot(2,1,1)
plt.plot(bump_amplitude, betx_sel_xsuite, label='xtrack')
plt.plot(bump_amplitude, betx_sel_ptc, label='ptc')
plt.legend()
plt.ylabel('betx [m]')

plt.subplot(2,1,2, sharex=sp1, sharey=sp1)
plt.plot(bump_amplitude, bety_sel_xsuite, label='xtrack')
plt.plot(bump_amplitude, bety_sel_ptc, label='ptc')
plt.legend()
plt.ylabel('bety [m]')
plt.xlabel('bump amplitude')




plt.show()