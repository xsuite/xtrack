import itertools
import numpy as np

import xtrack.mad_loader
from xtrack import MadLoader
import xtrack as xt
from scipy.constants import c as clight

import xpart as xp

from cpymad.madx import Madx

def test_non_zero_index():
    lst=[1,2,3,0,0,0]
    assert xtrack.mad_loader.non_zero_len([1,2,3,0,0,0])==3

def test_add_lists():
    a=[1,2,3,1,1,1]
    b=[1,1,1,4,5,6]
    c=xtrack.mad_loader.add_lists(a,b,8)
    assert c==[2, 3, 4, 5, 6, 7, 0, 0]


def test_add_lists():
    a=[1,2,3,1,1,1]
    b=[1,1,1,4,5,6,7,8]
    c=xtrack.mad_loader.add_lists(a,b,8)
    assert c==[2, 3, 4, 5, 6, 7, 7, 8]

    a=[1,2,3,1,1,1]
    b=[1,1,1,4,5,6,7,8]
    c=xtrack.mad_loader.add_lists(a,b,10)
    assert c==[2, 3, 4, 5, 6, 7, 7, 8,0,0]


def test_tilt_shift_and_errors():

    mad = Madx()

    src="""
    k1=0.2;
    tilt=0.1;

    elm: multipole,
            knl:={0.1,-k1,0.3},
            ksl={-0.1,0.2,-0.3,4},
            angle=0.1,
            tilt=0.2,
            lrad=1,
            apertype="rectellipse",
            aperture={0.1,0.2,0.11,0.22},
            aper_tol={0.1,0.2,0.3},
            aper_tilt:=tilt,
            aper_offset={0.2, 0.3};

    seq: sequence, l=1;
    elm1:elm, at=0;
    mk:marker, at=0.1;
    mk2:marker, at=0.1,aperture={0.1,0.2,0.11,0.22},apertype="rectellipse";
    elm2:elm, at=0.5;
    elm3:elm, at=0.5;
    endsequence;


    beam;
    use,sequence=seq;

    select,pattern=elm,flag=error;
    efcomp,order=0,radius=0.3,
    dkn={0.01,-0.02,0.03,0.2},
    dks={-0.01,0.02,-0.03,0.3,5},
    dknr={0.02,-0.04,0.06},
    dksr={-0.03,0.05,-0.07};

    ealign,
    DX=0.1, DY=0.3, DS=0.0,
    DPHI=0.0, DTHETA=0.0, DPSI=0.3,
    AREX=0.2, AREY=0.3;

    twiss,betx=1,bety=1,x=0.1,y=0.2,t=0.3;

    track,onepass;
    start,x=0.1,y=0.2,t=0.3;
    run,turns=1;
    endtrack;
    value,table(twiss,elm,x);
    value,table(tracksumm,x,2);
    value,table(twiss,elm,px);
    value,table(tracksumm,px,2);
    value,table(twiss,elm,y);
    value,table(tracksumm,y,2);
    value,table(twiss,elm,py);
    value,table(tracksumm,py,2);
    """

    mad.input(src)

    from xtrack import MadLoader

    def gen_options(opt1):
        for v in itertools.product(*([[True, False]]*len(opt1))):
            yield dict(zip(opt1,v))

    opt1=["enable_expressions",
        "enable_errors",
        "enable_apertures"]

    opt2=["skip_markers","merge_drifts","merge_multipoles"]


    for opt in gen_options(opt1):
        ml=MadLoader(mad.sequence.seq,**opt)
        line=ml.make_line()

        if opt['enable_apertures'] and opt['enable_errors']:
            line.element_names == (
                'seq$start', 'elm1_aper_tilt_entry', 'elm1_aper_offset_entry',
                'elm1_aper', 'elm1_aper_offset_exit', 'elm1_aper_tilt_exit',
                'elm1_tilt_entry', 'elm1_offset_entry', 'elm1', 'elm1_offset_exit',
                'elm1_tilt_exit', 'drift_0', 'mk', 'mk2_aper', 'mk2', 'drift_1',
                'elm2_aper_tilt_entry', 'elm2_aper_offset_entry', 'elm2_aper',
                'elm2_aper_offset_exit', 'elm2_aper_tilt_exit', 'elm2_tilt_entry',
                'elm2_offset_entry', 'elm2', 'elm2_offset_exit', 'elm2_tilt_exit',
                'elm3_aper_tilt_entry', 'elm3_aper_offset_entry', 'elm3_aper',
                'elm3_aper_offset_exit', 'elm3_aper_tilt_exit', 'elm3_tilt_entry',
                'elm3_offset_entry', 'elm3', 'elm3_offset_exit', 'elm3_tilt_exit',
                'drift_2', 'seq$end')

        elif opt['enable_apertures'] and not(opt['enable_errors']):
            line.element_names == (
                'seq$start', 'elm1_aper_tilt_entry', 'elm1_aper_offset_entry',
                'elm1_aper', 'elm1_aper_offset_exit', 'elm1_aper_tilt_exit',
                'elm1_tilt_entry', 'elm1', 'elm1_tilt_exit', 'drift_0', 'mk',
                'mk2_aper', 'mk2', 'drift_1', 'elm2_aper_tilt_entry',
                'elm2_aper_offset_entry', 'elm2_aper', 'elm2_aper_offset_exit',
                'elm2_aper_tilt_exit', 'elm2_tilt_entry', 'elm2', 'elm2_tilt_exit',
                'elm3_aper_tilt_entry', 'elm3_aper_offset_entry', 'elm3_aper',
                'elm3_aper_offset_exit', 'elm3_aper_tilt_exit',
                'elm3_tilt_entry','elm3', 'elm3_tilt_exit', 'drift_2', 'seq$end'
                )
        elif not(opt['enable_apertures']) and opt['enable_errors']:
            line.element_names == (
                'seq$start', 'elm1_tilt_entry', 'elm1_offset_entry', 'elm1',
                'elm1_offset_exit', 'elm1_tilt_exit', 'drift_0', 'mk', 'mk2',
                'drift_1', 'elm2_tilt_entry', 'elm2_offset_entry', 'elm2',
                'elm2_offset_exit', 'elm2_tilt_exit', 'elm3_tilt_entry',
                'elm3_offset_entry', 'elm3', 'elm3_offset_exit',
                'elm3_tilt_exit', 'drift_2', 'seq$end')
        elif not(opt['enable_apertures']) and not(opt['enable_errors']):
            line.element_names == (
                'seq$start', 'elm1_tilt_entry', 'elm1', 'elm1_tilt_exit',
                'drift_0', 'mk', 'mk2', 'drift_1', 'elm2_tilt_entry', 'elm2',
                'elm2_tilt_exit', 'elm3_tilt_entry', 'elm3', 'elm3_tilt_exit',
                'drift_2', 'seq$end')

        for nn in line.element_names:
            if 'tilt' in nn:
                assert isinstance(line[nn], xt.SRotation)
            elif 'offset' in nn:
                assert isinstance(line[nn], xt.XYShift)

        mad_elm2 = mad.sequence['seq'].expanded_elements['elm2']

        on_err = int(opt['enable_errors'])
        if opt['enable_apertures']:

            assert np.isclose(line['elm2_aper_tilt_entry'].angle,
                                mad_elm2.aper_tilt/np.pi*180,
                                rtol=0, atol=1e-13)
            assert np.isclose(line['elm2_aper_tilt_exit'].angle,
                                -mad_elm2.aper_tilt/np.pi*180,
                                rtol=0, atol=1e-13)


            assert np.isclose(line['elm2_aper_offset_entry'].dx,
                        on_err * mad_elm2.align_errors.arex + mad_elm2.aper_offset[0],
                        rtol=0, atol=1e-13)
            assert np.isclose(line['elm2_aper_offset_entry'].dy,
                        on_err * mad_elm2.align_errors.arey + mad_elm2.aper_offset[1],
                        rtol=0, atol=1e-13)
            assert np.isclose(line['elm2_aper_offset_exit'].dx,
                        -(on_err * mad_elm2.align_errors.arex + mad_elm2.aper_offset[0]),
                        rtol=0, atol=1e-13)
            assert np.isclose(line['elm2_aper_offset_exit'].dy,
                        -(on_err * mad_elm2.align_errors.arey + mad_elm2.aper_offset[1]),
                        rtol=0, atol=1e-13)

            assert isinstance(line['elm2_aper'], xt.LimitRectEllipse)
            assert line['elm2_aper'].max_x == .1
            assert line['elm2_aper'].max_y == .2
            assert line['elm2_aper'].a_squ == .11**2
            assert line['elm2_aper'].b_squ == .22**2

        assert np.isclose(line['elm2_tilt_entry'].angle,
                    (mad_elm2.tilt + on_err * mad_elm2.align_errors.dpsi)/np.pi*180,
                    rtol=0, atol=1e-13)
        assert np.isclose(line['elm2_tilt_exit'].angle,
                    -(mad_elm2.tilt + on_err * mad_elm2.align_errors.dpsi)/np.pi*180,
                    rtol=0, atol=1e-13)

        if opt['enable_errors']:
            assert np.isclose(line['elm2_offset_entry'].dx,
                                on_err * mad_elm2.align_errors.dx, rtol=0, atol=1e-13)
            assert np.isclose(line['elm2_offset_entry'].dy,
                                on_err * mad_elm2.align_errors.dy, rtol=0, atol=1e-13)
            assert np.isclose(line['elm2_offset_exit'].dx,
                                -on_err * mad_elm2.align_errors.dx, rtol=0, atol=1e-13)
            assert np.isclose(line['elm2_offset_exit'].dy,
                                -on_err * mad_elm2.align_errors.dy, rtol=0, atol=1e-13)

        for ii in range(line['elm2'].order+1):
            ref = 0
            if len(mad_elm2.knl)>ii:
                ref += mad_elm2.knl[ii]
            if opt['enable_errors'] and len(mad_elm2.field_errors.dkn)>ii:
                ref += mad_elm2.field_errors.dkn[ii]
            assert np.isclose(line['elm2'].knl[ii], ref, rtol=0, atol=1e-13)

        for ii in range(line['elm2'].order+1):
            ref = 0
            if len(mad_elm2.ksl)>ii:
                ref += mad_elm2.ksl[ii]
            if opt['enable_errors'] and len(mad_elm2.field_errors.dks)>ii:
                ref += mad_elm2.field_errors.dks[ii]
            assert np.isclose(line['elm2'].ksl[ii], ref, rtol=0, atol=1e-13)

    for opt in gen_options(opt2):
        ml=MadLoader(mad.sequence.seq,**opt)
        line=list(ml.iter_elements())



def test_matrix():
    mad = Madx()

    mad.input("""
    a11=1;
    m1: matrix,
          rm11:=a11,rm12=2,rm21=3,rm22=4,
          kick1=0.1,kick2=0.2,kick3=0.3;

    ss: sequence, l=1; m1: m1, at=0; endsequence;

    beam; use, sequence=ss;
    """)

    line=MadLoader(mad.sequence.ss).make_line()
    line=MadLoader(mad.sequence.ss,enable_expressions=True).make_line()
    line.vars['a11']=2.0
    assert line[1].m1[0,0]==line.vars['a11']._value

def test_srotation():
    mad = Madx()

    mad.input("""
    angle=0.2;
    rot: srotation,angle:=angle;

    ss: sequence, l=1; rot: rot, at=0; endsequence;

    beam; use, sequence=ss;
    """)

    line=MadLoader(mad.sequence.ss).make_line()
    line=MadLoader(mad.sequence.ss,enable_expressions=True).make_line()
    assert isinstance(line[1],xt.SRotation)
    line.vars['angle'] = 2.0
    assert line[1].angle == line.vars['angle']._value*180/np.pi

def test_xrotation():
    mad = Madx()

    mad.input("""
    angle=0.2;
    rot: xrotation,angle:=angle;

    ss: sequence, l=1; rot: rot, at=0; endsequence;

    beam; use, sequence=ss;
    """)

    line=MadLoader(mad.sequence.ss).make_line()
    line=MadLoader(mad.sequence.ss,enable_expressions=True).make_line()
    assert isinstance(line[1],xt.XRotation)
    line.vars['angle'] = 2.0
    assert line[1].angle == line.vars['angle']._value*180/np.pi

def test_yrotation():
    mad = Madx()

    mad.input("""
    angle=0.2;
    rot: yrotation,angle:=angle;

    ss: sequence, l=1; rot: rot, at=0; endsequence;

    beam; use, sequence=ss;
    """)

    line=MadLoader(mad.sequence.ss).make_line()
    line=MadLoader(mad.sequence.ss,enable_expressions=True).make_line()
    assert isinstance(line[1],xt.YRotation)
    line.vars['angle'] = 2.0
    assert line[1].angle == line.vars['angle']._value*180/np.pi

def test_mad_elements_import():

    mad = Madx()

    # Element definitions
    mad.input("""

    a = 1.;

    cav0: rfcavity, freq:=a*10, lag:=a*0.5, volt:=a*6;
    cav1: rfcavity, lag:=a*0.5, volt:=a*6, harmon:=a*8;
    wire1: wire, current:=a*5, l:=a*0, l_phy:=a*1, l_int:=a*2, xma:=a*1e-3, yma:=a*2e-3;
    mult0: multipole, knl:={a*1,a*2,a*3}, ksl:={a*4,a*5,a*6}, lrad:=a*1.1;
    mult1: multipole, knl={1,2,3,0}, ksl:={1,2,3};
    mult2: multipole, knl={1,2,3,0}, ksl={1,2,0,0,0};
    mult3: multipole, knl={1,2,3,0}, ksl:={1,2,0,b,0};
    kick0: kicker, hkick:=a*5, vkick:=a*6, lrad:=a*2.2;
    kick1: tkicker, hkick:=a*7, vkick:=a*8, lrad:=a*2.3;
    kick2: hkicker, kick:=a*3, lrad:=a*2.4;
    kick3: vkicker, kick:=a*4, lrad:=a*2.5;
    dipedge0: dipedge, h:=a*0.1, e1:=a*3, fint:=a*4, hgap:=a*0.02;
    rfm0: rfmultipole, volt:=a*2, lag:=a*0.5, freq:=a*100.,
                knl:={a*2,a*3}, ksl:={a*4,a*5},
                pnl:={a*0.3, a*0.4}, psl:={a*0.5, a*0.6};
    crab0: crabcavity, volt:=a*2, lag:=a*0.5, freq:=a*100.;
    crab1: crabcavity, volt:=a*2, lag:=a*0.5, freq:=a*100., tilt:=a*pi/2;
    """)

    matrix_m0 = np.random.randn(6)*1E-6
    matrix_m1 = np.reshape(np.random.randn(36),(6,6))
    mad.input(f"mat:matrix,l:=0.003*a,"
              f"kick1:={matrix_m0[0]}*a,kick2:={matrix_m0[1]}*a,"
              f"kick3:={matrix_m0[2]}*a,kick4:={matrix_m0[3]}*a,"
              f"kick5:={matrix_m0[4]}*a,kick6:={matrix_m0[5]}*a,"
              f"rm11:={matrix_m1[0,0]}*a,rm12:={matrix_m1[0,1]}*a,"
              f"rm13:={matrix_m1[0,2]}*a,rm14:={matrix_m1[0,3]}*a,"
              f"rm15:={matrix_m1[0,4]}*a,rm16:={matrix_m1[0,5]}*a,"
              f"rm21:={matrix_m1[1,0]}*a,rm22:={matrix_m1[1,1]}*a,"
              f"rm23:={matrix_m1[1,2]}*a,rm24:={matrix_m1[1,3]}*a,"
              f"rm25:={matrix_m1[1,4]}*a,rm26:={matrix_m1[1,5]}*a,"
              f"rm31:={matrix_m1[2,0]}*a,rm32:={matrix_m1[2,1]}*a,"
              f"rm33:={matrix_m1[2,2]}*a,rm34:={matrix_m1[2,3]}*a,"
              f"rm35:={matrix_m1[2,4]}*a,rm36:={matrix_m1[2,5]}*a,"
              f"rm41:={matrix_m1[3,0]}*a,rm42:={matrix_m1[3,1]}*a,"
              f"rm43:={matrix_m1[3,2]}*a,rm44:={matrix_m1[3,3]}*a,"
              f"rm45:={matrix_m1[3,4]}*a,rm46:={matrix_m1[3,5]}*a,"
              f"rm51:={matrix_m1[4,0]}*a,rm52:={matrix_m1[4,1]}*a,"
              f"rm53:={matrix_m1[4,2]}*a,rm54:={matrix_m1[4,3]}*a,"
              f"rm55:={matrix_m1[4,4]}*a,rm56:={matrix_m1[4,5]}*a,"
              f"rm61:={matrix_m1[5,0]}*a,rm62:={matrix_m1[5,1]}*a,"
              f"rm63:={matrix_m1[5,2]}*a,rm64:={matrix_m1[5,3]}*a,"
              f"rm65:={matrix_m1[5,4]}*a,rm66={matrix_m1[5,5]}*a;")

    # Sequence
    mad.input("""
    testseq: sequence, l=10;
    m0: mult0 at=0.1;
    m1: mult1 at=0.11;
    m2: mult2 at=0.12;
    m3: mult3 at=0.13;
    c0: cav0, at=0.2, apertype=circle, aperture=0.01;
    c1: cav1, at=0.2, apertype=circle, aperture=0.01;
    k0: kick0, at=0.3;
    k1: kick1, at=0.33;
    k2: kick2, at=0.34;
    k3: kick3, at=0.35;
    de0: dipedge0, at=0.38;
    r0: rfm0, at=0.4;
    cb0: crab0, at=0.41;
    cb1: crab1, at=0.42;
    w: wire1, at=1;
    mat0:mat, at=2+0.003/2;
    endsequence;
    """
    )

    # Beam
    mad.input("""
    beam, particle=proton, gamma=1.05, sequence=testseq;
    """)


    mad.use('testseq')

    seq = mad.sequence['testseq']

    for test_expressions in [True, False]:
        line = xt.Line.from_madx_sequence(sequence=seq,
                                          deferred_expressions=test_expressions)
        line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, gamma0=1.05)

        line = xt.Line.from_dict(line.to_dict()) # This calls the to_dict method fot all
                                                # elements


        assert len(line.element_names) == len(line.element_dict.keys())
        assert line.get_length() == 10

        assert isinstance(line['m0'], xt.Multipole)
        assert line.get_s_position('m0') == 0.1
        assert np.all(line['m0'].knl == np.array([1,2,3]))
        assert np.all(line['m0'].ksl == np.array([4,5,6]))
        assert line['m0'].hxl == 1
        assert line['m0'].hyl == 4
        assert line['m0'].length == 1.1
        assert len(line['m1'].knl)==3
        assert len(line['m1'].ksl)==3
        assert len(line['m2'].knl)==3
        assert len(line['m2'].ksl)==3
        if test_expressions:
          assert len(line['m3'].knl)==4
          assert len(line['m3'].ksl)==4
        else:
          assert len(line['m3'].knl)==3
          assert len(line['m3'].ksl)==3


        assert isinstance(line['k0'], xt.Multipole)
        assert line.get_s_position('k0') == 0.3
        assert np.all(line['k0'].knl == np.array([-5]))
        assert np.all(line['k0'].ksl == np.array([6]))
        assert line['k0'].hxl == 0
        assert line['k0'].hyl == 0
        assert line['k0'].length == 2.2

        assert isinstance(line['k1'], xt.Multipole)
        assert line.get_s_position('k1') == 0.33
        assert np.all(line['k1'].knl == np.array([-7]))
        assert np.all(line['k1'].ksl == np.array([8]))
        assert line['k1'].hxl == 0
        assert line['k1'].hyl == 0
        assert line['k1'].length == 2.3

        assert isinstance(line['k2'], xt.Multipole)
        assert line.get_s_position('k2') == 0.34
        assert np.all(line['k2'].knl == np.array([-3]))
        assert np.all(line['k2'].ksl == np.array([0]))
        assert line['k2'].hxl == 0
        assert line['k2'].hyl == 0
        assert line['k2'].length == 2.4

        assert isinstance(line['k3'], xt.Multipole)
        assert line.get_s_position('k3') == 0.35
        assert np.all(line['k3'].knl == np.array([0]))
        assert np.all(line['k3'].ksl == np.array([4]))
        assert line['k3'].hxl == 0
        assert line['k3'].hyl == 0
        assert line['k3'].length == 2.5

        assert isinstance(line['c0'], xt.Cavity)
        assert line.get_s_position('c0') == 0.2
        assert line['c0'].frequency == 10e6
        assert line['c0'].lag == 180
        assert line['c0'].voltage == 6e6

        assert isinstance(line['c1'], xt.Cavity)
        assert line.get_s_position('c1') == 0.2
        assert np.isclose(line['c1'].frequency, clight*line.particle_ref.beta0/10.*8,
                        rtol=0, atol=1e-7)
        assert line['c1'].lag == 180
        assert line['c1'].voltage == 6e6

        assert isinstance(line['de0'], xt.DipoleEdge)
        assert line.get_s_position('de0') == 0.38
        assert line['de0'].h == 0.1
        assert line['de0'].e1 == 3
        assert line['de0'].fint == 4
        assert line['de0'].hgap == 0.02

        assert isinstance(line['r0'], xt.RFMultipole)
        assert line.get_s_position('r0') == 0.4
        assert np.all(line['r0'].knl == np.array([2,3]))
        assert np.all(line['r0'].ksl == np.array([4,5]))
        assert np.all(line['r0'].pn == np.array([0.3*360,0.4*360]))
        assert np.all(line['r0'].ps == np.array([0.5*360,0.6*360]))
        assert line['r0'].voltage == 2e6
        assert line['r0'].order == 1
        assert line['r0'].frequency == 100e6
        assert line['r0'].lag == 180

        assert isinstance(line['cb0'], xt.RFMultipole)
        assert line.get_s_position('cb0') == 0.41
        assert len(line['cb0'].knl) == 1
        assert len(line['cb0'].ksl) == 1
        assert np.isclose(line['cb0'].knl[0], 2*1e6/line.particle_ref.p0c[0],
                        rtol=0, atol=1e-12)
        assert np.all(line['cb0'].ksl == 0)
        assert np.all(line['cb0'].pn == np.array([270]))
        assert np.all(line['cb0'].ps == 0.)
        assert line['cb0'].voltage == 0
        assert line['cb0'].order == 0
        assert line['cb0'].frequency == 100e6
        assert line['cb0'].lag == 0

        assert isinstance(line['cb1'], xt.RFMultipole)
        assert line.get_s_position('cb1') == 0.42
        assert len(line['cb1'].knl) == 1
        assert len(line['cb1'].ksl) == 1
        assert np.isclose(line['cb1'].ksl[0], -2*1e6/line.particle_ref.p0c[0],
                        rtol=0, atol=1e-12)
        assert np.all(line['cb1'].knl == 0)
        assert np.all(line['cb1'].ps == np.array([270]))
        assert np.all(line['cb1'].pn == 0.)
        assert line['cb1'].voltage == 0
        assert line['cb1'].order == 0
        assert line['cb1'].frequency == 100e6
        assert line['cb1'].lag == 0


        assert isinstance(line['w'], xt.Wire)
        assert line.get_s_position('w') == 1
        assert line['w'].L_phy == 1
        assert line['w'].L_int == 2
        assert line['w'].xma == 1e-3
        assert line['w'].yma == 2e-3

        assert isinstance(line['mat0'],xt.FirstOrderTaylorMap)
        assert line.get_s_position('mat0') == 2
        assert np.allclose(line['mat0'].m0,matrix_m0,rtol=0.0,atol=1E-12)
        assert np.allclose(line['mat0'].m1,matrix_m1,rtol=0.0,atol=1E-12)
