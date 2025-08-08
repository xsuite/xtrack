import itertools
import pathlib

import numpy as np
from cpymad.madx import Madx
from scipy.constants import c as clight
from scipy.special import factorial

import xobjects as xo
import xpart as xp
import xtrack as xt
import xtrack.mad_loader
from xtrack import MadLoader

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()


def test_non_zero_index():
    lst = [1, 2, 3, 0, 0, 0]
    assert xtrack.mad_loader.non_zero_len(lst) == 3


def test_add_lists_same_lengths():
    a = [1, 2, 3, 1, 1, 1]
    b = [1, 1, 1, 4, 5, 6]
    c = xtrack.mad_loader.add_lists(a, b, 8)
    assert c == [2, 3, 4, 5, 6, 7, 0, 0]


def test_add_lists_different_lengths_pick_longer():
    a = [1, 2, 3, 1, 1, 1]
    b = [1, 1, 1, 4, 5, 6, 7, 8]
    c = xtrack.mad_loader.add_lists(a, b, 8)
    assert c == [2, 3, 4, 5, 6, 7, 7, 8]


def test_add_lists_manually_extend():
    a = [1, 2, 3, 1, 1, 1]
    b = [1, 1, 1, 4, 5, 6, 7, 8]
    c = xtrack.mad_loader.add_lists(a, b, 10)
    assert c == [2, 3, 4, 5, 6, 7, 7, 8, 0, 0]


def test_tilt_shift_and_errors():
    mad = Madx(stdout=False)

    src = """
    k1=0.2;
    tilt=0.1;

    elm: multipole,
            knl:={0.1,-k1,0.3},
            ksl={0.,0.2,-0.3,4},
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
        for v in itertools.product(*([[True, False]] * len(opt1))):
            yield dict(zip(opt1, v))

    opt1 = ["enable_expressions",
        "enable_errors",
        "enable_apertures"]

    opt2 = ["skip_markers", "merge_drifts", "merge_multipoles"]

    for opt in gen_options(opt1):
        ml = MadLoader(mad.sequence.seq, **opt)
        line = ml.make_line()

        if opt['enable_apertures'] and opt['enable_errors']:
            assert np.all(np.array(line.element_names) == (
                'seq$start',
                'elm1_aper',
                'elm1',
                'drift_0', 'mk', 'mk2_aper', 'mk2', 'drift_1',
                'elm2_aper',
                'elm2',
                'elm3_aper',
                'elm3',
                'drift_2', 'seq$end'))

        elif opt['enable_apertures'] and not (opt['enable_errors']):
            assert np.all(np.array(line.element_names) == (
                'seq$start',
                'elm1_aper',
                'elm1',
                'drift_0',
                'mk',
                'mk2_aper', 'mk2', 'drift_1',
                'elm2_aper',
                'elm2',
                'elm3_aper',
                'elm3', 'drift_2', 'seq$end'
            ))
        elif not (opt['enable_apertures']) and opt['enable_errors']:
            assert np.all(np.array(line.element_names) == (
                'seq$start', 'elm1',
                'drift_0', 'mk', 'mk2',
                'drift_1', 'elm2',
                'elm3',
                'drift_2', 'seq$end'))
        elif not (opt['enable_apertures']) and not (opt['enable_errors']):
            assert np.all(np.array(line.element_names) == (
                'seq$start', 'elm1',
                'drift_0', 'mk', 'mk2', 'drift_1', 'elm2',
                'elm3',
                'drift_2', 'seq$end'))

        for nn in line.element_names:
            if 'tilt' in nn:
                assert isinstance(line[nn], xt.SRotation)
            elif 'offset' in nn:
                assert isinstance(line[nn], xt.XYShift)

        mad_elm2 = mad.sequence['seq'].expanded_elements['elm2']

        on_err = int(opt['enable_errors'])
        if opt['enable_apertures']:
            xo.assert_allclose(line['elm2_aper'].rot_s_rad,
                               mad_elm2.aper_tilt,
                               rtol=0, atol=1e-13)

            xo.assert_allclose(line['elm2_aper'].shift_x,
                               on_err * mad_elm2.align_errors.arex + mad_elm2.aper_offset[0],
                               rtol=0, atol=1e-13)
            xo.assert_allclose(line['elm2_aper'].shift_y,
                               on_err * mad_elm2.align_errors.arey + mad_elm2.aper_offset[1],
                               rtol=0, atol=1e-13)

            assert isinstance(line['elm2_aper'], xt.LimitRectEllipse)
            assert line['elm2_aper'].max_x == .1
            assert line['elm2_aper'].max_y == .2
            assert line['elm2_aper'].a_squ == .11 ** 2
            assert line['elm2_aper'].b_squ == .22 ** 2

        xo.assert_allclose(line['elm2'].rot_s_rad,
                           (mad_elm2.tilt + on_err * mad_elm2.align_errors.dpsi),
                           rtol=0, atol=1e-13)

        if opt['enable_errors']:
            xo.assert_allclose(line['elm2'].shift_x,
                               on_err * mad_elm2.align_errors.dx, rtol=0, atol=1e-13)
            xo.assert_allclose(line['elm2'].shift_y,
                               on_err * mad_elm2.align_errors.dy, rtol=0, atol=1e-13)

        for ii in range(line['elm2'].order + 1):
            ref = 0
            if len(mad_elm2.knl) > ii:
                ref += mad_elm2.knl[ii]
            if opt['enable_errors'] and len(mad_elm2.field_errors.dkn) > ii:
                ref += mad_elm2.field_errors.dkn[ii]
            xo.assert_allclose(line['elm2'].knl[ii], ref, rtol=0, atol=1e-13)

        for ii in range(line['elm2'].order + 1):
            ref = 0
            if len(mad_elm2.ksl) > ii:
                ref += mad_elm2.ksl[ii]
            if opt['enable_errors'] and len(mad_elm2.field_errors.dks) > ii:
                ref += mad_elm2.field_errors.dks[ii]
            xo.assert_allclose(line['elm2'].ksl[ii], ref, rtol=0, atol=1e-13)

    for opt in gen_options(opt2):
        ml = MadLoader(mad.sequence.seq, **opt)
        line = list(ml.iter_elements())


def test_thick_errors():
    mad = Madx()

    mad.input('''
        k0 = 0.1;
        k1 = 0.2;
        k2 = 0.3;
        k3 = 0.4;
        ks = 0.5;
        l = 5.;
        radius = 0.3;

        dknr0 = 0.02;
        dknr1 = -0.04;
        dknr2 = 0.06;
        dkns3 = 0.2;
        dksr0 = 0.03;
        dksr1 = -0.05;
        dksr2 = 0.07;
        dksr3 = 0.3;

        bend: sbend, k0 := k0, l = 5;
        quad: quadrupole, k1 := k1, l = 5.;
        sext: sextupole, k2 := k2, l = 5.;
        octu: octupole, k3 := k3, l = 5.;
        sole: solenoid, ks := ks, l = 5.;

        seq: sequence, l = 50;
            quad1: quad, at = 5;
            quad2: quad, at = 10;
            quad3: quad, at = 20;
            bend1: bend, at = 25;
            sext1: sext, at = 30;
            octu1: octu, at = 35;
            sole1: sole, at = 40;
        endsequence;

        beam;

        use, sequence = seq;

        select, pattern = quad, flag = error;
        select, pattern = bend, flag = error;
        select, pattern = sext, flag = error;
        select, pattern = octu, flag = error;
        select, pattern = sole, flag = error;
        efcomp, radius = radius,
            dknr = {dknr0, dknr1, dknr2, dknr3},
            dksr = {dksr0, dksr1, dksr2, dksr3};
    ''')

    line = xt.Line.from_madx_sequence(
        sequence=mad.sequence.seq,
        enable_field_errors=True,
        allow_thick=True,
    )

    def error(name, which):
        field_errors = mad.sequence.seq.expanded_elements[name].field_errors
        dk = getattr(field_errors, which)
        return dk[:6]

    xo.assert_allclose(line['bend1'].knl, error('bend1', 'dkn'), atol=0, rtol=1e-15)
    xo.assert_allclose(line['bend1'].ksl, error('bend1', 'dks'), atol=0, rtol=1e-15)

    xo.assert_allclose(line['quad1'].knl, error('quad1', 'dkn'), atol=0, rtol=1e-15)
    xo.assert_allclose(line['quad2'].knl, error('quad2', 'dkn'), atol=0, rtol=1e-15)
    xo.assert_allclose(line['quad3'].knl, error('quad3', 'dkn'), atol=0, rtol=1e-15)

    xo.assert_allclose(line['quad1'].ksl, error('quad1', 'dks'), atol=0, rtol=1e-15)
    xo.assert_allclose(line['quad2'].ksl, error('quad2', 'dks'), atol=0, rtol=1e-15)
    xo.assert_allclose(line['quad3'].ksl, error('quad3', 'dks'), atol=0, rtol=1e-15)

    xo.assert_allclose(line['sext1'].knl, error('sext1', 'dkn'), atol=0, rtol=1e-15)
    xo.assert_allclose(line['sext1'].ksl, error('sext1', 'dks'), atol=0, rtol=1e-15)

    xo.assert_allclose(line['octu1'].knl, error('octu1', 'dkn'), atol=0, rtol=1e-15)
    xo.assert_allclose(line['octu1'].ksl, error('octu1', 'dks'), atol=0, rtol=1e-15)

    xo.assert_allclose(line['sole1'].knl, error('sole1', 'dkn'), atol=0, rtol=1e-15)
    xo.assert_allclose(line['sole1'].ksl, error('sole1', 'dks'), atol=0, rtol=1e-15)


def test_matrix():
    mad = Madx(stdout=False)

    mad.input("""
    a11=1;
    mat: matrix,l=0.4,
          rm11:=a11,rm12=2,rm21=3,rm22=4,
          kick1=0.1,kick2=0.2,kick3=0.3;

    ss: sequence, l=1; mm: mat, at=0.3; endsequence;

    beam; use, sequence=ss;
    """)

    line = MadLoader(mad.sequence.ss).make_line()
    line = MadLoader(mad.sequence.ss, enable_expressions=True).make_line()
    line.vars['a11'] = 2.0
    assert line['mm'].m1[0, 0] == line.vars['a11']._value

    line.reset_s_at_end_turn

    part = xt.Particles()
    line['mm'].track(part)
    xo.assert_allclose(part.s, 0.4, atol=1e-12, rtol=0)


def test_srotation():
    mad = Madx(stdout=False)

    mad.input("""
    angle=0.2;
    rot: srotation,angle:=angle;

    ss: sequence, l=1; rot: rot, at=0; endsequence;

    beam; use, sequence=ss;
    """)

    line = MadLoader(mad.sequence.ss).make_line()
    line = MadLoader(mad.sequence.ss, enable_expressions=True).make_line()
    assert isinstance(line[1], xt.SRotation)
    line.vars['angle'] = 2.0
    assert line[1].angle == line.vars['angle']._value * 180 / np.pi


def test_thick_kicker_option():
    mad = Madx(stdout=False)

    mad.input("""
    vk: vkicker, l=2, kick=2;
    hk: hkicker, l=2, kick=3;
    ki: kicker, l=2, vkick=4, hkick=5;
    vk_thin: vkicker, lrad=2, kick=6;
    hk_thin: hkicker, lrad=2, kick=7;
    ki_thin: kicker, lrad=2, vkick=8, hkick=9;

    ss: sequence, l = 6;
        vk: vk, at = 1;
        hk: hk, at = 3;
        ki: ki, at = 5;
        vk_thin: vk_thin, at = 6;
        hk_thin: hk_thin, at = 6;
        ki_thin: ki_thin, at = 6;
    endsequence;

    beam; use, sequence=ss;
    """)
    line = MadLoader(mad.sequence.ss, enable_expressions=True, allow_thick=True, enable_thick_kickers=True).make_line()

    _, vk, hk, ki, vk_thin, hk_thin, ki_thin, _ = line.elements

    assert isinstance(vk, xt.Magnet)
    assert isinstance(hk, xt.Magnet)
    assert isinstance(ki, xt.Magnet)
    assert isinstance(vk_thin, xt.Multipole)
    assert isinstance(hk_thin, xt.Multipole)
    assert isinstance(ki_thin, xt.Multipole)

    def assert_integrated_strength_eq(value, expected):
        padded_expected = np.zeros_like(value)
        padded_expected[:len(expected)] = expected
        assert np.all(value == padded_expected)

    assert_integrated_strength_eq(vk.knl, [0])
    assert_integrated_strength_eq(vk.ksl, [2])
    assert vk.length == 2

    assert_integrated_strength_eq(hk.knl, [-3])
    assert_integrated_strength_eq(hk.ksl, [0])
    assert hk.length == 2

    assert_integrated_strength_eq(ki.knl, [-5])
    assert_integrated_strength_eq(ki.ksl, [4])
    assert ki.length == 2

    assert_integrated_strength_eq(vk_thin.knl, [0])
    assert_integrated_strength_eq(vk_thin.ksl, [6])
    assert vk_thin.length == 2

    assert_integrated_strength_eq(hk_thin.knl, [-7])
    assert_integrated_strength_eq(hk_thin.ksl, [0])
    assert hk_thin.length == 2

    assert_integrated_strength_eq(ki_thin.knl, [-9])
    assert_integrated_strength_eq(ki_thin.ksl, [8])
    assert ki_thin.length == 2


def test_xrotation():
    mad = Madx(stdout=False)

    mad.input("""
    angle=0.2;
    rot: xrotation,angle:=angle;

    ss: sequence, l=1; rot: rot, at=0; endsequence;

    beam; use, sequence=ss;
    """)

    line = MadLoader(mad.sequence.ss).make_line()
    line = MadLoader(mad.sequence.ss, enable_expressions=True).make_line()
    assert isinstance(line[1], xt.XRotation)
    line.vars['angle'] = 2.0
    assert line[1].angle == line.vars['angle']._value * 180 / np.pi


def test_yrotation():
    mad = Madx(stdout=False)

    mad.input("""
    angle=0.2;
    rot: yrotation,angle:=angle;

    ss: sequence, l=1; rot: rot, at=0; endsequence;

    beam; use, sequence=ss;
    """)

    line = MadLoader(mad.sequence.ss).make_line()
    line = MadLoader(mad.sequence.ss, enable_expressions=True).make_line()
    assert isinstance(line[1], xt.YRotation)
    line.vars['angle'] = 2.0
    assert line[1].angle == line.vars['angle']._value * 180 / np.pi


def test_mad_elements_import():
    mad = Madx(stdout=False)

    # Element definitions
    mad.input("""

    a = 1.;

    cav0: rfcavity, freq:=a*10, lag:=a*0.5, volt:=a*6;
    cav1: rfcavity, lag:=a*0.5, volt:=a*6, harmon:=a*8;
    wire1: wire, current:=a*5, l:=a*0, l_phy:=a*1, l_int:=a*2, xma:=a*1e-3, yma:=a*2e-3;
    mult0: multipole, knl:={a*1,a*2,a*3}, ksl:={0, a*5,a*6}, lrad:=a*1.1;
    mult1: multipole, knl={1,2,3,0}, ksl:={0,2,3};
    mult2: multipole, knl={1,2,3,0}, ksl={0,2,0,0,0};
    mult3: multipole, knl={1,2,3,0}, ksl:={0,2,0,b,0};
    kick0: kicker, hkick:=a*5, vkick:=a*6, lrad:=a*2.2;
    kick1: tkicker, hkick:=a*7, vkick:=a*8, lrad:=a*2.3;
    kick2: hkicker, kick:=a*3, lrad:=a*2.4;
    kick3: vkicker, kick:=a*4, lrad:=a*2.5;
    dipedge0: dipedge, h:=a*0.1, e1:=a*3, fint:=a*4, hgap:=a*0.02;
    rfm0: rfmultipole, volt:=a*2, lag:=a*0.5, freq:=a*100.,
                knl:={a*2,a*3}, ksl:={0,a*5},
                pnl:={a*0.3, a*0.4}, psl:={a*0.5, a*0.6};
    crab0: crabcavity, volt:=a*2, lag:=a*0.5, freq:=a*100.;
    crab1: crabcavity, volt:=a*2, lag:=a*0.5, freq:=a*100., tilt:=a*pi/2;
    oct0: marker, apertype=octagon, aperture:={a * 3, a * 6, a * pi/6, a * pi/3};
    """)

    matrix_m0 = np.random.randn(6) * 1E-6
    matrix_m1 = np.reshape(np.random.randn(36), (6, 6))
    mad.input(f"mat:matrix,l:=0.003*a,"
              f"kick1:={matrix_m0[0]}*a,kick2:={matrix_m0[1]}*a,"
              f"kick3:={matrix_m0[2]}*a,kick4:={matrix_m0[3]}*a,"
              f"kick5:={matrix_m0[4]}*a,kick6:={matrix_m0[5]}*a,"
              f"rm11:={matrix_m1[0, 0]}*a,rm12:={matrix_m1[0, 1]}*a,"
              f"rm13:={matrix_m1[0, 2]}*a,rm14:={matrix_m1[0, 3]}*a,"
              f"rm15:={matrix_m1[0, 4]}*a,rm16:={matrix_m1[0, 5]}*a,"
              f"rm21:={matrix_m1[1, 0]}*a,rm22:={matrix_m1[1, 1]}*a,"
              f"rm23:={matrix_m1[1, 2]}*a,rm24:={matrix_m1[1, 3]}*a,"
              f"rm25:={matrix_m1[1, 4]}*a,rm26:={matrix_m1[1, 5]}*a,"
              f"rm31:={matrix_m1[2, 0]}*a,rm32:={matrix_m1[2, 1]}*a,"
              f"rm33:={matrix_m1[2, 2]}*a,rm34:={matrix_m1[2, 3]}*a,"
              f"rm35:={matrix_m1[2, 4]}*a,rm36:={matrix_m1[2, 5]}*a,"
              f"rm41:={matrix_m1[3, 0]}*a,rm42:={matrix_m1[3, 1]}*a,"
              f"rm43:={matrix_m1[3, 2]}*a,rm44:={matrix_m1[3, 3]}*a,"
              f"rm45:={matrix_m1[3, 4]}*a,rm46:={matrix_m1[3, 5]}*a,"
              f"rm51:={matrix_m1[4, 0]}*a,rm52:={matrix_m1[4, 1]}*a,"
              f"rm53:={matrix_m1[4, 2]}*a,rm54:={matrix_m1[4, 3]}*a,"
              f"rm55:={matrix_m1[4, 4]}*a,rm56:={matrix_m1[4, 5]}*a,"
              f"rm61:={matrix_m1[5, 0]}*a,rm62:={matrix_m1[5, 1]}*a,"
              f"rm63:={matrix_m1[5, 2]}*a,rm64:={matrix_m1[5, 3]}*a,"
              f"rm65:={matrix_m1[5, 4]}*a,rm66={matrix_m1[5, 5]}*a;")

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
    oct: oct0, at=3;
    endsequence;
    """
              )

    # Beam
    mad.input(f"""
    beam, particle=ion, gamma=1.05, mass={xt.PROTON_MASS_EV / 1e9}, sequence=testseq;
    """)

    mad.use('testseq')

    seq = mad.sequence['testseq']

    for test_expressions in [True, False]:
        line = xt.Line.from_madx_sequence(sequence=seq,
                                          deferred_expressions=test_expressions,
                                          install_apertures=True)
        line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, gamma0=1.05)

        line = xt.Line.from_dict(line.to_dict())  # This calls the to_dict method fot all
        # elements

        assert len(line.element_names) == len(line.element_dict.keys())
        assert line.get_length() == 10

        assert isinstance(line['m0'], xt.Multipole)
        assert line.get_s_position('m0') == 0.1
        assert np.all(line['m0'].knl == np.array([1, 2, 3]))
        assert np.all(line['m0'].ksl == np.array([0, 5, 6]))
        assert line['m0'].hxl == 1
        assert line['m0'].length == 1.1
        assert len(line['m1'].knl) == 3
        assert len(line['m1'].ksl) == 3
        assert len(line['m2'].knl) == 3
        assert len(line['m2'].ksl) == 3
        if test_expressions:
            assert len(line['m3'].knl) == 4
            assert len(line['m3'].ksl) == 4
        else:
            assert len(line['m3'].knl) == 3
            assert len(line['m3'].ksl) == 3

        assert isinstance(line['k0'], xt.Multipole)
        assert line.get_s_position('k0') == 0.3
        assert np.all(line['k0'].knl == np.array([-5]))
        assert np.all(line['k0'].ksl == np.array([6]))
        assert line['k0'].hxl == 0
        assert line['k0'].length == 2.2

        assert isinstance(line['k1'], xt.Multipole)
        assert line.get_s_position('k1') == 0.33
        assert np.all(line['k1'].knl == np.array([-7]))
        assert np.all(line['k1'].ksl == np.array([8]))
        assert line['k1'].hxl == 0
        assert line['k1'].length == 2.3

        assert isinstance(line['k2'], xt.Multipole)
        assert line.get_s_position('k2') == 0.34
        assert np.all(line['k2'].knl == np.array([-3]))
        assert np.all(line['k2'].ksl == np.array([0]))
        assert line['k2'].hxl == 0
        assert line['k2'].length == 2.4

        assert isinstance(line['k3'], xt.Multipole)
        assert line.get_s_position('k3') == 0.35
        assert np.all(line['k3'].knl == np.array([0]))
        assert np.all(line['k3'].ksl == np.array([4]))
        assert line['k3'].hxl == 0
        assert line['k3'].length == 2.5

        assert isinstance(line['c0'], xt.Cavity)
        assert line.get_s_position('c0') == 0.2
        assert line['c0'].frequency == 10e6
        assert line['c0'].lag == 180
        assert line['c0'].voltage == 6e6

        assert isinstance(line['c1'], xt.Cavity)
        assert line.get_s_position('c1') == 0.2
        xo.assert_allclose(line['c1'].frequency, clight * line.particle_ref.beta0 / 10. * 8,
                           rtol=0, atol=1e-7)
        assert line['c1'].lag == 180
        assert line['c1'].voltage == 6e6

        assert isinstance(line['de0'], xt.DipoleEdge)
        assert line.get_s_position('de0') == 0.38
        assert line['de0'].k == 0.1
        assert line['de0'].e1 == 3
        assert line['de0'].fint == 4
        assert line['de0'].hgap == 0.02

        assert isinstance(line['r0'], xt.RFMultipole)
        assert line.get_s_position('r0') == 0.4
        assert np.all(line['r0'].knl == np.array([2, 3]))
        assert np.all(line['r0'].ksl == np.array([0, 5]))
        assert np.all(line['r0'].pn == np.array([0.3 * 360, 0.4 * 360]))
        assert np.all(line['r0'].ps == np.array([0.5 * 360, 0.6 * 360]))
        assert line['r0'].voltage == 2e6
        assert line['r0'].order == 1
        assert line['r0'].frequency == 100e6
        assert line['r0'].lag == 180

        assert isinstance(line['cb0'], xt.RFMultipole)
        assert line.get_s_position('cb0') == 0.41
        assert len(line['cb0'].knl) == 1
        assert len(line['cb0'].ksl) == 1
        xo.assert_allclose(line['cb0'].knl[0], 2 * 1e6 / line.particle_ref.p0c[0],
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
        xo.assert_allclose(line['cb1'].ksl[0], -2 * 1e6 / line.particle_ref.p0c[0],
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

        assert isinstance(line['mat0'], xt.FirstOrderTaylorMap)
        assert line.get_s_position('mat0') == 2
        xo.assert_allclose(line['mat0'].m0, matrix_m0, rtol=0.0, atol=1E-12)
        xo.assert_allclose(line['mat0'].m1, matrix_m1, rtol=0.0, atol=1E-12)

        assert isinstance(line['oct_aper'], xt.LimitPolygon)
        assert line.get_s_position('oct_aper') == 3
        x_1, x_2, y_1, y_2 = 3, 2 * np.sqrt(3), np.sqrt(3), 6
        expected_x_vertices = [x_1, x_2, -x_2, -x_1, -x_1, -x_2, x_2, x_1]
        expected_y_vertices = [y_1, y_2, y_2, y_1, -y_1, -y_2, -y_2, -y_1]
        xo.assert_allclose(line['oct_aper'].x_vertices, expected_x_vertices, atol=1e-14)
        xo.assert_allclose(line['oct_aper'].y_vertices, expected_y_vertices, atol=1e-14)


def test_selective_expr_import_and_replace_in_expr():
    # Load line with knobs on correctors only
    from cpymad.madx import Madx
    mad = Madx(stdout=False)
    mad.call(str(test_data_folder /
                 'hllhc14_no_errors_with_coupling_knobs/lhcb1_seq.madx'))
    mad.use(sequence='lhcb1')
    line = xt.Line.from_madx_sequence(mad.sequence.lhcb1,
                                      deferred_expressions=True,
                                      replace_in_expr={'bv_aux': 'bv_aux_lhcb1'},
                                      expressions_for_element_types=('kicker', 'hkicker', 'vkicker'))

    assert len(line.vars['bv_aux_lhcb1']._find_dependant_targets()) > 1
    assert 'bv_aux' not in line.vars

    assert line.element_refs['mqxfa.b3r5..1'].knl[1]._expr is None  # multipole
    assert line.element_refs['mcbxfbv.b2r1'].ksl[0]._expr is not None  # kicker


def test_load_madx_optics_file():
    collider = xt.load(
        test_data_folder / 'hllhc15_thick/hllhc15_collider_thick.json')
    collider.build_trackers()

    # Check varval behaviour
    collider.vars['on_x1'] = 40
    assert collider.varval['on_x1'] == 40
    assert collider.lhcb1.vars['on_x1']._value == 40
    assert collider.lhcb2.vars['on_x1']._value == 40
    assert collider.lhcb1.varval['on_x1'] == 40
    assert collider.lhcb2.varval['on_x1'] == 40

    collider.varval['on_x1'] = 50
    assert collider.vars['on_x1']._value == 50
    assert collider.lhcb1.vars['on_x1']._value == 50
    assert collider.lhcb2.vars['on_x1']._value == 50
    assert collider.lhcb1.varval['on_x1'] == 50
    assert collider.lhcb2.varval['on_x1'] == 50

    collider.lhcb1.varval['on_x1'] = 60
    assert collider.vars['on_x1']._value == 60
    assert collider.lhcb1.vars['on_x1']._value == 60
    assert collider.lhcb2.vars['on_x1']._value == 60
    assert collider.lhcb1.varval['on_x1'] == 60
    assert collider.lhcb2.varval['on_x1'] == 60

    collider.lhcb2.varval['on_x1'] = 70
    assert collider.vars['on_x1']._value == 70
    assert collider.lhcb1.vars['on_x1']._value == 70
    assert collider.lhcb2.vars['on_x1']._value == 70
    assert collider.lhcb1.varval['on_x1'] == 70
    assert collider.lhcb2.varval['on_x1'] == 70

    collider.vars['on_disp'] = 0  # more precise angle
    xo.assert_allclose(collider.twiss().lhcb1['px', 'ip1'], 70e-6, atol=1e-8, rtol=0)
    xo.assert_allclose(collider.twiss().lhcb2['px', 'ip1'], -70e-6, atol=1e-8, rtol=0)

    collider.vars.load_madx_optics_file(
        test_data_folder / 'hllhc15_thick/opt_round_300_1500.madx')

    collider._xdeps_manager.verify()

    tw = collider.twiss()
    xo.assert_allclose(tw.lhcb1.qx, 62.31000000, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb1.qy, 60.32000000, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb2.qx, 62.31000000, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb2.qy, 60.32000000, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb1['betx', 'ip1'], 0.30, atol=0, rtol=1e-4)
    xo.assert_allclose(tw.lhcb1['bety', 'ip1'], 0.30, atol=0, rtol=1e-4)
    xo.assert_allclose(tw.lhcb2['betx', 'ip1'], 0.30, atol=0, rtol=1e-4)
    xo.assert_allclose(tw.lhcb2['bety', 'ip1'], 0.30, atol=0, rtol=1e-4)

    # Check a knob
    collider.vars['on_x1'] = 30
    collider.vars['on_disp'] = 0
    tw = collider.twiss()
    xo.assert_allclose(tw.lhcb1['px', 'ip1'], 30e-6, atol=1e-9, rtol=0)
    xo.assert_allclose(tw.lhcb2['px', 'ip1'], -30e-6, atol=1e-9, rtol=0)

    collider.vars.load_madx_optics_file(
        test_data_folder / 'hllhc15_thick/opt_round_150_1500.madx')

    tw = collider.twiss()
    xo.assert_allclose(tw.lhcb1.qx, 62.31000000, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb1.qy, 60.32000000, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb2.qx, 62.31000000, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb2.qy, 60.32000000, atol=1e-6, rtol=0)
    xo.assert_allclose(tw.lhcb1['betx', 'ip1'], 0.15, atol=0, rtol=1e-6)
    xo.assert_allclose(tw.lhcb1['bety', 'ip1'], 0.15, atol=0, rtol=1e-6)
    xo.assert_allclose(tw.lhcb2['betx', 'ip1'], 0.15, atol=0, rtol=1e-6)
    xo.assert_allclose(tw.lhcb2['bety', 'ip1'], 0.15, atol=0, rtol=1e-6)

    # Check a knob
    collider.vars['on_x1'] = 10
    collider.vars['on_disp'] = 0
    tw = collider.twiss()
    xo.assert_allclose(tw.lhcb1['px', 'ip1'], 10e-6, atol=1e-9, rtol=0)
    xo.assert_allclose(tw.lhcb2['px', 'ip1'], -10e-6, atol=1e-9, rtol=0)

    # Try unregister/register

    collider.vars['ox_x1h'] = 20
    tw = collider.twiss()
    xo.assert_allclose(tw.lhcb1['px', 'ip1'], 10e-6, atol=1e-9, rtol=0)
    xo.assert_allclose(tw.lhcb2['px', 'ip1'], -10e-6, atol=1e-9, rtol=0)

    collider.vars['on_x8h'] = collider.vars['on_x2']
    collider.vars['on_x2'] = 25
    tw = collider.twiss()

    xo.assert_allclose(tw.lhcb1['px', 'ip8'], 25e-6, atol=1e-9, rtol=0)
    xo.assert_allclose(tw.lhcb2['px', 'ip8'], -25e-6, atol=1e-9, rtol=0)
    xo.assert_allclose(tw.lhcb1['py', 'ip8'], 0, atol=1e-9, rtol=0)
    xo.assert_allclose(tw.lhcb2['py', 'ip8'], 0, atol=1e-9, rtol=0)
    xo.assert_allclose(tw.lhcb1['py', 'ip2'], 25e-6, atol=1e-9, rtol=0)
    xo.assert_allclose(tw.lhcb2['py', 'ip2'], -25e-6, atol=1e-9, rtol=0)
    xo.assert_allclose(tw.lhcb1['px', 'ip2'], 0, atol=1e-9, rtol=0)
    xo.assert_allclose(tw.lhcb2['px', 'ip2'], 0, atol=1e-9, rtol=0)


def test_load_b2_with_bv_minus_one():
    test_data_folder_str = str(test_data_folder)

    mad1 = Madx(stdout=False)
    mad1.call(test_data_folder_str + '/hllhc15_thick/lhc.seq')
    mad1.call(test_data_folder_str + '/hllhc15_thick/hllhc_sequence.madx')
    mad1.input('beam, sequence=lhcb1, particle=proton, energy=7000;')
    mad1.use('lhcb1')
    mad1.input('beam, sequence=lhcb2, particle=proton, energy=7000, bv=-1;')
    mad1.use('lhcb2')
    mad1.call(test_data_folder_str + '/hllhc15_thick/opt_round_150_1500.madx')
    mad1.twiss()

    mad4 = Madx(stdout=False)
    mad4.input('mylhcbeam=4')
    mad4.call(test_data_folder_str + '/hllhc15_thick/lhcb4.seq')
    mad4.call(test_data_folder_str + '/hllhc15_thick/hllhc_sequence.madx')
    mad4.input('beam, sequence=lhcb2, particle=proton, energy=7000;')
    mad4.use('lhcb2')
    mad4.call(test_data_folder_str + '/hllhc15_thick/opt_round_150_1500.madx')
    mad4.twiss()

    for mad in [mad1, mad4]:
        mad.globals['vrf400'] = 16  # Check voltage expressions
        mad.globals['lagrf400.b2'] = 0.02  # Check lag expressions
        mad.globals['on_x1'] = 100  # Check kicker expressions
        mad.globals['on_sep2'] = 2  # Check kicker expressions
        mad.globals['on_x5'] = 123  # Check kicker expressions
        mad.globals['kqtf.b2'] = 1e-5  # Check quad expressions
        mad.globals['ksf.b2'] = 1e-3  # Check sext expressions
        mad.globals['kqs.l3b2'] = 1e-4  # Check skew expressions
        mad.globals['kss.a45b2'] = 1e-4  # Check skew sext expressions
        mad.globals['kof.a34b2'] = 3  # Check oct expressions
        mad.globals['on_crab1'] = -190  # Check cavity expressions
        mad.globals['on_crab5'] = -130  # Check cavity expressions
        mad.globals['on_sol_atlas'] = 1  # Check solenoid expressions
        mad.globals['kcdx3.r1'] = 1e-4  # Check thin decapole expressions
        mad.globals['kcdsx3.r1'] = 1e-4  # Check thin skew decapole expressions
        mad.globals['kctx3.l1'] = 1e-5  # Check thin dodecapole expressions
        mad.globals['kctsx3.r1'] = 1e-5  # Check thin skew dodecapole expressions

    line2 = xt.Line.from_madx_sequence(mad1.sequence.lhcb2,
                                       allow_thick=True,
                                       deferred_expressions=True,
                                       replace_in_expr={'bv_aux': 'bvaux_b2'})
    line2.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, p0c=7000e9)

    line4 = xt.Line.from_madx_sequence(mad4.sequence.lhcb2,
                                       allow_thick=True,
                                       deferred_expressions=True,
                                       replace_in_expr={'bv_aux': 'bvaux_b2'})
    line4.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, p0c=7000e9)

    # Bend done

    # Quadrupole
    xo.assert_allclose(line2['mq.27l2.b2'].k1, line4['mq.27l2.b2'].k1, rtol=0, atol=1e-12)
    xo.assert_allclose(line2['mqs.27l3.b2'].k1s, line4['mqs.27l3.b2'].k1s, rtol=0, atol=1e-12)

    tt2 = line2.get_table()
    tt4 = line4.get_table()

    tt2nodr = tt2.rows[tt2.element_type != 'Drift']
    tt4nodr = tt4.rows[tt4.element_type != 'Drift']

    # Check s
    l2names = list(tt2nodr.name)
    l4names = list(tt4nodr.name)

    l2names.remove('lhcb2$start')
    l2names.remove('lhcb2$end')
    l4names.remove('lhcb2$start')
    l4names.remove('lhcb2$end')

    assert set(l2names) == set(l4names)

    xo.assert_allclose(
        tt2nodr.rows[l2names].s, tt4nodr.rows[l2names].s, rtol=0, atol=1e-8)

    for nn in l2names:
        print(nn + '              ', end='\r', flush=True)
        if nn == '_end_point':
            continue
        e2 = line2[nn]
        e4 = line4[nn]
        d2 = e2.to_dict()
        d4 = e4.to_dict()
        for kk in d2.keys():
            if kk in ('__class__', 'model', 'side'):
                assert d2[kk] == d4[kk]
                continue
            xo.assert_allclose(d2[kk], d4[kk], rtol=1e-10, atol=1e-16)
