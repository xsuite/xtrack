from distutils.errors import PreprocessError
import itertools
import numpy as np

import xtrack.mad_loader
from xtrack import MadLoader
import xtrack as xt

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


def test_multipole():

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
            mad_elm2 = mad.sequence['seq'].expanded_elements['elm2']

            assert np.isclose(line['elm2_aper_tilt_entry'].angle,
                              mad_elm2.aper_tilt/np.pi*180,
                              rtol=0, atol=1e-13)
            assert np.isclose(line['elm2_aper_tilt_exit'].angle,
                              -mad_elm2.aper_tilt/np.pi*180,
                              rtol=0, atol=1e-13)


            assert np.isclose(line['elm2_aper_offset_entry'].dx,
                        mad_elm2.align_errors.arex + mad_elm2.aper_offset[0],
                        rtol=0, atol=1e-13)
            assert np.isclose(line['elm2_aper_offset_entry'].dy,
                        mad_elm2.align_errors.arey + mad_elm2.aper_offset[1],
                        rtol=0, atol=1e-13)
            assert np.isclose(line['elm2_aper_offset_exit'].dx,
                        -(mad_elm2.align_errors.arex + mad_elm2.aper_offset[0]),
                        rtol=0, atol=1e-13)
            assert np.isclose(line['elm2_aper_offset_exit'].dy,
                        -(mad_elm2.align_errors.arey + mad_elm2.aper_offset[1]),
                        rtol=0, atol=1e-13)

            # Add a check on the aperture

            assert np.isclose(line['elm2_tilt_entry'].angle,
                        (mad_elm2.tilt + mad_elm2.align_errors.dpsi)/np.pi*180,
                        rtol=0, atol=1e-13)
            assert np.isclose(line['elm2_tilt_exit'].angle,
                        -(mad_elm2.tilt + mad_elm2.align_errors.dpsi)/np.pi*180,
                        rtol=0, atol=1e-13)

            assert np.isclose(line['elm2_offset_entry'].dx,
                              mad_elm2.align_errors.dx, rtol=0, atol=1e-13)
            assert np.isclose(line['elm2_offset_entry'].dy,
                                mad_elm2.align_errors.dy, rtol=0, atol=1e-13)
            assert np.isclose(line['elm2_offset_exit'].dx,
                                -mad_elm2.align_errors.dx, rtol=0, atol=1e-13)
            assert np.isclose(line['elm2_offset_exit'].dy,
                                -mad_elm2.align_errors.dy, rtol=0, atol=1e-13)


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