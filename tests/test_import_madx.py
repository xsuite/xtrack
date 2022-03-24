from scipy.constants import c as clight
import numpy as np

from cpymad.madx import Madx
import xtrack as xt
import xpart as xp

def test_mad_element_import():

    mad = Madx()

    # Element definitions
    mad.input("""
    cav0: rfcavity, freq=10, lag=0.5, volt=6;
    cav1: rfcavity, lag=0.5, volt=6, harmon=8;
    wire1: wire, current=5, l=0, l_phy=1, l_int=2, xma=1e-3, yma=2e-3;

    mult0: multipole, knl={1,2,3}, ksl={4,5,6}, lrad=1.1;
    kick0: kicker, hkick=5, vkick=6, lrad=2.2;
    kick1: tkicker, hkick=7, vkick=8, lrad=2.3;
    kick2: hkicker, kick=3, lrad=2.4;
    kick3: vkicker, kick=4, lrad=2.5;

    dipedge0: dipedge, h=0.1, e1=3, fint=4, hgap=0.02;

    rfm0: rfmultipole, volt=2, lag=0.5, freq=100.,
                knl={2,3}, ksl={4,5},
                pnl={0.3, 0.4}, psl={0.5, 0.6};
    crab0: crabcavity, volt=2, lag=0.5, freq=100.;
    crab1: crabcavity, volt=2, lag=0.5, freq=100., tilt=pi/2;

    """)

    # Sequence
    mad.input("""

    testseq: sequence, l=10;
    m0: mult0 at=0.1;
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

    endsequence;
    """
    )

    # Beam
    mad.input("""
    beam, particle=proton, gamma=1.05, sequence=testseq;
    """)


    mad.use('testseq')

    seq = mad.sequence['testseq']
    line = xt.Line.from_madx_sequence(sequence=seq)
    line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, gamma0=1.05)

    line = xt.Line.from_dict(line.to_dict()) # This calls the to_dict_method fot all
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
    assert line['w'].wire_L_phy == 1
    assert line['w'].wire_L_int == 2
    assert line['w'].wire_xma == 1e-3
    assert line['w'].wire_yma == 2e-3
