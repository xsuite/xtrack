def test_hvkick():
    import xtrack as xt
    from cpymad.madx import Madx

    # Load a very simple sequence from MAD-X
    mad = Madx()
    mad.input("""
        seq: sequence, l=4;
        b1: sbend, at=0.5, angle=0.2, l=1;
        b2: sbend, at=2.5, angle=0.3, k0=0.15, l=1;
        m1: multipole, at=3, knl={0.1};
        k1: kicker, at=3, hkick=0.3, vkick=0.4;
        endsequence;
        beam;
        use,sequence=seq;
    """)

    line = xt.Line.from_madx_sequence(mad.sequence.seq)
    line.build_tracker()

    print('The line as imported from MAD-X:')
    line.get_table().show()

    # Shift and tilt selected elements
    line['b1'].rot_s_rad = 0.8
    line['b2'].rot_s_rad = -0.8

    tt = line.get_table(attr=True)
    tt.cols['s', 'element_type', 'angle_rad', 'rot_s_rad', 'k0l', 'k0sl', 'hkick', 'vkick']

    assert  tt['hkick','k1']==0.3
    assert  tt['vkick','k1']==0.4
    assert  tt['hkick','m1']==0
    assert  tt['vkick','m1']==0
    assert  tt['hkick','b1']==0
    assert  tt['vkick','b1']==0
    assert  tt['hkick','b2']==0.15
    assert  tt['vkick','b2']==0

