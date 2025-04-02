import math
import numpy as np
from collections import OrderedDict
from pathlib import Path

import pytest
import xobjects as xo
from cpymad.madx import Madx

from xdeps.refs import CompactFormatter
import xtrack as xt
from xtrack import Strategy, Uniform
from xtrack.mad_parser.loader import MadxLoader
from xtrack.mad_parser.parse import MadxOutputType, MadxParser

test_data_folder = (Path(__file__).parent / '../test_data').absolute()


def test_simple_parser():
    sequence = """
    if (version>=50401){option,-rbarc;};  ! to be ignored
    
    third = 1 / 3;
    power = 3^4;
    hello := third * twopi;
    mb.l = 1;
    qd.l = 0.5;
    offset := 0.1;

    mb: sbend, l := mb.l, angle := hello;
    qf: quadrupole, l := 1, k1 := 1;
    qd: quadrupole, l := 1, k1 := -1;
    
    line: sequence, l = 12;
        ip1: marker, at = 0;
        qf1: qf, at := 1 + offset, from = ip1, slot_id = 1;
        mb1: mb, at := 2 + offset, from = ip1, slot_id = 2;
        qd1: qd, at := 3 + offset, from = ip1, slot_id = 3;
    endsequence;

    mb1, k0 := hello, polarity = +1;
    qf1, knl := {0, 0, 0, 0.01, 0};
    qd1, knl := {0, 0, 0, -0.01, 0};
    
    return;  ! should also be ignored
    """

    env = xt.Environment()
    env['twopi'] = 2 * math.pi
    parser = MadxParser(vars=env.vars, functions=env.functions)
    result = parser.parse_string(sequence)

    expected = {
        'elements': {
            'mb': {
                'angle': env.vars['hello'],
                'l': env.vars['mb.l'],
                'parent': 'sbend',
            },
            'qf': {
                'k1': 1.0,
                'l': 1.0,
                'parent': 'quadrupole',
            },
            'qd': {
                'k1': -1.0,
                'l': 1.0,
                'parent': 'quadrupole',
            },
        },
        'lines': {
            'line': {
                'l': 12.0,
                'parent': 'sequence',
                'elements': {
                    'ip1': {
                        'at': 0.0,
                        'parent': 'marker',
                    },
                    'qf1': {
                        'at': 1.0 + env.vars['offset'],
                        'from': 'ip1',
                        'parent': 'qf',
                        'slot_id': 1.0,
                    },
                    'mb1': {
                        'at': 2.0 + env.vars['offset'],
                        'from': 'ip1',
                        'parent': 'mb',
                        'slot_id': 2.0,
                    },
                    'qd1': {
                        'at': 3.0 + env.vars['offset'],
                        'from': 'ip1',
                        'parent': 'qd',
                        'slot_id': 3.0,
                    },
                },
            },
        },
        'parameters': {
            'mb1': {
                'k0': env.vars['hello'],
                'polarity': 1.0,
            },
            'qf1': {
                'knl': [0.0, 0.0, 0.0, 0.01, 0.0],
            },
            'qd1': {
                'knl': [0.0, 0.0, 0.0, -0.01, 0.0],
            },
        },
    }

    def _order_madx_output(item):
        item['elements'] = OrderedDict(item['elements'])
        item['lines'] = OrderedDict(item['lines'])
        for line in item['lines'].values():
            line['elements'] = OrderedDict(line['elements'])
        item['parameters'] = OrderedDict(item['parameters'])

    _order_madx_output(expected)
    _order_madx_output(result)

    assert expected == result


@pytest.mark.parametrize(
    'input,value,expr',
    [
        ('a := 42;', 42, None),
        ('c := 3; d := 4; a := c^d;', 81, '(c ** d)'),
        ('c := 3; d := 4; a = c^d;', 81, None),
    ]
)
def test_parse_simple_expression(input, value, expr):
    env = xt.load_madx_lattice(string=input)
    assert env['a'] == value

    if expr is not None:
        formatter = CompactFormatter(None)
        assert env.get_expr('a')._formatted(formatter) == expr
    else:
        assert env.get_expr('a') is None
        env['c'] = 8
        assert env['a'] == value


@pytest.fixture(scope='module')
def example_sequence(temp_context_default_mod):
    sequence = """
    ll = 36;

    vk: vkicker, l=2, kick=3, tilt=-1;
    hk: hkicker, l=1, kick=6, tilt=-2;
    ki: kicker, l=2, vkick=3, hkick=4, tilt=1;
    tk: tkicker, l=1, vkick=3, hkick=4, tilt=2;
    in: instrument, l=2;
    mo: monitor, l=1;
    pl: placeholder, l=1;
    sb: sbend, l=2, angle=2, tilt=-2, k0=3, k1=1, k2=2, k1s=3, e1=2, e2=1, fint=3, fintx=2, hgap=1, h1=3, h2=2;
    rb: rbend, l=1.5, angle=2, tilt=-2, k0=3, k1=1, k2=2, k1s=3, e1=2, e2=1, fint=3, fintx=2, hgap=1, h1=3, h2=2;
    qu: quadrupole, l=2, k1=3, k1s=4, tilt=2;  ! ignore thick and ktap
    se: sextupole, L=1, K2=2, K2S=3, TILT=2;  ! ignore ktap
    oc: octupole, L=2, K3=3, K3S=2, TILT=2;
    ma: marker;
    rf: rfcavity, L=2, VOLT=1, LAG=2, FREQ=3, HARMON=2, NO_CAVITY_TOTALPATH;  ! ignore N_BESSEL
    mu: multipole, LRAD=1, TILT=2, KNL={3, 4, 5, 6}, KSL={1, 2, 3, 4};
    so: solenoid, l=2, ks=3;  ! ignore ksi
    
    rb_stage1: rbend, l=1;
    rb_stage2: rb_stage1, angle=2;

    ! Not yet implemented
    ! co: collimator, l=2, apertype=ellipse, aperture={0.01,0.005}, aper_offset={x,y}, aper_tol={corner_r,inner_x,inner_y};

    line: sequence, l = ll;
        vk1: vk, at = 1;
        hk1: hk, at = 3;
        ki1: ki, at = 5;
        tk1: tk, at = 7;
        in1: in, at = 9;
        mo1: mo, at = 11;
        pl1: pl, at = 13;
        sb1: sb, at = 15;
        rb1: rb, at = 17;
        qu1: qu, at = 19;
        se1: se, at = 21;
        oc1: oc, at = 23;
        ma1: ma, at = 25;
        rf1: rf, at = 27;
        mu1: mu, at = 29;
        so1: so, at = 31;

        rx1: rb_stage2, at = 33;
        rx2: rb_stage2, at = 35;
    endsequence;
    
    ! exactly the same as above, but to be parsed in reverse order
    line_reversed: sequence, l = ll;
        vk1: vk, at = 1;
        hk1: hk, at = 3;
        ki1: ki, at = 5;
        tk1: tk, at = 7;
        in1: in, at = 9;
        mo1: mo, at = 11;
        pl1: pl, at = 13;
        sb1: sb, at = 15;
        rb1: rb, at = 17;
        qu1: qu, at = 19;
        se1: se, at = 21;
        oc1: oc, at = 23;
        ma1: ma, at = 25;
        rf1: rf, at = 27;
        mu1: mu, at = 29;
        so1: so, at = 31;

        rx1: rb_stage2, at = 33;

    endsequence;

    rx2, angle = 1.5;
    """

    env = xt.load_madx_lattice(string=sequence, reverse_lines=['line_reversed'])

    def make_positions(line):
        tt = line.get_table()
        diffs = np.diff(tt['s'], append=tt['s', '_end_point'])
        s_centre = tt['s'] + 0.5 * diffs
        return dict(zip(tt['name'], s_centre))

    positions = make_positions(env['line'])
    positions_reversed = make_positions(env['line_reversed'])

    return env, positions, positions_reversed


def test_parsed_lines(example_sequence):
    env, _, _ = example_sequence
    assert env['line'].name == 'line'
    assert env['line_reversed'].name == 'line_reversed'


def test_vkick(example_sequence):
    env, positions, _ = example_sequence
    vk1 = env['vk1/line']
    xo.assert_allclose(positions['vk1/line'], 1)
    assert isinstance(vk1, xt.Multipole)
    assert vk1.length == 2
    assert vk1.knl[0] == 0
    assert vk1.ksl[0] == 3
    xo.assert_allclose(vk1.rot_s_rad, -1)


def test_hkick(example_sequence):
    env, positions, _ = example_sequence
    hk1 = env['hk1/line']
    xo.assert_allclose(positions['hk1/line'], 3)
    assert isinstance(hk1, xt.Multipole)
    assert hk1.length == 1
    assert hk1.knl[0] == -6
    assert hk1.ksl[0] == 0
    assert hk1.rot_s_rad == -2


def test_kick(example_sequence):
    env, positions, _ = example_sequence
    ki1 = env['ki1/line']
    xo.assert_allclose(positions['ki1/line'], 5)
    assert isinstance(ki1, xt.Multipole)
    assert ki1.length == 2
    assert ki1.knl[0] == -4
    assert ki1.ksl[0] == 3
    xo.assert_allclose(ki1.rot_s_rad, 1)


def test_tkick(example_sequence):
    env, positions, _ = example_sequence
    tk1 = env['tk1/line']
    xo.assert_allclose(positions['tk1/line'], 7)
    assert isinstance(tk1, xt.Multipole)
    assert tk1.length == 1
    assert tk1.knl[0] == -4
    assert tk1.ksl[0] == 3
    assert tk1.rot_s_rad == 2


def test_instrument(example_sequence):
    env, positions, _ = example_sequence
    in1 = env['in1/line']
    xo.assert_allclose(positions['in1/line'], 9)
    assert isinstance(in1, xt.Drift)
    assert in1.length == 2


def test_monitor(example_sequence):
    env, positions, _ = example_sequence
    mo1 = env['mo1/line']
    xo.assert_allclose(positions['mo1/line'], 11)
    assert isinstance(mo1, xt.Drift)
    assert mo1.length == 1


def test_placeholder(example_sequence):
    env, positions, _ = example_sequence
    pl1 = env['pl1/line']
    xo.assert_allclose(positions['pl1/line'], 13)
    assert isinstance(pl1, xt.Drift)
    assert pl1.length == 1


def test_sbend(example_sequence):
    env, positions, _ = example_sequence
    # sb: sbend, l=2, angle=2, tilt=-2, k0=3, k1=1, k2=2, k1s=3, e1=2, e2=1,
    #   fint=3, fintx=2, hgap=1;  ! thick, ktap, h1, h2 we ignore
    sb1 = env['sb1/line']
    xo.assert_allclose(positions['sb1/line'], 15)
    assert isinstance(sb1, xt.Bend)
    assert sb1.length == 2
    assert sb1.k0 == 3
    assert sb1.h == 2 / 2  # angle / l
    assert sb1.k1 == 1
    assert sb1.knl[0] == 0
    assert sb1.knl[1] == 0
    assert sb1.knl[2] == 2 * 2  # k2 * l
    assert sb1.ksl[0] == 0
    assert sb1.ksl[1] == 3 * 2  # k1s * l
    assert sb1.edge_entry_angle == 2
    assert sb1.edge_exit_angle == 1
    assert sb1.edge_entry_fint == 3
    assert sb1.edge_exit_fint == 2
    assert sb1.edge_entry_hgap == 1
    assert sb1.edge_exit_hgap == 1


def test_rbend(example_sequence):
    env, positions, _ = example_sequence
    # rb: rbend, l=2, angle=1.5, tilt=-2, k0=3, k1=1, k2=2, k1s=3, e1=2, e2=1,
    #   fint=3, fintx=2, hgap=1, h1=3, h2=2;  ! ditto
    rb1 = env['rb1/line']
    xo.assert_allclose(positions['rb1/line'], 17)
    assert isinstance(rb1, xt.RBend)

    angle = 2
    l = 1.5
    R = 0.5 * l / math.sin(0.5 * angle)
    l_curv = R * angle
    h = 1 / R

    assert rb1.length == l_curv
    assert rb1.k0 == 3
    assert rb1.h == h
    assert rb1.k1 == 1
    assert rb1.knl[0] == 0
    assert rb1.knl[1] == 0
    assert rb1.knl[2] == 2 * l  # k2 * l
    assert rb1.ksl[0] == 0
    assert rb1.ksl[1] == 3 * l  # k1s * l
    assert rb1.edge_entry_angle == 2
    assert rb1.edge_exit_angle == 1
    assert rb1.edge_entry_fint == 3
    assert rb1.edge_exit_fint == 2
    assert rb1.edge_entry_hgap == 1
    assert rb1.edge_exit_hgap == 1

def test_rbend_two_step(example_sequence):
    env, positions, _ = example_sequence
    rb1 = env['rx1/line']
    xo.assert_allclose(positions['rx1/line'], 33)
    assert isinstance(rb1, xt.RBend)

    angle = 2
    l = 1
    R = 0.5 * l / math.sin(0.5 * angle)
    l_curv = R * angle
    h = 1 / R

    assert rb1.length == l_curv
    assert rb1.h == h
    assert rb1.edge_entry_angle == 0
    assert rb1.edge_exit_angle == 0
    assert rb1.k0 == h

def test_rbend_set_params_after_lattice(example_sequence):
    env, positions, _ = example_sequence
    rb1 = env['rx2']
    assert positions['rx2'] < 35 # When changing the angle the position changes
                                 # (the magnet shortens)
    assert isinstance(rb1, xt.RBend)

    angle = 1.5
    l = 1
    R = 0.5 * l / math.sin(0.5 * angle)
    l_curv = R * angle
    h = 1 / R

    assert rb1.length == l_curv
    assert rb1.h == h
    assert rb1.edge_entry_angle == 0
    assert rb1.edge_exit_angle == 0
    assert rb1.k0 == h


def test_quadrupole(example_sequence):
    env, positions, _ = example_sequence
    # qu: quadrupole, l=2, k1=3, k1s=4, tilt=2;  ! ignore thick and ktap
    qu1 = env['qu1/line']
    xo.assert_allclose(positions['qu1/line'], 19)
    assert isinstance(qu1, xt.Quadrupole)
    assert qu1.length == 2
    assert qu1.k1 == 3
    assert qu1.k1s == 4
    assert qu1.rot_s_rad == 2


def test_sextupole(example_sequence):
    env, positions, _ = example_sequence
    # se: sextupole, L=1, K2=2, K2S=3, TILT=2;  ! ignore ktap
    se1 = env['se1/line']
    xo.assert_allclose(positions['se1/line'], 21)
    assert isinstance(se1, xt.Sextupole)
    assert se1.length == 1
    assert se1.k2 == 2
    assert se1.k2s == 3
    assert se1.rot_s_rad == 2


def test_octupole(example_sequence):
    env, positions, _ = example_sequence
    # oc: octupole, L=2, K3=3, K3S=2, TILT=2;
    oc1 = env['oc1/line']
    xo.assert_allclose(positions['oc1/line'], 23)
    assert isinstance(oc1, xt.Octupole)
    assert oc1.length == 2
    assert oc1.k3 == 3
    assert oc1.k3s == 2
    assert oc1.rot_s_rad == 2


def test_marker(example_sequence):
    env, positions, _ = example_sequence
    # ma: marker;
    ma1 = env['ma1']
    xo.assert_allclose(positions['ma1'], 25)
    assert isinstance(ma1, xt.Marker)


def test_rfcavity(example_sequence):
    env, positions, _ = example_sequence
    # rf: rfcavity, L=2, VOLT=1, LAG=2, FREQ=3, HARMON=2;  ! ignore N_BESSEL, NO_CAVITY_TOTALPATH
    rf1 = env['rf1/line']
    xo.assert_allclose(positions['rf1/line'], 27)
    assert isinstance(rf1, xt.Cavity)
    assert rf1.voltage == 1e6
    assert rf1.lag == 2 * 360
    assert rf1.frequency == 3e6


def test_multipole(example_sequence):
    env, positions, _ = example_sequence
    # mu: multipole, LRAD=1, TILT=2, KNL={3, 4, 5}, KSL={1, 2, 3};
    mu1 = env['mu1/line']
    xo.assert_allclose(positions['mu1/line'], 29)
    assert isinstance(mu1, xt.Multipole)
    assert mu1.length == 1
    assert mu1.knl[0] == 3
    assert mu1.knl[1] == 4
    assert mu1.knl[2] == 5
    assert mu1.knl[3] == 6
    assert mu1.ksl[0] == 1
    assert mu1.ksl[1] == 2
    assert mu1.ksl[2] == 3
    assert mu1.ksl[3] == 4
    assert mu1.rot_s_rad == 2


def test_solenoid(example_sequence):
    env, positions, _ = example_sequence
    # so: solenoid, l=2, ks=3;  ! ignore ksi
    so1 = env['so1/line']
    xo.assert_allclose(positions['so1/line'], 31)
    assert isinstance(so1, xt.Solenoid)
    assert so1.length == 2
    assert so1.ks == 3


def test_reversed_vkick(example_sequence):
    env, _, positions_reversed = example_sequence
    ivk1 = env['vk1/line_reversed']
    xo.assert_allclose(positions_reversed['vk1/line_reversed'], 36 - 1)
    assert isinstance(ivk1, xt.Multipole)
    assert ivk1.length == 2
    assert ivk1.knl[0] == 0
    assert ivk1.ksl[0] == -3
    xo.assert_allclose(ivk1.rot_s_rad,  1)


def test_reversed_hkick(example_sequence):
    env, _, positions_reversed = example_sequence
    hk1 = env['hk1/line_reversed']
    xo.assert_allclose(positions_reversed['hk1/line_reversed'], 36 - 3)
    assert isinstance(hk1, xt.Multipole)
    assert hk1.length == 1
    assert hk1.knl[0] == -6
    assert hk1.ksl[0] == 0
    assert hk1.rot_s_rad == 2


def test_reversed_kick(example_sequence):
    env, _, positions_reversed = example_sequence
    ki1 = env['ki1/line_reversed']
    xo.assert_allclose(positions_reversed['ki1/line_reversed'], 36 - 5)
    assert isinstance(ki1, xt.Multipole)
    assert ki1.length == 2
    assert ki1.knl[0] == -4
    assert ki1.ksl[0] == -3
    xo.assert_allclose(ki1.rot_s_rad, -1)


def test_reversed_tkick(example_sequence):
    env, _, positions_reversed = example_sequence
    tk1 = env['tk1/line_reversed']
    xo.assert_allclose(positions_reversed['tk1/line_reversed'], 36 - 7)
    assert isinstance(tk1, xt.Multipole)
    assert tk1.length == 1
    assert tk1.knl[0] == -4
    assert tk1.ksl[0] == -3
    assert tk1.rot_s_rad == -2


def test_reversed_instrument(example_sequence):
    env, _, positions_reversed = example_sequence
    in1 = env['in1/line_reversed']
    xo.assert_allclose(positions_reversed['in1/line_reversed'], 36 - 9)
    assert isinstance(in1, xt.Drift)
    assert in1.length == 2


def test_reversed_monitor(example_sequence):
    env, _, positions_reversed = example_sequence
    mo1 = env['mo1/line_reversed']
    xo.assert_allclose(positions_reversed['mo1/line_reversed'], 36 - 11)
    assert isinstance(mo1, xt.Drift)
    assert mo1.length == 1


def test_reversed_placeholder(example_sequence):
    env, _, positions_reversed = example_sequence
    pl1 = env['pl1/line_reversed']
    xo.assert_allclose(positions_reversed['pl1/line_reversed'], 36 - 13)
    assert isinstance(pl1, xt.Drift)
    assert pl1.length == 1


def test_reversed_sbend(example_sequence):
    env, _, positions_reversed = example_sequence
    # sb: sbend, l=2, angle=2, tilt=-2, k0=3, k1=1, k2=2, k1s=3, e1=2, e2=1,
    #   fint=3, fintx=2, hgap=1;  ! thick, ktap, h1, h2 we ignore
    sb1 = env['sb1/line_reversed']
    xo.assert_allclose(positions_reversed['sb1/line_reversed'], 36 - 15)
    assert isinstance(sb1, xt.Bend)
    assert sb1.length == 2
    assert sb1.k0 == 3
    assert sb1.h == 2 / 2  # angle / l
    assert sb1.k1 == -1
    assert sb1.knl[0] == 0
    assert sb1.knl[1] == 0
    assert sb1.knl[2] == 2 * 2  # k2 * l
    assert sb1.ksl[0] == 0
    assert sb1.ksl[1] == 3 * 2  # k1s * l
    assert sb1.edge_entry_angle == 1
    assert sb1.edge_exit_angle == 2
    assert sb1.edge_entry_fint == 2
    assert sb1.edge_exit_fint == 3
    assert sb1.edge_entry_hgap == 1
    assert sb1.edge_exit_hgap == 1


def test_reversed_rbend(example_sequence):
    env, _, positions_reversed = example_sequence    # rb: rbend, l=2, angle=1.5, tilt=-2, k0=3, k1=1, k2=2, k1s=3, e1=2, e2=1,
    #   fint=3, fintx=2, hgap=1, h1=3, h2=2;  ! ditto
    rb1 = env['rb1/line_reversed']
    xo.assert_allclose(positions_reversed['rb1/line_reversed'], 36 - 17)
    assert isinstance(rb1, xt.RBend)

    angle = 2
    l = 1.5
    R = 0.5 * l / math.sin(0.5 * angle)
    l_curv = R * angle
    h = 1 / R

    assert rb1.length == l_curv
    assert rb1.k0 == 3
    assert rb1.h == h
    assert rb1.k1 == -1
    assert rb1.knl[0] == 0
    assert rb1.knl[1] == 0
    assert rb1.knl[2] == 2 * l  # k2 * l
    assert rb1.ksl[0] == 0
    assert rb1.ksl[1] == 3 * l  # k1s * l
    assert rb1.edge_entry_angle == 1
    assert rb1.edge_exit_angle == 2
    assert rb1.edge_entry_fint == 2
    assert rb1.edge_exit_fint == 3
    assert rb1.edge_entry_hgap == 1
    assert rb1.edge_exit_hgap == 1


def test_reversed_quadrupole(example_sequence):
    env, _, positions_reversed = example_sequence
    # qu: quadrupole, l=2, k1=3, k1s=4, tilt=2;  ! ignore thick and ktap
    qu1 = env['qu1/line_reversed']
    xo.assert_allclose(positions_reversed['qu1/line_reversed'], 36 - 19)
    assert isinstance(qu1, xt.Quadrupole)
    assert qu1.length == 2
    assert qu1.k1 == -3
    assert qu1.k1s == 4
    assert qu1.rot_s_rad == -2


def test_reversed_sextupole(example_sequence):
    env, _, positions_reversed = example_sequence
    # se: sextupole, L=1, K2=2, K2S=3, TILT=2;  ! ignore ktap
    se1 = env['se1/line_reversed']
    xo.assert_allclose(positions_reversed['se1/line_reversed'], 36 - 21)
    assert isinstance(se1, xt.Sextupole)
    assert se1.length == 1
    assert se1.k2 == 2
    assert se1.k2s == -3
    assert se1.rot_s_rad == -2


def test_reversed_octupole(example_sequence):
    env, _, positions_reversed = example_sequence
    # oc: octupole, L=2, K3=3, K3S=2, TILT=2;
    oc1 = env['oc1/line_reversed']
    xo.assert_allclose(positions_reversed['oc1/line_reversed'], 36 - 23)
    assert isinstance(oc1, xt.Octupole)
    assert oc1.length == 2
    assert oc1.k3 == -3
    assert oc1.k3s == 2
    assert oc1.rot_s_rad == -2


def test_reversed_marker(example_sequence):
    env, _, positions_reversed = example_sequence    # ma: marker;
    ma1 = env['ma1']
    xo.assert_allclose(positions_reversed['ma1'], 36 - 25)
    assert isinstance(ma1, xt.Marker)


def test_reversed_rfcavity(example_sequence):
    env, _, positions_reversed = example_sequence    # ma: marker;    # rf: rfcavity, L=2, VOLT=1, LAG=2, FREQ=3, HARMON=2;  ! ignore N_BESSEL, NO_CAVITY_TOTALPATH
    rf1 = env['rf1/line_reversed']
    xo.assert_allclose(positions_reversed['rf1/line_reversed'], 36 - 27)
    assert isinstance(rf1, xt.Cavity)
    assert rf1.voltage == 1e6
    assert rf1.lag == 180 - 2 * 360
    assert rf1.frequency == 3e6


def test_reversed_multipole(example_sequence):
    env, _, positions_reversed = example_sequence
    # mu: multipole, LRAD=1, TILT=2, KNL={3, 4, 5}, KSL={1, 2, 3};
    mu1 = env['mu1/line_reversed']
    xo.assert_allclose(positions_reversed['mu1/line_reversed'], 36 - 29)
    assert isinstance(mu1, xt.Multipole)
    assert mu1.length == 1
    assert mu1.knl[0] == 3
    assert mu1.knl[1] == -4
    assert mu1.knl[2] == 5
    assert mu1.knl[3] == -6
    assert mu1.ksl[0] == -1
    assert mu1.ksl[1] == 2
    assert mu1.ksl[2] == -3
    assert mu1.ksl[3] == 4
    assert mu1.rot_s_rad == -2


def test_reversed_solenoid(example_sequence):
    env, _, positions_reversed = example_sequence
    # so: solenoid, l=2, ks=3;  ! ignore ksi
    so1 = env['so1/line_reversed']
    xo.assert_allclose(positions_reversed['so1/line_reversed'], 36 - 31)
    assert isinstance(so1, xt.Solenoid)
    assert so1.length == 2
    assert so1.ks == -3


def test_load_b2_with_bv_minus_one(tmp_path):
    test_data_folder_str = str(test_data_folder)

    mad = Madx(stdout=False)
    mad.call(test_data_folder_str + '/hllhc15_thick/lhc.seq')
    mad.call(test_data_folder_str + '/hllhc15_thick/hllhc_sequence.madx')
    mad.input('beam, sequence=lhcb1, particle=proton, energy=7000;')
    mad.use('lhcb1')
    mad.input('beam, sequence=lhcb2, particle=proton, energy=7000, bv=-1;')
    mad.use('lhcb2')
    mad.call(test_data_folder_str + '/hllhc15_thick/opt_round_150_1500.madx')
    mad.twiss()

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


    tmp_seq_path = str(tmp_path / 'sequence.seq')
    mad.input('set, format=".20g";')
    mad.save(file=tmp_seq_path)

    line2_ref = xt.Line.from_madx_sequence(
        sequence=mad.sequence.lhcb2,
        allow_thick=True,
        deferred_expressions=True,
        replace_in_expr={'bv_aux': 'bvaux_b2'},
    )
    line2_ref.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=7000e9)

    env = xt.load_madx_lattice(tmp_seq_path, reverse_lines=['lhcb2'])
    line2 = env['lhcb2']

    # Remove apertures, they are not supported in the cpymadloader
    line2.element_names = [
        name for name in line2.element_names
        if not isinstance(line2[name], xt.LimitRectEllipse)
    ]

    # Bend done

    # Quadrupole
    xo.assert_allclose(line2_ref['mq.27l2.b2'].k1, line2['mq.27l2.b2'].k1, rtol=0, atol=1e-12)
    xo.assert_allclose(line2_ref['mqs.27l3.b2'].k1s, line2['mqs.27l3.b2'].k1s, rtol=0, atol=1e-12)

    tt2 = line2_ref.get_table()
    tt4 = line2.get_table()

    tt2nodr = tt2.rows[tt2.element_type != 'Drift']
    tt4nodr = tt4.rows[tt4.element_type != 'Drift']

    # Check s
    l2names = list(tt2nodr.name)
    l4names = list(tt4nodr.name)

    l2names.remove('lhcb2$start')
    l2names.remove('lhcb2$end')

    assert l2names == [nn[:-len('/lhcb2')] if nn.endswith('/lhcb2') else nn for nn in l4names]

    # The tolerance is higher than usual, due to the difference in handling
    # rbend lengths between MAD-X and Xtrack (RBends' straight length is
    # the invariant in Xtrack when changing the angle, which is not the case
    # for MAD-X with `rbarc` == False).
    xo.assert_allclose(tt2nodr.rows[l2names].s, tt4nodr.rows[l4names].s, rtol=0, atol=1.5e-6)

    for nn in l4names:
        if nn == '_end_point':
            continue
        nn_straight = nn[:-len('/lhcb2')] if nn.endswith('/lhcb2') else nn
        e2 = line2_ref[nn_straight]
        e4 = line2[nn]
        d2 = e2.to_dict()
        d4 = e4.to_dict()
        is_rbend = isinstance(e4, xt.RBend)

        for kk in d2.keys():
            if kk in ('__class__', 'model', 'side'):
                assert d2[kk] == d4[kk]
                continue

            if kk in {
                'order',  # Always assumed to be 5, not always the same
                'frequency',  # If not specified, depends on the beam,
                              # so for now we ignore it
            }:
                continue

            if kk in {'knl', 'ksl'}:
                maxlen = max(len(d2[kk]), len(d4[kk]))
                lhs = np.pad(d2[kk], (0, maxlen - len(d2[kk])), mode='constant')
                rhs = np.pad(d4[kk], (0, maxlen - len(d4[kk])), mode='constant')
                xo.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-16)
                continue

            if is_rbend and kk in ('length', 'length_straight'):
                xo.assert_allclose(d2[kk], d4[kk], rtol=1e-7, atol=1e-6)
                continue

            if is_rbend and kk in ('h', 'k0'):
                xo.assert_allclose(d2[kk], d4[kk], rtol=1e-7, atol=5e-10)
                continue

            xo.assert_allclose(d2[kk], d4[kk], rtol=1e-10, atol=1e-16)


def test_line_syntax():
    sequence = """
    el1: drift, l=1;
    el2: drift, l=2;
    el3: drift, l=3;
    
    l1: line = (el1, el2, el3);
    l2: line = (-l1);
    l3: line = (3 * el1, 2 * el2);
    l4: line = (-l3, l3, el3);
    l5: line = (-2 * l4);
    l6: line = (3 * (el1, el2), -(el2, el1));
    """

    loader = MadxLoader()
    loader.load_string(sequence)
    env = loader.env

    l1 = env['l1']
    assert l1.name == 'l1'
    assert l1.element_names == ['el1', 'el2', 'el3']

    l2 = env['l2']
    assert l2.name == 'l2'
    assert l2.element_names == ['el3', 'el2', 'el1']

    l3 = env['l3']
    assert l3.name == 'l3'
    assert l3.element_names == 3 * ['el1'] + 2 * ['el2']

    l4 = env['l4']
    assert l4.name == 'l4'
    assert l4.element_names == 2 * ['el2'] + 6 * ['el1'] + 2 * ['el2'] + ['el3']

    l5 = env['l5']
    assert l5.name == 'l5'
    assert l5.element_names == 2 * (['el3'] + 2 * ['el2'] + 6 * ['el1'] + 2 * ['el2'])

    l6 = env['l6']
    assert l6.name == 'l6'
    assert l6.element_names == 4 * ['el1', 'el2']


def test_refer_and_thin_elements():
    sequence = """
    mb: sbend, l = 3;
    cav: rfcavity, l = 4;
    seq1: sequence, l = 10;
        mb1: mb, at = 1.5;
        cav1: cav, at = 5;
        mb2: mb, at = 8.5;
    endsequence;

    seq2: sequence, l = 10, refer = ENTRY;
        mb1: mb, at = 0;
        cav1: cav, at = 3;
        mb2: mb, at = 7;
    endsequence;
    """

    loader = MadxLoader()
    loader.load_string(sequence)
    env = loader.env

    seq1 = env['seq1']
    seq1.merge_consecutive_drifts()
    tt1 = seq1.get_table()

    seq2 = env['seq2']
    seq2.merge_consecutive_drifts()
    tt2 = seq2.get_table()

    assert np.all(tt1['element_type'] == tt2['element_type'])
    assert np.all(tt1['s'] == tt2['s'])


def test_import_seq_length():
    sequence = """
    qu: quadrupole, l=2, k1=3, k1s=4, tilt=2;  ! ignore thick and ktap

    ll := 3 * a;
    a = 10;

    line: sequence, l = ll;
        qu1: qu, at = 19;
    endsequence;
    """

    env = xt.load_madx_lattice(string=sequence)

    tt = env.line.get_table()
    assert np.all(tt.name == np.array(['drift_1', 'qu1', 'drift_2', '_end_point']))
    xo.assert_allclose(tt['s'], np.array([ 0., 18., 20., 30.]), rtol=0, atol=1e-15)
    assert env.line.builder.length == 30


def test_repeated_element_mad_behaviour():
    sequence = """
    mar: marker;
    ben: sbend, l=1;
    
    seq1: sequence, l=10;
      ee: mar, at=5;
    endsequence;
    
    seq2: sequence, l=10;
      ee: ben, at=5;  ! in MAD-X this definition will be ignored
    endsequence;
    """

    env = xt.load_madx_lattice(string=sequence)

    element = env['ee']
    assert env.seq1['ee'] == element
    assert env.seq2['ee'] == element


@pytest.mark.parametrize('aper_config', ['attached_to_marker', 'standalone'])
def test_apertures_on_markers(aper_config):
    if aper_config == 'attached_to_marker':
        sequence = """
            ! Attached to a marker
            m_circle: marker, apertype="circle", aperture={.2};
            m_ellipse: marker, apertype="ellipse", aperture={.2, .1};
            m_rectangle: marker, apertype="rectangle", aperture={.07, .05};
            m_rectellipse: marker, apertype="rectellipse", aperture={.2, .4, .25, .45};
            m_racetrack: marker, apertype="racetrack", aperture={.6,.4,.2,.1};
            m_octagon: marker, apertype="octagon", aperture={.4, .5, 0.5, 1.};
            m_polygon: marker, apertype="circle", aper_vx={+5.800e-2,+5.800e-2,-8.800e-2}, aper_vy={+3.500e-2,-3.500e-2,+0.000e+0};
            """
    else:
        sequence = """
            ! Standalone
            m_circle: circle, aperture={.2};
            m_ellipse: ellipse, aperture={.2, .1};
            m_rectangle: rectangle, aperture={.07, .05};
            m_rectellipse: rectellipse, aperture={.2, .4, .25, .45};
            m_racetrack: racetrack, aperture={.6,.4,.2,.1};
            m_octagon: octagon, aperture={.4, .5, 0.5, 1.};
            m_polygon: circle, aper_vx={+5.800e-2,+5.800e-2,-8.800e-2}, aper_vy={+3.500e-2,-3.500e-2,+0.000e+0};
            """

    sequence += """
        line: sequence, l=1;
            m_circle, at=0;
            m_ellipse, at=0.01;
            m_rectangle, at=0.02;
            m_rectellipse, at=0.03;
            m_racetrack, at=0.04;
            m_octagon, at=0.05;
            m_polygon, at=0.06;
        endsequence;
        """

    env = xt.load_madx_lattice(string=sequence)
    line = env.line

    apertures = [ee for ee in line.elements if ee.__class__.__name__.startswith('Limit')]

    assert isinstance(line[line['m_circle'].name_associated_aperture], xt.LimitEllipse)
    circ = apertures[0]
    assert circ.__class__.__name__ == 'LimitEllipse'
    xo.assert_allclose(circ.a_squ, .2**2, atol=1e-13, rtol=0)
    xo.assert_allclose(circ.b_squ, .2**2, atol=1e-13, rtol=0)

    assert isinstance(line[line['m_ellipse'].name_associated_aperture], xt.LimitEllipse)
    ellip = apertures[1]
    assert ellip.__class__.__name__ == 'LimitEllipse'
    xo.assert_allclose(ellip.a_squ, .2**2, atol=1e-13, rtol=0)
    xo.assert_allclose(ellip.b_squ, .1**2, atol=1e-13, rtol=0)

    assert isinstance(line[line['m_rectangle'].name_associated_aperture], xt.LimitRect)
    rect = apertures[2]
    assert rect.__class__.__name__ == 'LimitRect'
    assert rect.min_x == -.07
    assert rect.max_x == +.07
    assert rect.min_y == -.05
    assert rect.max_y == +.05

    assert isinstance(line[line['m_rectellipse'].name_associated_aperture], xt.LimitRectEllipse)
    rectellip = apertures[3]
    assert rectellip.max_x == .2
    assert rectellip.max_y == .4
    xo.assert_allclose(rectellip.a_squ, .25**2, atol=1e-13, rtol=0)
    xo.assert_allclose(rectellip.b_squ, .45**2, atol=1e-13, rtol=0)

    assert isinstance(line[line['m_racetrack'].name_associated_aperture], xt.LimitRacetrack)
    racetr = apertures[4]
    assert racetr.__class__.__name__ == 'LimitRacetrack'
    assert racetr.min_x == -.6
    assert racetr.max_x == +.6
    assert racetr.min_y == -.4
    assert racetr.max_y == +.4
    assert racetr.a == .2
    assert racetr.b == .1

    assert isinstance(line[line['m_octagon'].name_associated_aperture], xt.LimitPolygon)
    octag = apertures[5]
    assert octag.__class__.__name__ == 'LimitPolygon'
    assert octag._xobject.x_vertices[0] == 0.4
    xo.assert_allclose(octag._xobject.y_vertices[0], 0.4*np.tan(0.5), atol=1e-14, rtol=0)
    assert octag._xobject.y_vertices[1] == 0.5
    xo.assert_allclose(octag._xobject.x_vertices[1], 0.5/np.tan(1.), atol=1e-14, rtol=0)

    assert octag._xobject.y_vertices[2] == 0.5
    xo.assert_allclose(octag._xobject.x_vertices[2], -0.5/np.tan(1.), atol=1e-14, rtol=0)
    assert octag._xobject.x_vertices[3] == -0.4
    xo.assert_allclose(octag._xobject.y_vertices[3], 0.4*np.tan(0.5), atol=1e-14, rtol=0)


    assert octag._xobject.x_vertices[4] == -0.4
    xo.assert_allclose(octag._xobject.y_vertices[4], -0.4*np.tan(0.5), atol=1e-14, rtol=0)
    assert octag._xobject.y_vertices[5] == -0.5
    xo.assert_allclose(octag._xobject.x_vertices[5], -0.5/np.tan(1.), atol=1e-14, rtol=0)


    assert octag._xobject.y_vertices[6] == -0.5
    xo.assert_allclose(octag._xobject.x_vertices[6], 0.5/np.tan(1.), atol=1e-14, rtol=0)
    assert octag._xobject.x_vertices[7] == 0.4
    xo.assert_allclose(octag._xobject.y_vertices[7], -0.4*np.tan(0.5), atol=1e-14, rtol=0)

    assert isinstance(line[line['m_polygon'].name_associated_aperture], xt.LimitPolygon)
    polyg = apertures[6]
    assert polyg.__class__.__name__ == 'LimitPolygon'
    assert len(polyg._xobject.x_vertices) == 3
    assert len(polyg._xobject.y_vertices) == 3
    # The below assertions don't match test_apertures.py::test_mad_import
    # TODO: Figure out what mad_loader.py:971 is doing? It seems wrong
    assert np.all(polyg.x_vertices == [5.8e-2, 5.8e-2, -8.8e-2])
    assert np.all(polyg.y_vertices == [3.5e-2, -3.5e-2, 0.0])


def test_aperture_setting():
    sequence = """
    m_ellipse: marker, apertype="ellipse", aperture={.2, .1};
    m_aper: marker, apertype="rectangle", aperture={.3, .4};
    
    line: sequence, l=1;
        m_ellipse, at=0;
        m_aper, at=0.1;
    endsequence;
    
    m_ellipse, aperture={.3, .2};  ! no apertype
    m_aper, apertype="ellipse", aperture={.5, .6};  ! change apertype
    """

    env = xt.load_madx_lattice(string=sequence)
    line = env.line

    assert line['m_ellipse_aper'].a == .3
    assert line['m_ellipse_aper'].b == .2

    assert line['m_aper_aper'].a == .5
    assert line['m_aper_aper'].b == .6


def test_import_thick_with_apertures_and_slice():
    sequence = """
    k1=0.2;
    tilt=0.1;

    elm: sbend,
        k1:=k1,
        l=1,
        angle=0.1,
        tilt=0.2,
        apertype="rectellipse",
        aperture={0.1,0.2,0.11,0.22},
        aper_tol={0.1,0.2,0.3},
        aper_tilt:=tilt,
        aper_offset={0.2, 0.3};

    seq: sequence, l=1;
    elm: elm, at=0.5;
    endsequence;
    """

    env = xt.load_madx_lattice(string=sequence)
    line = env.seq

    def _assert_eq(a, b):
        xo.assert_allclose(a, b, atol=1e-16)

    _assert_eq(line[f'elm_aper'].rot_s_rad, 0.1)
    _assert_eq(line[f'elm_aper'].shift_x, 0.2)
    _assert_eq(line[f'elm_aper'].shift_y, 0.3)
    _assert_eq(line[f'elm_aper'].max_x, 0.1)
    _assert_eq(line[f'elm_aper'].max_y, 0.2)
    _assert_eq(line[f'elm_aper'].a_squ, 0.11 ** 2)
    _assert_eq(line[f'elm_aper'].b_squ, 0.22 ** 2)

    _assert_eq(line[f'elm'].rot_s_rad, 0.2)

    line.slice_thick_elements(slicing_strategies=[Strategy(Uniform(2))])

    assert np.all(line.get_table().rows['elm_entry':'elm_exit'].name == [
        'elm_entry',                    # entry marker
        'elm_aper..0',                  # entry edge aperture
        'elm..entry_map',               # entry edge (+transform)
        'drift_elm..0',                 # drift 0
        'elm_aper..1',                  # slice 1 aperture
        'elm..0',                       # slice 0 (+transform)
        'drift_elm..1',                 # drift 1
        'elm_aper..2',                  # slice 2 aperture
        'elm..1',                       # slice 2 (+transform)
        'drift_elm..2',                 # drift 2
        'elm_aper..3',                  # exit edge aperture
        'elm..exit_map',                # exit edge (+transform)
        'elm_exit',                     # exit marker
    ])

    line.build_tracker(compile=False) # To resolve parents

    for i in range(4):
        _assert_eq(line[f'elm_aper..{i}'].resolve(line).rot_s_rad, 0.1)
        _assert_eq(line[f'elm_aper..{i}'].resolve(line).shift_x, 0.2)
        _assert_eq(line[f'elm_aper..{i}'].resolve(line).shift_y, 0.3)
        _assert_eq(line[f'elm_aper..{i}'].resolve(line).max_x, 0.1)
        _assert_eq(line[f'elm_aper..{i}'].resolve(line).max_y, 0.2)
        _assert_eq(line[f'elm_aper..{i}'].resolve(line).a_squ, 0.11 ** 2)
        _assert_eq(line[f'elm_aper..{i}'].resolve(line).b_squ, 0.22 ** 2)

    for i in range(2):
        _assert_eq(line[f'elm..{i}']._parent.rot_s_rad, 0.2)