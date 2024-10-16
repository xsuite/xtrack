import math
from collections import OrderedDict

import pytest

import xtrack as xt
from xtrack.mad_parser.parse import MadxParser, MadxOutputType
from xtrack.mad_parser.loader import MadxLoader


def test_simple_parser():
    sequence = """
    if (version>=50401){option,-rbarc;};  ! to be ignored
    
    third = 1 / 3;
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

    expected: MadxOutputType = {
        'vars': {
            'third': {'deferred': False, 'expr': '(1.0 / 3.0)'},
            'hello': {'deferred': True, 'expr': '(third * twopi)'},
            'mb.l': {'deferred': False, 'expr': 1.0},
            'qd.l': {'deferred': False, 'expr': 0.5},
            'offset': {'deferred': True, 'expr': 0.1},
        },
        'elements': {
            'mb': {
                'angle': {'deferred': True, 'expr': 'hello'},
                'l': {'deferred': True, 'expr': 'mb.l'},
                'parent': 'sbend',
            },
            'qf': {
                'k1': {'deferred': True, 'expr': 1.0},
                'l': {'deferred': True, 'expr': 1.0},
                'parent': 'quadrupole',
            },
            'qd': {
                'k1': {'deferred': True, 'expr': -1.0},
                'l': {'deferred': True, 'expr': 1.0},
                'parent': 'quadrupole',
            },
        },
        'lines': {
            'line': {
                'l': {'deferred': False, 'expr': 12.0},
                'parent': 'sequence',
                'elements': {
                    'ip1': {
                        'at': {'deferred': False, 'expr': 0.0},
                        'parent': 'marker',
                    },
                    'qf1': {
                        'at': {'deferred': True, 'expr': '(1.0 + offset)'},
                        'from': {'deferred': False, 'expr': 'ip1'},
                        'parent': 'qf',
                        'slot_id': {'deferred': False, 'expr': 1.0},
                    },
                    'mb1': {
                        'at': {'deferred': True, 'expr': '(2.0 + offset)'},
                        'from': {'deferred': False, 'expr': 'ip1'},
                        'parent': 'mb',
                        'slot_id': {'deferred': False, 'expr': 2.0},
                    },
                    'qd1': {
                        'at': {'deferred': True, 'expr': '(3.0 + offset)'},
                        'from': {'deferred': False, 'expr': 'ip1'},
                        'parent': 'qd',
                        'slot_id': {'deferred': False, 'expr': 3.0},
                    },
                },
            },
        },
        'parameters': {
            'mb1': {
                'k0': {'deferred': True, 'expr': 'hello'},
                'polarity': {'deferred': False, 'expr': 1.0},
            },
            'qf1': {
                'knl': {'deferred': True, 'expr': [0.0, 0.0, 0.0, 0.01, 0.0]},
            },
            'qd1': {
                'knl': {'deferred': True, 'expr': [0.0, 0.0, 0.0, -0.01, 0.0]},
            },
        },
    }

    parser = MadxParser()
    result = parser.parse_string(sequence)

    def _order_madx_output(item):
        item['vars'] = OrderedDict(item['vars'])
        item['elements'] = OrderedDict(item['elements'])
        item['lines'] = OrderedDict(item['lines'])
        for line in item['lines'].values():
            line['elements'] = OrderedDict(line['elements'])
        item['parameters'] = OrderedDict(item['parameters'])

    _order_madx_output(expected)
    _order_madx_output(result)

    assert expected == result


@pytest.fixture(scope='module')
def example_sequence(temp_context_default_mod):
    sequence = """
    ll = 30;

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
    rf: rfcavity, L=2, VOLT=1, LAG=2, FREQ=3, HARMON=2;  ! ignore N_BESSEL, NO_CAVITY_TOTALPATH
    mu: multipole, LRAD=1, TILT=2, NL={3, 4, 5}, KSL={1, 2, 3};
    so: solenoid, l=2, ks=3;  ! ignore ksi

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
    endsequence;
    """

    loader = MadxLoader()
    builders = loader.load_string(sequence, build=False)

    line_builder, = builders
    line_builder.build()
    env = loader.env

    positions = {place.name: place.at for place in line_builder.components}

    return env, positions


def test_parsed_line(example_sequence):
    env, _ = example_sequence
    assert env['line'].name == 'line'


def test_vkick(example_sequence):
    env, positions = example_sequence
    vk1 = env['vk1']
    assert positions['vk1'] == 1
    assert isinstance(vk1, xt.Multipole)
    assert vk1.length == 2
    assert vk1.knl[0] == 0
    assert vk1.ksl[0] == 3
    assert vk1.rot_s_rad == -1

def test_hkick(example_sequence):
    env, positions = example_sequence
    hk1 = env['hk1']
    assert positions['hk1'] == 3
    assert isinstance(hk1, xt.Multipole)
    assert hk1.length == 1
    assert hk1.knl[0] == -6
    assert hk1.ksl[0] == 0
    assert hk1.rot_s_rad == -2

def test_kick(example_sequence):
    env, positions = example_sequence
    ki1 = env['ki1']
    assert positions['ki1'] == 5
    assert isinstance(ki1, xt.Multipole)
    assert ki1.length == 2
    assert ki1.knl[0] == -4
    assert ki1.ksl[0] == 3
    assert ki1.rot_s_rad == 1

def test_tkick(example_sequence):
    env, positions = example_sequence
    tk1 = env['tk1']
    assert positions['tk1'] == 7
    assert isinstance(tk1, xt.Multipole)
    assert tk1.length == 1
    assert tk1.knl[0] == -4
    assert tk1.ksl[0] == 3
    assert tk1.rot_s_rad == 2

def test_instrument(example_sequence):
    env, positions = example_sequence
    in1 = env['in1']
    assert positions['in1'] == 9
    assert isinstance(in1, xt.Drift)
    assert in1.length == 2

def test_monitor(example_sequence):
    env, positions = example_sequence
    mo1 = env['mo1']
    assert positions['mo1'] == 11
    assert isinstance(mo1, xt.Drift)
    assert mo1.length == 1

def test_placeholder(example_sequence):
    env, positions = example_sequence
    pl1 = env['pl1']
    assert positions['pl1'] == 13
    assert isinstance(pl1, xt.Drift)
    assert pl1.length == 1

def test_sbend(example_sequence):
    env, positions = example_sequence
    # sb: sbend, l=2, angle=2, tilt=-2, k0=3, k1=1, k2=2, k1s=3, e1=2, e2=1,
    #   fint=3, fintx=2, hgap=1;  ! thick, ktap, h1, h2 we ignore
    sb1 = env['sb1']
    assert positions['sb1'] == 15
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
    env, positions = example_sequence
    # rb: rbend, l=2, angle=1.5, tilt=-2, k0=3, k1=1, k2=2, k1s=3, e1=2, e2=1,
    #   fint=3, fintx=2, hgap=1, h1=3, h2=2;  ! ditto
    rb1 = env['rb1']
    assert positions['rb1'] == 17
    assert isinstance(rb1, xt.Bend)

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
    assert rb1.edge_entry_angle == 2 + angle / 2
    assert rb1.edge_exit_angle == 1 + angle / 2
    assert rb1.edge_entry_fint == 3
    assert rb1.edge_exit_fint == 2
    assert rb1.edge_entry_hgap == 1
    assert rb1.edge_exit_hgap == 1

def test_quadrupole(example_sequence):
    env, positions = example_sequence
    # qu: quadrupole, l=2, k1=3, k1s=4, tilt=2;  ! ignore thick and ktap
    qu1 = env['qu1']
    assert positions['qu1'] == 19
    assert isinstance(qu1, xt.Quadrupole)
    assert qu1.length == 2
    assert qu1.k1 == 3
    assert qu1.k1s == 4
    assert qu1.rot_s_rad == 2

def test_sextupole(example_sequence):
    env, positions = example_sequence
    # se: sextupole, L=1, K2=2, K2S=3, TILT=2;  ! ignore ktap
    se1 = env['se1']
    assert positions['se1'] == 21
    assert isinstance(se1, xt.Sextupole)
    assert se1.length == 1
    assert se1.k2 == 2
    assert se1.k2s == 3
    assert se1.rot_s_rad == 2

def test_octupole(example_sequence):
    env, positions = example_sequence
    # oc: octupole, L=2, K3=3, K3S=2, TILT=2;
    oc1 = env['oc1']
    assert positions['oc1'] == 23
    assert isinstance(oc1, xt.Octupole)
    assert oc1.length == 2
    assert oc1.k3 == 3
    assert oc1.k3s == 2
    assert oc1.rot_s_rad == 2

def test_marker(example_sequence):
    env, positions = example_sequence
    # ma: marker;
    ma1 = env['ma1']
    assert positions['ma1'] == 25
    assert isinstance(ma1, xt.Marker)

def test_rfcavity(example_sequence):
    env, positions = example_sequence
    # rf: rfcavity, L=2, VOLT=1, LAG=2, FREQ=3, HARMON=2;  ! ignore N_BESSEL, NO_CAVITY_TOTALPATH
    rf1 = env['rf1']
    assert positions['rf1'] == 27
    assert isinstance(rf1, xt.Cavity)
    assert rf1.voltage == 1e6
    assert rf1.lag == 2 * 360
    assert rf1.frequency == 3e6

def test_multipole(example_sequence):
    env, positions = example_sequence
    # mu: multipole, LRAD=1, TILT=2, NL={3, 4, 5}, KSL={1, 2, 3};
    mu1 = env['mu1']
    assert positions['mu1'] == 29
    assert isinstance(mu1, xt.Multipole)
    assert mu1.length == 1
    assert mu1.knl[0] == 3
    assert mu1.knl[1] == 4
    assert mu1.knl[2] == 5
    assert mu1.ksl[0] == 1
    assert mu1.ksl[1] == 2
    assert mu1.ksl[2] == 3
    assert mu1.rot_s_rad == 2

def test_solenoid(example_sequence):
    env, positions = example_sequence
    # so: solenoid, l=2, ks=3;  ! ignore ksi
    so1 = env['so1']
    assert positions['so1'] == 31
    assert isinstance(so1, xt.Solenoid)
    assert so1.length == 2
    assert so1.ks == 3
