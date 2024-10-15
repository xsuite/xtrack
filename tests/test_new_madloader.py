from collections import OrderedDict

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
    result = parser.parse(sequence)

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


def test_element_reversal():
    # Elements:
    # - vkicker, l=real, kick=real, tilt=real
    # - hkicker, ditto
    # - kicker, l=real, vkick=real, hkick=real, tilt=real
    # - tkicker, ditto
    # - collimator, l=real, apertype=string, aperture={values}, aper_offset={values}, aper_tol={values}
    # - instrument, l=real
    # - monitor, l=real
    # - placeholder, l=real
    # - sbend, l=real, angle=real, tilt=real, k0=real, k1=real, k2=real, k1s=real, e1=real, e2=real, fint=real, fintx=real, hgap=real, h1=real, h2=real, ktap=real, thick=logical
    # - rbend, ditto
    # - quadrupole, l=real, k1=real, k1s=real, tilt=real, ktap=real, thick=logical
    # - sextupole, L=real, K2=real, K2S=real, TILT=real, KTAP=real
    # - octupole, L=real, K3=real, K3S=real, TILT=real
    # - marker;
    # - rfcavity, L=real, VOLT=real, LAG=real, FREQ=real, HARMON=integer, N_BESSEL=integer, NO_CAVITY_TOTALPATH=logical
    # - multipole, LRAD=real, TILT=real, NL={real}, KSL={real}
    # - solenoid, l=real, ks=real, ksi=real
    pass
