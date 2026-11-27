import numpy as np
import pytest

import xtrack as xt
from xtrack.environment import _reverse_element


@pytest.mark.parametrize('element_class', [
    xt.Bend, xt.RBend, xt.Quadrupole, xt.Sextupole, xt.Octupole,
    xt.UniformSolenoid,
])
@pytest.mark.parametrize('flags', ['constant', 'expression', 'mixed'])
def test_reverse_element_edge_active(element_class, flags):
    env = xt.Environment()
    env['pi'] = np.pi
    env['entry_on'], env['exit_on'] = 0, 1
    entry = 0 if flags == 'constant' else 'entry_on'
    exit_ = 'exit_on' if flags == 'expression' else 1
    length = {'length_straight': 1.} if element_class is xt.RBend else {'length': 1.}
    env.new('magnet', element_class, **length,
            edge_entry_active=entry, edge_exit_active=exit_)
    magnet = env['magnet']

    _reverse_element(env, 'magnet')
    assert (magnet.edge_entry_active, magnet.edge_exit_active) == (1, 0)

    # Expressions must follow their physical face and remain live after reversal.
    if flags != 'constant':
        env['entry_on'] = 1
        env['exit_on'] = 0
        expected_entry = 0 if flags == 'expression' else 1
        assert (magnet.edge_entry_active, magnet.edge_exit_active) == (expected_entry, 1)

    _reverse_element(env, 'magnet')
    if flags == 'constant':
        assert (magnet.edge_entry_active, magnet.edge_exit_active) == (0, 1)
    else:
        assert (magnet.edge_entry_active, magnet.edge_exit_active) == (1, expected_entry)
        env['entry_on'], env['exit_on'] = 0, 1
        assert (magnet.edge_entry_active, magnet.edge_exit_active) == (0, 1)


def test_reverse_lines_edge_active():
    # The native loader also accepts Xtrack field names in the input.
    env = xt.load(string='''
        b: sbend, l=1, angle=.1, e1=.02, e2=.05,
           edge_entry_active=0, edge_exit_active=1;
        seq: sequence, l=2;
            b, at=1;
        endsequence;
    ''', format='madx', reverse_lines=['seq'])

    bend = env.seq['b']
    assert bend.edge_entry_angle == .05
    assert bend.edge_exit_angle == .02
    assert bend.edge_entry_active == 1
    assert bend.edge_exit_active == 0
