import copy
import gc
import pickle
import weakref

import pytest

import xtrack as xt


def test_multitwiss_attributes():
    tw = xt.MultiTwiss()
    tw.ring = 'first'
    assert tw['ring'] == 'first'
    tw['ring'] = 'second'
    assert tw.ring == 'second'
    tw._line_names = ['ring']
    assert tw['_line_names'] == ['ring']

    # Stored keys take precedence over dict methods, as with an instance dict.
    tw.items = 'items value'
    assert tw['items'] == tw.items == 'items value'
    del tw.items
    assert callable(tw.items)
    del tw.ring
    assert 'ring' not in tw
    assert not hasattr(tw, 'ring')
    with pytest.raises(AttributeError, match='ring'):
        _ = tw.ring
    with pytest.raises(AttributeError, match='ring'):
        del tw.ring
    with pytest.raises(KeyError, match='ring'):
        _ = tw['ring']


@pytest.mark.parametrize('duplicate', [copy.copy, copy.deepcopy,
                                     lambda obj: pickle.loads(pickle.dumps(obj))])
def test_multitwiss_copy(duplicate):
    tw = xt.MultiTwiss()
    tw.ring = [1, 2]
    tw._line_names = ['ring']
    copied = duplicate(tw)
    assert isinstance(copied, xt.MultiTwiss)
    assert copied is not tw
    assert copied == tw
    assert copied.ring == [1, 2]
    assert copied._line_names == ['ring']
    copied.ring = [3]
    assert tw.ring == [1, 2]


@pytest.mark.parametrize('matching', [False, True], ids=['twiss', 'match'])
def test_multitwiss_releases_results(monkeypatch, matching):
    line = xt.Line(elements={'map': xt.LineSegmentMap(qx=0.31, qy=0.32)})
    env = line.env
    env.lines['ring1'] = line
    env.new_line(name='ring2', components=['map'])
    for ll in env.lines.values():
        ll.particle_ref = xt.Particles(p0c=7e12)
        ll.twiss_default['method'] = '4d'
    env['qx'] = 0.31
    env.ref['map'].qx = env.ref['qx']

    if matching:
        opt = env.match(
            solve=False, vary=xt.Vary('qx', step=1e-6),
            targets=xt.Target('qx', value=0.315, line='ring1'))
        merit = opt.get_merit_function(return_scalar=True)

    refs = []
    original_twiss = env.twiss

    def observed_twiss(**kwargs):
        tw = original_twiss(**kwargs)
        refs.append([weakref.ref(tw), weakref.ref(tw.ring1), weakref.ref(tw.ring2)])
        return tw

    monkeypatch.setattr(env, 'twiss', observed_twiss)
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for ii in range(5):
            if matching:
                merit([0.31 + ii * 1e-4])
            else:
                env.twiss()

        # Matching keeps its latest result for inspection in _last_data.
        discarded = refs[:-1] if matching else refs
        assert len(refs) == 5
        assert all(ref() is None for result in discarded for ref in result)
        if matching:
            assert all(ref() is not None for ref in refs[-1])
    finally:
        if gc_was_enabled:
            gc.enable()
