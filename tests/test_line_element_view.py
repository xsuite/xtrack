import inspect

import pytest
import xtrack as xt
from xtrack.view import LineElementView, View


class TestElementView(LineElementView):
    __test__ = False

    def line_context(self, scale=1):
        """Return the line context and a user argument."""
        return self._view_line, self._view_name, scale

    @property
    def twice_length(self):
        return 2 * self.length

    @twice_length.setter
    def twice_length(self, value):
        self.length = value / 2


@pytest.fixture
def line(monkeypatch):
    # Use an existing beam element to avoid a new Xobject type/kernel.
    monkeypatch.setattr(xt.Quadrupole, '_line_view_class', TestElementView,
                        raising=False)
    env = xt.Environment()
    env['kq'] = 0.2
    env.new('q', xt.Quadrupole, length=1., k1='kq', knl=[0., 0.])
    env.new('d', xt.Drift, length=2.)
    return env.new_line(components=['q', 'd'])


@pytest.mark.parametrize('managed', [True, False])
def test_line_view_access(line, managed):
    if not managed:
        line.env._var_management = None
    raw = line.get('q')
    view = line['q']
    assert isinstance(view, LineElementView)
    assert isinstance(view, xt.Quadrupole)
    assert view._get_viewed_object() is raw
    assert view.line_context(2) == (line, 'q', 2)
    assert line[0].line_context() == (line, 'q', 1)
    assert 'line_context' in dir(view)
    assert 'twice_length' in dir(view)
    assert str(inspect.signature(view.line_context)) == '(scale=1)'
    assert view.line_context.__doc__.startswith('Return the line context')
    assert not hasattr(raw, 'line_context')
    with pytest.raises(AttributeError, match='read-only'):
        view._view_line = object()
    assert not hasattr(raw, '_view_line')

    view.twice_length = 4.
    assert raw.length == 2.
    assert view.twice_length == 4.
    view.knl[1] = 0.3
    assert raw.knl[1] == pytest.approx(0.3)

    if managed:
        view.k1 = '2*kq'
        view.knl[1] = '3*kq'
        line['kq'] = 0.4
        assert raw.k1 == pytest.approx(0.8)
        assert raw.knl[1] == pytest.approx(1.2)
        assert view._get_expr('k1') is not None
        assert isinstance(line['d'], View)
        assert not isinstance(line['d'], LineElementView)
    else:
        view.k1 = 0.7
        assert raw.k1 == pytest.approx(0.7)
        assert view._get_expr('k1') is None
        assert view.get_table()['expr', 'k1'] == 'None'
        assert line['d'] is line.get('d')

    copied = view.copy()
    assert isinstance(copied, xt.Quadrupole)
    assert not isinstance(copied, View)
    assert copied is not raw
    assert copied.length == raw.length
    copied.length = 5.
    assert raw.length == 2.


def test_line_view_shared_element_and_copy(line):
    other = line.env.new_line(components=['d', 'q'])
    assert line.get('q') is other.get('q')
    assert line['q'].line_context() == (line, 'q', 1)
    assert other['q'].line_context() == (other, 'q', 1)
    assert not hasattr(line.env['q'], 'line_context')
    assert not isinstance(line.env['q'], LineElementView)

    copied = line.copy()
    assert copied.get('q') is not line.get('q')
    assert copied['q'].line_context() == (copied, 'q', 1)
    shallow = line.copy(shallow=True)
    assert shallow.get('q') is line.get('q')
    assert shallow['q'].line_context() == (shallow, 'q', 1)


def test_view_unwraps_specialised_view(line):
    view = line['q']
    plain = View(view, line._xdeps_eref['q'], line._xdeps_eval.eval)
    assert plain._get_viewed_object() is line.get('q')
    assert not hasattr(plain, 'line_context')
