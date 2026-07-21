"""Tests for ``xtrack.tpsa``: the compiled bridge that tracks a 6D TPSA map through
Xtrack elements, on top of the standalone ``xgtpsa`` engine.

Skipped wholesale unless the GTPSA core is built (gtpsa_lib/build.sh).
"""

import gc
import os

import numpy as np
import pytest

import xobjects as xo
import xtrack as xt
import xgtpsa
import xtrack.tpsa as xtpsa
from xtrack.tpsa import _bridge_build
from xtrack.tpsa._bridge_particle import COORD_FIELDS, REF_FIELDS, XtBridgeParticle
from xtrack.tpsa.backend import num_bridge, registry_classes, type_id_for
from xtrack.tpsa.registry import COORDS, REF_VARS, TYPE_IDS
from xtrack.twiss import _6d_w_matrix


def _gtpsa_available():
    try:
        xgtpsa.lib()
        _bridge_build._bridge_sources()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _gtpsa_available(),
    reason="libgtpsa_core.so unavailable; run gtpsa_lib/build.sh",
)

P0C = 7e12
MASS0 = xt.PROTON_MASS_EV
X0 = dict(x=1e-4, px=1.5e-4, y=-1e-4, py=1e-4, zeta=1e-3, delta=2e-3)


def _backend():
    from xtrack.tracking_backends import _BACKENDS
    return _BACKENDS[xtpsa.ParticlesTpsa]


def _map(order=2, **kwargs):
    return xtpsa.ParticlesTpsa(order=order, p0c=P0C, mass0=MASS0,
                               **{**X0, **kwargs})


def _line(specs):
    """Build and prepare an ``xt.Line`` for tracking from ``[(name, element), ...]``."""
    line = xt.Line(elements=[e for _, e in specs],
                   element_names=[n for n, _ in specs])
    line.particle_ref = xt.Particles(p0c=P0C, mass0=MASS0, q0=1)
    line.build_tracker()
    return line


def _demo_line():
    return _line([('d0', xt.Drift(length=1.2)),
                  ('q', xt.Quadrupole(length=0.5, k1=0.08)),
                  ('d1', xt.DriftExact(length=0.9)),
                  ('b', xt.Bend(length=1.5, k0=0.008, angle=0.02)),
                  ('d2', xt.Drift(length=0.7))])


def _native_orbit(line, coords=None):
    p = line.build_particles(**(coords or X0))
    line.track(p)
    return np.array([float(getattr(p, c)[0]) for c in COORDS])


def _fd_jac(line, h=1e-7):
    """Central-difference Jacobian of the line map at ``X0``."""
    jac = np.zeros((6, 6))
    for j, c in enumerate(COORDS):
        plus, minus = dict(X0), dict(X0)
        plus[c] += h
        minus[c] -= h
        jac[:, j] = (_native_orbit(line, plus) - _native_orbit(line, minus)) / (2 * h)
    return jac


# --------------------------------------------------------------------------- #
# Descriptors: Sharing, truncation integrity
# --------------------------------------------------------------------------- #


def test_maps_share_descriptor_by_order():
    a, b = _map(order=2), _map(order=2)
    c = _map(order=3)
    assert a.descriptor is b.descriptor
    assert a.descriptor is not c.descriptor
    assert (a.order, a.n_variables) == (2, 6)
    assert (c.order, c.n_variables) == (3, 6)
    # the six coordinate series of one map share its descriptor
    assert all(s.descriptor is a.descriptor for s in a.coords)


def test_order_truncation_integrity():
    """A map only carries terms up to its own order, and the shared ones agree.

    Note: querying a monomial above the descriptor order is not a Python error.
    GTPSA aborts the process ("mad_tpsa.c: invalid monomial").
    """
    el = xt.Sextupole(length=0.3, k2=50.0)
    m2, m3 = _map(order=2), _map(order=3)
    el.track(m2)
    el.track(m3)

    assert m2.descriptor is not m3.descriptor

    low = m2.monomial_coeffs("px")
    high = m3.monomial_coeffs("px")

    # the order-2 map holds nothing above order 2 ...
    assert max(sum(mono) for mono in low) <= 2
    # ... while the order-3 map does, and those terms are real
    third = {mono: c for mono, c in high.items() if sum(mono) == 3}
    assert third and all(c != 0.0 for c in third.values())

    # every term they share is bit-identical
    for mono, coeff in low.items():
        assert high[mono] == coeff

    # a query at exactly the max order is safe
    assert m2.coefficient("px", (2, 0, 0, 0, 0, 0)) == low.get((2, 0, 0, 0, 0, 0), 0.0)


# --------------------------------------------------------------------------- #
# TPSA coefficients
# --------------------------------------------------------------------------- #


def test_tpsa_coefficient_forms():
    el = xt.Sextupole(length=0.3, k2=50.0)
    m = _map(order=3)
    el.track(m)
    series = m.x

    monos = [(0, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 1)]
    batch = series.coefficient(monos)
    assert batch.shape == (3,)
    for i, mono in enumerate(monos):
        single = series.coefficient(mono)
        assert isinstance(single, float)
        assert single == series.get(mono) == batch[i]

    # coefficient() and monomial_coeffs() agree term by term
    coeffs = series.monomial_coeffs()
    for mono, value in coeffs.items():
        assert series.coefficient(mono) == value
    assert all(abs(v) > 1e-14 for v in coeffs.values())
    assert len(series.monomial_coeffs(tol=1e30)) == 0

    # the ParticlesTpsa wrappers delegate to the same series
    assert m.coefficient('x', monos[1]) == series.get(monos[1])
    assert m.coefficient(0, monos[1]) == series.get(monos[1])
    assert m.monomial_coeffs('x') == coeffs
    assert set(m.monomial_coeffs()) == set(COORDS)

    with pytest.raises(ValueError, match='one monomial'):
        series.coefficient(np.zeros((2, 2, 6), dtype=int))


# --------------------------------------------------------------------------- #
# ParticlesTpsa surface
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('kwargs', [
    dict(p0c=P0C, mass0=MASS0, delta=2e-3),
    dict(energy0=P0C, mass0=MASS0),
    dict(gamma0=7460.0, mass0=MASS0),
])
def test_reference_algebra_matches_particles(kwargs):
    m = xtpsa.ParticlesTpsa(order=2, **kwargs)
    p = xt.Particles(**kwargs)
    for var in REF_VARS:
        expected = float(np.asarray(getattr(p, var)).reshape(-1)[0])
        xo.assert_allclose(getattr(m, var), expected, rtol=1e-12, atol=0)


def test_scalar_guard():
    with pytest.raises(ValueError, match='single map'):
        xtpsa.ParticlesTpsa(order=2, x=[1e-4, 2e-4], p0c=P0C, mass0=MASS0)


def test_fresh_map_is_identity():
    """A new map is the identity around its coordinates."""
    m = _map(order=3)
    xo.assert_allclose(m.const_part, [X0[c] for c in COORDS], rtol=0, atol=0)
    xo.assert_allclose(m.jacobian(), np.eye(6), rtol=0, atol=0)


def test_getattr_and_to_particles():
    m = _map(order=2)
    assert isinstance(m.x, xgtpsa.Tpsa)
    assert isinstance(m.beta0, float)
    with pytest.raises(AttributeError):
        m.bogus

    p = m.to_particles()
    assert isinstance(p, xt.Particles)
    for c in COORDS:
        assert float(getattr(p, c)[0]) == X0[c]


def test_from_coords_view():
    m = _map(order=2)
    view = xtpsa.ParticlesTpsa._from_coords(m.coords)
    assert view.coords[0] is m.coords[0]        # shared series, not copies
    assert view.order == 2
    assert view._bridge is None                 # not trackable
    xo.assert_allclose(view.const_part, m.const_part, rtol=0, atol=0)
    with pytest.raises(AttributeError, match='no reference particle'):
        view.beta0


def test_bridge_particle_struct():
    assert COORD_FIELDS == COORDS
    assert REF_FIELDS == REF_VARS

    ffi = xgtpsa.ffi()
    bufs = [ffi.new('double*', float(X0[c])) for c in COORDS]
    refs = {r: 1.0 for r in REF_VARS}
    bp = num_bridge(bufs, refs, line_length=26658.0)

    assert isinstance(bp, XtBridgeParticle)
    assert bp.state == 1 and bp.at_element == 0 and bp.track_flags == 0
    assert bp.line_length == 26658.0
    for c, buf in zip(COORDS, bufs):
        assert getattr(bp, c) == int(ffi.cast('uintptr_t', buf))
    for r in REF_VARS:
        assert getattr(bp, r) == 1.0


# --------------------------------------------------------------------------- #
# Bridging: TPSA tracking must reproduce native tracking
# --------------------------------------------------------------------------- #

ELEMENTS = [
    pytest.param(lambda: xt.Drift(length=1.2), id='Drift'),
    pytest.param(lambda: xt.DriftExact(length=0.9), id='DriftExact'),
    pytest.param(lambda: xt.Marker(), id='Marker'),
    pytest.param(lambda: xt.Quadrupole(length=1.0, k1=0.06), id='Quadrupole'),
    pytest.param(lambda: xt.Quadrupole(length=1.0, k1=0.06, num_multipole_kicks=3),
                 id='Quadrupole-kicks'),
    pytest.param(lambda: xt.Quadrupole(length=1.0, k1s=0.04), id='Quadrupole-skew'),
    pytest.param(lambda: xt.Sextupole(length=0.4, k2=1.5), id='Sextupole'),
    pytest.param(lambda: xt.Octupole(length=0.3, k3=60.0), id='Octupole'),
    pytest.param(lambda: xt.Bend(length=1.5, k0=0.008, angle=0.02), id='Bend'),
    pytest.param(lambda: xt.RBend(length_straight=2.0, k0=0.008, k1=0.02, angle=0.02),
                 id='RBend'),
    pytest.param(lambda: xt.Multipole(knl=[0.001, 0.02, 0.3], ksl=[0.0, 0.01, 0.0]),
                 id='Multipole-thin'),
    pytest.param(lambda: xt.Multipole(knl=[0.002], hxl=0.002), id='Multipole-hxl'),
    pytest.param(lambda: xt.Multipole(knl=[0.0, 0.05], length=0.5), id='Multipole-thick'),
    pytest.param(lambda: xt.UniformSolenoid(length=2.0, ks=0.1), id='UniformSolenoid'),
    pytest.param(lambda: xt.Cavity(length=0.5, voltage=2e6, frequency=400e6, phase=np.pi),
                 id='Cavity-thick'),
    pytest.param(lambda: xt.Cavity(voltage=1e6, frequency=400e6, phase=np.pi),
                 id='Cavity-thin'),
    pytest.param(lambda: xt.LimitRectEllipse(max_x=0.05, max_y=0.04, a=0.05, b=0.04),
                 id='LimitRectEllipse'),
]


@pytest.mark.parametrize('make_element', ELEMENTS)
def test_track_element_matches_native(make_element):
    element = make_element()
    reference = _native_orbit(_line([('e', element)]))

    m = _map(order=2)
    element.track(m)
    xo.assert_allclose(m.const_part, reference, rtol=0, atol=1e-13)


def test_track_line_matches_native():
    line = _demo_line()
    reference = _native_orbit(line)

    m = _map(order=2)
    line.track(m)
    xo.assert_allclose(m.const_part, reference, rtol=0, atol=1e-14)
    xo.assert_allclose(m.jacobian(), _fd_jac(line), rtol=0, atol=1e-6)


def test_track_line_partial_range():
    line = _demo_line()

    by_name = _map(order=2)
    line.track(by_name, ele_stop='d1')
    by_count = _map(order=2)
    line.track(by_count, num_elements=2)
    xo.assert_allclose(by_name.const_part, by_count.const_part, rtol=0, atol=0)

    from_name = _map(order=2)
    line.track(from_name, ele_start='q', ele_stop='b')
    from_index = _map(order=2)
    line.track(from_index, ele_start=1, ele_stop=3)
    xo.assert_allclose(from_name.const_part, from_index.const_part, rtol=0, atol=0)

    # a partial range is a prefix of the full one
    full = _map(order=2)
    line.track(full)
    assert not np.allclose(full.const_part, by_name.const_part)


def test_track_num_twin_bit_identical():
    element = xt.Quadrupole(length=0.5, k1=0.08)
    line = _line([('q', element)])

    p = line.build_particles(**X0)
    xtpsa.track_num_twin(element, p)
    twin = np.array([float(getattr(p, c)[0]) for c in COORDS])

    xo.assert_allclose(twin, _native_orbit(line), rtol=0, atol=0)

    with pytest.raises(TypeError, match='expects xt.Particles'):
        xtpsa.track_num_twin(element, _map())


def test_registry_typeids():
    classes = registry_classes()
    assert len(classes) == len(TYPE_IDS)
    for name, type_id in TYPE_IDS.items():
        assert classes[type_id].__name__ == name
        assert type_id_for(name) == type_id

    with pytest.raises(NotImplementedError, match="not in the TPSA bridge registry"):
        type_id_for('NotAnElement')

    # an unregistered element cannot be tracked as a map
    assert 'Translation' not in TYPE_IDS
    with pytest.raises(NotImplementedError, match='Translation'):
        xt.Translation(shift_x=1e-3).track(_map())


# --------------------------------------------------------------------------- #
# Monitors
# --------------------------------------------------------------------------- #

def test_one_turn_ebe_records_full_maps():
    line = _demo_line()
    m = _map(order=2)
    line.track(m, turn_by_turn_monitor='ONE_TURN_EBE')

    mon = line.record_last_track
    assert isinstance(mon, xtpsa.TpsaMonitor)
    n = len(line.element_names)
    assert len(mon) == n + 1
    assert mon.const_part.shape == (n + 1, 6)
    assert mon.jacobian().shape == (n + 1, 6, 6)
    assert 'n_slots=%d' % (n + 1) in repr(mon)

    # first slot is the map on entry, last slot is the map on exit
    xo.assert_allclose(mon.const_part[0], [X0[c] for c in COORDS], rtol=0, atol=0)
    xo.assert_allclose(mon.const_part[-1], m.const_part, rtol=0, atol=0)

    slot = mon[-1]
    assert isinstance(slot, xtpsa.ParticlesTpsa)
    xo.assert_allclose(slot.const_part, m.const_part, rtol=0, atol=0)

    series = mon.x
    assert len(series) == n + 1
    assert all(isinstance(t, xgtpsa.Tpsa) and t.order == 2 for t in series)
    with pytest.raises(AttributeError):
        mon.bogus


def test_tpsa_monitor_slot_guard():
    line = _demo_line()
    m = _map(order=2)
    mon = xtpsa.TpsaMonitor(1, m.descriptor)
    with pytest.raises(ValueError, match='slots'):
        line.track(m, turn_by_turn_monitor=mon)


def test_particles_monitor_true():
    line = _demo_line()
    m = _map(order=2)
    line.track(m, turn_by_turn_monitor=True)

    mon = line.record_last_track
    assert isinstance(mon, xt.ParticlesMonitor)
    recorded = np.array([float(getattr(mon, c)[0, 0]) for c in COORDS])
    xo.assert_allclose(recorded, [X0[c] for c in COORDS], rtol=0, atol=1e-14)


def test_placed_particles_monitor():
    """A ParticlesMonitor placed *in* the line records the map's orbit at its slot."""
    mon = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=1, particle_id_range=(0, 1))
    line = _line([('d0', xt.Drift(length=1.2)),
                  ('q', xt.Quadrupole(length=0.5, k1=0.08)),
                  ('b', xt.Bend(length=1.5, k0=0.008, angle=0.02)),
                  ('mon', mon),
                  ('d1', xt.Drift(length=0.7))])

    line.track(_map(order=2))
    recorded = np.array([float(getattr(mon, c)[0, 0]) for c in COORDS])

    # the same line without the monitor, tracked up to where the monitor sat
    plain = _line([('d0', xt.Drift(length=1.2)),
                   ('q', xt.Quadrupole(length=0.5, k1=0.08)),
                   ('b', xt.Bend(length=1.5, k0=0.008, angle=0.02)),
                   ('d1', xt.Drift(length=0.7))])
    up_to_monitor = _map(order=2)
    plain.track(up_to_monitor, ele_stop='d1')

    p = plain.build_particles(**X0)
    plain.track(p, ele_stop='d1')
    native = np.array([float(getattr(p, c)[0]) for c in COORDS])

    xo.assert_allclose(recorded, up_to_monitor.const_part, rtol=0, atol=1e-13)
    xo.assert_allclose(recorded, native, rtol=0, atol=1e-12)


def test_invalid_monitor():
    line = _demo_line()
    with pytest.raises(ValueError, match='invalid turn_by_turn_monitor'):
        line.track(_map(), turn_by_turn_monitor='NOPE')


# --------------------------------------------------------------------------- #
# Range and loss handling
# --------------------------------------------------------------------------- #

def test_range_errors():
    line = _demo_line()

    with pytest.raises(NotImplementedError, match='multi-turn'):
        line.track(_map(), num_turns=3)
    with pytest.raises(ValueError, match='ele_start'):
        line.track(_map(), ele_start=99)
    with pytest.raises(ValueError, match='both num_elements and ele_stop'):
        line.track(_map(), ele_stop=2, num_elements=1)
    with pytest.raises(NotImplementedError, match='wrap-around'):
        line.track(_map(), ele_start=2, ele_stop=1)


def test_map_loss_raises():
    line = _line([('d0', xt.Drift(length=1.2)),
                  ('ap', xt.LimitRectEllipse(max_x=1e-6, max_y=1e-6, a=1e-6, b=1e-6)),
                  ('d1', xt.Drift(length=0.9))])
    m = xtpsa.ParticlesTpsa(order=1, p0c=P0C, mass0=MASS0, x=1e-2)
    with pytest.raises(RuntimeError, match="lost at element index 1 \\('ap'\\)"):
        line.track(m)


# --------------------------------------------------------------------------- #
# The per-line ElementRefData cache
# --------------------------------------------------------------------------- #

def test_refdata_cache_released_with_line():
    backend = _backend()
    buffers_before = len(xo.context_default._buffers)
    entries_before = len(backend._refdata_cache)

    line = _demo_line()
    line.track(_map())
    assert len(backend._refdata_cache) == entries_before + 1

    del line
    gc.collect()
    # the cache holds the line weakly, so its ElementRefData (and buffer) go with it
    assert len(backend._refdata_cache) == entries_before
    assert len(xo.context_default._buffers) == buffers_before


def test_refdata_cache_reused_and_revalidated():
    backend = _backend()
    line = _demo_line()

    line.track(_map())
    entries = len(backend._refdata_cache)
    first = backend._refdata_cache[line][2]

    line.track(_map())
    assert len(backend._refdata_cache) == entries
    assert backend._refdata_cache[line][2] == first   # reused, not rebuilt

    # in-place parameter edits are seen without rebuilding (shared tracker buffer)
    before = _map()
    line.track(before)
    line.element_dict['q'].k1 = 0.5
    after = _map()
    line.track(after)
    assert not np.allclose(before.const_part, after.const_part)
    assert backend._refdata_cache[line][2] == first
    xo.assert_allclose(after.const_part, _native_orbit(line), rtol=0, atol=1e-14)

    # a stale element_names tuple invalidates the entry instead of being trusted
    names, erd, _ = backend._refdata_cache[line]
    backend._refdata_cache[line] = (('bogus',), erd, 0)
    rebuilt = _map()
    line.track(rebuilt)
    assert backend._refdata_cache[line][0] == names
    xo.assert_allclose(rebuilt.const_part, after.const_part, rtol=0, atol=0)

    # distinct lines get distinct entries
    other = _demo_line()
    other.track(_map())
    assert len(backend._refdata_cache) == entries + 1


# --------------------------------------------------------------------------- #
# Library loading and the compiled-bridge cache
# --------------------------------------------------------------------------- #

def test_lib_and_ffi_singletons():
    assert xgtpsa.lib() is xgtpsa.lib()
    assert xgtpsa.ffi() is xgtpsa.ffi()


def test_missing_lib_error(monkeypatch, tmp_path):
    monkeypatch.setenv("XGTPSA_LIB", str(tmp_path / "nope.so"))
    monkeypatch.setattr(xgtpsa._cffi, "_lib", None)
    with pytest.raises(RuntimeError, match='build.sh'):
        xgtpsa.lib()
    with pytest.raises(RuntimeError, match="build.sh"):
        xgtpsa.include_dir()
    assert not xgtpsa.have_core()


def test_bridge_sources_present():
    sources = _bridge_build._bridge_sources()
    assert all(os.path.exists(s) for s in sources)
    assert any(s.endswith('xt_bridge.cpp') for s in sources)
    assert any(os.sep + 'generated' + os.sep in s for s in sources)


def test_cache_key_is_content_addressed(monkeypatch, tmp_path):
    source = tmp_path / 'fake.hpp'
    source.write_text('one')
    monkeypatch.setattr(_bridge_build, "_bridge_sources", lambda: [str(source)])

    key = _bridge_build._bridge_cache_key("tpsa")
    assert key.startswith('bridge_tpsa_')
    assert _bridge_build._bridge_cache_key("tpsa") == key  # deterministic
    assert _bridge_build._bridge_cache_key("num") != key  # per flavor

    source.write_text('two')
    assert _bridge_build._bridge_cache_key("tpsa") != key  # content addressed


@pytest.mark.parametrize('flavor', ['num', 'tpsa'])
def test_bridge_lib_memoized(flavor):
    assert _bridge_build.bridge_lib(flavor) is _bridge_build.bridge_lib(flavor)


@pytest.mark.parametrize('flavor', ['num', 'tpsa'])
def test_bridge_lib_uses_disk_cache(monkeypatch, flavor):
    """With the in-process memo dropped, the module must load from the cached .so."""
    from xobjects.context_cpu import ContextCpu

    _bridge_build.bridge_lib(flavor)  # ensure it is built on disk
    monkeypatch.delitem(_bridge_build._bridge_modules, flavor)

    def no_compile(*args, **kwargs):
        raise AssertionError('rebuilt instead of using the cached .so')

    monkeypatch.setattr(ContextCpu, 'build_kernels', no_compile)

    kernels = _bridge_build.bridge_lib(flavor)
    assert f'xt_bridge_track_element_{flavor}' in kernels
    assert f'xt_bridge_track_line_{flavor}' in kernels


def test_bridge_lib_force_rebuilds(monkeypatch):
    from xobjects.context_cpu import ContextCpu, _so_for_module_name

    # Compile under a private module name and restore the memo afterwards, so that
    # under `pytest -n auto` this never rewrites the shared .so another worker is loading.
    cached = _bridge_build.bridge_lib("num")
    monkeypatch.setitem(_bridge_build._bridge_modules, "num", cached)
    module_name = 'bridge_num_forcetest_%d' % os.getpid()
    monkeypatch.setattr(_bridge_build, "_bridge_cache_key", lambda flavor: module_name)

    build_kernels = ContextCpu.build_kernels
    calls = []

    def spy(self, *args, **kwargs):
        calls.append(kwargs.get('module_name'))
        return build_kernels(self, *args, **kwargs)

    monkeypatch.setattr(ContextCpu, 'build_kernels', spy)

    try:
        kernels = _bridge_build.bridge_lib("num", force=True)
        assert calls == [module_name]      # force=True compiles, cache notwithstanding
        assert 'xt_bridge_track_element_num' in kernels
    finally:
        so = _so_for_module_name(module_name, _bridge_build._cache_dir())
        so.unlink(missing_ok=True)


def test_bridge_lib_bad_flavor():
    with pytest.raises(ValueError, match='flavor must be one of'):
        _bridge_build.bridge_lib("bogus")


def test_bridge_entry():
    fn, ffi = _bridge_build.bridge_entry("tpsa", "xt_bridge_track_line_tpsa")
    assert callable(fn)
    assert hasattr(ffi, 'cast')


# --- Knobs (parametric TPSA) --------------------------------------------- #

def _knob_line():
    """Toy line: knob-driven quads incl. a var->var chain (kqc = 2*klink)."""
    env = xt.Environment()
    env['kqa'] = 0.012
    env['kqb'] = -0.020
    env['klink'] = 0.003
    env['kqc'] = '2.0 * klink'
    return env.new_line(components=[
        env.new('mq1', xt.Quadrupole, length=1.0, k1='0.5*kqa + kqb'),
        env.new('mq2', xt.Quadrupole, length=1.0, k1='kqa'),
        env.new('mq3', xt.Quadrupole, length=1.0, k1='kqc'),
        env.new('d1', xt.Drift, length=2.0),
    ])


def test_knobs_target_enumeration():
    kn = xtpsa.Knobs(_knob_line(), ['kqa', 'kqb', 'klink'])
    assert len(kn) == 3
    assert kn._targets == [('mq1', 'k1'), ('mq2', 'k1'), ('mq3', 'k1')]


def test_knobs_bad_name():
    with pytest.raises(KeyError, match='not a line variable'):
        xtpsa.Knobs(_knob_line(), ['nope'])


def test_knobs_strength_jacobian_vs_fd():
    line = _knob_line()
    names = ['kqa', 'kqb', 'klink']
    kn = xtpsa.Knobs(line, names)
    sj = kn.strength_jacobian()          # self-binds, no descriptor needed

    def fd(elem, attr, knob, h=1e-6):
        v0 = line[knob]
        line[knob] = v0 + h
        hi = float(getattr(line.element_dict[elem], attr))
        line[knob] = v0 - h
        lo = float(getattr(line.element_dict[elem], attr))
        line[knob] = v0
        return (hi - lo) / (2 * h)

    expected = {('mq1', 'k1'): [0.5, 1.0, 0.0],
                ('mq2', 'k1'): [1.0, 0.0, 0.0],
                ('mq3', 'k1'): [0.0, 0.0, 2.0]}
    for t, grads in sj.items():
        assert np.allclose(grads, expected[t], atol=1e-12)
        assert np.allclose(grads, [fd(*t, n) for n in names], atol=1e-8)


def test_knobs_table_address_sanity_and_refresh():
    line = _knob_line()
    kn = xtpsa.Knobs(line, ['kqa', 'kqb', 'klink'])
    addrs, ptrs = kn.table()
    assert len(addrs) == len(ptrs) == 3
    # the recorded field address reads back the live strength
    for (e, a), addr in zip(kn._targets, addrs):
        read = xgtpsa.ffi().cast("double*", addr)[0]
        assert abs(read - float(getattr(line.element_dict[e], a))) < 1e-15

    # a knob change is picked up on the next table() (expansions rebuilt)
    line['kqa'] = 0.05
    kn.table()
    assert np.isclose(float(line.element_dict['mq2'].k1), 0.05)
    assert np.allclose(kn.strength_jacobian()[('mq2', 'k1')], [1.0, 0.0, 0.0], atol=1e-12)


def test_knobs_array_target_unsupported():
    env = xt.Environment()
    env['kk'] = 0.1
    line = env.new_line(components=[env.new('m', xt.Multipole, knl=[0, 'kk'])])
    with pytest.raises(NotImplementedError, match='array target'):
        xtpsa.Knobs(line, ['kk'])


def test_knobs_self_bind_matches_external_descriptor():
    """strength_jacobian is the same whether Knobs self-binds or a map bound it."""
    line = _knob_line()
    kn_self = xtpsa.Knobs(line, ['kqa', 'kqb'])
    self_sj = kn_self.strength_jacobian()

    kn_map = xtpsa.Knobs(line, ['kqa', 'kqb'])
    xtpsa.ParticlesTpsa(order=2, knobs=kn_map, p0c=P0C, mass0=MASS0)  # binds kn_map
    for t in kn_map._targets:
        assert np.allclose(kn_map.strength_jacobian()[t], self_sj[t], atol=1e-14)


def test_particles_tpsa_with_knobs_descriptor():
    line = _knob_line()
    kn = xtpsa.Knobs(line, ['kqa', 'kqb'], order=1)
    p = xtpsa.ParticlesTpsa(order=2, knobs=kn, p0c=P0C, mass0=MASS0, **X0)
    assert p.n_parameters == 2
    assert p.knob_names == ['kqa', 'kqb']
    assert p.descriptor.monomial_length == 8          # 6 coords + 2 params
    assert np.allclose(p.jacobian(), np.eye(6))       # identity map before tracking
    assert np.allclose(p.param_jacobian(), 0.0)       # no knob dependence yet


def test_parametric_track_matches_fd():
    """Route B end-to-end: one knobbed track gives d(coord)/d(knob) == central FD."""
    env = xt.Environment()
    env['kqf'] = 0.02
    env['kqd'] = -0.015
    env['ksx'] = 3.0
    line = env.new_line(components=[
        env.new('d0', xt.Drift, length=0.5),
        env.new('qf', xt.Quadrupole, length=1.0, k1='kqf'),
        env.new('d1', xt.Drift, length=1.2),
        env.new('sx', xt.Sextupole, length=0.3, k2='ksx'),
        env.new('d2', xt.Drift, length=1.2),
        env.new('qd', xt.Quadrupole, length=1.0, k1='kqd'),
        env.new('d3', xt.Drift, length=0.5),
    ])
    # deliberately NOT building the tracker here: the first track builds it and
    # relocates element buffers, which the knob-address table must account for.
    names = ['kqf', 'kqd', 'ksx']

    p = xtpsa.ParticlesTpsa(order=2, knobs=xtpsa.Knobs(line, names),
                            p0c=P0C, mass0=MASS0, **X0)
    line.track(p)
    pj = p.param_jacobian()                      # (6, 3)
    assert np.abs(pj).max() > 1e-6               # knob dependence is actually injected

    def orbit():
        q = xtpsa.ParticlesTpsa(order=1, p0c=P0C, mass0=MASS0, **X0)
        line.track(q)
        return q.const_part

    fd = np.zeros((6, len(names)))
    h = 1e-6
    for j, nm in enumerate(names):
        v0 = line[nm]
        line[nm] = v0 + h
        hi = orbit()
        line[nm] = v0 - h
        lo = orbit()
        line[nm] = v0
        fd[:, j] = (hi - lo) / (2 * h)
    assert np.allclose(pj, fd, atol=1e-8), np.abs(pj - fd).max()


# --------------------------------------------------------------------------- #
# Setters: set the const part (get0/set0), the Jacobian (get1/set1), any coeff
# --------------------------------------------------------------------------- #


def test_set_const_part_and_jacobian_round_trip():
    m = _map(order=3)
    orbit = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) * 1e-3
    m.set_const_part(orbit)
    xo.assert_allclose(m.const_part, orbit, rtol=0, atol=0)  # exact

    R = 0.1 * np.arange(36).reshape(6, 6) + np.eye(6)
    m.set_jacobian(R)
    xo.assert_allclose(m.jacobian(), R, rtol=0, atol=0)  # exact

    # setting the Jacobian leaves the const part untouched, and vice versa
    xo.assert_allclose(m.const_part, orbit, rtol=0, atol=0)
    m.set_const_part(np.zeros(6))
    xo.assert_allclose(m.jacobian(), R, rtol=0, atol=0)


def test_set_const_part_and_jacobian_shape_guards():
    m = _map(order=2)
    with pytest.raises(ValueError, match="length 6"):
        m.set_const_part(np.zeros(5))
    with pytest.raises(ValueError, match="6x6"):
        m.set_jacobian(np.zeros((6, 5)))


def test_set_coefficient():
    m = _map(order=3)
    m.set_coefficient("x", (2, 0, 0, 0, 0, 0), 0.777)
    assert m.coefficient("x", (2, 0, 0, 0, 0, 0)) == 0.777
    # index form selects the same output series
    m.set_coefficient(4, (0, 0, 0, 0, 0, 2), -1.5)  # zeta series, delta^2 term
    assert m.coefficient("zeta", (0, 0, 0, 0, 0, 2)) == -1.5

    # a malformed / beyond-order monomial raises rather than exit(1)-ing
    with pytest.raises(ValueError, match="invalid monomial"):
        m.set_coefficient("x", (0, 0, 0, 0, 0), 1.0)  # wrong length
    with pytest.raises(ValueError, match="invalid monomial"):
        m.set_coefficient("x", (3, 3, 0, 0, 0, 0), 1.0)  # order 6 > 3


def test_set_jacobian_leaves_parameter_columns():
    """set_jacobian writes the 6x6 variable block only; knob params stay untouched."""
    line = _knob_line()
    kn = xtpsa.Knobs(line, ["kqa", "kqb"], order=1)
    m = xtpsa.ParticlesTpsa(order=2, knobs=kn, p0c=P0C, mass0=MASS0, **X0)
    # seed a knob dependence in the order-1 param block via a raw coefficient
    m.set_coefficient("x", (0, 0, 0, 0, 0, 0, 1, 0), 0.25)  # d x / d kqa
    before = m.param_jacobian().copy()

    R = 0.1 * np.arange(36).reshape(6, 6) + np.eye(6)
    m.set_jacobian(R)
    xo.assert_allclose(m.jacobian(), R, rtol=0, atol=0)  # variable block set
    xo.assert_allclose(m.param_jacobian(), before, rtol=0, atol=0)  # params untouched
    assert m.param_jacobian()[0, 0] == 0.25


def test_set_jacobian_from_w_matrix():
    """seed the map's Jacobian from a Twiss W-matrix (_6d_w_matrix)."""
    from xtrack.twiss import _6d_w_matrix

    W = _6d_w_matrix(
        betx=12.0,
        bety=8.0,
        alfx=-1.5,
        alfy=0.7,
        bets=100.0,
        dx=1.2,
        dpx=0.03,
        dy=-0.4,
        dpy=0.01,
    )
    m = _map(order=2)
    m.set_jacobian(W)
    xo.assert_allclose(m.jacobian(), W, rtol=0, atol=0)
    xo.assert_allclose(m.jacobian()[0, 0], np.sqrt(12.0), rtol=1e-14, atol=0)

    # and from a real line.twiss() init row
    fodo = _line(
        [
            ("qf", xt.Quadrupole(length=0.5, k1=0.6)),
            ("d1", xt.Drift(length=1.0)),
            ("qd", xt.Quadrupole(length=0.5, k1=-0.6)),
            ("d2", xt.Drift(length=1.0)),
        ]
    )
    tw = fodo.twiss(method="4d")
    W2 = _6d_w_matrix(
        tw.betx[0],
        tw.bety[0],
        tw.alfx[0],
        tw.alfy[0],
        1.0,
        tw.dx[0],
        tw.dpx[0],
        tw.dy[0],
        tw.dpy[0],
    )
    m2 = _map(order=2)
    m2.set_jacobian(W2)
    xo.assert_allclose(m2.jacobian(), W2, rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# multi_element_monitor_at: full-map capture at only a few named positions,
# recorded in the single C track pass
# --------------------------------------------------------------------------- #


def _observe_line():
    return _line(
        [
            ("begin", xt.Marker()),
            ("d0", xt.Drift(length=1.2)),
            ("q1", xt.Quadrupole(length=0.5, k1=0.3)),
            ("ip8", xt.Marker()),
            ("b1", xt.Bend(length=2.0, k0=0.01, angle=0.02)),
            ("d1", xt.Drift(length=0.8)),
            ("end", xt.Marker()),
        ]
    )


def test_multi_element_monitor_matches_ebe():
    line = _observe_line()

    # EBE reference (full map at every position)
    ebe = _map(order=3)
    line.track(ebe, turn_by_turn_monitor="ONE_TURN_EBE")
    ebe_mon = line.record_last_track

    at = ["begin", "ip8", "b1", "end"]
    m = _map(order=3)
    line.track(m, multi_element_monitor_at=at)
    mon = line.record_multi_element_last_track

    assert isinstance(mon, xtpsa.TpsaMonitor)
    assert len(mon) == len(at)
    assert mon.obs_names == at  # given in ascending order here -> preserved

    names = list(line.element_names)
    index = {"begin": 0, "end": len(names)}
    for name in at:
        i = index.get(name, names.index(name))
        # full map (every monomial of every series) matches the EBE slot exactly
        for c in COORDS:
            assert mon.at(name).monomial_coeffs(c) == ebe_mon[i].monomial_coeffs(c)

    # 'begin' is the seed identity map; 'end' is the fully-tracked map
    xo.assert_allclose(
        mon.at("begin").const_part, [X0[c] for c in COORDS], rtol=0, atol=0
    )
    xo.assert_allclose(mon.at("begin").jacobian(), np.eye(6), rtol=0, atol=0)
    xo.assert_allclose(mon.at("end").const_part, m.const_part, rtol=0, atol=0)


def test_multi_element_monitor_out_of_order_and_indices():
    line = _observe_line()

    # positions out of order: slots fill ascending, obs_names records that order
    m = _map(order=2)
    line.track(m, multi_element_monitor_at=["end", "begin", "ip8"])
    mon = line.record_multi_element_last_track
    assert mon.obs_names == ["begin", "ip8", "end"]
    xo.assert_allclose(mon.at("begin").jacobian(), np.eye(6), rtol=0, atol=0)
    xo.assert_allclose(mon.at("end").const_part, m.const_part, rtol=0, atol=0)

    # integer / negative indices resolve like names (-1 -> the map after the line)
    m2 = _map(order=2)
    line.track(m2, multi_element_monitor_at=[0, 3, -1])
    mon2 = line.record_multi_element_last_track
    assert len(mon2) == 3
    xo.assert_allclose(mon2[0].jacobian(), np.eye(6), rtol=0, atol=0)


def test_multi_element_monitor_errors():
    line = _observe_line()
    with pytest.raises(ValueError, match="duplicate"):
        line.track(_map(order=2), multi_element_monitor_at=["ip8", "ip8"])
    # a monitor without named positions rejects .at()
    plain = xtpsa.TpsaMonitor(1, _map(order=2).descriptor)
    with pytest.raises(ValueError, match="no named positions"):
        plain.at("ip8")


def test_multi_element_monitor_records_parameters():
    """A parametric (knobbed) map is recorded with its parameter part, not just the 6 vars.

    ``mad_tpsa_copy`` snapshots the whole polynomial, so the knob columns (``param_jacobian``)
    and every parameter monomial survive into each slot and evolve through tracking.
    """
    line = _knob_line()
    line.build_tracker()
    kn = xtpsa.Knobs(line, ["kqa", "kqb"], order=1)

    def seeded():
        m = xtpsa.ParticlesTpsa(order=2, knobs=kn, p0c=P0C, mass0=MASS0, **X0)
        assert m.descriptor.monomial_length == 8  # 6 vars + 2 params
        m.set_coefficient("x", (0, 0, 0, 0, 0, 0, 1, 0), 0.5)  # d x  / d kqa
        m.set_coefficient("py", (0, 0, 0, 0, 0, 0, 0, 1), -0.3)  # d py / d kqb
        return m

    at = ["begin", "mq3", "end"]
    m = seeded()
    line.track(m, multi_element_monitor_at=at)
    mon = line.record_multi_element_last_track
    # slot maps carry the parameter dimension too
    assert mon.jacobian().shape[1:] == (6, 6)
    assert mon.at("begin").param_jacobian().shape == (6, 2)

    # 'begin', the injected knob dependence is recorded verbatim
    assert mon.at("begin").param_jacobian()[0, 0] == 0.5
    assert mon.at("begin").param_jacobian()[3, 1] == -0.3
    # 'end', the fully-tracked live map, and the params did evolve
    xo.assert_allclose(
        mon.at("end").param_jacobian(), m.param_jacobian(), rtol=0, atol=0
    )
    assert not np.allclose(mon.at("end").param_jacobian(), 0.0)

    # every parameter monomial matches a full ONE_TURN_EBE run at the same element
    ebe = seeded()
    line.track(ebe, turn_by_turn_monitor="ONE_TURN_EBE")
    ebe_mon = line.record_last_track
    i = line.element_names.index("mq3")
    for c in COORDS:
        assert mon.at("mq3").monomial_coeffs(c) == ebe_mon[i].monomial_coeffs(c)


# --------------------------------------------------------------------------- #
# TpsaOptics: optical functions (betx/bety/...) + knob derivatives read
# straight off a map's Jacobian (A-matrix).
# --------------------------------------------------------------------------- #


def test_optics_identity_roundtrip():
    """Reading optics off ``set_jacobian(_6d_w_matrix(...))`` returns the seed values."""
    W = _6d_w_matrix(3.0, 4.0, 0.7, -0.4, 1.0, 0.1, 0.02, -0.03, 0.05)
    m = _map(order=2)
    m.set_jacobian(W)
    o = m.optics()
    xo.assert_allclose(o.betx, 3.0, rtol=0, atol=1e-13)
    xo.assert_allclose(o.bety, 4.0, rtol=0, atol=1e-13)
    xo.assert_allclose(o.alfx, 0.7, rtol=0, atol=1e-13)
    xo.assert_allclose(o.alfy, -0.4, rtol=0, atol=1e-13)
    xo.assert_allclose([o.dx, o.dpx, o.dy, o.dpy], [0.1, 0.02, -0.03, 0.05],
                       rtol=0, atol=1e-13)
    assert set(o.to_dict()) == {"betx", "bety", "alfx", "alfy", "mux", "muy",
                                "dx", "dpx", "dy", "dpy"}


def test_optics_drift_propagation():
    """On-axis, optics after a drift follow the analytic beta(L) = b0 - 2 a0 L + g0 L^2."""
    L = 2.5
    line = _line([("d", xt.Drift(length=L))])
    betx0, alfx0 = 3.0, 0.7
    gamx0 = (1 + alfx0 ** 2) / betx0
    m = xtpsa.ParticlesTpsa(order=2, p0c=P0C, mass0=MASS0)  # on-axis identity
    m.set_jacobian(_6d_w_matrix(betx0, 4.0, alfx0, -0.4, 1.0, 0.0, 0.0, 0.0, 0.0))
    line.track(m)
    o = m.optics()
    xo.assert_allclose(o.betx, betx0 - 2 * alfx0 * L + gamx0 * L ** 2, rtol=0, atol=1e-12)
    xo.assert_allclose(o.alfx, alfx0 - gamx0 * L, rtol=0, atol=1e-12)


def test_optics_values_vs_twiss():
    """Optics off the tracked map match ``line.twiss`` at the same position."""
    line = _demo_line()
    init = dict(betx=3.0, bety=4.0, alfx=0.7, alfy=-0.4, dx=0.1, dpx=0.02, dy=0.0, dpy=0.0)
    tw = line.twiss(**init)
    m = _map(order=2, x=0, px=0, y=0, py=0, zeta=0, delta=0)
    m.set_jacobian(_6d_w_matrix(init["betx"], init["bety"], init["alfx"], init["alfy"],
                                1.0, init["dx"], init["dpx"], init["dy"], init["dpy"]))
    line.track(m, multi_element_monitor_at=["d1"])
    o = line.record_multi_element_last_track.at("d1").optics()
    i = list(line.element_names).index("d1")
    for name in ("betx", "bety", "alfx", "alfy", "dx", "mux"):
        xo.assert_allclose(getattr(o, name), getattr(tw, name)[i], rtol=1e-9, atol=1e-11)


def _param_map(A0, dA, order=2):
    """A knobbed map with Jacobian ``A0`` and injected ``d A(i,j)/d knob`` (``dA[(i,j)]``)."""
    line = _knob_line()
    line.build_tracker()
    kn = xtpsa.Knobs(line, ["kqa", "kqb"], order=1)
    m = xtpsa.ParticlesTpsa(order=order, knobs=kn, p0c=P0C, mass0=MASS0, **X0)
    m.set_jacobian(A0)
    nv = m.n_variables
    for (i, j), g in dA.items():
        for k, gk in enumerate(g):
            mono = [0] * (nv + len(g))
            mono[j] = 1
            mono[nv + k] = 1
            m.set_coefficient(COORDS[i], tuple(mono), gk)
    return m


def test_optics_knob_gradient():
    """d(optics)/d(knob) is the chain rule on the map's mixed coefficients."""
    A0 = _6d_w_matrix(3.0, 4.0, 0.7, -0.4, 1.0, 0.1, 0.02, 0.0, 0.0)
    dA = {(0, 0): [1.3, -0.5], (0, 1): [0.4, 0.9], (1, 0): [0.2, 0.1],
          (1, 1): [-0.7, 0.3], (0, 5): [0.05, -0.02]}
    o = _param_map(A0, dA).optics()

    # analytic: d betx = 2 A00 dA00 + 2 A01 dA01 ; d dx = dA05
    xo.assert_allclose(o.gradient("betx"),
                       2 * A0[0, 0] * np.array(dA[(0, 0)])
                       + 2 * A0[0, 1] * np.array(dA[(0, 1)]), rtol=0, atol=1e-13)
    xo.assert_allclose(o.gradient("dx"), dA[(0, 5)], rtol=0, atol=1e-13)

    # finite difference (along kqa) of betx built from A0 + h*dA
    def betx_at(h):
        Ah = A0.copy()
        for (i, j), g in dA.items():
            Ah[i, j] += h * g[0]
        mm = _map(order=1)
        mm.set_jacobian(Ah)
        return mm.optics().betx

    hh = 1e-6
    fd = (betx_at(hh) - betx_at(-hh)) / (2 * hh)
    xo.assert_allclose(o.gradient("betx")[0], fd, rtol=1e-6, atol=1e-8)


def test_optics_gradient_guards():
    """Gradients need knobs and order >= 2; values always work."""
    plain = _map(order=2)
    plain.set_jacobian(_6d_w_matrix(3.0, 4.0, 0.7, -0.4, 1.0, 0.1, 0.02, 0.0, 0.0))
    assert plain.optics().betx > 0                       # values fine without knobs
    with pytest.raises(ValueError, match="no knobs"):
        plain.optics().gradient("betx")

    line = _knob_line()
    line.build_tracker()
    kn = xtpsa.Knobs(line, ["kqa", "kqb"], order=1)
    m1 = xtpsa.ParticlesTpsa(order=1, knobs=kn, p0c=P0C, mass0=MASS0, **X0)
    with pytest.raises(ValueError, match="order >= 2"):
        m1.optics().gradient("betx")
    with pytest.raises(KeyError, match="unknown optical function"):
        _param_map(np.eye(6), {}).optics().gradient("nope")
