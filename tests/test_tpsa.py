import pathlib
import numpy as np
import pytest

import xobjects as xo
import madng_tpsa
import xtrack as xt
import xtrack.tpsa as xtpsa
from xobjects.test_helpers import allow_kernel_compilation

from xtrack._temp import lhc_match as lm
from xtrack.tpsa._knobs import KnobParameters
from xtrack.twiss.twiss_init import _6d_w_matrix

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()

P0C = 7e12
MASS0 = xt.PROTON_MASS_EV
COORDS = ("x", "px", "y", "py", "zeta", "delta")
# Off-axis seed with all six coordinates non-zero, so no residual hides in a zero.
X0 = dict(x=1e-4, px=1.5e-4, y=-1e-4, py=1e-4, zeta=1e-3, delta=2e-3)

_SUPPORTED_FLOAT_OR_TPSA_ELEMENTS = [
    ("b", xt.Bend, {"length": 1.0, "k0": 0.01}, "k0"),
    ("rb", xt.RBend, {"length_straight": 1.0, "k0": 0.01}, "k0"),
    ("q", xt.Quadrupole, {"length": 1.0, "k1": 0.1}, "k1"),
    ("s", xt.Sextupole, {"length": 1.0, "k2": 0.2}, "k2"),
    ("o", xt.Octupole, {"length": 1.0, "k3": 0.3}, "k3"),
    ("sol", xt.UniformSolenoid, {"length": 1.0, "ks": 0.01}, "ks"),
]


def _line(k1=0.1):
    elements = []
    element_names = []
    for name, element_cls, kwargs, field in _SUPPORTED_FLOAT_OR_TPSA_ELEMENTS:
        element_kwargs = dict(kwargs)
        if field == "k1":
            element_kwargs[field] = k1
        elements.append(element_cls(**element_kwargs))
        element_names.append(name)

    line = xt.Line(
        elements=elements,
        element_names=element_names,
    )
    line.particle_ref = xt.Particles(p0c=7e12, mass0=xt.PROTON_MASS_EV)
    line.build_tracker()
    return line


def _particle():
    return xt.Particles(
        x=1e-4,
        px=2e-5,
        y=0.0,
        py=0.0,
        zeta=0.0,
        delta=0.0,
        p0c=7e12,
        mass0=xt.PROTON_MASS_EV,
    )


def _map(order=1, descriptor=None):
    return xtpsa.ParticlesTpsa(
        order=order,
        descriptor=descriptor,
        x=1e-4, px=2e-5, y=0.0, py=0.0, zeta=0.0, delta=0.0,
        p0c=7e12,
        mass0=xt.PROTON_MASS_EV,
    )


@allow_kernel_compilation
def test_tpsa_tracking_error_marks_particle_lost():
    class IllegalTpsaOperation(xt.BeamElement):
        _extra_c_sources = [r"""
            #include "xtrack/headers/track.h"

            GPUFUN
            void IllegalTpsaOperation_track_local_particle(
                    IllegalTpsaOperationData el, LocalParticle* part0) {
                START_PER_PARTICLE_BLOCK(part0, part)
                    xt_num_t zero = 0.0;
                    LocalParticle_set_x(part, sqrt(zero));
                END_PER_PARTICLE_BLOCK;
            }
        """]

    line = xt.Line(elements=[IllegalTpsaOperation()])
    line.particle_ref = xt.Particles(p0c=P0C, mass0=MASS0)
    line.build_tracker(compile=False)
    particles = _map()

    with pytest.raises(RuntimeError, match="TPSA map lost"):
        line.track(particles)

    assert particles._xobject.state == -50


def test_tpsa_quadrupole_zero_strength_parameter_map():
    length = 2.0
    quadrupole = xt.Quadrupole(length=length, k1=0.0)
    line = xt.Line(elements=[quadrupole])
    line.particle_ref = xt.Particles(p0c=7e12, mass0=xt.PROTON_MASS_EV)
    descriptor = madng_tpsa.Descriptor(6, 1, num_params=1, param_order=1)
    quadrupole.k1 = descriptor.param(1, 0.0)
    line.build_tracker(compile=False)
    particles = _map(descriptor=descriptor)
    x0 = particles.const_part[0]
    px0 = particles.const_part[1]

    line.track(particles)

    # Expand the thick-quadrupole map about k1 = 0:
    # cos(sqrt(k1)L) = 1 - k1 L^2 / 2 and sinc(sqrt(k1)L) = 1 - k1 L^2 / 6.
    assert particles.sensitivity("x", 0) == pytest.approx(
        -0.5 * length**2 * x0 - length**3 * px0 / 6.0)
    assert particles.sensitivity("px", 0) == pytest.approx(
        -length * x0 - 0.5 * length**2 * px0)
    assert particles.sensitivity("zeta", 0) == pytest.approx(
        0.5 * length**2 * x0 * px0 + length**3 * px0**2 / 6.0)


def test_tpsa_quadrupole_zero_strength_parameter_element_track():
    length = 2.0
    element = xt.Quadrupole(length=length, k1=0.0)
    line = xt.Line(elements=[element])
    descriptor = madng_tpsa.Descriptor(6, 1, num_params=1, param_order=1)
    element.k1 = descriptor.param(1, 0.0)
    particles = _map(descriptor=descriptor)
    x0 = particles.const_part[0]
    px0 = particles.const_part[1]

    element.track(particles)

    assert particles.sensitivity("x", 0) == pytest.approx(
        -0.5 * length**2 * x0 - length**3 * px0 / 6.0)
    assert particles.sensitivity("px", 0) == pytest.approx(
        -length * x0 - 0.5 * length**2 * px0)
    assert particles.sensitivity("zeta", 0) == pytest.approx(
        0.5 * length**2 * x0 * px0 + length**3 * px0**2 / 6.0)


def test_particles_tpsa_uses_tracker_tpsa_config():
    line = _line()
    m = _map()
    line.track(m)

    assert line.tracker.config.XTRACK_TPSA_TRACK is True
    assert any(
        ("XTRACK_TPSA_TRACK", True) in key
        for key in line.tracker.track_kernel
    )


def test_particles_tpsa_requires_synrad_disabled():
    line = _line()
    line.config.XTRACK_MULTIPOLE_NO_SYNRAD = False
    m = _map()

    with pytest.raises(NotImplementedError, match="synchrotron radiation"):
        line.track(m)


def test_tpsa_line_track_matches_scalar_const_part():
    line_scalar = _line()
    part = _particle()
    line_scalar.track(part)

    line_tpsa = _line()
    m = _map()
    line_tpsa.track(m)

    assert np.allclose(
        m.const_part,
        [
            float(part.x[0]),
            float(part.px[0]),
            float(part.y[0]),
            float(part.py[0]),
            float(part.zeta[0]),
            float(part.delta[0]),
        ],
        rtol=0,
        atol=1e-15,
    )


def test_tpsa_element_track_matches_scalar_const_part():
    element_scalar = xt.Quadrupole(length=1.0, k1=0.1)
    part = _particle()
    element_scalar.track(part)

    element_tpsa = xt.Quadrupole(length=1.0, k1=0.1)
    m = _map()
    element_tpsa.track(m)

    assert np.allclose(
        m.const_part,
        [
            float(part.x[0]),
            float(part.px[0]),
            float(part.y[0]),
            float(part.py[0]),
            float(part.zeta[0]),
            float(part.delta[0]),
        ],
        rtol=0,
        atol=1e-15,
    )


def test_tpsa_multiturn_track_matches_scalar_const_part():
    line_scalar = _line()
    part = _particle()
    line_scalar.track(part, num_turns=2)

    line_tpsa = _line()
    m = _map()
    line_tpsa.track(m, num_turns=2)

    assert np.allclose(
        m.const_part,
        [
            float(part.x[0]),
            float(part.px[0]),
            float(part.y[0]),
            float(part.py[0]),
            float(part.zeta[0]),
            float(part.delta[0]),
        ],
        rtol=0,
        atol=1e-15,
    )


def test_tpsa_local_particle_freezing_and_turn_state():
    line = xt.Line(elements=[xt.Drift(length=1.0)])
    line.particle_ref = xt.Particles(p0c=7e12, mass0=xt.PROTON_MASS_EV)
    line.freeze_vars(["x"])
    line.build_tracker(use_prebuilt_kernels=False)
    m = _map()

    x_before = m.x.copy()
    line.track(m, num_turns=2)

    assert m.x == x_before
    assert m._xobject.at_turn == 2


def test_tpsa_reference_energy_change_matches_scalar():
    delta_p0c = 1e9
    line_scalar = xt.Line(elements=[xt.ReferenceEnergyIncrease(Delta_p0c=delta_p0c)])
    part = _particle()
    line_scalar.track(part)

    line_tpsa = xt.Line(elements=[xt.ReferenceEnergyIncrease(Delta_p0c=delta_p0c)])
    line_tpsa.build_tracker(use_prebuilt_kernels=False)
    m = _map()
    line_tpsa.track(m)

    assert m.p0c == pytest.approx(float(part.p0c[0]))
    assert m.delta.const_part == pytest.approx(float(part.delta[0]))


def test_scalar_track_tpsa_enabled_element_uses_const_part():
    line_scalar = _line(k1=0.125)
    part_scalar = _particle()
    line_scalar.track(part_scalar)

    line_tpsa_enabled = _line(k1=0.125)
    descriptor = madng_tpsa.Descriptor(6, 1, num_params=1, param_order=1)
    line_tpsa_enabled["q"].k1 = descriptor.param(1, 0.125)
    part_tpsa_enabled = _particle()
    line_tpsa_enabled.track(part_tpsa_enabled)

    assert np.allclose(
        [
            float(part_tpsa_enabled.x[0]),
            float(part_tpsa_enabled.px[0]),
            float(part_tpsa_enabled.y[0]),
            float(part_tpsa_enabled.py[0]),
            float(part_tpsa_enabled.zeta[0]),
            float(part_tpsa_enabled.delta[0]),
        ],
        [
            float(part_scalar.x[0]),
            float(part_scalar.px[0]),
            float(part_scalar.y[0]),
            float(part_scalar.py[0]),
            float(part_scalar.zeta[0]),
            float(part_scalar.delta[0]),
        ],
        rtol=0,
        atol=1e-15,
    )


def test_line_disable_tpsa_elements():
    line = _line(k1=0.125)
    descriptor = madng_tpsa.Descriptor(6, 1, num_params=1, param_order=1)
    line["q"].k1 = descriptor.param(1, 0.125)

    assert np.array_equal(line.attr['_tpsa_enabled'], [0, 0, 1, 0, 0, 0])

    line.disable_tpsa_elements()

    assert np.array_equal(line.attr['_tpsa_enabled'], np.zeros(6))
    assert line["q"].k1 == 0.125


def test_build_tracker_preserves_tpsa_enabled_elements_moved_to_common_buffer():
    ctx = xo.ContextCpu()
    buffer_a = ctx.new_buffer(capacity=1024)
    buffer_b = ctx.new_buffer(capacity=1024)

    q1 = xt.Quadrupole(length=1.0, k1=0.1, _buffer=buffer_a)
    q2 = xt.Quadrupole(length=1.0, k1=0.2, _buffer=buffer_b)
    descriptor = madng_tpsa.Descriptor(6, 1, num_params=1, param_order=1)
    k1_tpsa = descriptor.param(1, 0.2)
    q2.k1 = k1_tpsa
    q2_handles = q2._tpsa_handles

    line = xt.Line(elements=[q1, q2], element_names=["q1", "q2"])
    line.particle_ref = xt.Particles(p0c=7e12, mass0=xt.PROTON_MASS_EV)

    assert line._element_dict["q1"]._buffer is not line._element_dict["q2"]._buffer
    line.build_tracker(compile=False, use_prebuilt_kernels=False)

    q1_after = line._element_dict["q1"]
    q2_after = line._element_dict["q2"]
    assert q2_after is q2
    assert q2_after._tpsa_handles is q2_handles
    assert q1_after._buffer is q2_after._buffer
    assert q2_after._buffer is line.tracker._buffer
    assert q2_after._xobject._tpsa_enabled
    assert q2_after.k1 is k1_tpsa
    assert q2_after.k1.const_part == pytest.approx(0.2)

    scalar_line = xt.Line(
        elements=[
            xt.Quadrupole(length=1.0, k1=0.1),
            xt.Quadrupole(length=1.0, k1=0.2),
        ],
        element_names=["q1", "q2"],
    )
    scalar_line.particle_ref = xt.Particles(p0c=7e12, mass0=xt.PROTON_MASS_EV)
    scalar_line.build_tracker(use_prebuilt_kernels=False)

    p_ref = xt.Particles(
        x=1e-4, px=2e-5, y=0.0, py=0.0, zeta=0.0, delta=0.0,
        p0c=7e12, mass0=xt.PROTON_MASS_EV,
    )
    p_moved = p_ref.copy()
    scalar_line.track(p_ref)
    line.track(p_moved)

    assert np.allclose(
        [
            float(p_moved.x[0]),
            float(p_moved.px[0]),
            float(p_moved.y[0]),
            float(p_moved.py[0]),
            float(p_moved.zeta[0]),
            float(p_moved.delta[0]),
        ],
        [
            float(p_ref.x[0]),
            float(p_ref.px[0]),
            float(p_ref.y[0]),
            float(p_ref.py[0]),
            float(p_ref.zeta[0]),
            float(p_ref.delta[0]),
        ],
        rtol=0,
        atol=1e-15,
    )


def test_tpsa_enabled_element_copy_preserves_handles():
    q = xt.Quadrupole(length=1.0, k1=0.1)
    descriptor = madng_tpsa.Descriptor(6, 1, num_params=1, param_order=1)
    k1_tpsa = descriptor.param(1, 0.1)
    q.k1 = k1_tpsa

    q_copy = q.copy()

    assert q_copy is not q
    assert q_copy._xobject is not q._xobject
    assert q_copy._buffer is not q._buffer
    assert q_copy._xobject._tpsa_enabled
    assert q_copy._tpsa_descriptor is q._tpsa_descriptor
    assert q_copy._tpsa_handles is not q._tpsa_handles
    assert q_copy.k1 is k1_tpsa
    assert q_copy._field_raw_bits("k1") == q._field_raw_bits("k1")


def test_float_or_tpsa_field_assignment():
    line = _line()
    descriptor = madng_tpsa.Descriptor(6, 1, num_params=1, param_order=1)
    line["q"].k1 = descriptor.param(1, 0.1)

    assert line["q"]._tpsa_enabled
    assert line["q"].k1.const_part == pytest.approx(0.1)


@pytest.mark.parametrize("name, element_cls, kwargs, field", _SUPPORTED_FLOAT_OR_TPSA_ELEMENTS)
def test_supported_float_or_tpsa_elements_accept_tpsa_fields(
        name, element_cls, kwargs, field):
    element = element_cls(**kwargs)
    descriptor = madng_tpsa.Descriptor(6, 1, num_params=1, param_order=1)

    setattr(element, field, descriptor.param(1, kwargs[field]))

    assert element._tpsa_enabled
    assert getattr(element, field).const_part == pytest.approx(kwargs[field])


def test_tpsa_enabled_element_to_dict_is_rejected():
    line = _line()
    descriptor = madng_tpsa.Descriptor(6, 1, num_params=1, param_order=1)
    line["q"].k1 = descriptor.param(1, 0.1)

    with pytest.raises(NotImplementedError, match="Serializing TPSA-enabled"):
        line["q"].to_dict()


def test_tpsa_constructor_assignment_without_container_is_rejected():
    descriptor = madng_tpsa.Descriptor(6, 1, num_params=1, param_order=1)

    with pytest.raises(ValueError, match="without an owning container"):
        xt.Quadrupole(length=1.0, k1=descriptor.param(1, 0.1))


def test_parametric_element_field_tracks_with_shared_descriptor():
    line = _line()
    descriptor = madng_tpsa.Descriptor(6, 1, num_params=1, param_order=1)
    line["q"].k1 = descriptor.param(1, 0.1)

    m = _map(descriptor=descriptor)
    line.track(m)

    assert m.num_params == 1
    assert m.x.param_grad()[0] != 0


def test_particles_tpsa_rejects_descriptor_shape_mismatch():
    descriptor = madng_tpsa.Descriptor(5, 1)
    with pytest.raises(ValueError, match="6 variables"):
        _map(descriptor=descriptor)


def test_tpsa_match_optics():
    collider = xt.Environment.from_json(test_data_folder /
                    'hllhc15_thick/hllhc15_collider_thick.json')
    collider.vars.load(test_data_folder /
                    'hllhc15_thick/opt_round_150_1500.madx')

    line = collider.lhcb1
    tw0 = line.twiss()

    lm.set_var_limits_and_steps(collider)

    # Match with Xsuite Targets
    opt = line.match(
    solve=False,
    default_tol={None: 1e-8, 'betx': 1e-6, 'bety': 1e-6, 'alfx': 1e-6, 'alfy': 1e-6},
    start='s.ds.l8.b1', end='ip1',
    init=tw0, init_at=xt.START,
    vary=[
        # Only IR8 quadrupoles including DS
        xt.VaryList(['kq6.l8b1', 'kq7.l8b1', 'kq8.l8b1', 'kq9.l8b1', 'kq10.l8b1',
            'kqtl11.l8b1', 'kqt12.l8b1', 'kqt13.l8b1',
            'kq4.l8b1', 'kq5.l8b1', 'kq4.r8b1', 'kq5.r8b1',
            'kq6.r8b1', 'kq7.r8b1', 'kq8.r8b1', 'kq9.r8b1',
            'kq10.r8b1', 'kqtl11.r8b1', 'kqt12.r8b1', 'kqt13.r8b1'])],
    targets=[
        xt.TargetSet(at='ip8', tars=('betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx'), value=tw0, weight=1),
        xt.TargetSet(at='ip1', betx=0.15, bety=0.1, alfx=0, alfy=0, dx=0, dpx=0, weight=1),
        xt.TargetRelPhaseAdvance('mux', value = tw0['mux', 'ip1.l1'] - tw0['mux', 's.ds.l8.b1'], start='s.ds.l8.b1', end='ip1.l1', weight=1),
        xt.TargetRelPhaseAdvance('muy', value = tw0['muy', 'ip1.l1'] - tw0['muy', 's.ds.l8.b1'], start='s.ds.l8.b1', end='ip1.l1', weight=1),
    ],
    use_tpsa=True, tpsa_backend="madng_tpsa")

    # Assert that all variables are TPSAs
    for name in opt.actions[0].vary_names:
        assert isinstance(line.vars.val[name], madng_tpsa.tpsa.Tpsa)

    opt.step(30)

    assert opt._err.call_counter < 20
    assert len(opt.log()) < 10

    tw = line.twiss(init=tw0, start='s.ds.l8.b1', end='ip1')

    xo.assert_allclose(tw['betx', 'ip1'], 0.15, atol=1e-6, rtol=0)
    xo.assert_allclose(tw['bety', 'ip1'], 0.1, atol=1e-6, rtol=0)
    xo.assert_allclose(tw['alfx', 'ip1'], 0., atol=1e-6, rtol=0)
    xo.assert_allclose(tw['alfy', 'ip1'], 0., atol=1e-6, rtol=0)
    xo.assert_allclose(tw['dx', 'ip1'], 0., atol=1e-6, rtol=0)
    xo.assert_allclose(tw['dy', 'ip1'], 0., atol=1e-6, rtol=0)

    xo.assert_allclose(tw['betx', 'ip8'], tw0['betx', 'ip8'], atol=1e-6, rtol=0)
    xo.assert_allclose(tw['bety', 'ip8'], tw0['bety', 'ip8'], atol=1e-6, rtol=0)
    xo.assert_allclose(tw['alfx', 'ip8'], tw0['alfx', 'ip8'], atol=1e-6, rtol=0)
    xo.assert_allclose(tw['alfy', 'ip8'], tw0['alfy', 'ip8'], atol=1e-6, rtol=0)
    xo.assert_allclose(tw['dx', 'ip8'], tw0['dx', 'ip8'], atol=1e-6, rtol=0)
    xo.assert_allclose(tw['dy', 'ip8'], tw0['dy', 'ip8'], atol=1e-6, rtol=0)

    xo.assert_allclose(tw['mux', 'ip1.l1'] - tw['mux', 's.ds.l8.b1'], tw0['mux', 'ip1.l1'] - tw0['mux', 's.ds.l8.b1'], atol=1e-6, rtol=0)
    xo.assert_allclose(tw['muy', 'ip1.l1'] - tw['muy', 's.ds.l8.b1'], tw0['muy', 'ip1.l1'] - tw0['muy', 's.ds.l8.b1'], atol=1e-6, rtol=0)

    opt.reload(0)


    opt.actions[0].teardown()
    # Check for doubles in the variables after teardown
    assert opt.actions[0]._already_prepared is False
    for name in opt.actions[0].vary_names:
        assert isinstance(line.vars.val[name], float)

    # Match on full line without initial conditions
    opt = line.match(
    solve=False,
    default_tol={None: 1e-8, 'betx': 1e-6, 'bety': 1e-6, 'alfx': 1e-6, 'alfy': 1e-6},
    vary=[
        # Only IR8 quadrupoles including DS
        xt.VaryList(['kq6.l8b1', 'kq7.l8b1', 'kq8.l8b1', 'kq9.l8b1', 'kq10.l8b1',
            'kqtl11.l8b1', 'kqt12.l8b1', 'kqt13.l8b1',
            'kq4.l8b1', 'kq5.l8b1', 'kq4.r8b1', 'kq5.r8b1',
            'kq6.r8b1', 'kq7.r8b1', 'kq8.r8b1', 'kq9.r8b1',
            'kq10.r8b1', 'kqtl11.r8b1', 'kqt12.r8b1', 'kqt13.r8b1'])],
    targets=[
            xt.TargetSet(at='ip8', tars=('betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx'), value=tw0, weight=1),
            xt.TargetSet(at='ip1', betx=0.15, bety=0.1, alfx=0, alfy=0, dx=0, dpx=0, weight=1),
            xt.TargetRelPhaseAdvance('mux', value = tw0['mux', 'ip1.l1'] - tw0['mux', 's.ds.l8.b1'], start='s.ds.l8.b1', end='ip1.l1', weight=1),
            xt.TargetRelPhaseAdvance('muy', value = tw0['muy', 'ip1.l1'] - tw0['muy', 's.ds.l8.b1'], start='s.ds.l8.b1', end='ip1.l1', weight=1),
    ],
    use_tpsa=True, tpsa_backend="madng_tpsa")

    opt.step(30)

    assert opt._err.call_counter < 20
    assert len(opt.log()) < 10

    tw = line.twiss(init=tw0)

    xo.assert_allclose(tw['betx', 'ip1.l1'], 0.15, atol=1e-6, rtol=0)
    xo.assert_allclose(tw['bety', 'ip1.l1'], 0.1, atol=1e-6, rtol=0)
    xo.assert_allclose(tw['alfx', 'ip1.l1'], 0., atol=1e-6, rtol=0)
    xo.assert_allclose(tw['alfy', 'ip1.l1'], 0., atol=1e-6, rtol=0)
    xo.assert_allclose(tw['dx', 'ip1.l1'], 0., atol=1e-6, rtol=0)
    xo.assert_allclose(tw['dy', 'ip1.l1'], 0., atol=1e-6, rtol=0)

    xo.assert_allclose(tw['betx', 'ip8'], tw0['betx', 'ip8'], atol=1e-6, rtol=0)
    xo.assert_allclose(tw['bety', 'ip8'], tw0['bety', 'ip8'], atol=1e-6, rtol=0)
    xo.assert_allclose(tw['alfx', 'ip8'], tw0['alfx', 'ip8'], atol=1e-6, rtol=0)
    xo.assert_allclose(tw['alfy', 'ip8'], tw0['alfy', 'ip8'], atol=1e-6, rtol=0)
    xo.assert_allclose(tw['dx', 'ip8'], tw0['dx', 'ip8'], atol=1e-6, rtol=0)
    xo.assert_allclose(tw['dy', 'ip8'], tw0['dy', 'ip8'], atol=1e-6, rtol=0)

    xo.assert_allclose(tw['mux', 'ip1.l1'] - tw['mux', 's.ds.l8.b1'], tw0['mux', 'ip1.l1'] - tw0['mux', 's.ds.l8.b1'], atol=1e-6, rtol=0)
    xo.assert_allclose(tw['muy', 'ip1.l1'] - tw['muy', 's.ds.l8.b1'], tw0['muy', 'ip1.l1'] - tw0['muy', 's.ds.l8.b1'], atol=1e-6, rtol=0)

# Helpers for the map surface, optics and knob tests below

def _offaxis_map(order=2, descriptor=None, **coords):
    """A map seeded at X0 (or X0 with some coordinates overridden)."""
    return xtpsa.ParticlesTpsa(order=order, descriptor=descriptor, p0c=P0C,
                               mass0=MASS0, **{**X0, **coords})


def _demo_line():
    """Mixed thick line: drift, quad, exact drift, bend."""
    line = xt.Line(
        elements=[xt.Drift(length=1.2), xt.Quadrupole(length=0.5, k1=0.08),
                  xt.DriftExact(length=0.9),
                  xt.Bend(length=1.5, k0=0.008, angle=0.02),
                  xt.Drift(length=0.7)],
        element_names=["d0", "q", "d1", "b", "d2"])
    line.particle_ref = xt.Particles(p0c=P0C, mass0=MASS0, q0=1)
    line.build_tracker()
    return line


def _native_orbit(line, coords=None):
    """Native (doubles) tracked orbit of one particle through the whole line."""
    p = line.build_particles(**(coords or X0))
    line.track(p)
    return np.array([float(getattr(p, c)[0]) for c in COORDS])


def _fd_jacobian(line, h=1e-7):
    """Central-difference Jacobian of the native line map at X0."""
    jac = np.zeros((6, 6))
    for j, c in enumerate(COORDS):
        plus, minus = dict(X0), dict(X0)
        plus[c] += h
        minus[c] -= h
        jac[:, j] = (_native_orbit(line, plus) - _native_orbit(line, minus)) / (2 * h)
    return jac


def _knob_line():
    """Knob-driven quads, including a var->var chain (kqc = 2*klink)."""
    env = xt.Environment()
    env["kqa"] = 0.012
    env["kqb"] = -0.020
    env["klink"] = 0.003
    env["kqc"] = "2.0 * klink"
    line = env.new_line(components=[
        env.new("mq1", xt.Quadrupole, length=1.0, k1="0.5*kqa + kqb"),
        env.new("d0", xt.Drift, length=2.0),
        env.new("mq2", xt.Quadrupole, length=1.0, k1="kqa"),
        env.new("mq3", xt.Quadrupole, length=1.0, k1="kqc"),
    ])
    line.particle_ref = xt.Particles(p0c=P0C, mass0=MASS0)
    return line


def _nonlinear_knob_line():
    """A strength quadratic in the knob (k1 = kqa**2)."""
    env = xt.Environment()
    env["kqa"] = 0.012
    line = env.new_line(components=[
        env.new("mq1", xt.Quadrupole, length=1.0, k1="kqa * kqa"),
        env.new("d0", xt.Drift, length=2.0),
    ])
    line.particle_ref = xt.Particles(p0c=P0C, mass0=MASS0)
    return line


def _fodo_knob_line():
    """FODO-like line with knobbed quads and a knobbed sextupole."""
    env = xt.Environment()
    env["kqf"] = 0.02
    env["kqd"] = -0.015
    env["ksx"] = 3.0
    line = env.new_line(components=[
        env.new("d0", xt.Drift, length=0.5),
        env.new("qf", xt.Quadrupole, length=1.0, k1="kqf"),
        env.new("d1", xt.Drift, length=1.2),
        env.new("sx", xt.Sextupole, length=0.3, k2="ksx"),
        env.new("d2", xt.Drift, length=1.2),
        env.new("qd", xt.Quadrupole, length=1.0, k1="kqd"),
        env.new("d3", xt.Drift, length=0.5),
    ])
    line.particle_ref = xt.Particles(p0c=P0C, mass0=MASS0)
    return line


# ParticlesTpsa surface

def test_fresh_map_is_identity():
    m = _offaxis_map(order=3)
    xo.assert_allclose(m.const_part, [X0[c] for c in COORDS], rtol=0, atol=0)
    xo.assert_allclose(m.jacobian(), np.eye(6), rtol=0, atol=0)
    assert (m.order, m.num_vars, m.num_params) == (3, 6, 0)


def test_maps_of_the_same_order_share_a_descriptor():
    a, b, c = _offaxis_map(order=2), _offaxis_map(order=2), _offaxis_map(order=3)
    assert a.descriptor is b.descriptor
    assert a.descriptor is not c.descriptor
    assert all(s.descriptor is a.descriptor for s in a.coords)


def test_getattr_and_to_particles():
    m = _offaxis_map(order=2)
    assert isinstance(m.x, madng_tpsa.Tpsa)
    assert isinstance(m.beta0, float)
    with pytest.raises(AttributeError):
        m.bogus

    p = m.to_particles()
    assert isinstance(p, xt.Particles)
    for c in COORDS:
        assert float(getattr(p, c)[0]) == X0[c]


@pytest.mark.parametrize("kwargs", [
    dict(p0c=P0C, mass0=MASS0, delta=2e-3),
    dict(energy0=P0C, mass0=MASS0),
    dict(gamma0=7460.0, mass0=MASS0),
])
def test_reference_algebra_matches_particles(kwargs):
    m = xtpsa.ParticlesTpsa(order=2, **kwargs)
    p = xt.Particles(**kwargs)
    for var in ("q0", "mass0", "beta0", "gamma0", "p0c", "chi", "charge_ratio"):
        expected = float(np.asarray(getattr(p, var)).reshape(-1)[0])
        xo.assert_allclose(getattr(m, var), expected, rtol=1e-12, atol=0)


def test_from_coords_view_shares_series():
    m = _offaxis_map(order=2)
    view = xtpsa.ParticlesTpsa._from_coords(m.coords)
    assert view.coords[0] is m.coords[0]     # shared series, not copies
    assert view.order == 2
    xo.assert_allclose(view.const_part, m.const_part, rtol=0, atol=0)
    with pytest.raises(AttributeError, match="no reference particle"):
        view.beta0


def test_particles_tpsa_rejects_descriptor_order_mismatch():
    with pytest.raises(ValueError, match="descriptor is order 3, map asks for 2"):
        _offaxis_map(order=2, descriptor=madng_tpsa.Descriptor(6, 3))


def test_particles_tpsa_rejects_vector_coordinates():
    with pytest.raises(ValueError, match="single map"):
        xtpsa.ParticlesTpsa(order=1, x=[1e-4, 2e-4], p0c=P0C, mass0=MASS0)


def test_order_truncation_integrity():
    """A map carries terms only up to its own order, and shared ones are bit-identical."""
    element = xt.Sextupole(length=0.3, k2=50.0)
    m2, m3 = _offaxis_map(order=2), _offaxis_map(order=3)
    element.track(m2)
    element.track(m3)

    low = m2.monomial_coeffs("px")
    high = m3.monomial_coeffs("px")
    assert max(sum(mono) for mono in low) <= 2
    third = {mono: c for mono, c in high.items() if sum(mono) == 3}
    assert third and all(c != 0.0 for c in third.values())
    for mono, coeff in low.items():
        assert high[mono] == coeff


def test_coefficient_and_set_coefficient():
    m = _offaxis_map(order=3)
    m.set_coefficient("x", (2, 0, 0, 0, 0, 0), 0.777)
    assert m.coefficient("x", (2, 0, 0, 0, 0, 0)) == 0.777
    assert m.coefficient(0, (2, 0, 0, 0, 0, 0)) == 0.777   # index selects the same series
    m.set_coefficient(4, (0, 0, 0, 0, 0, 2), -1.5)         # zeta series, delta^2 term
    assert m.coefficient("zeta", (0, 0, 0, 0, 0, 2)) == -1.5

    assert set(m.monomial_coeffs()) == set(COORDS)
    assert m.monomial_coeffs("x") == m.x.monomial_coeffs()


def test_coefficient_rejects_invalid_monomials():
    """A malformed or beyond-order monomial raises instead of GTPSA exit(1)-ing."""
    m = _offaxis_map(order=3)
    with pytest.raises(ValueError, match="invalid monomial"):
        m.coefficient("x", (0, 0, 0, 0, 0))         # wrong length
    with pytest.raises(ValueError, match="invalid monomial"):
        m.coefficient("x", (3, 3, 0, 0, 0, 0))      # total order 6 > 3
    with pytest.raises(ValueError, match="invalid monomial"):
        m.set_coefficient("x", (3, 3, 0, 0, 0, 0), 1.0)


# Tracking a map against native tracking

def test_track_line_const_part_and_jacobian_vs_native():
    line = _demo_line()
    m = _offaxis_map(order=2)
    line.track(m)
    xo.assert_allclose(m.const_part, _native_orbit(line), rtol=0, atol=1e-14)
    xo.assert_allclose(m.jacobian(), _fd_jacobian(line), rtol=0, atol=1e-7)


def test_track_line_partial_range():
    line = _demo_line()

    by_name = _offaxis_map(order=2)
    line.track(by_name, ele_stop="d1")
    by_count = _offaxis_map(order=2)
    line.track(by_count, num_elements=2)
    xo.assert_allclose(by_name.const_part, by_count.const_part, rtol=0, atol=0)

    from_name = _offaxis_map(order=2)
    line.track(from_name, ele_start="q", ele_stop="b")
    from_index = _offaxis_map(order=2)
    line.track(from_index, ele_start=1, ele_stop=3)
    xo.assert_allclose(from_name.const_part, from_index.const_part, rtol=0, atol=0)

    full = _offaxis_map(order=2)
    line.track(full)
    assert not np.allclose(full.const_part, by_name.const_part)


def test_track_range_and_argument_errors():
    line = _demo_line()
    with pytest.raises(ValueError, match="Cannot use both num_elements and ele_stop"):
        line.track(_offaxis_map(), ele_stop=2, num_elements=1)
    with pytest.raises(TypeError, match="unsupported TPSA tracking arguments"):
        line.track(_offaxis_map(), bogus_kwarg=1)


def test_track_partial_range_multiturn_matches_scalar_semantics():
    line = _demo_line()
    multi_turn = _offaxis_map()
    line.track(multi_turn, ele_start=1, ele_stop=3, num_turns=3)

    line_repeated = _demo_line()
    expected = _offaxis_map()
    line_repeated.track(expected, ele_start=1)
    line_repeated.track(expected)
    line_repeated.track(expected, ele_stop=3)

    scalar_line = _demo_line()
    scalar_particle = scalar_line.build_particles(**X0)
    scalar_line.track(scalar_particle, ele_start=1, ele_stop=3, num_turns=3)

    xo.assert_allclose(multi_turn.const_part, expected.const_part, rtol=0, atol=0)
    xo.assert_allclose(multi_turn.jacobian(), expected.jacobian(), rtol=0, atol=0)
    xo.assert_allclose(
        multi_turn.const_part,
        [float(getattr(scalar_particle, cc)[0]) for cc in COORDS],
        rtol=0,
        atol=1e-15,
    )
    assert (
        multi_turn._xobject.at_turn
        == expected._xobject.at_turn
        == scalar_particle.at_turn[0]
    )
    assert (
        multi_turn._xobject.at_element
        == expected._xobject.at_element
        == scalar_particle.at_element[0]
    )


@pytest.mark.parametrize("kwargs, match", [
    (dict(freeze_longitudinal=True), "freeze_longitudinal"),
    (dict(backtrack=True), "backtracking"),
    (dict(with_progress=True), "progress"),
    (dict(turn_by_turn_monitor=True), "turn-by-turn"),
])
def test_track_unwired_features_raise(kwargs, match):
    line = _demo_line()
    with pytest.raises(NotImplementedError, match=match):
        line.track(_offaxis_map(), **kwargs)


def test_map_loss_raises():
    line = xt.Line(
        elements=[xt.Drift(length=1.2),
                  xt.LimitRectEllipse(max_x=1e-6, max_y=1e-6, a=1e-6, b=1e-6),
                  xt.Drift(length=0.9)],
        element_names=["d0", "ap", "d1"])
    line.particle_ref = xt.Particles(p0c=P0C, mass0=MASS0)
    line.build_tracker()
    m = xtpsa.ParticlesTpsa(order=1, p0c=P0C, mass0=MASS0, x=1e-2)
    with pytest.raises(RuntimeError, match=r"lost at element index 1 \('ap'\)"):
        line.track(m)


def test_tpsa_enabled_element_rejects_scalar_element_track():
    line = _line()
    descriptor = madng_tpsa.Descriptor(6, 1, num_params=1, param_order=1)
    line["q"].k1 = descriptor.param(1, 0.1)
    with pytest.raises(RuntimeError, match="Cannot track normal Particles"):
        line.element_dict["q"].track(_particle())


# Setters: const part (get0/set0), Jacobian (get1/set1), single coefficients

def test_set_const_part_and_jacobian_round_trip():
    m = _offaxis_map(order=3)
    orbit = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) * 1e-3
    m.set_const_part(orbit)
    xo.assert_allclose(m.const_part, orbit, rtol=0, atol=0)

    R = 0.1 * np.arange(36).reshape(6, 6) + np.eye(6)
    m.set_jacobian(R)
    xo.assert_allclose(m.jacobian(), R, rtol=0, atol=0)

    # the two setters do not disturb each other
    xo.assert_allclose(m.const_part, orbit, rtol=0, atol=0)
    m.set_const_part(np.zeros(6))
    xo.assert_allclose(m.jacobian(), R, rtol=0, atol=0)


def test_set_const_part_and_jacobian_shape_guards():
    m = _offaxis_map(order=2)
    with pytest.raises(ValueError, match="length 6"):
        m.set_const_part(np.zeros(5))
    with pytest.raises(ValueError, match="6x6"):
        m.set_jacobian(np.zeros((6, 5)))


def test_set_jacobian_leaves_parameter_columns():
    descriptor = madng_tpsa.Descriptor(6, 2, params=["kqa", "kqb"], param_order=1)
    m = _offaxis_map(order=2, descriptor=descriptor)
    m.set_coefficient("x", (0, 0, 0, 0, 0, 0, 1, 0), 0.25)   # d x / d kqa
    before = m.param_jacobian().copy()

    R = 0.1 * np.arange(36).reshape(6, 6) + np.eye(6)
    m.set_jacobian(R)
    xo.assert_allclose(m.jacobian(), R, rtol=0, atol=0)
    xo.assert_allclose(m.param_jacobian(), before, rtol=0, atol=0)
    assert m.sensitivity("x", 0) == 0.25
    with pytest.raises(TypeError, match="does not store parameter names"):
        m.sensitivity("x", "kqa")


def test_set_jacobian_from_w_matrix():
    W = _6d_w_matrix(betx=12.0, bety=8.0, alfx=-1.5, alfy=0.7, bets=100.0,
                     dx=1.2, dpx=0.03, dy=-0.4, dpy=0.01)
    m = _offaxis_map(order=2)
    m.set_jacobian(W)
    xo.assert_allclose(m.jacobian(), W, rtol=0, atol=0)
    xo.assert_allclose(m.jacobian()[0, 0], np.sqrt(12.0), rtol=1e-14, atol=0)


# MultiElementMonitor recording full maps in the single TPSA track pass

def test_multi_element_monitor_records_maps():
    line = _demo_line()
    m = _offaxis_map(order=2)
    line.track(m, multi_element_monitor_at=["q", "b"])
    mon = line.tracker.record_multi_element_last_track

    assert isinstance(mon, xt.MultiElementMonitor)
    assert list(mon.obs_names) == ["q", "b"]
    assert len(mon) == 2
    assert mon.map_jacobian().shape == (2, 6, 6)

    # a slot is recorded on entry, so it equals a track that stops there
    for name in ("q", "b"):
        up_to = _offaxis_map(order=2)
        line.track(up_to, ele_stop=name)
        recorded = mon.map_at(name)
        xo.assert_allclose(recorded.const_part, up_to.const_part, rtol=0, atol=0)
        xo.assert_allclose(recorded.jacobian(), up_to.jacobian(), rtol=0, atol=0)
        for c in COORDS:                       # the whole polynomial, not just order 1
            assert recorded.monomial_coeffs(c) == up_to.monomial_coeffs(c)
        # the doubles buffer holds the same orbit
        xo.assert_allclose(mon.get("x", name)[0, 0], up_to.const_part[0],
                           rtol=0, atol=0)


def test_multi_element_monitor_slot_order_and_lookup():
    line = _demo_line()
    m = _offaxis_map(order=2)
    line.track(m, multi_element_monitor_at=["b", "q"])   # not in line order
    mon = line.tracker.record_multi_element_last_track

    # slots follow the requested order, so slot 0 is 'b' even though 'q' comes first
    assert list(mon.obs_names) == ["b", "q"]
    xo.assert_allclose(mon.map_at(0).const_part, mon.map_at("b").const_part,
                       rtol=0, atol=0)
    xo.assert_allclose(mon.map_at(1).const_part, mon.map_at("q").const_part,
                       rtol=0, atol=0)
    # and 'q' sits earlier in the line, so its slot is the shorter track
    up_to_q = _offaxis_map(order=2)
    line.track(up_to_q, ele_stop="q")
    xo.assert_allclose(mon.map_at("q").const_part, up_to_q.const_part,
                       rtol=0, atol=0)
    with pytest.raises(KeyError, match="not a recorded location"):
        mon.map_at("nope")


def test_multi_element_monitor_records_every_turn():
    line = _demo_line()
    m = _offaxis_map(order=2)
    line.track(m, num_turns=2, multi_element_monitor_at=["q"])
    mon = line.tracker.record_multi_element_last_track

    first = _offaxis_map(order=2)
    line.track(first, ele_stop="q")
    xo.assert_allclose(mon.map_at("q", turn=0).const_part, first.const_part,
                       rtol=0, atol=0)

    second = _offaxis_map(order=2)
    line.track(second)                 # one full turn ...
    line.track(second, ele_stop="q")   # ... then up to q again
    xo.assert_allclose(mon.map_at("q", turn=1).const_part, second.const_part,
                       rtol=0, atol=0)


def test_multi_element_monitor_without_maps_rejects_map_at():
    line = _demo_line()
    line.track(_particle(), multi_element_monitor_at=["q"])
    mon = line.tracker.record_multi_element_last_track
    with pytest.raises(AttributeError, match="only holds doubles"):
        mon.map_at("q")


def test_multi_element_monitor_records_parameters():
    """A parametric map is recorded whole, so the knob columns survive into each slot."""
    line = _fodo_knob_line()
    names = ["kqf", "kqd", "ksx"]
    descriptor = madng_tpsa.Descriptor(6, 2, params=names, param_order=1)
    knobs = KnobParameters(line, names, descriptor)
    knobs.apply()

    m = _offaxis_map(order=2, descriptor=descriptor)
    line.track(m, multi_element_monitor_at=["sx", "d3"])
    mon = line.tracker.record_multi_element_last_track

    assert mon.map_at("sx").param_jacobian().shape == (6, 3)
    # 'd3' is the last element: its slot is the map at the end of the line minus d3
    up_to = _offaxis_map(order=2, descriptor=descriptor)
    line.track(up_to, ele_stop="d3")
    xo.assert_allclose(mon.map_at("d3").param_jacobian(), up_to.param_jacobian(),
                       rtol=0, atol=0)
    assert np.abs(mon.map_at("sx").param_jacobian()).max() > 1e-6
    knobs.teardown()


# MultiElementMonitor recording selected map coefficients instead of full maps

def _coefficient_reference(line, obs_names, monomial, coords, turn=0):
    """The same coefficients read off the full recorded maps."""
    ref_map = _offaxis_map(order=2)
    line.track(ref_map, num_turns=turn + 1, multi_element_monitor_at=obs_names)
    ref = line.tracker.record_multi_element_last_track
    return np.array([[ref.map_at(name, turn=turn).coefficient(c, monomial)
                      for c in coords] for name in obs_names])


def test_monitor_monomials_match_the_full_maps():
    line = _demo_line()
    obs = ["q", "b", "d2"]
    monomials = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2]], dtype=np.uint8)
    line.track(_offaxis_map(order=2), multi_element_monitor_at=obs,
               monitor_monomials=monomials)
    mon = line.tracker.record_multi_element_last_track

    assert mon.coefficients.shape == (1, len(obs), 2 * len(COORDS))
    assert mon._map_series is None                  # no full maps allocated
    for monomial in monomials:
        monomial = tuple(int(o) for o in monomial)
        xo.assert_allclose(
            mon.coefficient(monomial)[0],
            _coefficient_reference(line, obs, monomial, COORDS),
            rtol=0, atol=0)


def test_monitor_monomials_dict_selects_coordinates():
    line = _demo_line()
    obs = ["q", "d2"]
    monomial = (0, 0, 0, 0, 0, 2)
    line.track(_offaxis_map(order=2), multi_element_monitor_at=obs,
               monitor_monomials={monomial: ("x", "zeta")})
    mon = line.tracker.record_multi_element_last_track

    assert mon.monomial_slots == [(monomial, "x"), (monomial, "zeta")]
    xo.assert_allclose(mon.coefficient(monomial)[0],
                       _coefficient_reference(line, obs, monomial, ["x", "zeta"]),
                       rtol=0, atol=0)
    with pytest.raises(KeyError, match="was not recorded for"):
        mon.coefficient(monomial, coord="py")


def test_monitor_monomials_axes_and_turns():
    line = _demo_line()
    obs = ["q", "b"]
    monomial = (1, 0, 0, 0, 0, 0)
    line.track(_offaxis_map(order=2), num_turns=3, multi_element_monitor_at=obs,
               monitor_monomials={monomial: COORDS})
    mon = line.tracker.record_multi_element_last_track

    assert mon.coefficient(monomial).shape == (3, 2, 6)
    assert mon.coefficient(monomial, coord="x").shape == (3, 2)
    assert mon.coefficient(monomial, coord="x", obs_name="b").shape == (3,)
    assert np.shape(mon.coefficient(monomial, coord="x", obs_name="b", turn=2)) == ()

    for turn in range(3):
        xo.assert_allclose(
            mon.coefficient(monomial)[turn],
            _coefficient_reference(line, obs, monomial, COORDS, turn=turn),
            rtol=0, atol=0)
    with pytest.raises(IndexError, match="not recorded"):
        mon.coefficient(monomial, turn=3)


def test_monitor_monomials_leave_the_tracked_map_untouched():
    line = _demo_line()
    m = _offaxis_map(order=2)
    line.track(m, multi_element_monitor_at=["q"],
               monitor_monomials=np.array([[1, 0, 0, 0, 0, 0]], dtype=np.uint8))
    plain = _offaxis_map(order=2)
    line.track(plain)
    for c in COORDS:
        assert m.monomial_coeffs(c) == plain.monomial_coeffs(c)


def test_monitor_monomials_rejects_bad_requests():
    line = _demo_line()
    monomials = np.array([[1, 0, 0, 0, 0, 0]], dtype=np.uint8)
    with pytest.raises(ValueError, match="beyond the order"):
        line.track(_offaxis_map(order=2), multi_element_monitor_at=["q"],
                   monitor_monomials={(0, 0, 0, 0, 0, 3): "x"})
    with pytest.raises(ValueError, match="expected length 6"):
        line.track(_offaxis_map(order=2), multi_element_monitor_at=["q"],
                   monitor_monomials=np.array([[1, 0, 0, 0, 0]], dtype=np.uint8))
    with pytest.raises(ValueError, match="not an output coordinate"):
        line.track(_offaxis_map(order=2), multi_element_monitor_at=["q"],
                   monitor_monomials={(1, 0, 0, 0, 0, 0): "s"})
    with pytest.raises(ValueError, match="needs .multi_element_monitor_at."):
        line.track(_offaxis_map(order=2), monitor_monomials=monomials)
    with pytest.raises(ValueError, match="needs TPSA tracking"):
        line.track(_particle(), multi_element_monitor_at=["q"],
                   monitor_monomials=monomials)


def test_monitor_monomials_constant_part_is_the_scalar_particle():
    """The recorded slot is the same particle, at the same place and turn, as a
    doubles track through the same monitor."""
    line = _demo_line()
    obs = ["q", "b", "d2"]
    constant = (0,) * len(COORDS)

    p = line.build_particles(**X0)
    line.track(p, num_turns=3, multi_element_monitor_at=obs)
    scalar = line.tracker.record_multi_element_last_track

    line.track(_offaxis_map(order=2), num_turns=3, multi_element_monitor_at=obs,
               monitor_monomials={constant: COORDS})
    mon = line.tracker.record_multi_element_last_track

    for turn in range(3):
        for name in obs:
            recorded = mon.coefficient(constant, obs_name=name, turn=turn)
            native = [np.ravel(scalar.get(c, name, turn=turn))[0] for c in COORDS]
            xo.assert_allclose(recorded, native, rtol=0, atol=1e-14)
            # the monitor's own doubles buffer is written by the same C block from
            # the same series, but indexed (turn, particle, coord, location)
            own = [np.ravel(mon.get(c, name, turn=turn))[0] for c in COORDS]
            xo.assert_allclose(recorded, own, rtol=0, atol=0)

    # a swapped turn or location index cannot pass unnoticed
    orbit = mon.coefficient(constant, coord="x")
    assert len(set(orbit.ravel())) == orbit.size


def test_monitor_monomials_first_order_matches_scalar_differences():
    """Order-1 slots are the derivatives of that same scalar particle."""
    line = _demo_line()
    obs = ["q", "d2"]
    monomials = np.eye(len(COORDS), dtype=np.uint8)
    h = 1e-7

    displaced = {c: [] for c in COORDS}
    for varied in COORDS:                       # two particles per coordinate
        for sign in (+1, -1):
            for c in COORDS:
                displaced[c].append(X0[c] + sign * h * (c == varied))
    p = line.build_particles(**{c: np.array(v) for c, v in displaced.items()})
    line.track(p, num_turns=2, multi_element_monitor_at=obs)
    scalar = line.tracker.record_multi_element_last_track

    line.track(_offaxis_map(order=2), num_turns=2, multi_element_monitor_at=obs,
               monitor_monomials=monomials)
    mon = line.tracker.record_multi_element_last_track

    for turn in range(2):
        for name in obs:
            fd = np.zeros((len(COORDS), len(COORDS)))
            for i, ci in enumerate(COORDS):
                column = np.ravel(scalar.get(ci, name, turn=turn))
                for j in range(len(COORDS)):
                    fd[i, j] = (column[2 * j] - column[2 * j + 1]) / (2 * h)
            recorded = np.array([mon.coefficient(tuple(int(o) for o in monomial),
                                                 obs_name=name, turn=turn)
                                 for monomial in monomials]).T
            xo.assert_allclose(recorded, fd, rtol=0, atol=1e-8)


def test_monitor_full_maps_reject_coefficient():
    line = _demo_line()
    line.track(_offaxis_map(order=2), multi_element_monitor_at=["q"])
    mon = line.tracker.record_multi_element_last_track
    with pytest.raises(AttributeError, match="holds full TPSA maps"):
        mon.coefficient((1, 0, 0, 0, 0, 0))


# TpsaOptics

def test_optics_round_trip_from_w_matrix():
    W = _6d_w_matrix(3.0, 4.0, 0.7, -0.4, 1.0, 0.1, 0.02, -0.03, 0.05)
    m = _offaxis_map(order=2)
    m.set_jacobian(W)
    o = m.optics()
    xo.assert_allclose([o.betx, o.bety, o.alfx, o.alfy], [3.0, 4.0, 0.7, -0.4],
                       rtol=0, atol=1e-13)
    xo.assert_allclose([o.dx, o.dpx, o.dy, o.dpy], [0.1, 0.02, -0.03, 0.05],
                       rtol=0, atol=1e-13)
    assert set(o.to_dict()) == {"betx", "bety", "alfx", "alfy", "mux", "muy",
                                "dx", "dpx", "dy", "dpy"}


def test_optics_drift_propagation():
    """On-axis, beta after a drift follows beta(L) = b0 - 2 a0 L + g0 L^2."""
    L = 2.5
    line = xt.Line(elements=[xt.Drift(length=L)], element_names=["d"])
    line.particle_ref = xt.Particles(p0c=P0C, mass0=MASS0)
    line.build_tracker()

    betx0, alfx0 = 3.0, 0.7
    gamx0 = (1 + alfx0 ** 2) / betx0
    m = xtpsa.ParticlesTpsa(order=2, p0c=P0C, mass0=MASS0)
    m.set_jacobian(_6d_w_matrix(betx0, 4.0, alfx0, -0.4, 1.0, 0.0, 0.0, 0.0, 0.0))
    line.track(m)

    o = m.optics()
    xo.assert_allclose(o.betx, betx0 - 2 * alfx0 * L + gamx0 * L ** 2,
                       rtol=0, atol=1e-12)
    xo.assert_allclose(o.alfx, alfx0 - gamx0 * L, rtol=0, atol=1e-12)


def test_optics_values_vs_twiss():
    """Optics off a recorded map match line.twiss at the same element."""
    line = _demo_line()
    init = dict(betx=3.0, bety=4.0, alfx=0.7, alfy=-0.4,
                dx=0.1, dpx=0.02, dy=0.0, dpy=0.0)
    tw = line.twiss(**init)

    m = _offaxis_map(order=2, x=0, px=0, y=0, py=0, zeta=0, delta=0)
    m.set_jacobian(_6d_w_matrix(init["betx"], init["bety"], init["alfx"],
                                init["alfy"], 1.0, init["dx"], init["dpx"],
                                init["dy"], init["dpy"]))
    line.track(m, multi_element_monitor_at=["b"])
    o = line.tracker.record_multi_element_last_track.map_at("b").optics()

    i = list(line.element_names).index("b")
    for name in ("betx", "bety", "alfx", "alfy", "dx", "dpx", "mux", "muy"):
        xo.assert_allclose(getattr(o, name), tw[name][i], rtol=0, atol=1e-12)


def test_optics_parameter_gradient_vs_finite_differences():
    """d(optics)/d(knob) is the chain rule on the map's mixed coefficients."""
    A0 = _6d_w_matrix(3.0, 4.0, 0.7, -0.4, 1.0, 0.1, 0.02, 0.0, 0.0)
    dA = {(0, 0): [1.3, -0.5], (0, 1): [0.4, 0.9], (1, 0): [0.2, 0.1],
          (1, 1): [-0.7, 0.3], (0, 5): [0.05, -0.02]}

    descriptor = madng_tpsa.Descriptor(6, 2, params=["kqa", "kqb"], param_order=1)
    m = _offaxis_map(order=2, descriptor=descriptor)
    m.set_jacobian(A0)
    for (i, j), gradient in dA.items():
        for k, value in enumerate(gradient):
            mono = [0] * (6 + len(gradient))
            mono[j] = 1
            mono[6 + k] = 1
            m.set_coefficient(COORDS[i], tuple(mono), value)
    o = m.optics()

    # analytic: d betx = 2 A00 dA00 + 2 A01 dA01, d dx = dA05
    xo.assert_allclose(o.gradient("betx"),
                       2 * A0[0, 0] * np.array(dA[(0, 0)])
                       + 2 * A0[0, 1] * np.array(dA[(0, 1)]), rtol=0, atol=1e-13)
    xo.assert_allclose(o.gradient("dx"), dA[(0, 5)], rtol=0, atol=1e-13)
    assert set(o.gradients()) == set(o.to_dict())

    # betx built from A0 + h*dA, differentiated along the first parameter
    def betx_at(h):
        Ah = A0.copy()
        for (i, j), gradient in dA.items():
            Ah[i, j] += h * gradient[0]
        mm = _offaxis_map(order=1)
        mm.set_jacobian(Ah)
        return mm.optics().betx

    h = 1e-6
    xo.assert_allclose(o.gradient("betx")[0], (betx_at(h) - betx_at(-h)) / (2 * h),
                       rtol=1e-6, atol=1e-8)


def test_optics_gradient_guards():
    """Values need no parameters, gradients need parameters and order >= 2."""
    plain = _offaxis_map(order=2)
    plain.set_jacobian(_6d_w_matrix(3.0, 4.0, 0.7, -0.4, 1.0, 0.1, 0.02, 0.0, 0.0))
    assert plain.optics().betx > 0
    with pytest.raises(ValueError, match="no parameters"):
        plain.optics().gradient("betx")

    params = dict(params=["kqa", "kqb"], param_order=1)
    order_one = _offaxis_map(order=1, descriptor=madng_tpsa.Descriptor(6, 1, **params))
    with pytest.raises(ValueError, match="order >= 2"):
        order_one.optics().gradient("betx")

    parametric = _offaxis_map(order=2, descriptor=madng_tpsa.Descriptor(6, 2, **params))
    with pytest.raises(KeyError, match="unknown optical function"):
        parametric.optics().gradient("nope")


# KnobParameters: line variables held as GTPSA parameters

def test_knob_parameters_rejects_bad_names_and_shapes():
    line = _knob_line()
    with pytest.raises(KeyError, match="not a line variable"):
        KnobParameters(line, ["nope"], madng_tpsa.Descriptor(6, 2, num_params=1))
    with pytest.raises(ValueError, match="descriptor has 2 parameters, expected 1"):
        KnobParameters(line, ["kqa"], madng_tpsa.Descriptor(
            6, 2, params=["kqa", "kqb"], param_order=1))


def test_knob_parameters_apply_and_driven_elements():
    line = _knob_line()
    names = ["kqa", "kqb", "klink"]
    knobs = KnobParameters(line, names,
                           madng_tpsa.Descriptor(6, 2, params=names, param_order=1))
    assert len(knobs) == 3 and not knobs.applied
    knobs.teardown()          # a no-op before apply
    knobs.apply()
    assert knobs.applied
    assert "applied=True" in repr(knobs)

    # every element the expressions reach is enabled, with all its strength fields
    assert knobs.driven_elements() == [("mq1", "k1"), ("mq1", "k1s"),
                                       ("mq2", "k1"), ("mq2", "k1s"),
                                       ("mq3", "k1"), ("mq3", "k1s")]
    assert line.element_dict["mq1"].k1.const_part == pytest.approx(
        0.5 * 0.012 - 0.020)
    knobs.teardown()


def test_knob_parameters_strength_jacobian_vs_fd():
    """d strength / d knob through the xdeps expressions, incl. a var->var chain."""
    line = _knob_line()
    names = ["kqa", "kqb", "klink"]
    knobs = KnobParameters(line, names,
                           madng_tpsa.Descriptor(6, 2, params=names, param_order=1))
    knobs.apply()
    jacobian = knobs.strength_jacobian()

    expected = {("mq1", "k1"): [0.5, 1.0, 0.0],
                ("mq2", "k1"): [1.0, 0.0, 0.0],
                ("mq3", "k1"): [0.0, 0.0, 2.0]}
    for target, grads in expected.items():
        xo.assert_allclose(jacobian[target], grads, rtol=0, atol=1e-14)
    # the skew fields are along for the ride and carry no knob dependence
    for element in ("mq1", "mq2", "mq3"):
        xo.assert_allclose(jacobian[(element, "k1s")], np.zeros(3), rtol=0, atol=0)

    # central differences on the plain-double strengths
    knobs.teardown()

    def fd(element, attr, knob, h=1e-6):
        value0 = line[knob]
        line[knob] = value0 + h
        high = float(getattr(line.element_dict[element], attr))
        line[knob] = value0 - h
        low = float(getattr(line.element_dict[element], attr))
        line[knob] = value0
        return (high - low) / (2 * h)

    for (element, attr), grads in expected.items():
        xo.assert_allclose([fd(element, attr, n) for n in names], grads,
                           rtol=0, atol=1e-9)


def test_knob_parameters_refresh_moves_a_nonlinear_knob():
    """refresh re-seeds the parameters, so the gradient follows the knob value."""
    line = _nonlinear_knob_line()
    knobs = KnobParameters(line, ["kqa"], madng_tpsa.Descriptor(6, 2, num_params=1))
    knobs.apply()
    xo.assert_allclose(knobs.strength_jacobian()[("mq1", "k1")], [2 * 0.012],
                       rtol=0, atol=1e-14)

    for value in (0.05, -0.03, 0.11):
        knobs.refresh([value])
        xo.assert_allclose(knobs.strength_jacobian()[("mq1", "k1")], [2 * value],
                           rtol=0, atol=1e-14)
        assert line.element_dict["mq1"].k1.const_part == pytest.approx(value ** 2)
    knobs.teardown()


def test_knob_parameters_teardown_restores_doubles():
    line = _knob_line()
    names = ["kqa", "kqb", "klink"]
    knobs = KnobParameters(line, names,
                           madng_tpsa.Descriptor(6, 2, params=names, param_order=1))
    knobs.apply()
    knobs.teardown()

    assert not knobs.applied
    assert knobs.strength_jacobian() == {}
    for name in names:
        assert isinstance(line.vars.val[name], float)
    for element_name in ("mq1", "mq2", "mq3"):
        element = line.element_dict[element_name]
        assert not element._xobject._tpsa_enabled
        assert isinstance(element.k1, (float, np.floating))
    assert float(line.element_dict["mq2"].k1) == pytest.approx(0.012)


def test_parametric_track_matches_finite_differences():
    """One knobbed track gives d(coord)/d(knob) == central differences."""
    line = _fodo_knob_line()
    names = ["kqf", "kqd", "ksx"]
    descriptor = madng_tpsa.Descriptor(6, 2, params=names, param_order=1)
    knobs = KnobParameters(line, names, descriptor)
    knobs.apply()

    m = _offaxis_map(order=2, descriptor=descriptor)
    line.track(m)
    param_jacobian = m.param_jacobian()
    assert np.abs(param_jacobian).max() > 1e-6      # knob dependence is really there
    knobs.teardown()

    def orbit():
        q = _offaxis_map(order=1)
        line.track(q)
        return q.const_part

    fd = np.zeros((6, len(names)))
    h = 1e-6
    for j, name in enumerate(names):
        value0 = line[name]
        line[name] = value0 + h
        high = orbit()
        line[name] = value0 - h
        low = orbit()
        line[name] = value0
        fd[:, j] = (high - low) / (2 * h)

    xo.assert_allclose(param_jacobian, fd, rtol=0, atol=1e-9)


# ActionTpsaTrack: the native GTPSA match action

def _fodo_ring(num_cells=4):
    """Small FODO ring with the two quad families on knobs."""
    env = xt.Environment()
    env["kqf"] = 0.28
    env["kqd"] = -0.28
    components = []
    for i in range(num_cells):
        components += [
            env.new(f"qf{i}", xt.Quadrupole, length=0.5, k1="kqf"),
            env.new(f"d{i}a", xt.Drift, length=1.0),
            env.new(f"qd{i}", xt.Quadrupole, length=0.5, k1="kqd"),
            env.new(f"d{i}b", xt.Drift, length=1.0),
        ]
    line = env.new_line(components=components)
    line.particle_ref = xt.Particles(p0c=P0C, mass0=MASS0)
    return line


def test_action_tpsa_track_values_and_jacobian_vs_twiss():
    """Values match twiss on the same fixed init, the Jacobian matches its differences."""
    from xtrack.tpsa.match_action import ActionTpsaTrack

    line = _fodo_ring()
    tw0 = line.twiss(method="4d")
    start, end, at = "qf0", "d3b", "qd2"
    quantities = ["betx", "bety", "alfx", "dx", "mux"]

    action = ActionTpsaTrack(
        line, ["kqf", "kqd"],
        targets=[xt.Target(q, at=at, value=1.0) for q in quantities],
        tw_kwargs=dict(init=tw0, start=start, end=end), order=2)
    res = action.run()

    tw_ref = line.twiss(init=tw0, start=start, end=end)
    row = list(res["name"]).index(at)
    for q in quantities:
        xo.assert_allclose(res[q][row], tw_ref[q, at], rtol=0, atol=1e-13)

    jacobian = action.acquire_jacobian()
    action.teardown()
    assert not action._already_prepared
    assert isinstance(line.vars.val["kqf"], float)

    fd = np.zeros((len(quantities), 2))
    h = 1e-7
    for j, name in enumerate(["kqf", "kqd"]):
        value0 = line[name]
        line[name] = value0 + h
        high = line.twiss(init=tw0, start=start, end=end)
        line[name] = value0 - h
        low = line.twiss(init=tw0, start=start, end=end)
        line[name] = value0
        for i, q in enumerate(quantities):
            fd[i, j] = (high[q, at] - low[q, at]) / (2 * h)
    xo.assert_allclose(jacobian, fd, rtol=1e-6, atol=1e-6)


def test_action_tpsa_track_rejects_unsupported_targets():
    from xtrack.tpsa.match_action import ActionTpsaTrack

    line = _fodo_ring()
    tw0 = line.twiss(method="4d")
    kwargs = dict(tw_kwargs=dict(init=tw0, start="qf0", end="d3b"))

    action = ActionTpsaTrack(line, ["kqf"],
                             targets=[xt.Target("ddx", at="qd2", value=0.0)], **kwargs)
    with pytest.raises(ValueError):
        action.prepare()

    with pytest.raises(AssertionError):
        action = ActionTpsaTrack(
            line, ["kqf"],
            targets=[xt.TargetRelPhaseAdvance("dqx", value=0.0,
                                            start="qf0", end="qd2")], **kwargs)
