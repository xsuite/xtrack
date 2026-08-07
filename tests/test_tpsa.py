import numpy as np
import pytest

import xobjects as xo
import xgtpsa
import xtrack as xt
import xtrack.tpsa as xtpsa


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


def test_scalar_track_tpsa_enabled_element_uses_const_part():
    line_scalar = _line(k1=0.125)
    part_scalar = _particle()
    line_scalar.track(part_scalar)

    line_tpsa_enabled = _line(k1=0.125)
    descriptor = xgtpsa.Descriptor(6, 1, num_params=1, param_order=1)
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


def test_build_tracker_preserves_tpsa_enabled_elements_moved_to_common_buffer():
    ctx = xo.ContextCpu()
    buffer_a = ctx.new_buffer(capacity=1024)
    buffer_b = ctx.new_buffer(capacity=1024)

    q1 = xt.Quadrupole(length=1.0, k1=0.1, _buffer=buffer_a)
    q2 = xt.Quadrupole(length=1.0, k1=0.2, _buffer=buffer_b)
    descriptor = xgtpsa.Descriptor(6, 1, num_params=1, param_order=1)
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
    descriptor = xgtpsa.Descriptor(6, 1, num_params=1, param_order=1)
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
    descriptor = xgtpsa.Descriptor(6, 1, num_params=1, param_order=1)
    line["q"].k1 = descriptor.param(1, 0.1)

    assert line["q"]._tpsa_enabled
    assert line["q"].k1.const_part == pytest.approx(0.1)


@pytest.mark.parametrize("name, element_cls, kwargs, field", _SUPPORTED_FLOAT_OR_TPSA_ELEMENTS)
def test_supported_float_or_tpsa_elements_accept_tpsa_fields(
        name, element_cls, kwargs, field):
    element = element_cls(**kwargs)
    descriptor = xgtpsa.Descriptor(6, 1, num_params=1, param_order=1)

    setattr(element, field, descriptor.param(1, kwargs[field]))

    assert element._tpsa_enabled
    assert getattr(element, field).const_part == pytest.approx(kwargs[field])


def test_tpsa_enabled_element_to_dict_is_rejected():
    line = _line()
    descriptor = xgtpsa.Descriptor(6, 1, num_params=1, param_order=1)
    line["q"].k1 = descriptor.param(1, 0.1)

    with pytest.raises(NotImplementedError, match="Serializing TPSA-enabled"):
        line["q"].to_dict()


def test_tpsa_constructor_assignment_without_container_is_rejected():
    descriptor = xgtpsa.Descriptor(6, 1, num_params=1, param_order=1)

    with pytest.raises(ValueError, match="without an owning container"):
        xt.Quadrupole(length=1.0, k1=descriptor.param(1, 0.1))


def test_parametric_element_field_tracks_with_shared_descriptor():
    line = _line()
    descriptor = xgtpsa.Descriptor(6, 1, num_params=1, param_order=1)
    line["q"].k1 = descriptor.param(1, 0.1)

    m = _map(descriptor=descriptor)
    line.track(m)

    assert m.num_params == 1
    assert m.x.param_grad()[0] != 0


def test_particles_tpsa_rejects_descriptor_shape_mismatch():
    descriptor = xgtpsa.Descriptor(5, 1)
    with pytest.raises(ValueError, match="6 variables"):
        _map(descriptor=descriptor)
