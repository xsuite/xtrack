import numpy as np
import pytest

import xgtpsa
import xtrack as xt
import xtrack.tpsa as xtpsa
from xtrack.tracking_backends import _BACKENDS


def _line(k1=0.1):
    line = xt.Line(
        elements=[xt.Quadrupole(length=1.0, k1=k1)],
        element_names=["q"],
    )
    line.particle_ref = xt.Particles(p0c=7e12, mass0=xt.PROTON_MASS_EV)
    line.build_tracker(use_prebuilt_kernels=False)
    return line


def _map(order=1, descriptor=None):
    return xtpsa.ParticlesTpsa(
        order=order,
        descriptor=descriptor,
        x=1e-4,
        px=2e-5,
        y=0.0,
        py=0.0,
        zeta=0.0,
        delta=0.0,
        p0c=7e12,
        mass0=xt.PROTON_MASS_EV,
    )


def test_particles_tpsa_backend_registered():
    assert type(_BACKENDS[xtpsa.ParticlesTpsa]).__name__ == "IntegratedTpsaBackend"


def test_tpsa_line_track_matches_scalar_const_part():
    line_scalar = _line()
    part = xt.Particles(
        x=1e-4,
        px=2e-5,
        y=0.0,
        py=0.0,
        zeta=0.0,
        delta=0.0,
        p0c=7e12,
        mass0=xt.PROTON_MASS_EV,
    )
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


def test_float_or_tpsa_field_assignment_and_scalar_guard():
    line = _line()
    descriptor = xgtpsa.Descriptor(6, 1, num_params=1, param_order=1)
    line["q"].k1 = descriptor.param(1, 0.1)

    assert line["q"]._tpsa_enabled
    assert line["q"].k1.const_part == pytest.approx(0.1)

    part = xt.Particles(
        x=1e-4,
        px=2e-5,
        p0c=7e12,
        mass0=xt.PROTON_MASS_EV,
    )
    with pytest.raises(RuntimeError, match="TPSA-enabled"):
        line.track(part)


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
