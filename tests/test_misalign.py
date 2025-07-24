# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #
from cpymad.madx import Madx
import numpy as np
import pytest
import xobjects as xo
import xtrack as xt
import pymadng as ng


def theta_matrix(angle):
    """Positive angle move z towards x"""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]]
    )


def phi_matrix(angle):
    """Positive angle move z towards y"""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [[1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0], [0, 0, 0, 1]]
    )


def psi_matrix(angle):
    """Positive angle move x towards y"""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )


def translate_matrix(x, y, z):
    """Translation matrix in 3D"""
    return np.array(
        [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]]
    )


def curvature_matrix(length, angle=0.0, tilt=0.0):
    """Change frame of reference by an arc of length `length` and `angle`."""
    sinc = lambda x: np.sinc(x / np.pi)  # np.sinc is normalized to pi, so we divide by pi
    delta_x = -length * sinc(angle / 2) * np.sin(angle / 2)  # rho * (np.cos(angle) - 1)
    delta_theta = -angle
    delta_s = length * sinc(angle)  # rho * np.sin(angle)
    wedge = translate_matrix(delta_x, 0, delta_s) @ theta_matrix(delta_theta)
    return psi_matrix(tilt) @ wedge @ psi_matrix(-tilt)


def particle_pos_in_frame(part, idx, frame):
    """Transform particle coordinates to a new frame defined by `frame_matrix`."""
    part_x, part_y = part.x[idx], part.y[idx]
    coords = frame[:3, 3] + part_x * frame[:3, 0] + part_y * frame[:3, 1]
    return coords


def particles_from_madng(tbl, beta, at='$end', slice_=0):
    df = tbl.to_df()
    row = df[(df['name'] == at) & (df['slc'] == slice_)]
    coords = {coord: np.array(row[coord]) for coord in
        ['x', 'px', 'y', 'py', 't', 'pt']}
    coords['zeta'] = coords.pop('t') * beta
    coords['ptau'] = coords.pop('pt')
    return xt.Particles(beta0=beta, **coords)


@pytest.mark.parametrize(
    'angle,tilt',
    [
        (0, 0),
        (0.3, 0),
    ],
    ids=['straight', 'curved']
)
def test_misalign_drift(angle, tilt):
    # Element parameters
    length = 20

    # Misalignment parameters
    dx = -15
    dy = 7
    ds = 20
    theta = -0.1  # rad
    phi = 0.11  # rad
    psi = 0.7  # rad
    anchor = 0.4  # fraction of the element length for the misalignment

    # Initial particle coordinates
    p0 = xt.Particles(
        x=[0, 0.2, -0.4, -0.6, 0.8],
        y=[0, 0.2, 0.4, -0.6, -0.8],
        px=[0, -0.01, -0.01, 0.01, 0.01],
        py=[0, -0.01, 0.01, -0.01, 0.01],
    )

    p_expected = p0.copy()
    if angle:
        element = xt.Bend(
            angle=angle,
            length=length,
            k0=0,
            rot_s_rad=tilt,
            model='rot-kick-rot',
        )
    else:
        element = xt.DriftExact(length=length)
    element.track(p_expected)

    p_misaligned_entry = p0.copy()
    mis_entry = xt.Misalignment(
        dx=dx, dy=dy, ds=ds,
        theta=theta, phi=phi, psi=psi,
        length=length, angle=angle, tilt=tilt,
        anchor=anchor, is_exit=False,
    )
    mis_entry.track(p_misaligned_entry)

    p_misaligned_exit = p_misaligned_entry.copy()
    element.track(p_misaligned_exit)

    p_aligned_exit = p_misaligned_exit.copy()
    mis_exit = xt.Misalignment(
        dx=dx, dy=dy, ds=ds,
        theta=theta, phi=phi, psi=psi,
        length=length, angle=angle, tilt=tilt,
        anchor=anchor, is_exit=True,
    )
    mis_exit.track(p_aligned_exit)

    # Compare the trajectory with and without misalignment
    xo.assert_allclose(p_expected.x, p_aligned_exit.x, atol=1e-14, rtol=1e-9)
    xo.assert_allclose(p_expected.px, p_aligned_exit.px, atol=1e-14, rtol=1e-9)
    xo.assert_allclose(p_expected.y, p_aligned_exit.y, atol=1e-14, rtol=1e-9)
    xo.assert_allclose(p_expected.py, p_aligned_exit.py, atol=1e-14, rtol=1e-9)

    # Check that the intermediate points (entry and exit in the misaligned frame)
    # still lie on the straight line
    for idx, (x, px, y, py, delta) in enumerate(zip(p0.x, p0.px, p0.y, p0.py, p0.delta)):
        pz = np.sqrt((1 + delta) ** 2 - px ** 2 - py ** 2)
        xp = px / pz  # = dpx / ds
        yp = py / pz  # = dpy / ds
        dp_ds = np.array([xp, yp, 1])  # = dp / ds

        coords_at_start = np.array([x, y, 0])

        part_length = anchor * length
        part_angle = anchor * angle
        to_entry = (
                curvature_matrix(part_length, part_angle, tilt)
                @ translate_matrix(dx, dy, ds)
                @ theta_matrix(theta)
                @ phi_matrix(phi)
                @ psi_matrix(psi)
                @ np.linalg.inv(curvature_matrix(part_length, part_angle, tilt))
        )
        coords_misaligned_entry = particle_pos_in_frame(p_misaligned_entry, idx, to_entry)
        calculated_dp_ds_misaligned_entry = coords_misaligned_entry - coords_at_start
        cross_misaligned_entry = np.cross(dp_ds, calculated_dp_ds_misaligned_entry)
        xo.assert_allclose(cross_misaligned_entry, 0, atol=1e-13, rtol=1e-9)

        to_exit = to_entry @ curvature_matrix(length, angle, tilt)
        coords_misaligned_exit = particle_pos_in_frame(p_misaligned_exit, idx, to_exit)
        calculated_dp_ds_misaligned_exit = coords_misaligned_exit - coords_at_start
        cross_misaligned_exit = np.cross(dp_ds, calculated_dp_ds_misaligned_exit)
        xo.assert_allclose(cross_misaligned_exit, 0, atol=1e-13, rtol=1e-9)


def test_misalign_vs_madng():
    # Element parameters
    length = 5
    ks = 0.5

    # Misalignment parameters
    dx = 0.1
    dy = 0.2
    ds = 0.3
    theta = 0.1  # rad
    phi = 0.2  # rad
    psi = 0.5  # rad

    # Track in Xsuite
    p0 = xt.Particles(x=0.2, y=-0.6, px=-0.01, py=0.02)
    line = xt.Line(elements=[
        xt.Misalignment(
            dx=dx, dy=dy, ds=ds,
            theta=theta, phi=phi, psi=psi,
            length=length, angle=0,
            anchor=0, is_exit=False,
        ),
        xt.Solenoid(length=length, ks=ks),
        xt.Misalignment(
            dx=dx, dy=dy, ds=ds,
            theta=theta, phi=phi, psi=psi,
            length=length, angle=0,
            anchor=0, is_exit=True,
        ),
    ])
    p_xt = p0.copy()
    line.track(p_xt)

    # MAD-NG
    mng = ng.MAD()
    X0 = [
        {
            'x': float(p0.x[i]),
            'px': float(p0.px[i]),
            'y': float(p0.y[i]),
            'py': float(p0.py[i]),
            't': float(p0.zeta[i] / p0.beta0[0]),
            'pt': float(p0.ptau[i]),
        } for i in range(len(p0.x))
    ]

    mng['ks'] = ks
    mng['length'] = length
    mng['dx'] = dx
    mng['dy'] = dy
    mng['ds'] = ds
    mng['theta'] = theta
    mng['phi'] = phi
    mng['psi'] = psi

    mng['X0'] = X0
    mng['beta'] = p0.beta0[0]

    ng_script = """
    local sequence, solenoid in MAD.element
    local track, beam in MAD

    local elem
    local misalign = {
        dx=dx,
        dy=dy,
        ds=ds,
        dtheta=theta,
        dphi=phi,
        dpsi=psi,
        -- anchor not supported yet
    }

    elem = solenoid 'elem' {
        l=length,
        ks=ks,
        misalign=misalign,
    }

    local seq = sequence 'seq' { refer='entry',
        elem { at=0 }
    }

    mybeam = beam { particle='proton', beta=beta }
    tbl = track {
        X0=X0,
        sequence=seq,
        nturn=1,
        beam=mybeam,
        observe=0,
        aperture={ kind='square', 100 },
    }
    """
    mng.send(ng_script)

    p_ng = particles_from_madng(mng.tbl, mng.mybeam.beta, at='$end')

    xo.assert_allclose(p_ng.x, p_xt.x, atol=1e-14, rtol=1e-9)
    xo.assert_allclose(p_ng.px, p_xt.px, atol=1e-14, rtol=1e-9)
    xo.assert_allclose(p_ng.y, p_xt.y, atol=1e-14, rtol=1e-9)
    xo.assert_allclose(p_ng.py, p_xt.py, atol=1e-14, rtol=1e-9)
