# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #
import numpy as np
import xobjects as xo
import xtrack as xt


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


def curvature_matrix(length, angle):
    """Change frame of reference by an arc of length `length` and `angle`."""
    sinc = lambda x: np.sinc(x / np.pi)  # np.sinc is normalized to pi, so we divide by pi
    delta_x = - length * sinc(angle / 2) * np.sin(angle / 2)  # rho * (np.cos(angle) - 1)
    delta_theta = -angle
    delta_s = length * sinc(angle)  # rho * np.sin(angle)
    return translate_matrix(delta_x, 0, delta_s) @ theta_matrix(delta_theta)


def particle_pos_in_frame(part, idx, frame):
    """Transform particle coordinates to a new frame defined by `frame_matrix`."""
    part_x, part_y = part.x[idx], part.y[idx]
    coords = frame[:3, 3] + part_x * frame[:3, 0] + part_y * frame[:3, 1]
    return coords


def test_misalign_straight_drift():
    # Element parameters
    length = 20
    angle = 0

    # Misalignment parameters
    dx = -15
    dy = 7
    ds = 20
    theta = -0.1  # rad
    phi = 0.11  # rad
    psi = np.pi / 4  # rad
    anchor = 0.4  # fraction of the element length for the misalignment

    # Initial particle coordinates
    p0 = xt.Particles(
        x=[0, 0.2, -0.4, -0.6, 0.8],
        y=[0, 0.2, 0.4, -0.6, -0.8],
        px=[0, -0.01, -0.01, 0.01, 0.01],
        py=[0, -0.01, 0.01, -0.01, 0.01],
    )

    p_expected = p0.copy()
    element_straight = xt.Solenoid(length=length)
    element_straight.track(p_expected)

    p_misaligned_entry = p0.copy()
    mis_entry = xt.Misalignment(dx=dx, dy=dy, ds=ds, theta=theta, phi=phi, psi=psi, length=length, anchor=anchor, angle=angle, is_exit=False)
    mis_entry.track(p_misaligned_entry)

    p_misaligned_exit = p_misaligned_entry.copy()
    element_straight.track(p_misaligned_exit)

    p_aligned_exit = p_misaligned_exit.copy()
    mis_exit = xt.Misalignment(dx=dx, dy=dy, ds=ds, theta=theta, phi=phi, psi=psi, length=length, anchor=anchor, angle=angle, is_exit=True)
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

        to_entry = (
                curvature_matrix(anchor * length, 0)
                @ translate_matrix(dx, dy, ds)
                @ theta_matrix(theta)
                @ phi_matrix(phi)
                @ psi_matrix(psi)
                @ np.linalg.inv(curvature_matrix(anchor * length, 0))
        )
        coords_misaligned_entry = particle_pos_in_frame(p_misaligned_entry, idx, to_entry)
        calculated_dp_ds_misaligned_entry = coords_misaligned_entry - coords_at_start
        cross_misaligned_entry = np.cross(dp_ds, calculated_dp_ds_misaligned_entry)
        xo.assert_allclose(cross_misaligned_entry, 0, atol=1e-14, rtol=1e-9)

        to_exit = to_entry @ curvature_matrix(length, 0)
        coords_misaligned_exit = particle_pos_in_frame(p_misaligned_exit, idx, to_exit)
        calculated_dp_ds_misaligned_exit = coords_misaligned_exit - coords_at_start
        cross_misaligned_exit = np.cross(dp_ds, calculated_dp_ds_misaligned_exit)
        xo.assert_allclose(cross_misaligned_exit, 0, atol=1e-14, rtol=1e-9)
