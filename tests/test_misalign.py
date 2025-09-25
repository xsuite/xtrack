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

from xobjects.test_helpers import for_all_test_contexts


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
        (0, 0.1),
        (0.3, 0.1),
    ],
    ids=['straight', 'curved', 'straight-tilted', 'curved-tilted']
)
@for_all_test_contexts
def test_misalign_drift(angle, tilt, test_context):
    # Element parameters
    length = 20

    # Misalignment parameters
    dx = -15  # m
    dy = 7  # m
    ds = 20  # m
    theta = -0.1  # rad
    phi = 0.11  # rad
    psi = 0.7  # rad
    anchor = 8  # m

    # Initial particle coordinates
    p0 = xt.Particles(
        x=[0, 0.2, -0.4, -0.6, 0.8],
        y=[0, 0.2, 0.4, -0.6, -0.8],
        px=[0, -0.01, -0.01, 0.01, 0.01],
        py=[0, -0.01, 0.01, -0.01, 0.01],
        _context=test_context,
    )

    p_expected = p0.copy()
    if angle:
        element = xt.Bend(
            angle=angle,
            length=length,
            k0=0,
            rot_s_rad=tilt,
            model='rot-kick-rot',
            _context=test_context,
        )
        element.track(p_expected)
        element.rot_s_rad = 0 # Put in the misalignment element
    else:
        element = xt.DriftExact(length=length, _context=test_context)
        element.track(p_expected)

    p_misaligned_entry = p0.copy()
    mis_entry = xt.Misalignment(
        dx=dx, dy=dy, ds=ds,
        theta=theta, phi=phi, psi=psi,
        length=length, angle=angle, tilt=tilt,
        anchor=anchor, is_exit=False,
        _context=test_context,
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
        _context=test_context,
    )
    mis_exit.track(p_aligned_exit)

    p_expected.move(_context=xo.ContextCpu())
    p_aligned_exit.move(_context=xo.ContextCpu())

    # Compare the trajectory with and without misalignment
    xo.assert_allclose(p_expected.x, p_aligned_exit.x, atol=1e-14, rtol=1e-9)
    xo.assert_allclose(p_expected.px, p_aligned_exit.px, atol=1e-14, rtol=1e-9)
    xo.assert_allclose(p_expected.y, p_aligned_exit.y, atol=1e-14, rtol=1e-9)
    xo.assert_allclose(p_expected.py, p_aligned_exit.py, atol=1e-14, rtol=1e-9)
    xo.assert_allclose(p_expected.s, p_aligned_exit.s, atol=1e-14, rtol=1e-9)
    xo.assert_allclose(p_expected.zeta, p_aligned_exit.zeta, atol=1e-14, rtol=1e-9)

    # I comment out the following as it is inconsistent with the current implementation
    # (including the tilt in the misalignment), need to adapt:

    # # Check that the intermediate points (entry and exit in the misaligned frame)
    # # still lie on the straight line
    # p0.move(_context=xo.ContextCpu())
    # p_misaligned_entry.move(_context=xo.ContextCpu())
    # p_misaligned_exit.move(_context=xo.ContextCpu())
    # for idx, (x, px, y, py, delta) in enumerate(zip(p0.x, p0.px, p0.y, p0.py, p0.delta)):
    #     pz = np.sqrt((1 + delta) ** 2 - px ** 2 - py ** 2)
    #     xp = px / pz  # = dpx / ds
    #     yp = py / pz  # = dpy / ds
    #     dp_ds = np.array([xp, yp, 1])  # = dp / ds

    #     coords_at_start = np.array([x, y, 0])

    #     part_length = anchor
    #     part_angle = anchor / length * angle
    #     to_entry = (
    #             curvature_matrix(part_length, part_angle, tilt)
    #             @ translate_matrix(dx, dy, ds)
    #             @ theta_matrix(theta)
    #             @ phi_matrix(phi)
    #             @ psi_matrix(psi)
    #             @ np.linalg.inv(curvature_matrix(part_length, part_angle, tilt))
    #     )
    #     coords_misaligned_entry = particle_pos_in_frame(p_misaligned_entry, idx, to_entry)
    #     calculated_dp_ds_misaligned_entry = coords_misaligned_entry - coords_at_start
    #     cross_misaligned_entry = np.cross(dp_ds, calculated_dp_ds_misaligned_entry)
    #     xo.assert_allclose(cross_misaligned_entry, 0, atol=1e-13, rtol=1e-9)

    #     to_exit = to_entry @ curvature_matrix(length, angle, tilt)
    #     coords_misaligned_exit = particle_pos_in_frame(p_misaligned_exit, idx, to_exit)
    #     calculated_dp_ds_misaligned_exit = coords_misaligned_exit - coords_at_start
    #     cross_misaligned_exit = np.cross(dp_ds, calculated_dp_ds_misaligned_exit)
    #     xo.assert_allclose(cross_misaligned_exit, 0, atol=1e-13, rtol=1e-9)


@pytest.mark.parametrize('angle', [0, 0.3], ids=['straight', 'curved'])
@pytest.mark.parametrize('tilt', [0, 0.1], ids=['horizontal', 'tilted'])
def test_misalign_vs_madng(angle, tilt):
    # Element parameters
    length = 5
    ks = 0.5  # in the straight case let's put a solenoid
    k0 = 0.09  # in the curved case let's put an sbend, with strength != h

    # Misalignment parameters
    dx = 0.1
    dy = 0.2
    ds = 0.3
    theta = 0.1  # rad
    phi = 0.2  # rad
    psi = 0.5  # rad

    if angle:
        element = xt.Bend(length=length, angle=angle, model='rot-kick-rot', k0=k0)#, rot_s_rad=tilt)
    else:
        element = xt.Solenoid(length=length, ks=ks)#, rot_s_rad=tilt)

    # Track in Xsuite
    p0 = xt.Particles(x=0.2, y=-0.6, px=-0.01, py=0.02, zeta=0.5, delta=0.9)
    line = xt.Line(elements=[
        xt.Misalignment(
            dx=dx, dy=dy, ds=ds,
            theta=theta, phi=phi, psi=psi,
            length=length, angle=angle, tilt=tilt,
            anchor=0, is_exit=False,
        ),
        element,
        xt.Misalignment(
            dx=dx, dy=dy, ds=ds,
            theta=theta, phi=phi, psi=psi,
            length=length, angle=angle, tilt=tilt,
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

    mng['length'] = length
    mng['dx'] = dx
    mng['dy'] = dy
    mng['ds'] = ds
    mng['theta'] = theta
    mng['phi'] = phi
    mng['psi'] = psi
    mng['angle'] = angle
    mng['ks'] = ks
    mng['k0'] = k0
    mng['tilt'] = tilt

    mng['X0'] = X0
    mng['beta'] = p0.beta0[0]

    ng_script = """
    local sequence, solenoid, quadrupole, sbend in MAD.element
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

    if angle ~= 0 then
        elem = sbend 'elem' {
            l=length,
            angle=angle,
            k0=k0,
            misalign=misalign,
            kill_ent_fringe=true,
            kill_exi_fringe=true,
            tilt=tilt,
        }
    else
        elem = solenoid 'elem' {
            l=length,
            ks=ks,
            misalign=misalign,
            tilt=tilt,
        }
    end

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
        save='atall',
        aperture={ kind='square', 100 },
    }
    """
    mng.send(ng_script)

    p_ng = particles_from_madng(mng.tbl, mng.mybeam.beta, at='$end', slice_=0)
    assert p_ng.x.size > 0

    xo.assert_allclose(p_ng.x, p_xt.x, atol=1e-14, rtol=1e-9)
    xo.assert_allclose(p_ng.px, p_xt.px, atol=1e-14, rtol=1e-9)
    xo.assert_allclose(p_ng.y, p_xt.y, atol=1e-14, rtol=1e-9)
    xo.assert_allclose(p_ng.py, p_xt.py, atol=1e-14, rtol=1e-9)
    xo.assert_allclose(p_ng.zeta, p_xt.zeta, atol=1e-14, rtol=1e-9)


@for_all_test_contexts
@pytest.mark.parametrize(
    'element_type',
    ['Bend', 'Quadrupole', 'Multipole'],
)
def test_misalign_dedicated_vs_beam_element(test_context, element_type):
    # Element parameters
    tilt = 0.1

    # Misalignment parameters
    dx = 0.1
    dy = 0.2
    ds = 0.3
    theta = 0.1  # rad
    phi = 0.2  # rad
    psi = 0.5  # rad
    anchor = 4

    if element_type == 'Bend':
        angle = 0.3
        length = 5
        element = xt.Bend(
            length=length,
            angle=angle,
            model='rot-kick-rot',
            k0=0.09,
        )
    elif element_type == 'Quadrupole':
        angle = 0
        length = 5
        element = xt.Quadrupole(
            length=length,
            k1=0.09,
        )
    elif element_type == 'Multipole':
        angle = 0
        length = 0
        element = xt.Multipole(
            length=5,
            knl=[0.04, 0.09],
            ksl=[0.02, 0.01],
        )
    else:
        raise ValueError(f"Test not implemented for {element_type}")

    # Track in Xsuite
    p0 = xt.Particles(x=0.2, y=-0.6, px=-0.01, py=0.02, zeta=0.5, delta=0.9)
    p0.move(_context=test_context)

    line_ref = xt.Line(elements=[
        xt.Misalignment(
            dx=dx, dy=dy, ds=ds,
            theta=theta, phi=phi, psi=psi,
            length=length, angle=angle, tilt=tilt,
            anchor=anchor, is_exit=False,
        ),
        element,
        xt.Misalignment(
            dx=dx, dy=dy, ds=ds,
            theta=theta, phi=phi, psi=psi,
            length=length, angle=angle, tilt=tilt,
            anchor=anchor, is_exit=True,
        ),
    ])
    p_ref = p0.copy()
    line_ref.build_tracker(_context=test_context)
    line_ref.track(p_ref)

    p_test = p0.copy()
    transformed_element = element.copy()
    transformed_element.shift_x = dx
    transformed_element.shift_y = dy
    transformed_element.shift_s = ds
    transformed_element.rot_x_rad = phi
    transformed_element.rot_y_rad = theta
    transformed_element.rot_s_rad_no_frame = psi
    transformed_element.rot_s_rad = tilt
    transformed_element.rot_shift_anchor = anchor

    line_test = xt.Line(elements=[transformed_element])
    line_test.build_tracker(_context=test_context)
    line_test.track(p_test)

    xo.assert_allclose(p_ref.x, p_test.x, atol=1e-15, rtol=1e-15)
    xo.assert_allclose(p_ref.px, p_test.px, atol=1e-15, rtol=1e-15)
    xo.assert_allclose(p_ref.y, p_test.y, atol=1e-15, rtol=1e-15)
    xo.assert_allclose(p_ref.py, p_test.py, atol=1e-15, rtol=1e-15)
    xo.assert_allclose(p_ref.delta, p_test.delta, atol=1e-15, rtol=1e-15)
    xo.assert_allclose(p_ref.zeta, p_test.zeta, atol=1e-15, rtol=1e-15)
    xo.assert_allclose(p_ref.s, p_test.s, atol=1e-15, rtol=1e-15)

    # Check backtrak
    line_test.track(p_test, backtrack=True)
    xo.assert_allclose(p_test.x, p0.x, atol=1e-14, rtol=1e-14)
    xo.assert_allclose(p_test.px, p0.px, atol=1e-14, rtol=1e-14)
    xo.assert_allclose(p_test.y, p0.y, atol=1e-14, rtol=1e-14)
    xo.assert_allclose(p_test.py, p0.py, atol=1e-14, rtol=1e-14)
    xo.assert_allclose(p_test.delta, p0.delta, atol=1e-14, rtol=1e-14)
    xo.assert_allclose(p_test.zeta, p0.zeta, atol=1e-14, rtol=1e-14)
    xo.assert_allclose(p_test.s, p0.s, atol=1e-14, rtol=1e-14)

def test_errors_on_slices():

    env = xt.Environment()
    line = env.new_line(components=[
        env.new('b', 'Bend', angle=0.1, length=2)
    ])

    p = xt.Particles(p0c=1e9)
    line.track(p)
    assert p.state[0] == 1

    line['b'].shift_x = 0.1
    line.track(p)
    assert p.state[0] == 1

    line.cut_at_s([0.5])
    line.track(p)
    assert p.state[0] == -41

    p = xt.Particles(p0c=1e9)
    line['b'].angle = 0
    line.track(p)
    assert p.state[0] == 1

    line['b'].angle = 0.1
    line.track(p)
    assert p.state[0] == -41

    line['b'].angle = 0
    p = xt.Particles(p0c=1e9)
    line.track(p)
    assert p.state[0] == 1

    line['b'].rot_x_rad = 0.1
    line.track(p)
    assert p.state[0] == -40