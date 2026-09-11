# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

from dataclasses import dataclass

import numpy as np


__all__ = ['CCSFrame', 'Frame']


@dataclass
class CCSFrame:
    """Pose in the CERN Coordinate System.

    The x axis points towards a positive bend, y follows the beam, and z
    points upwards. ``theta_gon`` is in gradians (gon), while ``phi`` and
    ``psi`` are in radians.
    """

    x: float = 0
    y: float = 0
    z: float = 0
    theta_gon: float = 0
    phi: float = 0
    psi: float = 0


_SURVEY_TO_CCS = np.array([
    [-1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
])
_CCS_TO_SURVEY = _SURVEY_TO_CCS.T
_GON_TO_RAD = np.pi / 200
_RAD_TO_GON = 200 / np.pi


def _angles_from_E_matrix(E_matrix):
    """Return MAD-X survey angles for one or more orientation matrices."""
    E_matrix = np.asarray(E_matrix)
    theta = np.arctan2(E_matrix[..., 0, 2], E_matrix[..., 2, 2])
    psi = np.arctan2(E_matrix[..., 1, 0], E_matrix[..., 1, 1])
    phi = np.arctan2(
        E_matrix[..., 1, 2], E_matrix[..., 1, 1] / np.cos(psi))
    return theta, phi, psi


class Frame:
    """Mutable local reference frame used by survey propagation.

    The homogeneous 4 x 4 matrix maps local ``(x, y, s, 1)`` coordinates to
    global ``(X, Y, Z, 1)`` coordinates. Transformations are applied in the
    local frame and mutate this object in place.

    Custom beam elements can participate in survey propagation by exposing a
    ``track_frame(frame, backtrack=False)`` method and applying their geometry
    through the transformation methods of the supplied frame.
    """

    def __init__(self, matrix=None):
        if matrix is None:
            matrix = np.eye(4)
        else:
            matrix = np.array(matrix, dtype=float, copy=True)

        if matrix.shape != (4, 4):
            raise ValueError(
                f"Frame matrix must have shape (4, 4), got {matrix.shape}")

        self.matrix = matrix

    def __repr__(self):
        values = (
            *self.XYZ,
            self.theta,
            self.phi,
            self.psi,
        )
        X, Y, Z, theta, phi, psi = (
            f'{float(value):.12g}' for value in values)
        return (
            f'Frame(X={X}, Y={Y}, Z={Z}, '
            f'theta={theta}, phi={phi}, psi={psi})'
        )

    @classmethod
    def from_survey(cls, XYZ, E_matrix):
        matrix = np.eye(4)
        matrix[:3, :3] = E_matrix
        matrix[:3, 3] = XYZ
        return cls(matrix)

    @classmethod
    def from_survey_angles(cls, X=0, Y=0, Z=0,
                           theta=0, phi=0, psi=0):
        return cls.from_survey(
            XYZ=np.array([X, Y, Z]),
            E_matrix=cls.E_matrix_from_angles(theta, phi, psi),
        )

    @classmethod
    def from_ccs(cls, ccs: CCSFrame):
        """Build a survey frame from CERN Coordinate System coordinates."""
        ccs_XYZ = np.array([ccs.x, ccs.y, ccs.z])
        ccs_E_matrix = cls._ccs_E_matrix_from_angles(
            ccs.theta_gon * _GON_TO_RAD, ccs.phi, ccs.psi)
        return cls.from_survey(
            XYZ=_CCS_TO_SURVEY @ ccs_XYZ,
            E_matrix=(
                _CCS_TO_SURVEY @ ccs_E_matrix @ _SURVEY_TO_CCS),
        )

    @staticmethod
    def E_matrix_from_angles(theta, phi, psi):
        """Build the MAD-X-compatible survey orientation matrix."""
        costhe = np.cos(theta)
        cosphi = np.cos(phi)
        cospsi = np.cos(psi)
        sinthe = np.sin(theta)
        sinphi = np.sin(phi)
        sinpsi = np.sin(psi)

        E_matrix = np.zeros((3, 3))
        E_matrix[0, 0] = +costhe * cospsi - sinthe * sinphi * sinpsi
        E_matrix[0, 1] = -costhe * sinpsi - sinthe * sinphi * cospsi
        E_matrix[0, 2] = sinthe * cosphi
        E_matrix[1, 0] = cosphi * sinpsi
        E_matrix[1, 1] = cosphi * cospsi
        E_matrix[1, 2] = sinphi
        E_matrix[2, 0] = -sinthe * cospsi - costhe * sinphi * sinpsi
        E_matrix[2, 1] = +sinthe * sinpsi - costhe * sinphi * cospsi
        E_matrix[2, 2] = costhe * cosphi
        return E_matrix

    @staticmethod
    def _ccs_E_matrix_from_angles(theta, phi, psi):
        costhe = np.cos(theta)
        cosphi = np.cos(phi)
        cospsi = np.cos(psi)
        sinthe = np.sin(theta)
        sinphi = np.sin(phi)
        sinpsi = np.sin(psi)

        theta_matrix = np.array([
            [costhe, sinthe, 0],
            [-sinthe, costhe, 0],
            [0, 0, 1],
        ])
        phi_matrix = np.array([
            [1, 0, 0],
            [0, cosphi, -sinphi],
            [0, sinphi, cosphi],
        ])
        psi_matrix = np.array([
            [cospsi, 0, -sinpsi],
            [0, 1, 0],
            [sinpsi, 0, cospsi],
        ])
        return theta_matrix @ phi_matrix @ psi_matrix

    def to_ccs(self) -> CCSFrame:
        """Return this frame as principal-angle CCS coordinates."""
        ccs_XYZ = _SURVEY_TO_CCS @ self.XYZ
        ccs_E_matrix = (
            _SURVEY_TO_CCS @ self.E_matrix @ _CCS_TO_SURVEY)
        psi = np.arctan2(ccs_E_matrix[2, 0], ccs_E_matrix[2, 2])
        cosphi = np.hypot(ccs_E_matrix[0, 1], ccs_E_matrix[1, 1])
        phi = np.arctan2(ccs_E_matrix[2, 1], cosphi)
        theta_rad = np.arctan2(
            ccs_E_matrix[0, 1], ccs_E_matrix[1, 1])
        return CCSFrame(
            x=ccs_XYZ[0],
            y=ccs_XYZ[1],
            z=ccs_XYZ[2],
            theta_gon=theta_rad * _RAD_TO_GON,
            phi=phi,
            psi=psi,
        )

    @property
    def XYZ(self):
        """Global position of the local-frame origin, as a writable view."""
        return self.matrix[:3, 3]

    @XYZ.setter
    def XYZ(self, value):
        self.matrix[:3, 3] = value

    @property
    def X(self):
        """Global horizontal coordinate of the local-frame origin."""
        return self.XYZ[0]

    @X.setter
    def X(self, value):
        self.XYZ[0] = value

    @property
    def Y(self):
        """Global vertical coordinate of the local-frame origin."""
        return self.XYZ[1]

    @Y.setter
    def Y(self, value):
        self.XYZ[1] = value

    @property
    def Z(self):
        """Global longitudinal coordinate of the local-frame origin."""
        return self.XYZ[2]

    @Z.setter
    def Z(self, value):
        self.XYZ[2] = value

    @property
    def E_matrix(self):
        """Local-to-global basis-vector matrix, as a writable view."""
        return self.matrix[:3, :3]

    @E_matrix.setter
    def E_matrix(self, value):
        self.matrix[:3, :3] = value

    @property
    def ex(self):
        """Global components of the local horizontal unit vector."""
        return self.E_matrix[:, 0]

    @ex.setter
    def ex(self, value):
        self.E_matrix[:, 0] = value

    @property
    def ey(self):
        """Global components of the local vertical unit vector."""
        return self.E_matrix[:, 1]

    @ey.setter
    def ey(self, value):
        self.E_matrix[:, 1] = value

    @property
    def ez(self):
        """Global components of the local longitudinal unit vector."""
        return self.E_matrix[:, 2]

    @ez.setter
    def ez(self, value):
        self.E_matrix[:, 2] = value

    @property
    def theta(self):
        """Principal MAD-X survey angle theta."""
        return _angles_from_E_matrix(self.E_matrix)[0]

    @property
    def phi(self):
        """Principal MAD-X survey angle phi."""
        return _angles_from_E_matrix(self.E_matrix)[1]

    @property
    def psi(self):
        """Principal MAD-X survey angle psi."""
        return _angles_from_E_matrix(self.E_matrix)[2]

    def copy(self):
        return Frame(self.matrix)

    def inverse(self):
        """Return the inverse homogeneous transform as a new frame."""
        return Frame(np.linalg.inv(self.matrix))

    def __matmul__(self, other):
        """Compose two frames without mutating either operand."""
        if not isinstance(other, Frame):
            return NotImplemented
        return Frame(self.matrix @ other.matrix)

    def transform(self, displacement=None, rotation_matrix=None):
        """Apply a local rigid transform represented by vector and matrix."""
        initial_E_matrix = self.E_matrix.copy()
        if displacement is not None:
            self.XYZ[:] = self.XYZ + initial_E_matrix @ displacement
        if rotation_matrix is not None:
            self.E_matrix[:] = initial_E_matrix @ rotation_matrix
        return self

    def _translate(self, dx=0, dy=0, ds=0):
        if dx != 0 or dy != 0 or ds != 0:
            self.transform(displacement=np.array([dx, dy, ds]))
        return self

    def translate_x(self, dx):
        """Translate along the local horizontal axis."""
        return self._translate(dx=dx)

    def translate_y(self, dy):
        """Translate along the local vertical axis."""
        return self._translate(dy=dy)

    def translate_s(self, ds):
        """Translate along the local longitudinal axis."""
        return self._translate(ds=ds)

    def _rotate(self, rotation_matrix):
        return self.transform(rotation_matrix=rotation_matrix)

    def rotate_x(self, angle):
        """Rotate around the local horizontal axis."""
        if angle == 0:
            return self
        c = np.cos(angle)
        s = np.sin(angle)
        return self._rotate(np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ]))

    def rotate_y(self, angle):
        """Rotate around the local vertical axis."""
        if angle == 0:
            return self
        c = np.cos(angle)
        s = np.sin(angle)
        return self._rotate(np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c],
        ]))

    def rotate_s(self, angle):
        """Rotate around the local longitudinal axis."""
        if angle == 0:
            return self
        c = np.cos(angle)
        s = np.sin(angle)
        return self._rotate(np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ]))

    def arc(self, length=0, angle=0, tilt=0):
        """Advance along a MAD-X-compatible arc in the local frame."""
        if angle == 0:
            return self.translate_s(length)

        ca = np.cos(angle)
        sa = np.sin(angle)
        ct = np.cos(tilt)
        st = np.sin(tilt)

        displacement = np.array([
            -0.5 * length * angle * np.sinc(angle / (2 * np.pi))**2,
            0,
            length * np.sinc(angle / np.pi),
        ])
        bend_rotation = np.array([
            [ca, 0, -sa],
            [0, 1, 0],
            [sa, 0, ca],
        ])
        tilt_rotation = np.array([
            [ct, -st, 0],
            [st, ct, 0],
            [0, 0, 1],
        ])
        inverse_tilt_rotation = np.array([
            [ct, st, 0],
            [-st, ct, 0],
            [0, 0, 1],
        ])

        return self.transform(
            displacement=tilt_rotation @ displacement,
            rotation_matrix=(
                tilt_rotation @ bend_rotation @ inverse_tilt_rotation),
        )

    def arc_x(self, length=0, angle=0):
        """Advance along an arc in the local horizontal-longitudinal plane."""
        return self.arc(length=length, angle=angle, tilt=0)

    def arc_y(self, length=0, angle=0):
        """Advance along an arc in the local vertical-longitudinal plane."""
        return self.arc(length=length, angle=angle, tilt=np.pi / 2)
