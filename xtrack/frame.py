# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np


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

    @classmethod
    def from_xyz_matrix(cls, XYZ, E_matrix):
        matrix = np.eye(4)
        matrix[:3, :3] = E_matrix
        matrix[:3, 3] = XYZ
        return cls(matrix)

    @classmethod
    def from_xyz_angles(cls, X=0, Y=0, Z=0,
                        theta=0, phi=0, psi=0):
        return cls.from_xyz_matrix(
            XYZ=np.array([X, Y, Z]),
            E_matrix=cls.rotation_from_angles(theta, phi, psi),
        )

    @staticmethod
    def rotation_from_angles(theta, phi, psi):
        """Build the MAD-X-compatible survey orientation matrix."""
        costhe = np.cos(theta)
        cosphi = np.cos(phi)
        cospsi = np.cos(psi)
        sinthe = np.sin(theta)
        sinphi = np.sin(phi)
        sinpsi = np.sin(psi)

        rotation = np.zeros((3, 3))
        rotation[0, 0] = +costhe * cospsi - sinthe * sinphi * sinpsi
        rotation[0, 1] = -costhe * sinpsi - sinthe * sinphi * cospsi
        rotation[0, 2] = sinthe * cosphi
        rotation[1, 0] = cosphi * sinpsi
        rotation[1, 1] = cosphi * cospsi
        rotation[1, 2] = sinphi
        rotation[2, 0] = -sinthe * cospsi - costhe * sinphi * sinpsi
        rotation[2, 1] = +sinthe * sinpsi - costhe * sinphi * cospsi
        rotation[2, 2] = costhe * cosphi
        return rotation

    @property
    def xyz(self):
        """Global position of the local-frame origin, as a writable view."""
        return self.matrix[:3, 3]

    @xyz.setter
    def xyz(self, value):
        self.matrix[:3, 3] = value

    @property
    def rotation(self):
        """Local-to-global orientation matrix, as a writable view."""
        return self.matrix[:3, :3]

    @rotation.setter
    def rotation(self, value):
        self.matrix[:3, :3] = value

    def copy(self):
        return Frame(self.matrix)

    def inverse(self):
        """Return the inverse rigid transform as a new frame."""
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation.T
        matrix[:3, 3] = -self.rotation.T @ self.xyz
        return Frame(matrix)

    def __matmul__(self, other):
        if not isinstance(other, Frame):
            return NotImplemented
        return Frame(self.matrix @ other.matrix)

    def transform(self, displacement=None, rotation=None):
        """Apply a local rigid transform represented by vector and matrix."""
        initial_rotation = self.rotation.copy()
        if displacement is not None:
            self.xyz[:] = self.xyz + initial_rotation @ displacement
        if rotation is not None:
            self.rotation[:] = initial_rotation @ rotation
        return self

    def _trans(self, dx=0, dy=0, ds=0):
        if dx != 0 or dy != 0 or ds != 0:
            self.transform(displacement=np.array([dx, dy, ds]))
        return self

    def trans_x(self, dx):
        """Translate along the local horizontal axis."""
        return self._trans(dx=dx)

    def trans_y(self, dy):
        """Translate along the local vertical axis."""
        return self._trans(dy=dy)

    def trans_s(self, ds):
        """Translate along the local longitudinal axis."""
        return self._trans(ds=ds)

    def _rot(self, rotation):
        return self.transform(rotation=rotation)

    def rot_x(self, angle):
        """Rotate around the local horizontal axis."""
        if angle == 0:
            return self
        c = np.cos(angle)
        s = np.sin(angle)
        return self._rot(np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ]))

    def rot_y(self, angle):
        """Rotate around the local vertical axis."""
        if angle == 0:
            return self
        c = np.cos(angle)
        s = np.sin(angle)
        return self._rot(np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c],
        ]))

    def rot_s(self, angle):
        """Rotate around the local longitudinal axis."""
        if angle == 0:
            return self
        c = np.cos(angle)
        s = np.sin(angle)
        return self._rot(np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ]))

    def arc(self, length=0, angle=0, tilt=0):
        """Advance along a MAD-X-compatible arc in the local frame."""
        if angle == 0:
            return self.trans_s(length)

        rho = length / angle
        ca = np.cos(angle)
        sa = np.sin(angle)
        ct = np.cos(tilt)
        st = np.sin(tilt)

        displacement = np.array([rho * (ca - 1), 0, rho * sa])
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
            rotation=tilt_rotation @ bend_rotation @ inverse_tilt_rotation,
        )

    def arc_x(self, length=0, angle=0):
        """Advance along an arc in the local horizontal-longitudinal plane."""
        return self.arc(length=length, angle=angle, tilt=0)

    def arc_y(self, length=0, angle=0):
        """Advance along an arc in the local vertical-longitudinal plane."""
        return self.arc(length=length, angle=angle, tilt=np.pi / 2)
