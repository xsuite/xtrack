import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


class LDBPoint:
    """
    Point in the space of the layout database.

    Typically beams move along the X axis.
    The Y axis is horizontal and points towards the center of the ring.
    The Z axis is vertical and points upwards. This sets a left-handed
    coordinate system despite rotations follow a right handed definitions.

    Local coordinates are:
    - x: along the beam (S)
    - y: horizontal (U)
    - z: vertical (V)
    - rx: rotation around S (B)
    - ry: rotation around U (A)
    - rz: rotation around V (C)

    """

    format = "%.10g"

    def ldb2mad(p):
        return [-p[1], p[2], p[0]]

    def mad2ldb(p):
        return [p[2], -p[0], p[1]]

    def ldb2ccs(p):
        return [p[1], p[0], p[2]]

    def ccs2ldb(p):
        return [p[1], p[0], p[2]]

    @staticmethod
    def rx_matrix(angle):
        """Positive angle move y towards z and z towards -y"""
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array(
            [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]]
        )

    @staticmethod
    def ry_matrix(angle):
        """Positive angle move z towards x and x towards z"""
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array(
            [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]]
        )

    @staticmethod
    def rz_matrix(angle):
        """Positive angle move x towards y and y towards -x"""
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array(
            [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

    @staticmethod
    def matrix_to_rxyz(matrix):
        rz = -np.arctan2(matrix[0, 1], matrix[0, 0])
        cy = np.hypot(matrix[2, 2], matrix[1, 2])
        ry = np.arctan2(matrix[0, 2], cy)
        rx = -np.arctan2(matrix[1, 2], matrix[2, 2])
        return rx, ry, rz

    @staticmethod
    def rxryrz_to_matrix(rx, ry, rz):
        return (
            LDBPoint.rx_matrix(rx)
            @ LDBPoint.ry_matrix(ry)
            @ LDBPoint.rz_matrix(rz)
        )

    @staticmethod
    def sympy_rotation_matrix():
        """
        Matrix([
        [            cy*cz,           -cy*sz,     sy],
        [ cx*sz + cz*sx*sy, cx*cz - sx*sy*sz, -cy*sx],
        [-cx*cz*sy + sx*sz, cx*sy*sz + cz*sx,  cx*cy]])
        """
        import sympy

        rx, ry, rz = sympy.symbols("rx ry rz", real=True)
        # cx, cy, cz = sympy.symbols("cx cy cz")
        # sx, sy, sz = sympy.symbols("sx sy sz")
        cx, cy, cz = sympy.cos(rx), sympy.cos(ry), sympy.cos(rz)
        sx, sy, sz = sympy.sin(rx), sympy.sin(ry), sympy.sin(rz)
        mx = sympy.Matrix([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        my = sympy.Matrix([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        mz = sympy.Matrix([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        mm = mx @ my @ mz
        return mm

    @classmethod
    def from_ccspoint(cls, ccsp):
        mat = np.eye(4)
        mat[:3, 3] = LDBPoint.ccs2ldb(ccsp.xyz)
        mat[:3, 0] = LDBPoint.ccs2ldb(ccsp.dy)
        mat[:3, 1] = LDBPoint.ccs2ldb(ccsp.dx)
        mat[:3, 2] = LDBPoint.ccs2ldb(ccsp.dz)
        return cls(mat)

    def to_ccspoint(self):
        mat = np.eye(4)
        mat[:3, 3] = LDBPoint.ldb2ccs(self.xyz)
        mat[:3, 0] = LDBPoint.ldb2ccs(self.dy)
        mat[:3, 1] = LDBPoint.ldb2ccs(self.dx)
        mat[:3, 2] = LDBPoint.ldb2ccs(self.dz)
        return CCSPoint(mat)

    @classmethod
    def from_madpoint(cls, madp):
        mat = np.eye(4)
        mat[:3, 3] = LDBPoint.mad2ldb(madp.xyz)
        mat[:3, 0] = LDBPoint.mad2ldb(madp.dz)
        mat[:3, 1] = LDBPoint.mad2ldb(-madp.dx)
        mat[:3, 2] = LDBPoint.mad2ldb(madp.dy)
        return cls(mat)

    def to_madpoint(self):
        mat = np.eye(4)
        mat[:3, 3] = LDBPoint.ldb2mad(self.xyz)
        mat[:3, 0] = LDBPoint.ldb2mad(-self.dy)
        mat[:3, 1] = LDBPoint.ldb2mad(self.dz)
        mat[:3, 2] = LDBPoint.ldb2mad(self.dx)
        return MADPoint(mat)

    def __init__(self, src=None, x=0, y=0, z=0, rx=0, ry=0, rz=0):
        if src is None:
            self.matrix = np.eye(4)
            self.set_rxryrz(rx, ry, rz)
            self.xyz = np.array([x, y, z])
        elif isinstance(src, self.__class__):
            self.matrix = src.matrix.copy()
        else:
            self.matrix = np.array(src)

    def set_rxryrz_rad(self, rx, ry, rz):
        """Set angles in matrix

        This s local rotation around x then y then z
        """
        self.matrix[:3, :3] = LDBPoint.rxryrz_to_matrix(rx, ry, rz)[:3, :3]
        return self

    def set_rxryrz(self, rx, ry, rz):
        """Set angles in matrix

        This s local rotation around x then y then z
        """
        rx = np.deg2rad(rx)
        ry = np.deg2rad(ry)
        rz = np.deg2rad(rz)
        return self.set_rxryrz_rad(rx, ry, rz)

    def get_rxryrz_rad(self):
        """Return angles from matrix"

        This is local rotation around x then y then z
        """
        return LDBPoint.matrix_to_rxyz(self.matrix)

    def get_rxryrz(self):
        """Return angles from matrix"

        This is local rotation around x then y then z
        """
        return [np.rad2deg(vv) for vv in self.get_rxryrz_rad()]

    @property
    def rxyz_rad(self):
        """Return angles from matrix"""
        return self.get_rxryrz_rad()

    @rxyz_rad.setter
    def rxyz_rad(self, value):
        self.set_rxryrz_rad(*value)

    @property
    def rxyz(self):
        """Return angles from matrix"""
        return self.get_rxryrz()

    def rx_rad(self, angle):
        """Rotate around local x axis"""
        self.matrix @= LDBPoint.rx_matrix(angle)
        return self

    def ry_rad(self, angle):
        """Rotate around local y axis"""
        self.matrix @= LDBPoint.ry_matrix(angle)
        return self

    def rz_rad(self, angle):
        """Rotate around local z axis"""
        self.matrix @= LDBPoint.rz_matrix(angle)
        return self

    def rx(self, angle):
        """Rotate around local x axis"""
        return self.rx_rad(np.deg2rad(angle))

    def ry(self, angle):
        """Rotate around local y axis"""
        return self.ry_rad(np.deg2rad(angle))

    def rz(self, angle):
        """Rotate around local z axis"""
        return self.rz_rad(np.deg2rad(angle))

    @property
    def xyz(self):
        return self.matrix[:3, 3]

    @xyz.setter
    def xyz(self, value):
        self.matrix[:3, 3] = value

    @property
    def rmat(self):
        return self.matrix[:3, :3]

    @rmat.setter
    def rmat(self, value):
        self.matrix[:3, :3] = value

    @property
    def dx(self):
        return self.matrix[:3, 0]

    @dx.setter
    def dx(self, value):
        self.matrix[:3, 0] = value

    @property
    def dy(self):
        return self.matrix[:3, 1]

    @dy.setter
    def dy(self, value):
        self.matrix[:3, 1] = value

    @property
    def dz(self):
        return self.matrix[:3, 2]

    @dz.setter
    def dz(self, value):
        self.matrix[:3, 2] = value

    @property
    def x(self):
        return self.matrix[0, 3]

    @x.setter
    def x(self, value):
        self.matrix[0, 3] = value

    @property
    def y(self):
        return self.matrix[1, 3]

    @y.setter
    def y(self, value):
        self.matrix[1, 3] = value

    @property
    def z(self):
        return self.matrix[2, 3]

    @z.setter
    def z(self, value):
        self.matrix[2, 3] = value

    def moveby(self, dx=0, dy=0, dz=0):
        """Translate by a local vector"""
        self.matrix @= np.array(
            [[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]]
        )
        return self

    def tx(self, dx):
        return self.moveby(dx=dx)

    def ty(self, dy):
        return self.moveby(dy=dy)

    def tz(self, dz):
        return self.moveby(dz=dz)

    def arcby(self, length=0, angle=0, roll=0):
        """Move by an arc of angle and roll tangent to local x axis"

        If roll is 0, the arc is in the plane (x,y).
        If roll is 90, the arc is in the plane (y,z) going towards positive z.
        """
        if angle == 0:
            self.tx(length)
        else:
            rho = length / angle
            ca = np.cos(angle)
            sa = np.sin(angle)
            cr = np.cos(roll)
            sr = np.sin(roll)
            vv = np.array([rho * sa, rho * (1 - ca), 0])
            rot_v = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
            rot_s = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            roti_s = np.array([[1, 0, 0], [0, cr, sr], [0, -sr, cr]])
            vv = rot_s @ vv
            ss = rot_s @ rot_v @ roti_s
            mat = self.rmat
            pos = mat @ vv + self.xyz
            self.matrix[:3, :3] = mat @ ss
            self.matrix[:3, 3] = pos
        return self

    def arcby_deg(self, length=0, angle=0, roll=0):
        """as arcby but with angles in degrees"""

        return self.arcby(length, np.deg2rad(angle), np.deg2rad(roll))

    def clone(self):
        return LDBPoint(self.matrix)

    @property
    def c(self):
        return self.clone()

    def origin(self):
        self.matrix = np.eye(4)
        return self

    def __repr__(self):
        xyz = [self.format % vv for vv in self.xyz]
        rxyz = [self.format % vv for vv in self.rxyz]

        return f"LDBPoint(x={xyz[0]}, y={xyz[1]}, z={xyz[2]}, rx={rxyz[0]}, ry={rxyz[1]}, rz={rxyz[2]})"

    def __eq__(self, other):
        return np.allclose(self.matrix, other.matrix)

    def __matmul__(self, other):
        return LDBPoint(self.matrix @ other.matrix)

    def inv(self):
        return LDBPoint(np.linalg.inv(self.matrix))

    def distance(self, other):
        return np.sqrt(((other.xyz - self.xyz) ** 2).sum())

    def plot_xy(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        x = [self.x]
        y = [self.y]
        ax.plot(x, y, "o", **kwargs)
        return ax


class MADPoint:
    """
    Point in the MADX code

    Local coordinates are:
    - x: horizontal positive x points opposite to a positive bending angle
    - y: vertical
    - z: beam direction
    - theta: rotation around y
    - phi: rotation around x
    - psi: rotation around z
    """

    format = "%.10g"

    @staticmethod
    def mad2ccs(p):
        return [-p[0], p[2], p[1]]

    @staticmethod
    def ccs2mad(p):
        return [-p[0], p[2], p[1]]

    @staticmethod
    def theta_matrix(angle):
        """Positive angle move z towards x"""
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array(
            [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]]
        )

    @staticmethod
    def phi_matrix(angle):
        """Positive angle move z towards y"""
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array(
            [[1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0], [0, 0, 0, 1]]
        )

    @staticmethod
    def psi_matrix(angle):
        """Positive angle move x towards y"""
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array(
            [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

    @staticmethod
    def sympy_rotation_matrix():
        """
        Matrix([
        [            cy*cz,           -cy*sz,     sy],
        [ cx*sz + cz*sx*sy, cx*cz - sx*sy*sz, -cy*sx],
        [-cx*cz*sy + sx*sz, cx*sy*sz + cz*sx,  cx*cy]])
        """
        import sympy

        theta, phi, psi = sympy.symbols("theta phi psi", real=True)
        cx, cy, cz = sympy.symbols("cx cy cz")
        sx, sy, sz = sympy.symbols("sx sy sz")
        # cx, cy, cz = sympy.cos(rx), sympy.cos(ry), sympy.cos(rz)
        # sx, sy, sz = sympy.sin(rx), sympy.sin(ry), sympy.sin(rz)
        theta = sympy.Matrix([[cx, 0, sx], [0, 1, 0], [-sx, 0, cx]])
        phi = sympy.Matrix([[1, 0, 0], [0, cy, sy], [0, -sy, cy]])
        psi = sympy.Matrix([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        mm = theta @ phi @ psi
        return mm

    @classmethod
    def from_survey(self, table, row=0):
        return MADPoint(
            x=table.x[row],
            y=table.y[row],
            z=table.z[row],
            theta=table.theta[row],
            phi=table.phi[row],
            psi=table.psi[row],
        )

    @classmethod
    def from_ccspoint(cls, ccsp):
        mat = np.eye(4)
        mat[:3, 3] = MADPoint.ccs2mad(ccsp.xyz)
        mat[:3, 0] = MADPoint.ccs2mad(-ccsp.dx)
        mat[:3, 1] = MADPoint.ccs2mad(ccsp.dz)
        mat[:3, 2] = MADPoint.ccs2mad(ccsp.dy)
        return cls(mat)

    def to_ccspoint(self):
        mat = np.eye(4)
        mat[:3, 3] = MADPoint.mad2ccs(self.xyz)
        mat[:3, 0] = MADPoint.mad2ccs(-self.dx)
        mat[:3, 1] = MADPoint.mad2ccs(self.dz)
        mat[:3, 2] = MADPoint.mad2ccs(self.dy)
        return CCSPoint(mat)

    def __init__(self, src=None, x=0, y=0, z=0, theta=0, phi=0, psi=0):
        if src is None:
            self.matrix = np.eye(4)
            self.set_theta_phi_psi(theta, phi, psi)
            self.xyz = np.array([x, y, z])
        elif isinstance(src, self.__class__):
            self.matrix = src.matrix.copy()
        else:
            self.matrix = np.array(src)

    def set_theta_phi_psi(self, theta, phi, psi):
        self.matrix[:3, :3] = (
            MADPoint.theta_matrix(theta)
            @ MADPoint.phi_matrix(phi)
            @ MADPoint.psi_matrix(psi)
        )[:3, :3]

    def get_theta_phi_psi(self):
        matrix = self.matrix
        psi = np.arctan2(matrix[1, 0], matrix[1, 1])
        cy = np.hypot(matrix[1, 0], matrix[1, 1])
        phi = np.arctan2(matrix[1, 2], cy)
        theta = np.arctan2(matrix[0, 2], matrix[2, 2])
        return theta, phi, psi

    @property
    def xyz(self):
        return self.matrix[:3, 3]

    @xyz.setter
    def xyz(self, value):
        self.matrix[:3, 3] = value

    @property
    def rmat(self):
        return self.matrix[:3, :3]

    @rmat.setter
    def rmat(self, value):
        self.matrix[:3, :3] = value

    @property
    def x(self):
        return self.matrix[0, 3]

    @x.setter
    def x(self, value):
        self.matrix[0, 3] = value

    @property
    def y(self):
        return self.matrix[1, 3]

    @y.setter
    def y(self, value):
        self.matrix[1, 3] = value

    @property
    def z(self):
        return self.matrix[2, 3]

    @z.setter
    def z(self, value):
        self.matrix[2, 3] = value

    @property
    def dx(self):
        return self.matrix[:3, 0]

    @dx.setter
    def dx(self, value):
        self.matrix[:3, 0] = value

    @property
    def dy(self):
        return self.matrix[:3, 1]

    @dy.setter
    def dy(self, value):
        self.matrix[:3, 1] = value

    @property
    def dz(self):
        return self.matrix[:3, 2]

    @dz.setter
    def dz(self, value):
        self.matrix[:3, 2] = value

    def clone(self):
        return MADPoint(self.matrix)

    def __eq__(self, other):
        return np.allclose(self.matrix, other.matrix)

    @property
    def c(self):
        return self.clone()

    def __repr__(self):
        xyz = [self.format % vv for vv in self.xyz]
        an = [self.format % vv for vv in self.get_theta_phi_psi()]

        return f"MADPoint(x={xyz[0]}, y={xyz[1]}, z={xyz[2]}, theta={an[0]}, phi={an[1]}, psi={an[2]})"

    def distance(self, other):
        return np.sqrt(np.sum((self.xyz - other.xyz) ** 2))

    def rtheta(self, angle):
        """Rotate around x"""
        self.matrix @= MADPoint.theta_matrix(angle)
        return self

    def rphi(self, angle):
        """Rotate around y"""
        self.matrix @= MADPoint.phi_matrix(angle)
        return self

    def rpsi(self, angle):
        """Rotate around z"""
        self.matrix @= MADPoint.psi_matrix(angle)
        return self

    def moveby(self, dx=0, dy=0, dz=0):
        """Translate by a local vector"""
        self.matrix @= np.array(
            [[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]]
        )
        return self

    def tx(self, dx):
        """Translate by a local vector along x"""
        return self.moveby(dx=dx)

    def ty(self, dy):
        """Translate by a local vector along y"""
        return self.moveby(dy=dy)

    def tz(self, dz):
        """Translate by a local vector along z"""
        return self.moveby(dz=dz)

    def arcby(self, length=0, angle=0, roll=0):
        """Move by an arc of angle and roll tangent to local z axis

        If roll is 0, the arc is in the plane (x,z).
        If roll is 90, the arc is in the plane (y,z) going towards positive y.
        """
        if angle == 0:
            self.tz(length)
        else:
            rho = length / angle
            ca = np.cos(angle)
            sa = np.sin(angle)
            cr = np.cos(roll)
            sr = np.sin(roll)
            vv = np.array([0, rho * (1 - ca), rho * sa])
            rot_v = np.array([[ca, 0, -sa], [0, 1, 0], [sa, 0, ca]])
            rot_s = np.array([[cr, sr, 0], [-sr, cr, 0], [0, 0, 1]])
            roti_s = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
            vv = rot_s @ vv
            ss = rot_s @ rot_v @ roti_s
            mat = self.rmat
            pos = mat @ vv + self.xyz
            self.matrix[:3, :3] = mat @ ss
            self.matrix[:3, 3] = pos
        return self

    def inv(self):
        """Return the inverse of the point"""
        return MADPoint(np.linalg.inv(self.matrix))

    def __matmul__(self, other):
        return MADPoint(self.matrix @ other.matrix)

    def to_ldbpoint(self):
        return LDBPoint.from_madpoint(self)


class CCSPoint:
    """
    Cern coordinate system point

    Default
    - x: horizontal positive points towards a positive bending angle
    - y: beam direction
    - z: up
    - theta: rotation around local z
    - phi: rotation around local x
    - psi: rotation around local y
    """

    format = "%.10g"

    @staticmethod
    def theta_matrix(angle):
        """Positive angle move y towards x"""
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array(
            [[c, s, 0, 0], [-s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

    @staticmethod
    def phi_matrix(angle):
        """Positive angle move y towards z"""
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array(
            [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]]
        )

    @staticmethod
    def psi_matrix(angle):
        """Positive angle move x towards negative z"""
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array(
            [[c, 0, -s, 0], [0, 1, 0, 0], [s, 0, c, 0], [0, 0, 0, 1]]
        )

    @staticmethod
    def sympy_rotation_matrix():
        """
                Matrix([
        [cy*cz - sx*sy*sz, -cx*sz,  cy*sx*sz + cz*sy],
        [cy*sz + cz*sx*sy,  cx*cz, -cy*cz*sx + sy*sz],
        [          -cx*sy,     sx,             cx*cy]])
        """
        import sympy

        theta, phi, psi = sympy.symbols("theta phi psi", real=True)
        cx, cy, cz = sympy.symbols("cx cy cz")
        sx, sy, sz = sympy.symbols("sx sy sz")
        # cx, cy, cz = sympy.cos(rx), sympy.cos(ry), sympy.cos(rz)
        # sx, sy, sz = sympy.sin(rx), sympy.sin(ry), sympy.sin(rz)
        theta = sympy.Matrix([[cz, sz, 0], [-sz, cz, 0], [0, 0, 1]])
        phi = sympy.Matrix([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        psi = sympy.Matrix([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]])
        mm = theta @ phi @ psi
        return mm

    def __init__(
        self, src=None, x=0, y=0, z=0, theta=0, phi=0, psi=0, theta_gon=None
    ):
        if src is None:
            self.matrix = np.eye(4)
            if theta_gon is not None:
                theta = theta_gon / 200 * np.pi
            self.set_theta_phi_psi(theta, phi, psi)
            self.xyz = np.array([x, y, z])
        elif isinstance(src, self.__class__):
            self.matrix = src.matrix.copy()
        else:
            self.matrix = np.array(src)

    def set_theta_phi_psi(self, theta, phi, psi):
        self.matrix[:3, :3] = (
            CCSPoint.theta_matrix(theta)
            @ CCSPoint.phi_matrix(phi)
            @ CCSPoint.psi_matrix(psi)
        )[:3, :3]

    def get_theta_phi_psi(self):
        matrix = self.matrix
        psi = np.arctan2(matrix[2, 0], matrix[2, 2])
        cx = np.hypot(matrix[0, 1], matrix[1, 1])
        phi = np.arctan2(matrix[2, 1], cx)
        theta = np.arctan2(matrix[0, 1], matrix[1, 1])
        return theta, phi, psi

    @property
    def xyz(self):
        return self.matrix[:3, 3]

    @xyz.setter
    def xyz(self, value):
        self.matrix[:3, 3] = value

    @property
    def rmat(self):
        return self.matrix[:3, :3]

    @rmat.setter
    def rmat(self, value):
        self.matrix[:3, :3] = value

    @property
    def x(self):
        return self.matrix[0, 3]

    @x.setter
    def x(self, value):
        self.matrix[0, 3] = value

    @property
    def y(self):
        return self.matrix[1, 3]

    @y.setter
    def y(self, value):
        self.matrix[1, 3] = value

    @property
    def z(self):
        return self.matrix[2, 3]

    @z.setter
    def z(self, value):
        self.matrix[2, 3] = value

    @property
    def dx(self):
        return self.matrix[:3, 0]

    @dx.setter
    def dx(self, value):
        self.matrix[:3, 0] = value

    @property
    def dy(self):
        return self.matrix[:3, 1]

    @dy.setter
    def dy(self, value):
        self.matrix[:3, 1] = value

    @property
    def dz(self):
        return self.matrix[:3, 2]

    @dz.setter
    def dz(self, value):
        self.matrix[:3, 2] = value

    @property
    def theta(self):
        return self.get_theta_phi_psi()[0]

    @theta.setter
    def theta(self, value):
        _, phi, psi = self.get_theta_phi_psi()
        self.set_theta_phi_psi(value, phi, psi)

    @property
    def theta_gon(self):
        return self.theta / np.pi * 200

    @theta_gon.setter
    def theta_gon(self, value):
        theta = value / 200 * np.pi
        _, phi, psi = self.get_theta_phi_psi()
        self.set_theta_phi_psi(theta, phi, psi)

    @property
    def phi(self):
        return self.get_theta_phi_psi()[1]

    @phi.setter
    def phi(self, value):
        theta, _, psi = self.get_theta_phi_psi()
        self.set_theta_phi_psi(theta, value, psi)

    @property
    def psi(self):
        return self.get_theta_phi_psi()[2]

    @psi.setter
    def psi(self, value):
        theta, phi, _ = self.get_theta_phi_psi()
        self.set_theta_phi_psi(theta, phi, value)

    def clone(self):
        return CCSPoint(self.matrix)

    def __eq__(self, other):
        return np.allclose(self.matrix, other.matrix)

    @property
    def c(self):
        return self.clone()

    def distance(self, other):
        return np.sqrt(np.sum((self.xyz - other.xyz) ** 2))

    def __repr__(self):
        xyz = [self.format % vv for vv in self.xyz]
        an = [self.format % vv for vv in self.get_theta_phi_psi()]

        return f"CCSPoint(x={xyz[0]}, y={xyz[1]}, z={xyz[2]}, theta={an[0]}, phi={an[1]}, psi={an[2]})"

    def to_madpoint(self):
        return MADPoint.from_ccspoint(self)

    def to_ldbpoint(self):
        return LDBPoint.from_ccspoint(self)


class LDBMultiLine:
    """ "
    A list of segments in the LDB
    """

    def __init__(self, points=None):
        if points is None:
            self.points = []
        else:
            self.points = points

    def plot_xy(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        x = [p.x for p in self.points]
        y = [p.y for p in self.points]
        ax.plot(x, y, **kwargs)
        return ax

    def __repr__(self):
        return f"<LDBMultiLine {len(self.points)} points>"


class LDBLine:
    """
    A line in the LDB
    """

    def __init__(self, a, b):
        self.a = a
        self.b = b

    @property
    def mid(self):
        midpoint = self.a.clone()
        midpoint.xyz = (self.a.xyz + self.b.xyz) / 2
        return midpoint

    @property
    def length(self):
        return self.a.distance(self.b)

    @property
    def points(self):
        return [self.a, self.b]

    def __repr__(self):
        return f"LDBLine({self.a}, {self.b})"

    def plot_xy(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        x = [p.x for p in self.points]
        y = [p.y for p in self.points]
        ax.plot(x, y, **kwargs)
        return ax


class LDBPath:
    def __init__(self, segments, start=None):
        self.segments = segments
        self.length, self.angle, self.roll = np.array(segments).T
        self.dcum = np.r_[[0], np.cumsum(self.length)]
        self.points = []
        if start is None:
            self.start = LDBPoint()
        else:
            self.start = start
        st = self.start.clone()
        for length, angle, roll in segments:
            self.points.append(st)
            st = st.c.arcby(length, angle, roll)
        self.points.append(st)

    def get_point(self, s):
        if s < 0 or s > self.dcum[-1]:
            return ValueError("out of range")
        i = np.searchsorted(self.dcum, s, "right") - 1
        ds = s - self.dcum[i]
        if ds == 0:
            return self.points[i]
        else:
            return self.points[i].c.arcby(
                ds, self.angle[i] / self.length[i] * ds, self.roll[i]
            )

    def get_curve(self, steps=11):
        points = []
        for i in range(1, len(self.dcum)):
            ss = np.linspace(self.dcum[i - 1], self.dcum[i], steps)
            for s in ss:
                points.append(self.get_point(s))
        return points

    def plot_xy(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        points = self.get_curve()
        x = [p.x for p in points]
        y = [p.y for p in points]
        ax.plot(x, y)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")
        return ax


class XYStrip:
    def __init__(self, path, width=0.1):
        self.path = path
        self.width = width

    def plot_xy(self, ax=None, color="blue", **kwargs):
        if ax is None:
            ax = plt.gca()
        points = self.path.get_curve()
        x = [p.x for p in points]
        y = [p.y for p in points]
        width = np.broadcast_to(self.width, (len(points),))
        pp = [p.c.tx(w / 2) for p, w in zip(points, width)]
        pm = [p.c.tx(-w / 2) for p, w in zip(points, width)]
        ax.plot(x, y, color=color, **kwargs)
        xy = [(p.x, p.y) for p in pp]
        xy.extend((p.x, p.y) for p in pm[::-1])
        pol = Polygon(xy, closed=True, fill=True, color=color, alpha=0.2)
        ax.add_patch(pol)
        return ax
