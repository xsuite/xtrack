import numpy as np
import matplotlib.pyplot as plt


def plot_point(ax, p, **kwargs):
    ax.scatter3D(p[2], p[0], p[1], **kwargs)


def plot_plane(ax, plane, size=5.0, resolution=2, color="cyan", alpha=0.2):
    plane = np.asarray(plane)

    origin = plane[:3, 3]
    x_axis = plane[:3, 0]
    y_axis = plane[:3, 1]

    u = np.linspace(-size, size, resolution)
    v = np.linspace(-size, size, resolution)

    U, V = np.meshgrid(u, v)

    X = origin[0] + x_axis[0]*U + y_axis[0]*V
    Y = origin[1] + x_axis[1]*U + y_axis[1]*V
    Z = origin[2] + x_axis[2]*U + y_axis[2]*V

    ax.plot_surface(Z, X, Y, color=color, alpha=alpha)


def arc_matrix(length: float, angle: float, tilt: float, eps: float = 1e-9) -> np.ndarray:
    if abs(angle) < eps:
        T = np.eye(4, dtype=float)
        T[2, 3] = length
        return T

    ct, st = np.cos(tilt), np.sin(tilt)
    ca, sa = np.cos(angle), np.sin(angle)

    dx = length * (ca - 1.0) / angle
    ds = length * sa / angle

    T = np.array([
        [ct * ca, -st,     -ct * sa,  ct * dx],
        [st * ca,  ct,     -st * sa,  st * dx],
        [sa,       0.0,     ca,       ds],
        [0.0,      0.0,     0.0,      1.0],
    ], dtype=float)
    return T


def plane_pose_to_point_and_normal(plane):
    plane = np.asarray(plane)
    T = plane[:3, 3]
    n = plane[:3, 2]
    return T, n


def line_segment_plane_intersect(start, end, plane_point, plane_normal, eps=1e-9):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    T = plane_point
    n = plane_normal

    # A - T and B - T
    ta = start - T
    tb = end - T

    # Dot products
    n_dot_ta = np.dot(n, ta)   # n · (A - T)
    n_dot_tb = np.dot(n, tb)   # n · (B - T)

    # Degenerate segment: zero length
    ab = end - start
    length_sq = np.dot(ab, ab)

    if length_sq < eps * eps:
        if abs(n_dot_ta) < eps:
            return 0.0
        return np.nan

    # n · (B - A)
    n_dot_ab = n_dot_tb - n_dot_ta

    if abs(n_dot_ab) < eps:
        a_on_plane = abs(n_dot_ta) < eps
        b_on_plane = abs(n_dot_tb) < eps

        if a_on_plane and b_on_plane:
            return 0.0
        if a_on_plane:
            return 0.0
        if b_on_plane:
            return 1.0

        return np.nan

    return -n_dot_ta / n_dot_ab


def curvilinear_to_cartesian_point(p_curv, h):
    """
    Curvilinear (x, y, s) -> Cartesian (X, Y, Z)
    """
    x, y, s = np.asarray(p_curv, dtype=float)

    if np.isclose(h, 0.0):
        return np.array([x, y, s], dtype=float)

    R = 1.0 / h
    theta = h * s
    c = np.cos(theta)
    sn = np.sin(theta)

    X = (R + x) * c - R
    Y = y
    Z = (R + x) * sn

    return np.array([X, Y, Z], dtype=float)


def cartesian_to_curvilinear_point(p_cart, h):
    """
    Cartesian (X, Y, Z) -> Curvilinear (x, y, s)
    """
    X, Y, Z = np.asarray(p_cart, dtype=float)

    if np.isclose(h, 0.0):
        return np.array([X, Y, Z], dtype=float)

    R = 1.0 / h

    # Since X + R = (R + x) cos(theta), Z = (R + x) sin(theta)
    rho = np.hypot(X + R, Z)

    x = rho - R
    y = Y
    theta = np.arctan2(Z, X + R)
    s = theta / h

    return np.array([x, y, s], dtype=float)


def cartesian_vector_to_curvilinear_at_point(p_curv, v_cart, h):
    """
    Convert an attached Cartesian vector to curvilinear components
    using the local inverse Jacobian for your arc_matrix convention.
    """
    x, _, s = np.asarray(p_curv, dtype=float)
    VX, VY, VZ = np.asarray(v_cart, dtype=float)

    if np.isclose(h, 0.0):
        return np.array([VX, VY, VZ], dtype=float)

    theta = h * s
    c = np.cos(theta)
    sn = np.sin(theta)
    scale_s = 1.0 + h * x

    if np.isclose(scale_s, 0.0):
        raise ValueError("Jacobian is singular at x = -1/h.")

    vx = c * VX + sn * VZ
    vy = VY
    vs = (-sn * VX + c * VZ) / scale_s

    return np.array([vx, vy, vs], dtype=float)


ax = plt.figure().add_subplot(projection='3d')
# ax_str = plt.figure().add_subplot(projection='3d')

# Plot the frame
h = 0.1
origin = np.array([[0, 0, 0, 1]]).T
points = np.array([arc_matrix(s, s * h, 0) @ origin for s in np.linspace(-2, 12, 100)])
ax.plot3D(points[:, 2], points[:, 0], points[:, 1], color='gray', linestyle='--')

# Intersecting plane
s_plane = 5
plane = arc_matrix(s_plane, s_plane * h, 0)
plot_plane(ax, plane)

for th in np.linspace(0, 2 * np.pi, 20):
    # Generate face 1
    x0_loc = np.cos(th) * 4
    y0_loc = np.sin(th) * 4
    z0_loc = 0
    p0 = np.array([[x0_loc, y0_loc, z0_loc]]).T
    p0_hom = np.vstack([p0, 1])
    p0 = p0_hom

    # Generate face 2
    x1_loc = x0_loc
    y1_loc = y0_loc
    z1_loc = 0
    s1 = 10
    p1 = np.array([[x1_loc, y1_loc, z1_loc]]).T
    p1_hom = np.vstack([p1, 1])
    p1 = arc_matrix(s1, s1 * h, 0) @ p1_hom

    # Plot the faces
    plot_point(ax, p0, c='r')
    plot_point(ax, p1, c='b')

    # Plane point and normal
    p_plane, n_plane = plane_pose_to_point_and_normal(plane)

    # Intersect in straight frame
    t_straight = line_segment_plane_intersect(p0[:3].flatten(), p1[:3].flatten(), p_plane, n_plane)
    p_interp_straight = p0[:3] + t_straight * (p1[:3] - p0[:3])
    plot_point(ax, p_interp_straight, c='cyan')

    # Intersect in curved frame
    p0_curv = cartesian_to_curvilinear_point(p0[:3].flatten(), h)
    p1_curv = cartesian_to_curvilinear_point(p1[:3].flatten(), h)
    p_plane_curv = cartesian_to_curvilinear_point(p_plane, h)
    n_plane_curv = cartesian_vector_to_curvilinear_at_point(p_plane_curv, n_plane, h)
    t_curv = line_segment_plane_intersect(p0_curv[:3].flatten(), p1_curv[:3].flatten(), p_plane_curv, n_plane_curv)
    p_interp_curv = p0_curv[:3] + t_curv * (p1_curv[:3] - p0_curv[:3])
    p_interp_projected_back = curvilinear_to_cartesian_point(p_interp_curv, h)
    plot_point(ax, p_interp_projected_back, c='green')


ax.set_xlabel('Z [m]')
ax.set_ylabel('X [m]')
ax.set_zlabel('Y [m]')
ax.auto_scale_xyz([0, 12], [-6, 6], [-6, 6])
plt.show()