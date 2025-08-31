from cernlayoutdb import MADPoint
import pyvista as pv
import numpy as np


def plot_frame_zyx(madpoint, plotter, scale=1.0):
    origin = madpoint.xyz
    dz = madpoint.dz * scale
    dx = madpoint.dx * scale
    dy = madpoint.dy * scale

    # Z-direction (blue)
    plotter.add_arrows(origin, dz, mag=1.0, color='blue')
    # X-direction (red)
    plotter.add_arrows(origin, dx, mag=1.0, color='red')
    # Y-direction (green)
    plotter.add_arrows(origin, dy, mag=1.0, color='green')

    # Add origin as a sphere or point
    plotter.add_mesh(pv.Sphere(radius=0.05, center=origin), color='black')

    # Add dashed guideline along dx and dy
    # line_x = pv.Line(origin - 5 * dx, origin + 20 * dx)
    # line_y = pv.Line(origin - 5 * dy, origin + 20 * dy)
    # plotter.add_mesh(line_x, color='black', style='wireframe', opacity=0.4)
    # plotter.add_mesh(line_y, color='black', style='wireframe', opacity=0.4)
    plane_center = origin
    plane_normal = np.cross(dx, dy)
    plane = pv.Plane(
        center=plane_center, direction=plane_normal, i_size=40, j_size=40
    )
    plotter.add_mesh(plane, color='black', opacity=0.1, style='wireframe')


def plot_trajectory(p0, p1, frame0, frame1, plotter, **kwargs):
    xyz0 = np.array([frame0.xyz + x * frame0.dx + y * frame0.dy for x, y in zip(p0.x, p0.y)])
    xyz1 = np.array([frame1.xyz + x * frame1.dx + y * frame1.dy for x, y in zip(p1.x, p1.y)])

    for v0, v1 in zip(xyz0, xyz1):
        line = pv.Line(v0, v1)
        plotter.add_mesh(line, color=kwargs.get('color', 'blue'), line_width=kwargs.get('linewidth', 4))


def plot_trajectory_drift(p0, frame0, length, plotter, **kwargs):
    for x, px, y, py, delta in zip(p0.x, p0.px, p0.y, p0.py, p0.delta):
        pz = np.sqrt((1 + delta) ** 2 - px ** 2 - py ** 2)
        xp = px / pz  # = dpx / ds
        yp = py / pz  # = dpy / ds
        dp_ds = xp * frame0.dx + yp * frame0.dy + frame0.dz
        start = frame0.xyz + x * frame0.dx + y * frame0.dy
        end = start + length * dp_ds

        line = pv.Line(start, end)
        plotter.add_mesh(line, color=kwargs.get('color', 'blue'), line_width=kwargs.get('linewidth', 4))



def plot_point(p0, frame0, plotter, shape='sphere', size=0.1, **kwargs):
    xyz0 = np.array([frame0.xyz + x * frame0.dx + y * frame0.dy for x, y in zip(p0.x, p0.y)])

    def _plot_single_point(v0):
        if shape == 'sphere':
            sphere = pv.Sphere(radius=kwargs.get('radius', size), center=v0)
            plotter.add_mesh(sphere, color=kwargs.get('color', 'red'))
        elif shape == 'cube':
            cube = pv.Cube(
                center=v0,
                x_length=kwargs.get('size', 2 * size),
                y_length=kwargs.get('size', 2 * size),
                z_length=kwargs.get('size', 2 * size),
            )
            plotter.add_mesh(cube, color=kwargs.get('color', 'red'))
        if shape == 'diamond':
            def _make_pyramid(direction):
                return pv.Pyramid([
                    np.array([-size, -size, 0]),
                    np.array([size, -size, 0]),
                    np.array([size, size, 0]),
                    np.array([-size, size, 0]),
                    np.array([0, 0, direction * np.sqrt(2) * size]),
                ]).rotate_z(45).translate(v0)
            plotter.add_mesh(_make_pyramid(1), color=kwargs.get('color', 'red'))
            plotter.add_mesh(_make_pyramid(-1), color=kwargs.get('color', 'red'))

    for v0 in xyz0:
        _plot_single_point(v0)


def sinc(x):
    return np.sinc(x / np.pi)  # np.sinc is normalized to pi, so we divide by pi


def draw_bend_3d(entry, length, angle, tilt, plotter, width=10, resolution=100, **kwargs):
    """Draw a straight or bent cuboid using extrusion in PyVista."""
    # Define a square cross-section in YZ plane
    half = width / 2
    diamond = pv.Polygon(
        normal=[0, 0, 1],
        n_sides=4,
        radius=half,
    )
    square = diamond.rotate_z(45, point=diamond.center, inplace=False)
    rho = length / angle if angle != 0 else 0

    # Since we rotate around the origin, offset the square by radius of curvature
    square.translate(xyz=[rho, 0, 0], inplace=True)

    # Extrude the square to make a cuboid or a wedge, depending on the curvature
    if angle:
        swept = square.extrude_rotate(
            resolution=resolution,
            angle=np.rad2deg(-angle),
            rotation_axis=np.array([0, 1, 0]),
            capping=True,
        )
    else:
        swept = square.extrude(
            vector=[0, 0, length],
            capping=True,
        )

    # Move back to the origin
    swept.translate(xyz=[-rho, 0, 0], inplace=True)

    # Apply the roll
    swept.rotate_z(np.rad2deg(tilt), inplace=True)

    # Move to the target reference frame
    swept.transform(entry.matrix, inplace=True)

    # Add to plotter
    plotter.add_mesh(
        swept,
        color=kwargs.pop('c', 'gray'),
        opacity=kwargs.pop('opacity', 0.2),
        **kwargs,
    )
