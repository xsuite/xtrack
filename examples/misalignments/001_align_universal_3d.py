from cernlayoutdb import MADPoint
import sys
import pyvista as pv
import xtrack as xt
import numpy as np
import pymadng as ng
from ducktrack.elements import Misalign, Realign


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


def wedge(length, angle):
    delta_x = - length * sinc(angle / 2) * np.sin(angle / 2)  # rho * (np.cos(angle) - 1)
    delta_theta = -angle
    delta_s = length * sinc(angle)  # rho * np.sin(angle)
    return MADPoint(src=(
            MADPoint(x=delta_x, z=delta_s).matrix
            @ MADPoint(theta=delta_theta).matrix
    ))


def draw_bend_3d(entry, length, angle, plotter, width=10, resolution=100, **kwargs):
    """Draw a straight or bent cuboid using extrusion in PyVista."""
    # Define a square cross-section in YZ plane
    half = width / 2
    square = pv.Polygon(
        normal=[0, 0, 1],
        n_sides=4,
        radius=half,
    )
    square = square.rotate_z(45, point=square.center)
    rho = length / angle if angle != 0 else 0
    square = square.translate(xyz=[rho, 0, 0])

    # Extrude the square along the spline
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

    # Transform the swept solid to the global frame
    swept = swept.translate(xyz=[-rho, 0, 0])
    swept.transform(entry.matrix, inplace=True)

    # Add to plotter
    plotter.add_mesh(swept, color=kwargs.pop('c', 'gray'), opacity=kwargs.pop('opacity', 0.2), **kwargs)


# # Element parameters
# length = 20
# angle = 0.3  # rad
#
# # Misalignment parameters
# dx = -15
# dy = 7
# dz = 20
# theta = -0.1  # rad
# phi = 0.11  # rad
# psi = np.pi / 4  # rad
# f = 0.4  # fraction of the element length for the misalignment

# Element parameters
length = 20
angle = 0.3  # rad
k0 = 0

# Misalignment parameters
dx = -15
dy = 7
dz = 23
theta = -0.1  # rad
phi = 0.11  # rad
psi = np.pi / 4  # rad
f = 0.4  # fraction of the element length for the misalignment

element = (
    xt.Bend(length=length, angle=angle, model='rot-kick-rot', k0=k0)
    if angle else xt.Solenoid(length=length)
)

ax = pv.Plotter()
ax.show_axes()
ax.enable_parallel_projection()  # For orthographic projection


# Calculate the transformations using pylayout and draw the world
misalign = MADPoint(x=dx, y=dy, z=dz, theta=theta, phi=phi, psi=psi)
first_half = wedge(f * length, f * angle)
second_half = wedge((1 - f) * length, (1 - f) * angle)
bend = wedge(length, angle)

p0 = MADPoint()
p1 = MADPoint(src=first_half.matrix @ misalign.matrix @ np.linalg.inv(first_half.matrix))  # element entry
pf = MADPoint(src=p1.matrix @ first_half.matrix)
p2 = MADPoint(src=first_half.matrix @ misalign.matrix @ second_half.matrix)  # element exit
p3 = MADPoint(src=p0.matrix @ bend.matrix)

plot_frame_zyx(p0, ax)
plot_frame_zyx(first_half, ax)
plot_frame_zyx(p1, ax)
plot_frame_zyx(pf, ax)
plot_frame_zyx(p2, ax)
plot_frame_zyx(p3, ax)

# Draw the elements
mis_entry = first_half.matrix @ misalign.matrix @ np.linalg.inv(first_half.matrix)
draw_bend_3d(p0, length, angle, plotter=ax)
draw_bend_3d(MADPoint(src=mis_entry), length, angle, plotter=ax, c='pink')


# Track the particles part by part
theta0, phi0, psi0 = p1.get_theta_phi_psi()

line = xt.Line(
    elements=[
        xt.XYShift(dx=p1.x, dy=p1.y),
        xt.Solenoid(length=p1.z),
        xt.YRotation(angle=np.rad2deg(theta0)),
        xt.XRotation(angle=np.rad2deg(-phi0)),  # angle flip, unsure why
        xt.SRotation(angle=np.rad2deg(psi0)),
    ]
)

line2 = xt.Line(elements=[element])

matrix_exit = p2.matrix
realign = np.linalg.inv(matrix_exit) @ bend.matrix
dr = MADPoint(src=realign)
theta1, phi1, psi1 = dr.get_theta_phi_psi()

line3 = xt.Line(
    elements=[
        xt.XYShift(dx=dr.x, dy=dr.y),
        xt.Solenoid(length=dr.z),
        xt.YRotation(angle=np.rad2deg(theta1)),
        xt.XRotation(angle=np.rad2deg(-phi1)),  # angle flip, unsure why
        xt.SRotation(angle=np.rad2deg(psi1)),
    ]
)

pp0 = xt.Particles(
    x=[0, 0.2, -0.4, -0.6, 0.8],
    y=[0, 0.2, 0.4, -0.6, -0.8],
    px=[0, -0.01, -0.01, 0.01, 0.01],
    py=[0, -0.01, 0.01, -0.01, 0.01],
)
pp1 = pp0.copy()

# From 0 to the misaligned element entry
line.track(pp1)
plot_trajectory(pp0, pp1, p0, p1, plotter=ax, color='red')

# From the misaligned element entry to the misaligned element exit
pp1old = pp1.copy()
line2.track(pp1)
plot_trajectory(pp1old, pp1, p1, p2, plotter=ax, color='orange')

# From the misaligned element exit to the straight-through element exit
pp1old = pp1.copy()
line3.track(pp1)
plot_trajectory(pp1old, pp1, p2, p3, plotter=ax, color='green')


# Track straight-through without misalignment
# pp2 = pp0.copy()
# line2.track(pp2)
# plot_trajectory(pp0, pp2, p0, p3, plotter=ax, opacity=0.2)
plot_trajectory_drift(pp0, p0, length=50, plotter=ax, color='gray', opacity=0.2)


# Test proper Misalignment elements
pp_element = pp0.copy()
plot_point(pp_element, p0, plotter=ax, shape='cube', color='black')
mis_entry = Misalign(dx=dx, dy=dy, ds=dz, theta=theta, phi=phi, psi=psi, length=length, location=f, angle=angle)
# mis_entry = xt.beam_elements.elements.Misalignment(dx=dx, dy=dy, ds=dz, theta=theta, phi=phi, psi=psi, length=length, location=f, angle=angle, is_exit=False)
mis_entry.track(pp_element)
plot_point(pp_element, p1, plotter=ax, shape='cube', color='red')
line2.track(pp_element)
plot_point(pp_element, p2, plotter=ax, shape='cube', color='orange')
mis_exit = Realign(dx=dx, dy=dy, ds=dz, theta=theta, phi=phi, psi=psi, length=length, location=f, angle=angle)
# mis_exit = xt.beam_elements.elements.Misalignment(dx=dx, dy=dy, ds=dz, theta=theta, phi=phi, psi=psi, length=length, location=f, angle=angle, is_exit=True)
mis_exit.track(pp_element)
plot_point(pp_element, p3, plotter=ax, shape='cube', color='green')


# Benchmark against MAD-NG
if f:
    print("Skipping MAD-NG tracking for elements with misalignment fraction f != 0")
    sys.exit()

mng = ng.MAD()
# X0 = np.array([pp0.x, pp0.px, pp0.y, pp0.py, pp0.zeta / pp0.beta0, pp0.ptau]).T
X0 = [
    {
        'x': float(pp0.x[i]),
        'px': float(pp0.px[i]),
        'y': float(pp0.y[i]),
        'py': float(pp0.py[i]),
        't': float(pp0.zeta[i] / pp0.beta0[0]),
        'pt': float(pp0.ptau[i]),
    } for i in range(len(pp0.x))
]

mng['length'] = length
mng['angle'] = angle
mng['k0'] = k0
mng['dx'] = dx
mng['dy'] = dy
mng['dz'] = dz
mng['theta'] = theta
mng['phi'] = phi
mng['psi'] = psi
mng['f'] = f

mng['X0'] = X0
mng['beta'] = pp0.beta0[0]

ng_script = """
local sequence, quadrupole, sbend in MAD.element
local track, beam in MAD

local elem = sbend 'elem' {
    l=length,
    angle=angle,
    k0=k0,
    misalign={
        dx=dx,
        dy=dy,
        ds=dz,
        dtheta=theta,
        dphi=phi,
        dpsi=psi
        -- location not supported yet
    }
}

local seq = sequence 'seq' { refer='entry',
    elem { at=0 }
}

mybeam = beam { particle='proton', beta=beta }
tbl = track {X0=X0, sequence=seq, nturn=1, beam=mybeam, observe=0, save='atall', aperture = { kind='square', 100 }}
"""
mng.send(ng_script)
tracked_df = mng.tbl.to_df()


def madng_to_particles(df, beta, at='$end', slice_=0):
    row = df[(df['name'] == at) & (df['slc'] == slice_)]
    coords = {coord: np.array(row[coord]) for coord in ['x', 'px', 'y', 'py', 't', 'pt']}
    coords['zeta'] = coords.pop('t') * beta
    coords['ptau'] = coords.pop('pt')
    return xt.Particles(beta0=beta, **coords)


pp_madng_0 = madng_to_particles(tracked_df, mng.mybeam.beta, at='elem', slice_=-1)  # before everything
pp_madng_1 = madng_to_particles(tracked_df, mng.mybeam.beta, at='elem', slice_=-3)  # after entry misalignment
pp_madng_2 = madng_to_particles(tracked_df, mng.mybeam.beta, at='elem', slice_=-2)  # after element
pp_madng_3 = madng_to_particles(tracked_df, mng.mybeam.beta, at='$end', slice_=0)  # after exit misalignment


plot_point(pp_madng_0, p0, shape='diamond', plotter=ax, color='black')
plot_point(pp_madng_1, p1, shape='diamond', plotter=ax, color='red')
plot_point(pp_madng_2, p2, shape='diamond', plotter=ax, color='orange')
plot_point(pp_madng_3, p3, shape='diamond', plotter=ax, color='green')

