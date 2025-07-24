from typing import Literal

from cernlayoutdb import MADPoint
import sys
import xtrack as xt
import numpy as np
import pymadng as ng
from ducktrack.elements import Misalign, Realign
from cpymad.madx import Madx
from plotting import *


def wedge(length, angle):
    delta_x = - length * sinc(angle / 2) * np.sin(angle / 2)  # rho * (np.cos(angle) - 1)
    delta_theta = -angle
    delta_s = length * sinc(angle)  # rho * np.sin(angle)
    return MADPoint(src=(
            MADPoint(x=delta_x, z=delta_s).matrix
            @ MADPoint(theta=delta_theta).matrix
    ))


def wedge_roll(length, angle, tilt):
    wedge_matrix = wedge(length, angle).matrix
    return MADPoint(src=(
        MADPoint(psi=tilt).matrix
        @ wedge_matrix
        @ MADPoint(psi=-tilt).matrix
    ))

# Use xtrack elements or ducktrack?
mode: Literal['xt', 'ducktrack'] = 'ducktrack'

# Element parameters
length = 20
angle = 0.3  # rad
tilt = np.pi / 2  # rad, should point generally downwards
k0 = 0
k1 = 0

# Misalignment parameters
dx = -15
dy = 7
dz = 23
theta = -0.1  # rad
phi = 0.11  # rad
psi = 0.2  # rad
f = 0.4  # fraction of the element length for the misalignment

element = (
    xt.Bend(length=length, angle=angle, model='rot-kick-rot', k0=k0, rot_s_rad=tilt)
    if angle else xt.DriftExact(length=length, rot_s_rad=tilt)
)
# element = xt.Quadrupole(length=length, k1=k1, rot_s_rad=tilt)

ax = pv.Plotter()
ax.show_axes()
ax.enable_parallel_projection()  # For orthographic projection


# Calculate the transformations using pylayout and draw the world
misalign = MADPoint(x=dx, y=dy, z=dz, theta=theta, phi=phi, psi=psi)
first_half = wedge_roll(f * length, f * angle, tilt)
second_half = wedge_roll((1 - f) * length, (1 - f) * angle, tilt)
bend = wedge_roll(length, angle, tilt)

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
draw_bend_3d(p0, length, angle, tilt, plotter=ax)
draw_bend_3d(MADPoint(src=mis_entry), length, angle, tilt, plotter=ax, c='pink')


# Track the particles part by part
theta0, phi0, psi0 = p1.get_theta_phi_psi()

line = xt.Line(
    elements=[
        xt.XYShift(dx=p1.x, dy=p1.y),
        xt.DriftExact(length=p1.z),
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
        xt.DriftExact(length=dr.z),
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
pref = pp0.copy()
line2.reset_s_at_end_turn = False
line2.track(pref)
# plot_trajectory(pp0, pp2, p0, p3, plotter=ax, color='black', opacity=0.7)
plot_trajectory_drift(pp0, p0, length=50, plotter=ax, color='gray', opacity=0.2)


# Test proper Misalignment elements
pp_element = pp0.copy()
plot_point(pp_element, p0, plotter=ax, shape='cube', color='black')

if mode == 'ducktrack':
    mis_entry = Misalign(dx=dx, dy=dy, ds=dz, theta=theta, phi=phi, psi=psi, length=length, anchor=f, angle=angle, tilt=tilt)  # noqa
else:
    mis_entry = xt.Misalignment(dx=dx, dy=dy, ds=dz, theta=theta, phi=phi, psi=psi, length=length, anchor=f, angle=angle, tilt=tilt, is_exit=False)
mis_entry.track(pp_element)

plot_point(pp_element, p1, plotter=ax, shape='cube', color='red')
pprec1 = pp_element.copy()
line2.track(pp_element)
plot_point(pp_element, p2, plotter=ax, shape='cube', color='orange')

if mode == 'ducktrack':
    mis_exit = Realign(dx=dx, dy=dy, ds=dz, theta=theta, phi=phi, psi=psi, length=length, anchor=f, angle=angle, tilt=tilt)  # noqa
else:
    mis_exit = xt.Misalignment(dx=dx, dy=dy, ds=dz, theta=theta, phi=phi, psi=psi, length=length, anchor=f, angle=angle, tilt=tilt, is_exit=True)
pprec2 = pp_element.copy()
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
mng['beta'] = pp0.beta0[0]
mng['k0'] = k0
mng['k1'] = k1
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
local sequence, quadrupole, sbend, drift in MAD.element
local track, beam in MAD

local elem
local misalign = {
    dx=dx,
    dy=dy,
    ds=dz,
    dtheta=theta,
    dphi=phi,
    dpsi=psi
    -- anchor not supported yet
}

if k0 ~= 0 then
    elem = sbend 'elem' {
        l=length,
        angle=angle,
        k0=k0,
        misalign=misalign
    }
elseif k1 ~= 0 then
    elem = quadrupole 'elem' {
        l=length,
        k1=k1,
        misalign=misalign
    }
else
    elem = drift 'elem' {
        l=length,
        misalign=misalign
    }
end

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


# MAD-X
mad = Madx()

madx_particles = '\n'.join([
    f'start, x={pp0.x[i]}, px={pp0.px[i]}, y={pp0.y[i]}, py={pp0.py[i]};'
    for i in range(len(pp0.x))
])

mad.input(
    f"""
    beam, particle=proton, beta={pp0.beta0[0]};

    quad: quadrupole,
        l={length},
        k1={k1},
        dx={dx},
        dy={dy},
        ds={dz},
        dtheta={theta},
        dphi={phi},
        dpsi={psi};

    seq: sequence, l=10, refer=entry;
        quad, at=0;
    endsequence;

    use, sequence=seq;

    track;
        {madx_particles}
        run, turns=1;
    endtrack;
"""
)

turn = mad.table.tracksumm.turn
mask = np.where(turn == 1)
pp_madx_3 = xt.Particles(
    x=mad.table.tracksumm.x[mask],
    px=mad.table.tracksumm.px[mask],
    y=mad.table.tracksumm.y[mask],
    py=mad.table.tracksumm.py[mask],
)
plot_point(pp_madx_3, p3, shape='sphere', plotter=ax, color='green')
