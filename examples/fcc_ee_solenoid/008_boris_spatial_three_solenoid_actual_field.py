import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

from tilted_solenoid import TiltedSolenoid
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField


THETA = -0.015
PARTICLE = 'positron'
ENERGY0 = 45.6e9

BETX = 0.09
BETY = 0.0007

MAIN_FIELD_HALF_RANGE = 7.0
COMP_PREVIOUS_HALF_RANGE = 1.0
COMP_EXTRA_RANGE_EACH_SIDE = 3.0
COMP_FIELD_HALF_RANGE = COMP_PREVIOUS_HALF_RANGE + COMP_EXTRA_RANGE_EACH_SIDE

COMP_SOLENOID_DISTANCE_FROM_IP = 12.0
COMP_LEFT_CENTER = -COMP_SOLENOID_DISTANCE_FROM_IP - COMP_PREVIOUS_HALF_RANGE
COMP_RIGHT_CENTER = COMP_SOLENOID_DISTANCE_FROM_IP + COMP_PREVIOUS_HALF_RANGE

SLICE_LENGTH = 0.01
BORIS_STEPS_PER_SLICE = 10


# Actual field models. The main solenoid is tilted; the two anti-solenoids use
# the same axisymmetric field model as the FCC compensation solenoids.
main_field_model = TiltedSolenoid(L=1.23 * 2, a=0.13, B0=2.0, theta=THETA)
comp_field_model = SolenoidField(L=1.5, a=0.03, B0=1.0, z0=0.0)


# Set the anti-solenoid scale from the actual field integrals over the extended
# BorisSpatial integration windows used in this script.
main_integral_s = np.linspace(
    -MAIN_FIELD_HALF_RANGE, MAIN_FIELD_HALF_RANGE, 4001)
comp_integral_s_local = np.linspace(
    -COMP_FIELD_HALF_RANGE, COMP_FIELD_HALF_RANGE, 4001)
zero_main = np.zeros_like(main_integral_s)
zero_comp = np.zeros_like(comp_integral_s_local)

main_bs_integral = np.trapezoid(
    main_field_model.get_field(
        zero_main, zero_main, main_integral_s,
    )[2],
    main_integral_s,
)
comp_bs_integral_unscaled = np.trapezoid(
    comp_field_model.get_field(
        zero_comp, zero_comp, comp_integral_s_local,
    )[2],
    comp_integral_s_local,
)
comp_scale_b = -main_bs_integral / comp_bs_integral_unscaled / 2.0


def combined_field(x, y, s):
    x = np.asarray(x)
    y = np.asarray(y)
    s = np.asarray(s)

    bx = np.zeros_like(s, dtype=float)
    by = np.zeros_like(s, dtype=float)
    bs = np.zeros_like(s, dtype=float)

    main_mask = np.abs(s) <= MAIN_FIELD_HALF_RANGE
    if np.any(main_mask):
        bx_m, by_m, bs_m = main_field_model.get_field(
            x[main_mask],
            y[main_mask],
            s[main_mask],
        )
        bx[main_mask] += bx_m
        by[main_mask] += by_m
        bs[main_mask] += bs_m

    comp_left_s = s - COMP_LEFT_CENTER
    comp_left_mask = np.abs(comp_left_s) <= COMP_FIELD_HALF_RANGE
    if np.any(comp_left_mask):
        bx_c, by_c, bs_c = comp_field_model.get_field(
            x[comp_left_mask],
            y[comp_left_mask],
            comp_left_s[comp_left_mask],
        )
        bx[comp_left_mask] += comp_scale_b * bx_c
        by[comp_left_mask] += comp_scale_b * by_c
        bs[comp_left_mask] += comp_scale_b * bs_c

    comp_right_s = s - COMP_RIGHT_CENTER
    comp_right_mask = np.abs(comp_right_s) <= COMP_FIELD_HALF_RANGE
    if np.any(comp_right_mask):
        bx_c, by_c, bs_c = comp_field_model.get_field(
            x[comp_right_mask],
            y[comp_right_mask],
            comp_right_s[comp_right_mask],
        )
        bx[comp_right_mask] += comp_scale_b * bx_c
        by[comp_right_mask] += comp_scale_b * by_c
        bs[comp_right_mask] += comp_scale_b * bs_c

    return bx, by, bs


# Build one BorisSpatialIntegrator per slice over the union of the extended
# field windows, with an IP marker inserted at global s=0.
s_start_system = COMP_LEFT_CENTER - COMP_FIELD_HALF_RANGE
s_end_system = COMP_RIGHT_CENTER + COMP_FIELD_HALF_RANGE
n_slices = int(np.ceil((s_end_system - s_start_system) / SLICE_LENGTH))
s_edges = np.linspace(s_start_system, s_end_system, n_slices + 1)

elements = []
element_names = []
for ii in range(n_slices):
    s_start = s_edges[ii]
    s_end = s_edges[ii + 1]

    if np.isclose(s_start, 0.0, atol=1e-12):
        elements.append(xt.Marker())
        element_names.append('ip')
    elif s_start < 0.0 < s_end:
        raise RuntimeError('The BorisSpatial slice grid should contain s=0.')

    elements.append(xt.BorisSpatialIntegrator(
        fieldmap_callable=combined_field,
        s_start=s_start,
        s_end=s_end,
        n_steps=BORIS_STEPS_PER_SLICE,
    ))
    element_names.append(f'boris_{ii:04d}')

line_full = xt.Line(elements=elements, element_names=element_names)

ip_index = element_names.index('ip')
line_twiss = xt.Line(
    elements=[
        element.copy() if hasattr(element, 'copy') else element
        for element in elements[ip_index:]
    ],
    element_names=element_names[ip_index:],
)

particle_ref = xt.Particles(PARTICLE, energy0=ENERGY0)
line_full.particle_ref = particle_ref.copy()
line_twiss.particle_ref = particle_ref.copy()


# Symplecticity and open twiss with the same initial conditions used in 007.
S = xt.linear_normal_form.S
R_obj = line_full.get_R_matrix(
    particle_on_co=particle_ref.copy(),
    include_collective=True,
)
RR = R_obj['R_matrix']
symplectic_error = np.linalg.norm(RR.T @ S @ RR - S, ord=2)
det_r_error = abs(abs(np.linalg.det(RR)) - 1.0)

tw = line_twiss.twiss(
    betx=BETX,
    bety=BETY,
    include_collective=True,
)
s_from_ip = tw.s - tw['s', 'ip']


plt.close('all')

fig_orbit, axes_orbit = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
axes_orbit[0].plot(s_from_ip, tw.x, color='C0')
axes_orbit[1].plot(s_from_ip, tw.y, color='C0')
axes_orbit[0].set_ylabel('x [m]')
axes_orbit[1].set_ylabel('y [m]')
axes_orbit[1].set_xlabel('s - s_ip [m]')
for ax in axes_orbit:
    ax.grid(True, alpha=0.3)
fig_orbit.suptitle('BorisSpatial actual-field orbit, initialized at IP')
fig_orbit.tight_layout()

fig_dy, ax_dy = plt.subplots(figsize=(10, 4))
ax_dy.plot(s_from_ip, tw.dy, color='C0')
ax_dy.set_xlabel('s - s_ip [m]')
ax_dy.set_ylabel('dy [m]')
ax_dy.set_title('BorisSpatial actual-field vertical dispersion')
ax_dy.grid(True, alpha=0.3)
fig_dy.tight_layout()

fig_coupling, axes_coupling = plt.subplots(
    2, 1, figsize=(10, 7), sharex=True)
axes_coupling[0].plot(s_from_ip, tw.betx2, color='C0')
axes_coupling[1].plot(s_from_ip, tw.bety1, color='C0')
axes_coupling[0].set_ylabel('betx2 [m]')
axes_coupling[1].set_ylabel('bety1 [m]')
axes_coupling[1].set_xlabel('s - s_ip [m]')
for ax in axes_coupling:
    ax.grid(True, alpha=0.3)
fig_coupling.suptitle('BorisSpatial actual-field linear coupling')
fig_coupling.tight_layout()

fig_alpha_coupling, axes_alpha_coupling = plt.subplots(
    2, 1, figsize=(10, 7), sharex=True)
axes_alpha_coupling[0].plot(s_from_ip, tw.alfx2, color='C0')
axes_alpha_coupling[1].plot(s_from_ip, tw.alfy1, color='C0')
axes_alpha_coupling[0].set_ylabel('alfx2')
axes_alpha_coupling[1].set_ylabel('alfy1')
axes_alpha_coupling[1].set_xlabel('s - s_ip [m]')
for ax in axes_alpha_coupling:
    ax.grid(True, alpha=0.3)
fig_alpha_coupling.suptitle('BorisSpatial actual-field alpha coupling')
fig_alpha_coupling.tight_layout()


print('BorisSpatial actual-field three-solenoid system')
print(f'  main active window = +/-{MAIN_FIELD_HALF_RANGE:.6g} m from IP')
print(
    '  compensation active window = '
    f'+/-{COMP_FIELD_HALF_RANGE:.6g} m around each anti-solenoid center')
print(
    f'  compensation centers = {COMP_LEFT_CENTER:.6g} m, '
    f'{COMP_RIGHT_CENTER:.6g} m')
print(f'  slice length = {SLICE_LENGTH:.6g} m')
print(f'  Boris steps per slice = {BORIS_STEPS_PER_SLICE}')
print(f'  number of BorisSpatialIntegrator elements = {n_slices}')
print(f'  main Bs integral = {main_bs_integral:.12e} T m')
print(
    '  one compensation Bs integral = '
    f'{comp_scale_b * comp_bs_integral_unscaled:.12e} T m')
print(
    '  total Bs integral = '
    f'{main_bs_integral + 2 * comp_scale_b * comp_bs_integral_unscaled:.12e} '
    'T m')
print(f'  compensation scale_b = {comp_scale_b:.12e}')
print(f'  full system length = {line_full.get_length():.12e} m')
print(f'  twiss line length = {line_twiss.get_length():.12e} m')
print(
    f'  twiss end: x = {tw.x[-1]:+.12e} m, '
    f'y = {tw.y[-1]:+.12e} m, dy = {tw.dy[-1]:+.12e} m')
print(
    f'  twiss end: betx2 = {tw.betx2[-1]:+.12e} m, '
    f'bety1 = {tw.bety1[-1]:+.12e} m')
print(
    f'  twiss end: alfx2 = {tw.alfx2[-1]:+.12e}, '
    f'alfy1 = {tw.alfy1[-1]:+.12e}')
print(f'  symplectic error ||R.T S R - S||_2 = {symplectic_error:.12e}')
print(f'  determinant error ||det(R)| - 1| = {det_r_error:.12e}')

plt.show()
