from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

from tilted_solenoid import TiltedSolenoid
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField


HERE = Path(__file__).parent

THETA = -0.015
PARTICLE = 'positron'
ENERGY0 = 45.6e9

MAX_MULTIPOLE_ORDER = 1
DERIVATIVE_STEP = 5e-4
SPLINE_INTEGRAL_POINTS = 10
SPLINE_STEPS_PER_POINT = 10
TAPER_LENGTH = 0.15  # m

BETX = 0.09
BETY = 0.0007

MAIN_SOLENOID_S_AXIS = np.linspace(-2.399, 2.399, 201)
COMP_SOLENOID_S_AXIS = np.linspace(-1.0, 1.0, 201)
COMP_SOLENOID_DISTANCE_FROM_IP = 12.0


def smooth_edge_taper(s_axis, taper_length):
    taper = np.ones_like(s_axis)
    s_min = s_axis[0]
    s_max = s_axis[-1]

    left = s_axis < s_min + taper_length
    u_left = (s_axis[left] - s_min) / taper_length
    taper[left] = u_left**3 * (10.0 - 15.0 * u_left + 6.0 * u_left**2)

    right = s_axis > s_max - taper_length
    u_right = (s_max - s_axis[right]) / taper_length
    taper[right] = u_right**3 * (10.0 - 15.0 * u_right + 6.0 * u_right**2)

    taper[0] = 0.0
    taper[-1] = 0.0
    return taper


def extract_tapered_field_data(field_model, s_axis):
    zero = np.zeros_like(s_axis)
    bx0, by0, bs = field_model.get_field(zero, zero, s_axis)

    bx = {0: np.array(bx0, copy=True)}
    by = {0: np.array(by0, copy=True)}

    if MAX_MULTIPOLE_ORDER > 0:
        bx_derivatives = field_model.compute_pure_field_derivatives(
            s=s_axis,
            direction='x',
            step=DERIVATIVE_STEP,
            component='x',
            max_order=MAX_MULTIPOLE_ORDER,
            min_order=1,
        )
        by_derivatives = field_model.compute_pure_field_derivatives(
            s=s_axis,
            direction='x',
            step=DERIVATIVE_STEP,
            component='y',
            max_order=MAX_MULTIPOLE_ORDER,
            min_order=1,
        )
        for order in range(1, MAX_MULTIPOLE_ORDER + 1):
            bx[order] = bx_derivatives[order]
            by[order] = by_derivatives[order]

    bs = np.array(bs, copy=True)
    taper = smooth_edge_taper(s_axis, TAPER_LENGTH)

    bs *= taper
    for order in range(MAX_MULTIPOLE_ORDER + 1):
        bx[order] = np.array(bx[order], copy=True) * taper
        by[order] = np.array(by[order], copy=True) * taper

    bs_s_derivative = np.gradient(bs, s_axis, edge_order=2)
    bx_s_derivatives = {
        order: np.gradient(values, s_axis, edge_order=2)
        for order, values in bx.items()
    }
    by_s_derivatives = {
        order: np.gradient(values, s_axis, edge_order=2)
        for order, values in by.items()
    }

    bs_s_derivative[0] = 0.0
    bs_s_derivative[-1] = 0.0
    for order in range(MAX_MULTIPOLE_ORDER + 1):
        bx_s_derivatives[order][0] = 0.0
        bx_s_derivatives[order][-1] = 0.0
        by_s_derivatives[order][0] = 0.0
        by_s_derivatives[order][-1] = 0.0

    n_intervals = len(s_axis) - 1
    s_integral = np.array([
        np.linspace(s_axis[ii], s_axis[ii + 1], SPLINE_INTEGRAL_POINTS)
        for ii in range(n_intervals)
    ])

    bs_integral_average = (
        np.trapezoid(
            np.interp(s_integral, s_axis, bs),
            s_integral,
        )
        / np.diff(s_axis)
    )
    bx_integral_average = {
        order: (
            np.trapezoid(
                np.interp(s_integral, s_axis, bx[order]),
                s_integral,
            )
            / np.diff(s_axis)
        )
        for order in range(MAX_MULTIPOLE_ORDER + 1)
    }
    by_integral_average = {
        order: (
            np.trapezoid(
                np.interp(s_integral, s_axis, by[order]),
                s_integral,
            )
            / np.diff(s_axis)
        )
        for order in range(MAX_MULTIPOLE_ORDER + 1)
    }

    return {
        's_axis': s_axis,
        'bs': bs,
        'bx': bx,
        'by': by,
        'bs_s_derivative': bs_s_derivative,
        'bx_s_derivatives': bx_s_derivatives,
        'by_s_derivatives': by_s_derivatives,
        'bs_integral_average': bs_integral_average,
        'bx_integral_average': bx_integral_average,
        'by_integral_average': by_integral_average,
    }


def build_splineboris_line(name, field_data, scale_b):
    s_axis = field_data['s_axis']
    elements = []
    element_names = []
    name_width = len(str(len(s_axis) - 2))

    for ii in range(len(s_axis) - 1):
        length = s_axis[ii + 1] - s_axis[ii]

        bs = xt.Spline4(
            val_start=field_data['bs'][ii],
            der_start=field_data['bs_s_derivative'][ii],
            val_end=field_data['bs'][ii + 1],
            der_end=field_data['bs_s_derivative'][ii + 1],
            integral=field_data['bs_integral_average'][ii],
        )

        bx = []
        by = []
        for order in range(MAX_MULTIPOLE_ORDER + 1):
            bx.append(xt.Spline4(
                val_start=field_data['bx'][order][ii],
                der_start=field_data['bx_s_derivatives'][order][ii],
                val_end=field_data['bx'][order][ii + 1],
                der_end=field_data['bx_s_derivatives'][order][ii + 1],
                integral=field_data['bx_integral_average'][order][ii],
            ))
            by.append(xt.Spline4(
                val_start=field_data['by'][order][ii],
                der_start=field_data['by_s_derivatives'][order][ii],
                val_end=field_data['by'][order][ii + 1],
                der_end=field_data['by_s_derivatives'][order][ii + 1],
                integral=field_data['by_integral_average'][order][ii],
            ))

        elements.append(xt.SplineBoris(
            bs=bs,
            bx=tuple(bx),
            by=tuple(by),
            length=length,
            n_steps=SPLINE_STEPS_PER_POINT,
            scale_b=scale_b,
        ))
        element_names.append(f'{name}_{ii:0{name_width}d}')

    return xt.Line(elements=elements, element_names=element_names)


def build_variable_solenoid_line(name, field_data, scale_b, rigidity0):
    s_axis = field_data['s_axis']
    elements = []
    element_names = []
    name_width = len(str(len(s_axis) - 2))

    bs = scale_b * field_data['bs']
    bx = scale_b * field_data['bx'][0]
    by = scale_b * field_data['by'][0]

    ks = bs / rigidity0
    k0s = bx / rigidity0
    k0 = by / rigidity0

    for ii in range(len(s_axis) - 1):
        length = s_axis[ii + 1] - s_axis[ii]
        elements.append(xt.VariableSolenoid(
            length=length,
            ks_profile=[ks[ii], ks[ii + 1]],
            knl=[0.5 * (k0[ii] + k0[ii + 1]) * length],
            ksl=[0.5 * (k0s[ii] + k0s[ii + 1]) * length],
        ))
        element_names.append(f'{name}_{ii:0{name_width}d}')

    return xt.Line(elements=elements, element_names=element_names)


def assemble_three_solenoid_system(line_comp_left, line_main, line_comp_right):
    return xt.Line(
        elements=(
            list(line_comp_left.elements)
            + [xt.Drift(length=drift_between_comp_and_main)]
            + list(line_main.elements[:len(line_main.elements) // 2])
            + [xt.Marker()]
            + list(line_main.elements[len(line_main.elements) // 2:])
            + [xt.Drift(length=drift_between_comp_and_main)]
            + list(line_comp_right.elements)
        ),
        element_names=(
            list(line_comp_left.element_names)
            + ['drift_comp_left_to_main']
            + list(line_main.element_names[:len(line_main.element_names) // 2])
            + ['ip']
            + list(line_main.element_names[len(line_main.element_names) // 2:])
            + ['drift_main_to_comp_right']
            + list(line_comp_right.element_names)
        ),
    )


# Field models and tapered SplineBoris data for the main and compensation
# solenoid geometries used around each FCC-ee IP.
main_field_model = TiltedSolenoid(L=1.23 * 2, a=0.13, B0=2.0, theta=THETA)
comp_field_model = SolenoidField(L=1.5, a=0.03, B0=1.0, z0=0.0)

main_field_data = extract_tapered_field_data(
    main_field_model, MAIN_SOLENOID_S_AXIS)
comp_field_data = extract_tapered_field_data(
    comp_field_model, COMP_SOLENOID_S_AXIS)

main_bs_integral = np.trapezoid(
    main_field_data['bs'], main_field_data['s_axis'])
comp_bs_integral_unscaled = np.trapezoid(
    comp_field_data['bs'], comp_field_data['s_axis'])
comp_scale_b = -main_bs_integral / comp_bs_integral_unscaled / 2.0

particle_ref = xt.Particles(PARTICLE, energy0=ENERGY0)
rigidity0 = particle_ref.rigidity0[0]

main_half_length = (
    MAIN_SOLENOID_S_AXIS[-1] - MAIN_SOLENOID_S_AXIS[0]) / 2.0
drift_between_comp_and_main = (
    COMP_SOLENOID_DISTANCE_FROM_IP - main_half_length)

# Build SplineBoris and VariableSolenoid versions of the same longitudinal
# layout used in the FCC installation.
spline_comp_left = build_splineboris_line(
    'spline_comp_left', comp_field_data, comp_scale_b)
spline_main = build_splineboris_line('spline_main', main_field_data, 1.0)
spline_comp_right = build_splineboris_line(
    'spline_comp_right', comp_field_data, comp_scale_b)

varsol_comp_left = build_variable_solenoid_line(
    'varsol_comp_left', comp_field_data, comp_scale_b, rigidity0)
varsol_main = build_variable_solenoid_line(
    'varsol_main', main_field_data, 1.0, rigidity0)
varsol_comp_right = build_variable_solenoid_line(
    'varsol_comp_right', comp_field_data, comp_scale_b, rigidity0)

line_systems = {
    'SplineBoris': assemble_three_solenoid_system(
        spline_comp_left, spline_main, spline_comp_right),
    'VariableSolenoid': assemble_three_solenoid_system(
        varsol_comp_left, varsol_main, varsol_comp_right),
}

for line in line_systems.values():
    line.particle_ref = particle_ref.copy()


# Symplecticity metric from examples/boris_spatial/004_study_convergence.py,
# and open twiss initialized at the IP for both models.
S = xt.linear_normal_form.S
symplectic_errors = {}
det_r_errors = {}
twiss_results = {}

for name, line in line_systems.items():
    R_obj = line.get_R_matrix(particle_on_co=particle_ref.copy())
    RR = R_obj['R_matrix']
    symplectic_errors[name] = np.linalg.norm(RR.T @ S @ RR - S, ord=2)
    det_r_errors[name] = abs(abs(np.linalg.det(RR)) - 1.0)
    twiss_results[name] = line.twiss(init_at='ip', betx=BETX, bety=BETY)

plt.close('all')

fig_orbit, axes_orbit = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for name, tw in twiss_results.items():
    s_from_ip = tw.s - tw['s', 'ip']
    axes_orbit[0].plot(s_from_ip, tw.x, label=name)
    axes_orbit[1].plot(s_from_ip, tw.y, label=name)
axes_orbit[0].set_ylabel('x [m]')
axes_orbit[1].set_ylabel('y [m]')
axes_orbit[1].set_xlabel('s - s_ip [m]')
for ax in axes_orbit:
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
fig_orbit.suptitle('Three-solenoid open-twiss orbit, initialized at IP')
fig_orbit.tight_layout()

fig_dy, ax_dy = plt.subplots(figsize=(10, 4))
for name, tw in twiss_results.items():
    s_from_ip = tw.s - tw['s', 'ip']
    ax_dy.plot(s_from_ip, tw.dy, label=name)
ax_dy.set_xlabel('s - s_ip [m]')
ax_dy.set_ylabel('dy [m]')
ax_dy.set_title('Three-solenoid vertical dispersion')
ax_dy.grid(True, alpha=0.3)
ax_dy.legend(loc='best')
fig_dy.tight_layout()

fig_coupling, axes_coupling = plt.subplots(
    2, 1, figsize=(10, 7), sharex=True)
for name, tw in twiss_results.items():
    s_from_ip = tw.s - tw['s', 'ip']
    axes_coupling[0].plot(s_from_ip, tw.betx2, label=name)
    axes_coupling[1].plot(s_from_ip, tw.bety1, label=name)
axes_coupling[0].set_ylabel('betx2 [m]')
axes_coupling[1].set_ylabel('bety1 [m]')
axes_coupling[1].set_xlabel('s - s_ip [m]')
for ax in axes_coupling:
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
fig_coupling.suptitle('Three-solenoid linear coupling')
fig_coupling.tight_layout()

fig_alpha_coupling, axes_alpha_coupling = plt.subplots(
    2, 1, figsize=(10, 7), sharex=True)
for name, tw in twiss_results.items():
    s_from_ip = tw.s - tw['s', 'ip']
    axes_alpha_coupling[0].plot(s_from_ip, tw.alfx2, label=name)
    axes_alpha_coupling[1].plot(s_from_ip, tw.alfy1, label=name)
axes_alpha_coupling[0].set_ylabel('alfx2')
axes_alpha_coupling[1].set_ylabel('alfy1')
axes_alpha_coupling[1].set_xlabel('s - s_ip [m]')
for ax in axes_alpha_coupling:
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
fig_alpha_coupling.suptitle('Three-solenoid alpha coupling')
fig_alpha_coupling.tight_layout()

print('Tapered three-solenoid system comparison')
print(f'  taper length = {TAPER_LENGTH:.6g} m')
print(f'  twiss init at IP: betx = {BETX:.12e} m, bety = {BETY:.12e} m')
print(f'  main Bs integral = {main_bs_integral:.12e} T m')
print(
    '  one compensation Bs integral = '
    f'{comp_scale_b * comp_bs_integral_unscaled:.12e} T m')
print(
    '  total Bs integral = '
    f'{main_bs_integral + 2 * comp_scale_b * comp_bs_integral_unscaled:.12e} '
    'T m')
print(f'  compensation scale_b = {comp_scale_b:.12e}')
for name, line in line_systems.items():
    tw = twiss_results[name]
    print(f'  {name}:')
    print(f'    system length = {line.get_length():.12e} m')
    print(f'    number of elements = {len(line.elements)}')
    print(
        f'    twiss end: x = {tw.x[-1]:+.12e} m, '
        f'y = {tw.y[-1]:+.12e} m, dy = {tw.dy[-1]:+.12e} m')
    print(
        f'    twiss end: betx2 = {tw.betx2[-1]:+.12e} m, '
        f'bety1 = {tw.bety1[-1]:+.12e} m')
    print(
        '    symplectic error ||R.T S R - S||_2 = '
        f'{symplectic_errors[name]:.12e}')
    print(
        f'    determinant error ||det(R)| - 1| = {det_r_errors[name]:.12e}')

plt.show()
