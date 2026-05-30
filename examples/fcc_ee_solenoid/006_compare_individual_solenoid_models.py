from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

from tilted_solenoid import TiltedSolenoid


HERE = Path(__file__).parent

THETA = -0.015
SOLENOID_LENGTH = 2.399 * 2
SOLENOID_RADIUS = 0.13
SOLENOID_FIELD = 2.0
N_SLICES = 201
PARTICLE = 'positron'
ENERGY0 = 45.6e9

MAX_MULTIPOLE_ORDER = 1
DERIVATIVE_STEP = 5e-4
SPLINE_INTEGRAL_POINTS = 10
SPLINE_STEPS_PER_POINT = 10
BORIS_STEPS_PER_SLICE = 10

# Switches for the SplineBoris construction. The default here is the raw tilted
# field map, including on-axis transverse field and first transverse derivative.
USE_PIECEWISE_LINEAR_SPLINES = False
FORCE_IDEAL_SOLENOID_TRANSVERSE_FIELD = False
FORCE_ZERO_FIELD_AT_SOLENOID_ENDS = False

TWISS_BETX_AT_IP = 0.09
TWISS_BETY_AT_IP = 0.0007


# Field map and reference particle.
field_model = TiltedSolenoid(
    L=1.23 * 2,
    a=SOLENOID_RADIUS,
    B0=SOLENOID_FIELD,
    theta=THETA,
)

particle_ref = xt.Particles(PARTICLE, energy0=ENERGY0)
rigidity0 = particle_ref.rigidity0[0]

s_axis = np.linspace(-SOLENOID_LENGTH / 2, SOLENOID_LENGTH / 2, N_SLICES)
zero = np.zeros_like(s_axis)
bx_axis, by_axis, bs_axis = field_model.get_field(zero, zero, s_axis)


# Extract the field and first transverse derivatives used by the SplineBoris
# line. This is done directly from the tilted field map, not from saved 005 data.
bx_values = {0: bx_axis}
by_values = {0: by_axis}
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
        bx_values[order] = bx_derivatives[order]
        by_values[order] = by_derivatives[order]

bs_values = np.array(bs_axis, copy=True)
bx_values = {
    order: np.array(values, copy=True)
    for order, values in bx_values.items()
}
by_values = {
    order: np.array(values, copy=True)
    for order, values in by_values.items()
}

if FORCE_ZERO_FIELD_AT_SOLENOID_ENDS:
    bs_values[0] = 0.0
    bs_values[-1] = 0.0
    for order in range(MAX_MULTIPOLE_ORDER + 1):
        bx_values[order][0] = 0.0
        bx_values[order][-1] = 0.0
        by_values[order][0] = 0.0
        by_values[order][-1] = 0.0

bs_s_derivative = np.gradient(bs_values, s_axis, edge_order=2)
bx_s_derivatives = {
    order: np.gradient(values, s_axis, edge_order=2)
    for order, values in bx_values.items()
}
by_s_derivatives = {
    order: np.gradient(values, s_axis, edge_order=2)
    for order, values in by_values.items()
}
ideal_dbx_dx = -0.5 * bs_s_derivative
ideal_dbx_dx_s_derivative = np.gradient(ideal_dbx_dx, s_axis, edge_order=2)


# Build the full and right-half SplineBoris lines, with the IP marker at s=0.
splineboris_elements_left = []
splineboris_names_left = []
splineboris_elements_right = []
splineboris_names_right = []

for ii in range(len(s_axis) - 1):
    s_start = s_axis[ii]
    s_end = s_axis[ii + 1]
    length = s_end - s_start

    if USE_PIECEWISE_LINEAR_SPLINES:
        bs_derivative = (bs_values[ii + 1] - bs_values[ii]) / length
        bs_integral_average = 0.5 * (bs_values[ii] + bs_values[ii + 1])
    else:
        s_integral = np.linspace(s_start, s_end, SPLINE_INTEGRAL_POINTS)
        bs_integral_average = (
            np.trapezoid(
                field_model.get_field(
                    np.zeros_like(s_integral),
                    np.zeros_like(s_integral),
                    s_integral,
                )[2],
                s_integral,
            )
            / length
        )

    bs_spline = xt.Spline4(
        val_start=bs_values[ii],
        der_start=(bs_derivative if USE_PIECEWISE_LINEAR_SPLINES
                   else bs_s_derivative[ii]),
        val_end=bs_values[ii + 1],
        der_end=(bs_derivative if USE_PIECEWISE_LINEAR_SPLINES
                 else bs_s_derivative[ii + 1]),
        integral=bs_integral_average,
    )

    bx_splines = []
    by_splines = []
    for order in range(MAX_MULTIPOLE_ORDER + 1):
        if FORCE_IDEAL_SOLENOID_TRANSVERSE_FIELD:
            if order == 1:
                if USE_PIECEWISE_LINEAR_SPLINES:
                    ideal_value = -0.5 * bs_derivative
                    bx_val_start = ideal_value
                    bx_val_end = ideal_value
                    bx_der_start = 0.0
                    bx_der_end = 0.0
                    bx_integral_average = ideal_value
                else:
                    bx_val_start = ideal_dbx_dx[ii]
                    bx_val_end = ideal_dbx_dx[ii + 1]
                    bx_der_start = ideal_dbx_dx_s_derivative[ii]
                    bx_der_end = ideal_dbx_dx_s_derivative[ii + 1]
                    bx_integral_average = (
                        -0.5 * (bs_values[ii + 1] - bs_values[ii])
                        / length
                    )
            else:
                bx_val_start = 0.0
                bx_val_end = 0.0
                bx_der_start = 0.0
                bx_der_end = 0.0
                bx_integral_average = 0.0

            by_val_start = 0.0
            by_val_end = 0.0
            by_der_start = 0.0
            by_der_end = 0.0
            by_integral_average = 0.0

        elif USE_PIECEWISE_LINEAR_SPLINES:
            bx_val_start = bx_values[order][ii]
            bx_val_end = bx_values[order][ii + 1]
            by_val_start = by_values[order][ii]
            by_val_end = by_values[order][ii + 1]
            bx_derivative = (bx_val_end - bx_val_start) / length
            by_derivative = (by_val_end - by_val_start) / length
            bx_der_start = bx_derivative
            bx_der_end = bx_derivative
            by_der_start = by_derivative
            by_der_end = by_derivative
            bx_integral_average = 0.5 * (bx_val_start + bx_val_end)
            by_integral_average = 0.5 * (by_val_start + by_val_end)

        else:
            s_integral = np.linspace(s_start, s_end, SPLINE_INTEGRAL_POINTS)
            bx_val_start = bx_values[order][ii]
            bx_val_end = bx_values[order][ii + 1]
            by_val_start = by_values[order][ii]
            by_val_end = by_values[order][ii + 1]
            bx_der_start = bx_s_derivatives[order][ii]
            bx_der_end = bx_s_derivatives[order][ii + 1]
            by_der_start = by_s_derivatives[order][ii]
            by_der_end = by_s_derivatives[order][ii + 1]

            if order == 0:
                bx_integral_values = field_model.get_field(
                    np.zeros_like(s_integral),
                    np.zeros_like(s_integral),
                    s_integral,
                )[0]
                by_integral_values = field_model.get_field(
                    np.zeros_like(s_integral),
                    np.zeros_like(s_integral),
                    s_integral,
                )[1]
            else:
                bx_integral_values = field_model.compute_pure_field_derivatives(
                    s=s_integral,
                    direction='x',
                    step=DERIVATIVE_STEP,
                    component='x',
                    max_order=order,
                    min_order=order,
                )[order]
                by_integral_values = field_model.compute_pure_field_derivatives(
                    s=s_integral,
                    direction='x',
                    step=DERIVATIVE_STEP,
                    component='y',
                    max_order=order,
                    min_order=order,
                )[order]

            bx_integral_average = (
                np.trapezoid(bx_integral_values, s_integral) / length)
            by_integral_average = (
                np.trapezoid(by_integral_values, s_integral) / length)

        bx_splines.append(xt.Spline4(
            val_start=bx_val_start,
            der_start=bx_der_start,
            val_end=bx_val_end,
            der_end=bx_der_end,
            integral=bx_integral_average,
        ))
        by_splines.append(xt.Spline4(
            val_start=by_val_start,
            der_start=by_der_start,
            val_end=by_val_end,
            der_end=by_der_end,
            integral=by_integral_average,
        ))

    element = xt.SplineBoris(
        bs=bs_spline,
        bx=tuple(bx_splines),
        by=tuple(by_splines),
        length=length,
        n_steps=SPLINE_STEPS_PER_POINT,
    )

    if s_end <= 0.0:
        splineboris_elements_left.append(element)
        splineboris_names_left.append(f'splineboris_l_{ii:03d}')
    else:
        splineboris_elements_right.append(element)
        splineboris_names_right.append(f'splineboris_r_{ii:03d}')

line_splineboris_full = xt.Line(
    elements=(splineboris_elements_left + [xt.Marker()]
              + splineboris_elements_right),
    element_names=(splineboris_names_left + ['ip']
                   + splineboris_names_right),
)
line_splineboris_half = xt.Line(
    elements=[xt.Marker()] + splineboris_elements_right,
    element_names=['ip'] + splineboris_names_right,
)


# Build the corresponding VariableSolenoid line. The longitudinal field goes in
# ks_profile, while the transverse field along the reference trajectory is added
# as integrated dipole kicks, following examples/solenoid/005_on_beam_reference.py.
varsol_elements_left = []
varsol_names_left = []
varsol_elements_right = []
varsol_names_right = []

ks = bs_axis / rigidity0
k0 = by_axis / rigidity0
k0s = bx_axis / rigidity0

for ii in range(len(s_axis) - 1):
    s_start = s_axis[ii]
    s_end = s_axis[ii + 1]
    length = s_end - s_start
    element = xt.VariableSolenoid(
        length=length,
        ks_profile=[ks[ii], ks[ii + 1]],
        knl=[0.5 * (k0[ii] + k0[ii + 1]) * length],
        ksl=[0.5 * (k0s[ii] + k0s[ii + 1]) * length],
    )

    if s_end <= 0.0:
        varsol_elements_left.append(element)
        varsol_names_left.append(f'varsol_l_{ii:03d}')
    else:
        varsol_elements_right.append(element)
        varsol_names_right.append(f'varsol_r_{ii:03d}')

line_varsol_full = xt.Line(
    elements=varsol_elements_left + [xt.Marker()] + varsol_elements_right,
    element_names=varsol_names_left + ['ip'] + varsol_names_right,
)
line_varsol_half = xt.Line(
    elements=[xt.Marker()] + varsol_elements_right,
    element_names=['ip'] + varsol_names_right,
)


# Build the direct BorisSpatialIntegrator line, one spatial integrator per
# slice, using the tilted field map directly.
boris_elements_left = []
boris_names_left = []
boris_elements_right = []
boris_names_right = []

for ii in range(len(s_axis) - 1):
    s_start = s_axis[ii]
    s_end = s_axis[ii + 1]
    element = xt.BorisSpatialIntegrator(
        fieldmap_callable=field_model.get_field,
        s_start=s_start,
        s_end=s_end,
        n_steps=BORIS_STEPS_PER_SLICE,
    )

    if s_end <= 0.0:
        boris_elements_left.append(element)
        boris_names_left.append(f'boris_l_{ii:03d}')
    else:
        boris_elements_right.append(element)
        boris_names_right.append(f'boris_r_{ii:03d}')

line_boris_full = xt.Line(
    elements=boris_elements_left + [xt.Marker()] + boris_elements_right,
    element_names=boris_names_left + ['ip'] + boris_names_right,
)
line_boris_half = xt.Line(
    elements=[xt.Marker()] + boris_elements_right,
    element_names=['ip'] + boris_names_right,
)


# Assign the same reference particle to all lines.
lines_full = {
    'SplineBoris': line_splineboris_full,
    'VariableSolenoid': line_varsol_full,
    'BorisSpatialIntegrator': line_boris_full,
}
lines_half = {
    'SplineBoris': line_splineboris_half,
    'VariableSolenoid': line_varsol_half,
    'BorisSpatialIntegrator': line_boris_half,
}

for line in list(lines_full.values()) + list(lines_half.values()):
    line.particle_ref = particle_ref.copy()


# Open twiss for the right half-solenoid, starting at the IP marker.
twiss_half = {}
for name, line in lines_half.items():
    twiss_half[name] = line.twiss4d(
        betx=TWISS_BETX_AT_IP,
        bety=TWISS_BETY_AT_IP,
        strengths=True,
        include_collective=(name == 'BorisSpatialIntegrator'),
    )


# Comparison plots.
plt.close('all')

fig_orbit, axes_orbit = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for name, tw in twiss_half.items():
    axes_orbit[0].plot(tw.s, tw.x, label=name)
    axes_orbit[1].plot(tw.s, tw.y, label=name)
axes_orbit[0].set_ylabel('x [m]')
axes_orbit[1].set_ylabel('y [m]')
axes_orbit[1].set_xlabel('s from IP [m]')
for ax in axes_orbit:
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
fig_orbit.suptitle('Half-solenoid open-twiss orbit')
fig_orbit.tight_layout()

fig_dy, ax_dy = plt.subplots(figsize=(10, 4))
for name, tw in twiss_half.items():
    ax_dy.plot(tw.s, tw.dy, label=name)
ax_dy.set_xlabel('s from IP [m]')
ax_dy.set_ylabel('dy [m]')
ax_dy.set_title('Half-solenoid vertical dispersion')
ax_dy.grid(True, alpha=0.3)
ax_dy.legend(loc='best')
fig_dy.tight_layout()

fig_coupling, axes_coupling = plt.subplots(
    2, 1, figsize=(10, 7), sharex=True)
for name, tw in twiss_half.items():
    axes_coupling[0].plot(tw.s, tw.betx2, label=name)
    axes_coupling[1].plot(tw.s, tw.bety1, label=name)
axes_coupling[0].set_ylabel('betx2 [m]')
axes_coupling[1].set_ylabel('bety1 [m]')
axes_coupling[1].set_xlabel('s from IP [m]')
for ax in axes_coupling:
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
fig_coupling.suptitle('Half-solenoid linear coupling')
fig_coupling.tight_layout()


print('Built full lines with IP marker in the middle:')
for name, line in lines_full.items():
    print(f'  {name}: {len(line.element_names)} elements')
print('Computed open twiss for right half-solenoid.')
for name, tw in twiss_half.items():
    print(
        f'  {name}: x_end={tw.x[-1]:+.6e} m, '
        f'y_end={tw.y[-1]:+.6e} m, dy_end={tw.dy[-1]:+.6e} m, '
        f'betx2_end={tw.betx2[-1]:+.6e} m, '
        f'bety1_end={tw.bety1[-1]:+.6e} m')

plt.show()
