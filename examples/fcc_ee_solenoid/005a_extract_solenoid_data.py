import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
from pathlib import Path

from tilted_solenoid import TiltedSolenoid
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField

from solenoid_helpers_005 import (
    compute_bs_integrals,
    format_bs_integrals_title,
    validate_max_multipole_order,
    zero_negligible_central_derivative_values,
)


HERE = Path(__file__).parent
FIELD_DATA_JSON = HERE / '005_solenoid_field_data.json'

DERIVATIVE_STEP = 5e-4
MAX_MULTIPOLE_ORDER = 4
SPLINE_INTEGRAL_POINTS = 10
SPLINE_STEPS_PER_POINT = 10
ZERO_CENTRAL_DERIVATIVES_FROM_ORDER = 2
ZERO_CENTRAL_DERIVATIVES_HALF_LENGTH = 0.25

THETA = -0.015
N_SLICES_MAIN_SOLENOID = 201
N_SLICES_COMP_SOLENOID = 201


# Define solenoids and longitudinal sampling grids.
validate_max_multipole_order(MAX_MULTIPOLE_ORDER)

main_s_axis = np.linspace(-2.399, 2.399, N_SLICES_MAIN_SOLENOID)
comp_s_axis = np.linspace(-1.0, 1.0, N_SLICES_COMP_SOLENOID)

main_field_model = TiltedSolenoid(L=1.23 * 2, a=0.13, B0=2.0, theta=THETA)
comp_field_model = SolenoidField(L=1.5, a=0.03, B0=1.0, z0=0.0)

_, _, bz_main = main_field_model.get_field(
    np.zeros_like(main_s_axis), np.zeros_like(main_s_axis), main_s_axis)
_, _, bz_comp = comp_field_model.get_field(
    np.zeros_like(comp_s_axis), np.zeros_like(comp_s_axis), comp_s_axis)

specs = {
    'main_solenoid': {
        'field_model': main_field_model,
        's_axis': main_s_axis,
        'scale_b': 1.0,
    },
    'compensation_solenoid': {
        'field_model': comp_field_model,
        's_axis': comp_s_axis,
        'scale_b': (
            -np.trapezoid(bz_main, main_s_axis)
            / np.trapezoid(bz_comp, comp_s_axis)
            / 2.0
        ),
    },
}

# Extract fields and transverse derivatives at the SplineBoris knots.
fields = {}
for name, spec in specs.items():
    field_model = spec['field_model']
    s_axis = spec['s_axis']
    zero = np.zeros_like(s_axis)

    bx = {0: field_model.get_field(zero, zero, s_axis)[0]}
    by = {0: field_model.get_field(zero, zero, s_axis)[1]}
    bs = field_model.get_field(zero, zero, s_axis)[2]

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

    for order in range(MAX_MULTIPOLE_ORDER + 1):
        bx[order], by[order] = zero_negligible_central_derivative_values(
            bx[order], by[order], s_axis, order,
            ZERO_CENTRAL_DERIVATIVES_FROM_ORDER,
            ZERO_CENTRAL_DERIVATIVES_HALF_LENGTH,
        )

    # Extract interval averages used as the integral constraint of each spline.
    n_intervals = len(s_axis) - 1
    s_integral = np.array([
        np.linspace(s_axis[ii], s_axis[ii + 1], SPLINE_INTEGRAL_POINTS)
        for ii in range(n_intervals)
    ])
    s_integral_flat = s_integral.ravel()
    zero_integral = np.zeros_like(s_integral_flat)

    bs_integral_values = field_model.get_field(
        zero_integral, zero_integral, s_integral_flat)[2]
    bs_integral_values = bs_integral_values.reshape(n_intervals, -1)

    bx_integral_values = {
        0: field_model.get_field(
            zero_integral, zero_integral, s_integral_flat)[0]
    }
    by_integral_values = {
        0: field_model.get_field(
            zero_integral, zero_integral, s_integral_flat)[1]
    }

    if MAX_MULTIPOLE_ORDER > 0:
        bx_integral_derivatives = field_model.compute_pure_field_derivatives(
            s=s_integral_flat,
            direction='x',
            step=DERIVATIVE_STEP,
            component='x',
            max_order=MAX_MULTIPOLE_ORDER,
            min_order=1,
        )
        by_integral_derivatives = field_model.compute_pure_field_derivatives(
            s=s_integral_flat,
            direction='x',
            step=DERIVATIVE_STEP,
            component='y',
            max_order=MAX_MULTIPOLE_ORDER,
            min_order=1,
        )

        for order in range(1, MAX_MULTIPOLE_ORDER + 1):
            bx_integral_values[order] = bx_integral_derivatives[order]
            by_integral_values[order] = by_integral_derivatives[order]

    for order in range(MAX_MULTIPOLE_ORDER + 1):
        bx_integral_values[order], by_integral_values[order] = (
            zero_negligible_central_derivative_values(
                bx_integral_values[order],
                by_integral_values[order],
                s_integral_flat,
                order,
                ZERO_CENTRAL_DERIVATIVES_FROM_ORDER,
                ZERO_CENTRAL_DERIVATIVES_HALF_LENGTH,
            )
        )

    lengths = np.diff(s_axis)
    fields[name] = {
        'name': name,
        's_axis': s_axis,
        'bs': bs,
        'bx': bx,
        'by': by,
        'scale_b': spec['scale_b'],
        'bs_integral_average': (
            np.trapezoid(bs_integral_values, s_integral) / lengths),
        'bx_integral_average': {
            order: (
                np.trapezoid(
                    bx_integral_values[order].reshape(n_intervals, -1),
                    s_integral,
                )
                / lengths
            )
            for order in range(MAX_MULTIPOLE_ORDER + 1)
        },
        'by_integral_average': {
            order: (
                np.trapezoid(
                    by_integral_values[order].reshape(n_intervals, -1),
                    s_integral,
                )
                / lengths
            )
            for order in range(MAX_MULTIPOLE_ORDER + 1)
        },
    }

# Prepare a JSON-friendly dictionary with all data needed to build the lines.
output_data = {
    'metadata': {
        'max_multipole_order': MAX_MULTIPOLE_ORDER,
        'derivative_step': DERIVATIVE_STEP,
        'zero_central_derivatives_from_order': (
            ZERO_CENTRAL_DERIVATIVES_FROM_ORDER),
        'zero_central_derivatives_half_length': (
            ZERO_CENTRAL_DERIVATIVES_HALF_LENGTH),
        'spline_integral_points': SPLINE_INTEGRAL_POINTS,
        'spline_steps_per_point': SPLINE_STEPS_PER_POINT,
    },
    'fields': {
        name: {
            'name': field['name'],
            's_axis': field['s_axis'].tolist(),
            'bs': field['bs'].tolist(),
            'bx': {
                str(order): values.tolist()
                for order, values in field['bx'].items()
            },
            'by': {
                str(order): values.tolist()
                for order, values in field['by'].items()
            },
            'scale_b': field['scale_b'],
            'bs_integral_average': field['bs_integral_average'].tolist(),
            'bx_integral_average': {
                str(order): values.tolist()
                for order, values in field['bx_integral_average'].items()
            },
            'by_integral_average': {
                str(order): values.tolist()
                for order, values in field['by_integral_average'].items()
            },
        }
        for name, field in fields.items()
    },
}

# Save extracted data.
bs_integrals = compute_bs_integrals(fields)
bs_integrals_title = format_bs_integrals_title(bs_integrals)

xt.json.dump(output_data, FIELD_DATA_JSON, indent=1)

print(f'Wrote {FIELD_DATA_JSON}')
print(bs_integrals_title)

# Plot the extracted data for inspection.
plt.close('all')
fig_extracted_fields, axes_extracted_fields = plt.subplots(
    len(fields), 3,
    figsize=(15, 4.0 * len(fields)),
    squeeze=False,
)

for row, (name, field) in enumerate(fields.items()):
    s_axis = field['s_axis']
    scale_b = field['scale_b']

    ax = axes_extracted_fields[row, 0]
    ax.plot(s_axis, scale_b * field['bs'], label='B_s')
    ax.set_ylabel(f'{name}\nB_s [T]')
    ax.grid(True, alpha=0.3)

    ax = axes_extracted_fields[row, 1]
    for order in range(MAX_MULTIPOLE_ORDER + 1):
        label = 'B_x' if order == 0 else f'd^{order} B_x / dx^{order}'
        ax.plot(s_axis, scale_b * field['bx'][order], label=label)
    ax.set_ylabel('B_x derivatives')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)

    ax = axes_extracted_fields[row, 2]
    for order in range(MAX_MULTIPOLE_ORDER + 1):
        label = 'B_y' if order == 0 else f'd^{order} B_y / dx^{order}'
        ax.plot(s_axis, scale_b * field['by'][order], label=label)
    ax.set_ylabel('B_y derivatives')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)

for ax in axes_extracted_fields[-1, :]:
    ax.set_xlabel('s [m]')

fig_extracted_fields.suptitle(
    'Extracted SplineBoris inputs from field maps '
    f'(max multipole order {MAX_MULTIPOLE_ORDER})\n'
    f'{bs_integrals_title}'
)
fig_extracted_fields.tight_layout()

plt.show()
