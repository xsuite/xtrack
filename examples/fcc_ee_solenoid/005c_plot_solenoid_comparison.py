import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
from pathlib import Path

from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField

from solenoid_helpers_005 import (
    compute_bs_integrals,
    format_bs_integrals_title,
    load_field_data_json,
    validate_max_multipole_order,
    zero_negligible_central_derivative_values,
)


HERE = Path(__file__).parent
FIELD_DATA_JSON = HERE / '005_solenoid_field_data.json'
LINES_JSON = HERE / '005_solenoid_lines.json'

DERIVATIVE_STEP = 5e-4
SPLINE_STEPS_PER_POINT = 10
ZERO_CENTRAL_DERIVATIVES_FROM_ORDER = 2
ZERO_CENTRAL_DERIVATIVES_HALF_LENGTH = 0.25
X_FIELD_COMPARISON = 0.0
Y_FIELD_COMPARISON = 0.0


def sample_splineboris_line(line, s0, x=0.0, y=0.0):
    s_out = []
    bx_out = []
    by_out = []
    bs_out = []

    s_start = s0
    for element in line.elements:
        s_local = np.linspace(0, element.length, SPLINE_STEPS_PER_POINT + 1)
        bx, by, bs = element.get_field(
            np.full_like(s_local, x),
            np.full_like(s_local, y),
            s_local,
        )

        s_out.append(s_start + s_local)
        bx_out.append(bx)
        by_out.append(by)
        bs_out.append(bs)
        s_start += element.length

    return (
        np.concatenate(s_out),
        np.concatenate(bx_out),
        np.concatenate(by_out),
        np.concatenate(bs_out),
    )


metadata, fields = load_field_data_json(FIELD_DATA_JSON)
max_multipole_order = metadata['max_multipole_order']
validate_max_multipole_order(max_multipole_order)

line_data = xt.json.load(LINES_JSON)
lines = {
    name: xt.Line.from_dict(line_dict)
    for name, line_dict in line_data['lines'].items()
}

bs_integrals = compute_bs_integrals(fields)
bs_integrals_title = format_bs_integrals_title(bs_integrals)

plt.close('all')

fig_field_comparison, axes_field_comparison = plt.subplots(
    len(fields), 3,
    figsize=(15, 4.0 * len(fields)),
    squeeze=False,
)

components = [
    ('B_x [T]', 'bx', 0),
    ('B_y [T]', 'by', 0),
    ('B_s [T]', 'bs', None),
]

sampled_lines = {}
for row, (name, field) in enumerate(fields.items()):
    s_model, bx_model, by_model, bs_model = sample_splineboris_line(
        lines[name],
        s0=field['s_axis'][0],
        x=X_FIELD_COMPARISON,
        y=Y_FIELD_COMPARISON,
    )
    sampled_lines[name] = {
        's': s_model,
        'bx': bx_model,
        'by': by_model,
        'bs': bs_model,
    }

    field_values = {
        'bx': field['scale_b'] * field['bx'][0],
        'by': field['scale_b'] * field['by'][0],
        'bs': field['scale_b'] * field['bs'],
    }
    model_values = {
        'bx': bx_model,
        'by': by_model,
        'bs': bs_model,
    }

    for col, (ylabel, component, _) in enumerate(components):
        ax = axes_field_comparison[row, col]
        field_interp = np.interp(
            s_model, field['s_axis'], field_values[component])
        ax.plot(field['s_axis'], field_values[component], '.',
                label='field-map data')
        ax.plot(s_model, model_values[component], '--', label='SplineBoris')
        ax.plot(
            s_model,
            model_values[component] - field_interp,
            ':',
            label='difference',
        )
        ax.set_ylabel(f'{name}\n{ylabel}')
        ax.grid(True, alpha=0.3)
        if row == 0 and col == 0:
            ax.legend(loc='best')

for ax in axes_field_comparison[-1, :]:
    ax.set_xlabel('s [m]')

fig_field_comparison.suptitle(
    'Saved field-map data and isolated SplineBoris-line comparison '
    f'at x={X_FIELD_COMPARISON:g} m, y={Y_FIELD_COMPARISON:g} m\n'
    f'{bs_integrals_title}'
)
fig_field_comparison.tight_layout()

derivative_comparison_data = {}
offsets = np.arange(-4, 5)
zero_offset_index = np.where(offsets == 0)[0][0]

for name, field in fields.items():
    bx_at_offsets = []
    by_at_offsets = []
    bs_at_offsets = []
    s_model = None

    for offset in offsets:
        s_curr, bx_curr, by_curr, bs_curr = sample_splineboris_line(
            lines[name],
            s0=field['s_axis'][0],
            x=X_FIELD_COMPARISON + offset * DERIVATIVE_STEP,
            y=Y_FIELD_COMPARISON,
        )
        if s_model is None:
            s_model = s_curr
        bx_at_offsets.append(bx_curr)
        by_at_offsets.append(by_curr)
        bs_at_offsets.append(bs_curr)

    bx_at_offsets = np.array(bx_at_offsets)
    by_at_offsets = np.array(by_at_offsets)
    bs_at_offsets = np.array(bs_at_offsets)

    derivatives = {
        0: {
            's': s_model,
            'bx': bx_at_offsets[zero_offset_index],
            'by': by_at_offsets[zero_offset_index],
            'bs': bs_at_offsets[zero_offset_index],
        },
    }

    for order in range(1, max_multipole_order + 1):
        coefficients = SolenoidField.finite_difference_coefficients(
            offsets, order)
        bx_model = (
            np.tensordot(coefficients, bx_at_offsets, axes=(0, 0))
            / DERIVATIVE_STEP**order
        )
        by_model = (
            np.tensordot(coefficients, by_at_offsets, axes=(0, 0))
            / DERIVATIVE_STEP**order
        )
        bs_model = (
            np.tensordot(coefficients, bs_at_offsets, axes=(0, 0))
            / DERIVATIVE_STEP**order
        )
        bx_model, by_model = zero_negligible_central_derivative_values(
            bx_model, by_model, s_model, order,
            ZERO_CENTRAL_DERIVATIVES_FROM_ORDER,
            ZERO_CENTRAL_DERIVATIVES_HALF_LENGTH,
        )

        derivatives[order] = {
            's': s_model,
            'bx': bx_model,
            'by': by_model,
            'bs': bs_model,
        }

    derivative_comparison_data[name] = derivatives

derivative_comparison_figures = {}
derivative_comparison_axes = {}

for order in range(1, max_multipole_order + 1):
    fig, axes = plt.subplots(
        len(fields), 2,
        figsize=(12, 4.0 * len(fields)),
        squeeze=False,
    )

    for row, (name, field) in enumerate(fields.items()):
        s_model = derivative_comparison_data[name][order]['s']
        field_values = [
            field['scale_b'] * field['bx'][order],
            field['scale_b'] * field['by'][order],
        ]
        model_values = [
            derivative_comparison_data[name][order]['bx'],
            derivative_comparison_data[name][order]['by'],
        ]

        for col, component in enumerate(('x', 'y')):
            ax = axes[row, col]
            field_interp = np.interp(
                s_model, field['s_axis'], field_values[col])
            ax.plot(field['s_axis'], field_values[col], '.',
                    label='field-map data')
            ax.plot(s_model, model_values[col], '--', label='SplineBoris')
            ax.plot(
                s_model,
                model_values[col] - field_interp,
                ':',
                label='difference',
            )

            if order == 0:
                derivative_label = f'B_{component} [T]'
            else:
                derivative_label = f'd^{order} B_{component} / dx^{order}'
            ax.set_ylabel(f'{name}\n{derivative_label}')
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(loc='best')

    for ax in axes[-1, :]:
        ax.set_xlabel('s [m]')

    fig.suptitle(
        f'Transverse derivative comparison, order {order} '
        f'at x={X_FIELD_COMPARISON:g} m, y={Y_FIELD_COMPARISON:g} m\n'
        f'{bs_integrals_title}'
    )
    fig.tight_layout()

    derivative_comparison_figures[order] = fig
    derivative_comparison_axes[order] = axes

print(f'Loaded {FIELD_DATA_JSON}')
print(f'Loaded {LINES_JSON}')
print(bs_integrals_title)

plt.show()
