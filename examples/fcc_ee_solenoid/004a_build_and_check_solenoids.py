from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

from spline_boris_setup import (
    assemble_three_solenoid_system,
    build_splineboris_line,
    build_variable_solenoid_line,
    extract_tapered_field_data,
    sample_splineboris_line,
    sample_splineboris_line_on_s,
    smooth_edge_taper,
    symplectic_error,
)
from tilted_solenoid import TiltedSolenoid
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField


HERE = Path(__file__).parent
OUTPUT_LINES_JSON = HERE / '004_solenoid_lines.json'

THETA = -0.015
PARTICLE = 'positron'
ENERGY0 = 45.6e9

MAX_TRANSVERSE_DERIVATIVE_ORDER = 4
MAX_TRANSVERSE_DERIVATIVE_ORDER_FOR_SPLINE = 4
DERIVATIVE_STEP = 5e-4
SPLINE_INTEGRAL_POINTS = 10
DECREASE_S_POLY_ORDER_WITH_TRANSVERSE_ORDER = True
S_DERIVATIVE_SPLINE_ORDER = 4
MAX_S_DERIVATIVE_PLOT_ORDER = 5
SPLINE_STEPS_PER_POINT = 10
TAPER_LENGTH = 0.15  # m

USE_NEAR_AXIS_SIMPLIFIED_MODEL = False
SAVE_SOLENOID_LINES_JSON = True
X_FIELD_COMPARISON = 0
Y_FIELD_COMPARISON = 0
PLOT_MAIN_SOLENOID = True
PLOT_COMPENSATION_SOLENOID = False

MIXED_DERIVATIVE_SPECS = [
    {
        'component': 'bx',
        'transverse_direction': 'x',
        'transverse_order': 2,
        's_order': 2,
    },
]

BETX = 0.09
BETY = 0.0007

MAIN_SOLENOID_S_AXIS = np.linspace(-2.399, 2.399, 201)
COMP_SOLENOID_S_AXIS = np.linspace(-1.0, 1.0, 201)
COMP_SOLENOID_DISTANCE_FROM_IP = 12.0

assert MAX_TRANSVERSE_DERIVATIVE_ORDER_FOR_SPLINE <= xt.SplineBoris._SB_MAX_MULTIPOLE_ORDER - 1
assert MAX_TRANSVERSE_DERIVATIVE_ORDER <= 5
assert S_DERIVATIVE_SPLINE_ORDER == 4
assert MAX_S_DERIVATIVE_PLOT_ORDER <= 5
for spec in MIXED_DERIVATIVE_SPECS:
    assert spec['component'] in ('bx', 'by', 'bs')
    assert spec['transverse_direction'] in ('x', 'y')
    assert spec['transverse_order'] >= 0
    assert spec['s_order'] >= 0

# Build the two physical solenoid models and extract the tapered field data.
main_field_model = TiltedSolenoid(L=1.23 * 2, a=0.13, B0=2.0, theta=THETA)
comp_field_model = SolenoidField(L=1.5, a=0.03, B0=1.0, z0=0.0)

field_extraction_kwargs = {
    'max_transverse_derivative_order': MAX_TRANSVERSE_DERIVATIVE_ORDER,
    'derivative_step': DERIVATIVE_STEP,
    'spline_integral_points': SPLINE_INTEGRAL_POINTS,
    'taper_length': TAPER_LENGTH,
    's_derivative_spline_order': S_DERIVATIVE_SPLINE_ORDER,
    'max_s_derivative_plot_order': MAX_S_DERIVATIVE_PLOT_ORDER,
    'decrease_s_poly_order_with_transverse_order': (
        DECREASE_S_POLY_ORDER_WITH_TRANSVERSE_ORDER),
}

main_field_data = extract_tapered_field_data(
    name='main_solenoid',
    field_model=main_field_model,
    s_axis=MAIN_SOLENOID_S_AXIS,
    **field_extraction_kwargs)
comp_field_data = extract_tapered_field_data(
    name='compensation_solenoid',
    field_model=comp_field_model,
    s_axis=COMP_SOLENOID_S_AXIS,
    **field_extraction_kwargs)

main_bs_integral = np.trapezoid(
    main_field_data['bs'], main_field_data['s_axis'])
comp_bs_integral_unscaled = np.trapezoid(
    comp_field_data['bs'], comp_field_data['s_axis'])
comp_scale_b = -main_bs_integral / comp_bs_integral_unscaled / 2.0

particle_ref = xt.Particles(PARTICLE, energy0=ENERGY0)
rigidity0 = particle_ref.rigidity0[0]


# Build the isolated SplineBoris templates that will be installed later.
splineboris_build_kwargs = {
    'max_transverse_derivative_order_for_spline': (
        MAX_TRANSVERSE_DERIVATIVE_ORDER_FOR_SPLINE),
    'spline_steps_per_point': SPLINE_STEPS_PER_POINT,
    'use_near_axis_simplified_model': USE_NEAR_AXIS_SIMPLIFIED_MODEL,
}

line_main_solenoid = build_splineboris_line(
    name='main_solenoid',
    field_data=main_field_data,
    scale_b=1.0,
    **splineboris_build_kwargs)
line_compensation_solenoid = build_splineboris_line(
    name='compensation_solenoid',
    field_data=comp_field_data,
    scale_b=comp_scale_b,
    **splineboris_build_kwargs)
line_main_solenoid.particle_ref = particle_ref.copy()
line_compensation_solenoid.particle_ref = particle_ref.copy()

if SAVE_SOLENOID_LINES_JSON:
    output_data = {
        'metadata': {
            'max_transverse_derivative_order': (
                MAX_TRANSVERSE_DERIVATIVE_ORDER),
            'max_transverse_derivative_order_for_spline': (
                MAX_TRANSVERSE_DERIVATIVE_ORDER_FOR_SPLINE),
            'decrease_s_poly_order_with_transverse_order': (
                DECREASE_S_POLY_ORDER_WITH_TRANSVERSE_ORDER),
            'spline_s_polynomial_degree_rule': (
                'max(0, 4 - transverse_derivative_order)'
                if DECREASE_S_POLY_ORDER_WITH_TRANSVERSE_ORDER
                else '4'),
            'derivative_step': DERIVATIVE_STEP,
            'spline_integral_points': SPLINE_INTEGRAL_POINTS,
            's_derivative_spline_order': S_DERIVATIVE_SPLINE_ORDER,
            'max_s_derivative_plot_order': MAX_S_DERIVATIVE_PLOT_ORDER,
            'spline_steps_per_point': SPLINE_STEPS_PER_POINT,
            'taper_length': TAPER_LENGTH,
            'use_near_axis_simplified_model': (
                USE_NEAR_AXIS_SIMPLIFIED_MODEL),
            'main_bs_integral': main_bs_integral,
            'comp_bs_integral_unscaled': comp_bs_integral_unscaled,
            'comp_scale_b': comp_scale_b,
        },
        'lines': {
            'main_solenoid': line_main_solenoid.to_dict(),
            'compensation_solenoid': line_compensation_solenoid.to_dict(),
        },
    }
    xt.json.dump(output_data, OUTPUT_LINES_JSON, indent=1)

############################################
# ----------- Checks and plots ----------- #
############################################

# Build a complete local three-solenoid system for checks before FCC install.
main_half_length = (
    MAIN_SOLENOID_S_AXIS[-1] - MAIN_SOLENOID_S_AXIS[0]) / 2.0
drift_between_comp_and_main = (
    COMP_SOLENOID_DISTANCE_FROM_IP - main_half_length)

spline_comp_left = build_splineboris_line(
    name='spline_comp_left',
    field_data=comp_field_data,
    scale_b=comp_scale_b,
    **splineboris_build_kwargs)
spline_main = build_splineboris_line(
    name='spline_main',
    field_data=main_field_data,
    scale_b=1.0,
    **splineboris_build_kwargs)
spline_comp_right = build_splineboris_line(
    name='spline_comp_right',
    field_data=comp_field_data,
    scale_b=comp_scale_b,
    **splineboris_build_kwargs)

varsol_comp_left = build_variable_solenoid_line(
    name='varsol_comp_left',
    field_data=comp_field_data,
    scale_b=comp_scale_b,
    rigidity0=rigidity0)
varsol_main = build_variable_solenoid_line(
    name='varsol_main',
    field_data=main_field_data,
    scale_b=1.0,
    rigidity0=rigidity0)
varsol_comp_right = build_variable_solenoid_line(
    name='varsol_comp_right',
    field_data=comp_field_data,
    scale_b=comp_scale_b,
    rigidity0=rigidity0)

line_systems = {
    'SplineBoris': assemble_three_solenoid_system(
        line_comp_left=spline_comp_left,
        line_main=spline_main,
        line_comp_right=spline_comp_right,
        drift_between_comp_and_main=drift_between_comp_and_main),
    'VariableSolenoid': assemble_three_solenoid_system(
        line_comp_left=varsol_comp_left,
        line_main=varsol_main,
        line_comp_right=varsol_comp_right,
        drift_between_comp_and_main=drift_between_comp_and_main),
}
for line in line_systems.values():
    line.particle_ref = particle_ref.copy()


# Twiss checks for the local three-solenoid systems.
twiss_results = {}
twiss_results['SplineBoris'] = line_systems['SplineBoris'].twiss(
    init_at='ip', betx=BETX, bety=BETY)
twiss_results['VariableSolenoid'] = line_systems['VariableSolenoid'].twiss(
    init_at='ip', betx=BETX, bety=BETY)

main_symplectic_error = symplectic_error(
    line=line_main_solenoid, particle_ref=particle_ref)
comp_symplectic_error = symplectic_error(
    line=line_compensation_solenoid, particle_ref=particle_ref)


# Plots: compare field-map extraction against the built SplineBoris lines.
plt.close('all')

comparison_fields = {
}
if PLOT_MAIN_SOLENOID:
    comparison_fields['main_solenoid'] = {
        'field_data': main_field_data,
        'field_model': main_field_model,
        'line': line_main_solenoid,
        'scale_b': 1.0,
    }
if PLOT_COMPENSATION_SOLENOID:
    comparison_fields['compensation_solenoid'] = {
        'field_data': comp_field_data,
        'field_model': comp_field_model,
        'line': line_compensation_solenoid,
        'scale_b': comp_scale_b,
    }

sampled_lines = {}
for name, item in comparison_fields.items():
    field_data = item['field_data']
    s_model, bx_model, by_model, bs_model = sample_splineboris_line(
        line=item['line'],
        s0=field_data['s_axis'][0],
        spline_steps_per_point=SPLINE_STEPS_PER_POINT,
        x=X_FIELD_COMPARISON,
        y=Y_FIELD_COMPARISON,
    )
    sampled_lines[name] = {
        's': s_model,
        'bx': bx_model,
        'by': by_model,
        'bs': bs_model,
    }

field_components = [
    ('B_s [T]', 'bs'),
    ('B_x [T]', 'bx'),
    ('B_y [T]', 'by'),
]

shared_x_axes_by_solenoid = {}

if comparison_fields:
    fig_fields, axes_fields = plt.subplots(
        len(comparison_fields), 3,
        figsize=(15, 4.0 * len(comparison_fields)),
        squeeze=False,
        num=1000,
    )

    for row, (name, item) in enumerate(comparison_fields.items()):
        field_data = item['field_data']
        scale_b = item['scale_b']
        field_values = {
            'bs': scale_b * field_data['bs'],
            'bx': scale_b * field_data['bx'][0],
            'by': scale_b * field_data['by'][0],
        }
        model_values = sampled_lines[name]

        for col, (ylabel, component) in enumerate(field_components):
            ax = axes_fields[row, col]
            ax.plot(
                field_data['s_axis'], field_values[component],
                '-', label='field-map data')
            ax.plot(
                model_values['s'], model_values[component],
                '--', label='SplineBoris')
            ax.set_ylabel(f'{name}\n{ylabel}')
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(loc='best')

        axes_to_share = axes_fields[row, :].ravel()
        if name not in shared_x_axes_by_solenoid:
            shared_x_axes_by_solenoid[name] = axes_to_share[0]
        for ax in axes_to_share:
            if ax is not shared_x_axes_by_solenoid[name]:
                ax.sharex(shared_x_axes_by_solenoid[name])

    for ax in axes_fields[-1, :]:
        ax.set_xlabel('s [m]')
    fig_fields.suptitle(
        'Tapered field-map data and SplineBoris-line comparison '
        f'at x={X_FIELD_COMPARISON:g} m, y={Y_FIELD_COMPARISON:g} m')
    fig_fields.tight_layout()

s_derivative_model_data = {}
u_integral = np.linspace(-1.0, 1.0, SPLINE_INTEGRAL_POINTS)
splineboris_poly_order = min(4, SPLINE_INTEGRAL_POINTS - 1)
max_s_derivative_plot_order = min(
    MAX_S_DERIVATIVE_PLOT_ORDER,
    5)

for name, item in comparison_fields.items():
    field_data = item['field_data']
    s_axis = field_data['s_axis']
    line = item['line']
    n_intervals = len(line.elements)

    s_model = np.empty((n_intervals, SPLINE_INTEGRAL_POINTS))
    bx_model = np.empty_like(s_model)
    by_model = np.empty_like(s_model)
    bs_model = np.empty_like(s_model)

    for ii, element in enumerate(line.elements):
        length = element.length
        s_local = np.linspace(0.0, length, SPLINE_INTEGRAL_POINTS)
        s_model[ii] = s_axis[ii] + s_local
        bx_model[ii], by_model[ii], bs_model[ii] = element.get_field(
            np.full_like(s_local, X_FIELD_COMPARISON),
            np.full_like(s_local, Y_FIELD_COMPARISON),
            s_local,
        )

    s_derivative_model_data[name] = {}
    model_values = {
        'bx': bx_model,
        'by': by_model,
        'bs': bs_model,
    }

    for derivative_order in range(max_s_derivative_plot_order + 1):
        s_derivative_model_data[name][derivative_order] = {
            's': s_model.ravel(),
        }

        for component, values in model_values.items():
            derivative_values = np.empty_like(values)
            for ii, element in enumerate(line.elements):
                if derivative_order == 0:
                    derivative_values[ii] = values[ii]
                else:
                    coefficients = np.polyfit(
                        u_integral, values[ii], splineboris_poly_order)
                    derivative_coefficients = np.polyder(
                        coefficients, derivative_order)
                    derivative_values[ii] = (
                        np.polyval(derivative_coefficients, u_integral)
                        * (2.0 / element.length)**derivative_order)

            s_derivative_model_data[name][derivative_order][component] = (
                derivative_values.ravel())

for name, item in comparison_fields.items():
    field_data = item['field_data']
    scale_b = item['scale_b']
    figure_number_offset = 300 if name == 'compensation_solenoid' else 200

    for component_index, (component, label) in enumerate((
            ('bx', 'B_x'),
            ('by', 'B_y'),
            ('bs', 'B_s'))):
        fig_s_derivatives, axes_s_derivatives = plt.subplots(
            2, 3, figsize=(15, 8),
            num=figure_number_offset + 10 * component_index)
        axes_s_derivatives_flat = axes_s_derivatives.ravel()
        if name not in shared_x_axes_by_solenoid:
            shared_x_axes_by_solenoid[name] = axes_s_derivatives_flat[0]
        for ax in axes_s_derivatives_flat:
            if ax is not shared_x_axes_by_solenoid[name]:
                ax.sharex(shared_x_axes_by_solenoid[name])

        for derivative_order in range(max_s_derivative_plot_order + 1):
            ax = axes_s_derivatives_flat[derivative_order]
            field_plot_data = field_data['s_derivative_plot_data'][
                derivative_order]
            model_plot_data = s_derivative_model_data[name][derivative_order]
            ax.plot(
                field_plot_data['s'],
                scale_b * field_plot_data[component],
                '-',
                label='field-map data')
            ax.plot(
                model_plot_data['s'],
                model_plot_data[component],
                '--',
                label='SplineBoris')
            if derivative_order == 0:
                ax.set_ylabel(label)
            else:
                ax.set_ylabel(
                    f'd^{derivative_order} {label} / ds^{derivative_order}')
            ax.set_xlabel('s [m]')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'order {derivative_order}')

        for ax in axes_s_derivatives_flat[max_s_derivative_plot_order + 1:]:
            ax.set_visible(False)

        axes_s_derivatives_flat[0].legend(loc='best')
        fig_s_derivatives.suptitle(
            f'Longitudinal derivative comparison for {name}, '
            f'{label} at x={X_FIELD_COMPARISON:g} m, '
            f'y={Y_FIELD_COMPARISON:g} m')
        fig_s_derivatives.tight_layout()

offsets = np.arange(-4, 5)
zero_offset_index = np.where(offsets == 0)[0][0]
derivative_comparison_data = {}

for name, item in comparison_fields.items():
    field_data = item['field_data']
    bx_at_offsets = []
    by_at_offsets = []
    bs_at_offsets = []
    s_model = None

    for offset in offsets:
        s_curr, bx_curr, by_curr, bs_curr = sample_splineboris_line(
            line=item['line'],
            s0=field_data['s_axis'][0],
            spline_steps_per_point=SPLINE_STEPS_PER_POINT,
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

    derivative_comparison_data[name] = {
        0: {
            's': s_model,
            'bx': bx_at_offsets[zero_offset_index],
            'by': by_at_offsets[zero_offset_index],
            'bs': bs_at_offsets[zero_offset_index],
        }
    }

    for order in range(1, MAX_TRANSVERSE_DERIVATIVE_ORDER + 1):
        coefficients = SolenoidField.finite_difference_coefficients(
            offsets, order)
        derivative_comparison_data[name][order] = {
            's': s_model,
            'bx': (
                np.tensordot(coefficients, bx_at_offsets, axes=(0, 0))
                / DERIVATIVE_STEP**order
            ),
            'by': (
                np.tensordot(coefficients, by_at_offsets, axes=(0, 0))
                / DERIVATIVE_STEP**order
            ),
            'bs': (
                np.tensordot(coefficients, bs_at_offsets, axes=(0, 0))
                / DERIVATIVE_STEP**order
            ),
        }

vertical_derivative_comparison_data = {}

for name, item in comparison_fields.items():
    field_data = item['field_data']
    bx_at_offsets = []
    by_at_offsets = []
    bs_at_offsets = []
    s_model = None

    for offset in offsets:
        s_curr, bx_curr, by_curr, bs_curr = sample_splineboris_line(
            line=item['line'],
            s0=field_data['s_axis'][0],
            spline_steps_per_point=SPLINE_STEPS_PER_POINT,
            x=X_FIELD_COMPARISON,
            y=Y_FIELD_COMPARISON + offset * DERIVATIVE_STEP,
        )
        if s_model is None:
            s_model = s_curr
        bx_at_offsets.append(bx_curr)
        by_at_offsets.append(by_curr)
        bs_at_offsets.append(bs_curr)

    bx_at_offsets = np.array(bx_at_offsets)
    by_at_offsets = np.array(by_at_offsets)
    bs_at_offsets = np.array(bs_at_offsets)

    vertical_derivative_comparison_data[name] = {
        0: {
            's': s_model,
            'bx': bx_at_offsets[zero_offset_index],
            'by': by_at_offsets[zero_offset_index],
            'bs': bs_at_offsets[zero_offset_index],
        }
    }

    for order in range(1, MAX_TRANSVERSE_DERIVATIVE_ORDER + 1):
        coefficients = SolenoidField.finite_difference_coefficients(
            offsets, order)
        vertical_derivative_comparison_data[name][order] = {
            's': s_model,
            'bx': (
                np.tensordot(coefficients, bx_at_offsets, axes=(0, 0))
                / DERIVATIVE_STEP**order
            ),
            'by': (
                np.tensordot(coefficients, by_at_offsets, axes=(0, 0))
                / DERIVATIVE_STEP**order
            ),
            'bs': (
                np.tensordot(coefficients, bs_at_offsets, axes=(0, 0))
                / DERIVATIVE_STEP**order
            ),
        }

for name, item in comparison_fields.items():
    field_data = item['field_data']
    scale_b = item['scale_b']
    figure_number_offset = 700 if name == 'compensation_solenoid' else 600

    transverse_plot_specs = [
        ('bx', 'x', 'B_x', field_data['bx'], derivative_comparison_data[name]),
        (
            'bx', 'y', 'B_x',
            field_data['bx_y_pure'],
            vertical_derivative_comparison_data[name],
        ),
        ('by', 'x', 'B_y', field_data['by'], derivative_comparison_data[name]),
        (
            'by', 'y', 'B_y',
            field_data['by_y_pure'],
            vertical_derivative_comparison_data[name],
        ),
        (
            'bs', 'x', 'B_s',
            field_data['bs_x_pure'],
            derivative_comparison_data[name],
        ),
        (
            'bs', 'y', 'B_s',
            field_data['bs_y_pure'],
            vertical_derivative_comparison_data[name],
        ),
    ]

    for plot_index, (
            component, direction, label, field_values_by_order,
            model_data_by_order) in enumerate(transverse_plot_specs):
        fig_derivatives, axes_derivatives = plt.subplots(
            2, 3, figsize=(15, 8),
            num=figure_number_offset + plot_index)
        axes_derivatives_flat = axes_derivatives.ravel()
        if name not in shared_x_axes_by_solenoid:
            shared_x_axes_by_solenoid[name] = axes_derivatives_flat[0]
        for ax in axes_derivatives_flat:
            if ax is not shared_x_axes_by_solenoid[name]:
                ax.sharex(shared_x_axes_by_solenoid[name])

        for order in range(MAX_TRANSVERSE_DERIVATIVE_ORDER + 1):
            ax = axes_derivatives_flat[order]
            model_data = model_data_by_order[order]
            ax.plot(
                field_data['s_axis'],
                scale_b * field_values_by_order[order],
                '-',
                label='field-map data')
            ax.plot(
                model_data['s'],
                model_data[component],
                '--',
                label='SplineBoris')
            if order == 0:
                ax.set_ylabel(label)
            else:
                ax.set_ylabel(f'd^{order} {label} / d{direction}^{order}')
            ax.set_xlabel('s [m]')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'order {order}')

        for ax in axes_derivatives_flat[
                MAX_TRANSVERSE_DERIVATIVE_ORDER + 1:]:
            ax.set_visible(False)

        axes_derivatives_flat[0].legend(loc='best')
        fig_derivatives.suptitle(
            f'Tapered transverse derivative comparison for {name}: '
            f'{label} vs {direction} at x={X_FIELD_COMPARISON:g} m, '
            f'y={Y_FIELD_COMPARISON:g} m')
        fig_derivatives.tight_layout()

mixed_component_index = {'bx': 0, 'by': 1, 'bs': 2}
mixed_component_label = {'bx': 'B_x', 'by': 'B_y', 'bs': 'B_s'}

for name, item in comparison_fields.items():
    field_data = item['field_data']
    field_model = item['field_model']
    s_axis = field_data['s_axis']
    scale_b = item['scale_b']
    figure_number_offset = 1400 if name == 'compensation_solenoid' else 1300
    s_mixed = field_data['s_derivative_plot_data'][0]['s']
    taper_mixed = smooth_edge_taper(
        s_axis=s_mixed, taper_length=TAPER_LENGTH)

    for spec_index, spec in enumerate(MIXED_DERIVATIVE_SPECS):
        component = spec['component']
        transverse_direction = spec['transverse_direction']
        transverse_order = spec['transverse_order']
        s_order = spec['s_order']

        offsets = np.arange(-4, 5)
        coefficients = SolenoidField.finite_difference_coefficients(
            offsets, transverse_order)
        field_values_at_offsets = []
        model_values_at_offsets = []

        for offset in offsets:
            if transverse_direction == 'x':
                x_values = np.full_like(
                    s_mixed,
                    X_FIELD_COMPARISON + offset * DERIVATIVE_STEP)
                y_values = np.full_like(s_mixed, Y_FIELD_COMPARISON)
            elif transverse_direction == 'y':
                x_values = np.full_like(s_mixed, X_FIELD_COMPARISON)
                y_values = np.full_like(
                    s_mixed,
                    Y_FIELD_COMPARISON + offset * DERIVATIVE_STEP)
            else:
                raise ValueError(
                    "transverse_direction must be either 'x' or 'y'")

            field_values_at_offsets.append(
                field_model.get_field(
                    x_values, y_values, s_mixed
                )[mixed_component_index[component]] * taper_mixed)
            model_values_at_offsets.append(
                sample_splineboris_line_on_s(
                    line=item['line'],
                    s_axis=s_axis,
                    s_eval=s_mixed,
                    x=x_values[0], y=y_values[0],
                )[mixed_component_index[component]])

        field_mixed_derivative = (
            np.tensordot(
                coefficients,
                np.array(field_values_at_offsets),
                axes=(0, 0))
            / DERIVATIVE_STEP**transverse_order
        )
        model_mixed_derivative = (
            np.tensordot(
                coefficients,
                np.array(model_values_at_offsets),
                axes=(0, 0))
            / DERIVATIVE_STEP**transverse_order
        )

        for _ in range(s_order):
            field_mixed_derivative = np.gradient(
                field_mixed_derivative, s_mixed, edge_order=2)
            model_mixed_derivative = np.gradient(
                model_mixed_derivative, s_mixed, edge_order=2)

        fig_mixed, ax_mixed = plt.subplots(
            1, 1, figsize=(10, 5),
            num=figure_number_offset + spec_index)
        if name in shared_x_axes_by_solenoid:
            ax_mixed.sharex(shared_x_axes_by_solenoid[name])
        else:
            shared_x_axes_by_solenoid[name] = ax_mixed

        ax_mixed.plot(
            s_mixed,
            scale_b * field_mixed_derivative,
            '-',
            label='field-map data')
        ax_mixed.plot(
            s_mixed,
            model_mixed_derivative,
            '--',
            label='SplineBoris')
        total_order = transverse_order + s_order
        ax_mixed.set_ylabel(
            f'd^{total_order} {mixed_component_label[component]} / '
            f'd{transverse_direction}^{transverse_order} ds^{s_order}')
        ax_mixed.set_xlabel('s [m]')
        ax_mixed.grid(True, alpha=0.3)
        ax_mixed.legend(loc='best')
        fig_mixed.suptitle(
            f'Mixed derivative comparison for {name}: '
            f'd^{total_order} {mixed_component_label[component]} / '
            f'd{transverse_direction}^{transverse_order} ds^{s_order}')
        fig_mixed.tight_layout()


# Plots: local three-solenoid checks.
fig_orbit_coupling, axes_orbit_coupling = plt.subplots(
    2, 2, figsize=(12, 7), sharex=True, num=1001)
for name, tw in twiss_results.items():
    s_from_ip = tw.s - tw['s', 'ip']
    axes_orbit_coupling[0, 0].plot(s_from_ip, tw.x, label=name)
    axes_orbit_coupling[1, 0].plot(s_from_ip, tw.y, label=name)
    axes_orbit_coupling[0, 1].plot(s_from_ip, tw.betx2, label=name)
    axes_orbit_coupling[1, 1].plot(s_from_ip, tw.bety1, label=name)
axes_orbit_coupling[0, 0].set_ylabel('x [m]')
axes_orbit_coupling[1, 0].set_ylabel('y [m]')
axes_orbit_coupling[0, 1].set_ylabel('betx2 [m]')
axes_orbit_coupling[1, 1].set_ylabel('bety1 [m]')
axes_orbit_coupling[1, 0].set_xlabel('s - s_ip [m]')
axes_orbit_coupling[1, 1].set_xlabel('s - s_ip [m]')
for ax in axes_orbit_coupling.ravel():
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
fig_orbit_coupling.suptitle(
    'Three-solenoid orbit and beta coupling, initialized at IP')
fig_orbit_coupling.tight_layout()

print('004a build and check tapered solenoids')
print(f'  output lines json = {OUTPUT_LINES_JSON}')
print(f'  main Bs integral = {main_bs_integral:.12e} T m')
print(
    '  2 x compensation Bs integral = '
    f'{2 * comp_scale_b * comp_bs_integral_unscaled:.12e} T m')

print('  Symplectic checks:')
print(
    '    main SplineBoris line: '
    f'{main_symplectic_error:.12e}')
print(
    '    compensation SplineBoris line: '
    f'{comp_symplectic_error:.12e}')

for label, tw in twiss_results.items():
    print(f'  {label}:')
    print(
        f'    betx2_end = {tw.betx2[-1]:+.12e} m, '
        f'bety1_end = {tw.bety1[-1]:+.12e} m')

plt.show()
