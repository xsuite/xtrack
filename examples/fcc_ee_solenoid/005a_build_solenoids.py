from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

from tilted_solenoid import TiltedSolenoid
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField


HERE = Path(__file__).parent

THETA = -0.015
DERIVATIVE_STEP = 5e-4
MAX_MULTIPOLE_ORDER = 4
ZERO_CENTRAL_DERIVATIVES_FROM_ORDER = 2
ZERO_CENTRAL_DERIVATIVES_HALF_LENGTH = 0.25
SPLINE_STEPS_PER_POINT = 10
SPLINE_INTEGRAL_POINTS = 10
N_SLICES_MAIN_SOLENOID = 201
N_SLICES_COMP_SOLENOID = 201
SOL_ORBIT_CORRECTOR_DS = 1.8
X_FIELD_COMPARISON = 0.0
Y_FIELD_COMPARISON = 0.0


@dataclass
class SolenoidSpec:
    name: str
    field_model: object
    s_axis: np.ndarray
    scale_b: float


@dataclass
class ExtractedFieldData:
    name: str
    s_axis: np.ndarray
    bs: np.ndarray
    bx: list[np.ndarray]
    by: list[np.ndarray]
    scale_b: float


def zero_negligible_central_derivatives(values_by_order, s_axis):
    if ZERO_CENTRAL_DERIVATIVES_HALF_LENGTH <= 0:
        return values_by_order

    values_by_order = [np.array(values, copy=True) for values in values_by_order]
    mask_center = np.abs(s_axis) <= ZERO_CENTRAL_DERIVATIVES_HALF_LENGTH

    for order in range(ZERO_CENTRAL_DERIVATIVES_FROM_ORDER, len(values_by_order)):
        values_by_order[order][mask_center] = 0.0

    return values_by_order


def zero_negligible_central_derivative_values(bx, by, s_axis, derivative_order):
    if (
        derivative_order < ZERO_CENTRAL_DERIVATIVES_FROM_ORDER
        or ZERO_CENTRAL_DERIVATIVES_HALF_LENGTH <= 0
    ):
        return bx, by

    bx = np.array(bx, copy=True)
    by = np.array(by, copy=True)
    mask_center = np.abs(s_axis) <= ZERO_CENTRAL_DERIVATIVES_HALF_LENGTH
    bx[mask_center] = 0.0
    by[mask_center] = 0.0
    return bx, by


def _component_derivatives(field_model, s_axis, component, max_order):
    component_index = {'x': 0, 'y': 1, 'z': 2}[component]
    zero = np.zeros_like(s_axis)

    values = [field_model.get_field(zero, zero, s_axis)[component_index]]
    if max_order == 0:
        return values

    derivatives = field_model.compute_pure_field_derivatives(
        s=s_axis,
        direction='x',
        step=DERIVATIVE_STEP,
        component=component,
        max_order=max_order,
        min_order=1,
    )
    values.extend(derivatives[order] for order in range(1, max_order + 1))
    return values


def extract_required_field_data(spec, max_multipole_order):
    bx = _component_derivatives(
        spec.field_model, spec.s_axis, 'x', max_multipole_order)
    by = _component_derivatives(
        spec.field_model, spec.s_axis, 'y', max_multipole_order)

    bx = zero_negligible_central_derivatives(bx, spec.s_axis)
    by = zero_negligible_central_derivatives(by, spec.s_axis)

    return ExtractedFieldData(
        name=spec.name,
        s_axis=spec.s_axis,
        bs=_component_derivatives(
            spec.field_model, spec.s_axis, 'z', max_order=0)[0],
        bx=bx,
        by=by,
        scale_b=spec.scale_b,
    )


def _make_spline4(values, s_derivatives, integral_values, s_integral, ii, length):
    return xt.Spline4(
        val_start=values[ii],
        der_start=s_derivatives[ii],
        val_end=values[ii + 1],
        der_end=s_derivatives[ii + 1],
        integral=np.trapezoid(integral_values, s_integral) / length,
    )


def build_splineboris_line(spec, extracted, max_multipole_order):
    s_axis = extracted.s_axis
    n_intervals = len(s_axis) - 1

    s_integral = np.array([
        np.linspace(s_axis[ii], s_axis[ii + 1], SPLINE_INTEGRAL_POINTS)
        for ii in range(n_intervals)
    ])
    extracted_integral = extract_required_field_data(
        SolenoidSpec(
            name=spec.name,
            field_model=spec.field_model,
            s_axis=s_integral.ravel(),
            scale_b=spec.scale_b,
        ),
        max_multipole_order=max_multipole_order,
    )

    bs_integral = extracted_integral.bs.reshape(n_intervals, -1)
    bx_integral = [
        values.reshape(n_intervals, -1) for values in extracted_integral.bx]
    by_integral = [
        values.reshape(n_intervals, -1) for values in extracted_integral.by]

    bs_s_derivative = np.gradient(extracted.bs, s_axis, edge_order=2)
    bx_s_derivatives = [
        np.gradient(values, s_axis, edge_order=2) for values in extracted.bx]
    by_s_derivatives = [
        np.gradient(values, s_axis, edge_order=2) for values in extracted.by]

    elements = []
    names = []
    name_width = len(str(max(0, n_intervals - 1)))
    for ii in range(n_intervals):
        length = s_axis[ii + 1] - s_axis[ii]
        s_int = s_integral[ii]

        bs = _make_spline4(
            extracted.bs, bs_s_derivative, bs_integral[ii],
            s_int, ii, length)

        bx = []
        by = []
        for order in range(max_multipole_order + 1):
            bx.append(_make_spline4(
                extracted.bx[order], bx_s_derivatives[order],
                bx_integral[order][ii], s_int, ii, length))
            by.append(_make_spline4(
                extracted.by[order], by_s_derivatives[order],
                by_integral[order][ii], s_int, ii, length))

        elements.append(xt.SplineBoris(
            bs=bs,
            bx=tuple(bx),
            by=tuple(by),
            length=length,
            n_steps=SPLINE_STEPS_PER_POINT,
            scale_b=spec.scale_b,
        ))
        names.append(f'{spec.name}_splineboris_{ii:0{name_width}d}')

    return xt.Line(elements=elements, element_names=names)


def sample_splineboris_line(line, s0, x=0.0, y=0.0):
    s_out = []
    bx_out = []
    by_out = []
    bs_out = []

    s_start = s0
    for element in line.elements:
        s_local = np.linspace(0, element.length, SPLINE_STEPS_PER_POINT + 1)
        x_local = np.full_like(s_local, x)
        y_local = np.full_like(s_local, y)
        bx, by, bs = element.get_field(x_local, y_local, s_local)

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


def sample_splineboris_derivatives(line, s0, x_eval, y_eval, derivative_order):
    return sample_splineboris_derivatives_up_to_order(
        line, s0, x_eval, y_eval, derivative_order)[derivative_order]


def sample_splineboris_derivatives_up_to_order(
        line, s0, x_eval, y_eval, max_derivative_order):

    offsets = np.arange(-4, 5)

    bx_at_offsets = []
    by_at_offsets = []
    bs_at_offsets = []

    s_out = None
    for offset in offsets:
        s_curr, bx_curr, by_curr, bs_curr = sample_splineboris_line(
            line,
            s0=s0,
            x=x_eval + offset * DERIVATIVE_STEP,
            y=y_eval,
        )
        if s_out is None:
            s_out = s_curr
        bx_at_offsets.append(bx_curr)
        by_at_offsets.append(by_curr)
        bs_at_offsets.append(bs_curr)

    bx_at_offsets = np.array(bx_at_offsets)
    by_at_offsets = np.array(by_at_offsets)
    bs_at_offsets = np.array(bs_at_offsets)

    derivatives_by_order = {}
    zero_offset_index = np.where(offsets == 0)[0][0]
    derivatives_by_order[0] = (
        s_out,
        bx_at_offsets[zero_offset_index],
        by_at_offsets[zero_offset_index],
        bs_at_offsets[zero_offset_index],
    )

    for derivative_order in range(1, max_derivative_order + 1):
        coefficients = SolenoidField.finite_difference_coefficients(
            offsets, derivative_order)

        bx = (
            np.tensordot(coefficients, bx_at_offsets, axes=(0, 0))
            / DERIVATIVE_STEP**derivative_order
        )
        by = (
            np.tensordot(coefficients, by_at_offsets, axes=(0, 0))
            / DERIVATIVE_STEP**derivative_order
        )
        bs = (
            np.tensordot(coefficients, bs_at_offsets, axes=(0, 0))
            / DERIVATIVE_STEP**derivative_order
        )

        bx, by = zero_negligible_central_derivative_values(
            bx, by, s_out, derivative_order)
        derivatives_by_order[derivative_order] = (s_out, bx, by, bs)

    return derivatives_by_order


def compute_field_map_derivatives(
        field_model, s_axis, x_eval, y_eval, derivative_order):
    return compute_field_map_derivatives_up_to_order(
        field_model, s_axis, x_eval, y_eval, derivative_order)[derivative_order]


def compute_field_map_derivatives_up_to_order(
        field_model, s_axis, x_eval, y_eval, max_derivative_order):
    offsets = np.arange(-4, 5)

    bx_at_offsets = []
    by_at_offsets = []
    bs_at_offsets = []

    for offset in offsets:
        bx_curr, by_curr, bs_curr = field_model.get_field(
            np.full_like(s_axis, x_eval + offset * DERIVATIVE_STEP),
            np.full_like(s_axis, y_eval),
            s_axis,
        )
        bx_at_offsets.append(bx_curr)
        by_at_offsets.append(by_curr)
        bs_at_offsets.append(bs_curr)

    bx_at_offsets = np.array(bx_at_offsets)
    by_at_offsets = np.array(by_at_offsets)
    bs_at_offsets = np.array(bs_at_offsets)

    derivatives_by_order = {}
    zero_offset_index = np.where(offsets == 0)[0][0]
    derivatives_by_order[0] = (
        bx_at_offsets[zero_offset_index],
        by_at_offsets[zero_offset_index],
        bs_at_offsets[zero_offset_index],
    )

    for derivative_order in range(1, max_derivative_order + 1):
        coefficients = SolenoidField.finite_difference_coefficients(
            offsets, derivative_order)

        bx = (
            np.tensordot(coefficients, bx_at_offsets, axes=(0, 0))
            / DERIVATIVE_STEP**derivative_order
        )
        by = (
            np.tensordot(coefficients, by_at_offsets, axes=(0, 0))
            / DERIVATIVE_STEP**derivative_order
        )
        bs = (
            np.tensordot(coefficients, bs_at_offsets, axes=(0, 0))
            / DERIVATIVE_STEP**derivative_order
        )

        bx, by = zero_negligible_central_derivative_values(
            bx, by, s_axis, derivative_order)
        derivatives_by_order[derivative_order] = (bx, by, bs)

    return derivatives_by_order


def compute_bs_integrals(extracted_fields):
    return {
        extracted.name: np.trapezoid(
            extracted.scale_b * extracted.bs, extracted.s_axis)
        for extracted in extracted_fields
    }


def format_bs_integrals_title(bs_integrals):
    entries = [
        f'{name}={value:.6g}' for name, value in bs_integrals.items()]
    entries.append(f'sum={sum(bs_integrals.values()):.6g}')
    return 'int Bs ds [T m]: ' + ', '.join(entries)


def plot_extracted_fields(
        extracted_fields, max_multipole_order, bs_integrals_title):
    fig, axes = plt.subplots(
        len(extracted_fields), 3,
        figsize=(15, 4.0 * len(extracted_fields)),
        squeeze=False,
    )

    for row, extracted in enumerate(extracted_fields):
        s_axis = extracted.s_axis
        scale_b = extracted.scale_b

        ax = axes[row, 0]
        ax.plot(s_axis, scale_b * extracted.bs, label='B_s')
        ax.set_ylabel(f'{extracted.name}\nB_s [T]')
        ax.grid(True, alpha=0.3)

        ax = axes[row, 1]
        for order in range(max_multipole_order + 1):
            label = 'B_x' if order == 0 else f'd^{order} B_x / dx^{order}'
            ax.plot(s_axis, scale_b * extracted.bx[order], label=label)
        ax.set_ylabel('B_x derivatives')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)

        ax = axes[row, 2]
        for order in range(max_multipole_order + 1):
            label = 'B_y' if order == 0 else f'd^{order} B_y / dx^{order}'
            ax.plot(s_axis, scale_b * extracted.by[order], label=label)
        ax.set_ylabel('B_y derivatives')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel('s [m]')

    fig.suptitle(
        'Extracted SplineBoris inputs from field maps '
        f'(max multipole order {max_multipole_order})\n'
        f'{bs_integrals_title}'
    )
    fig.tight_layout()
    return fig, axes


def _derivative_label(component, derivative_order):
    if derivative_order == 0:
        return f'B_{component} [T]'
    return f'd^{derivative_order} B_{component} / dx^{derivative_order}'


def plot_field_comparison(specs, lines, x_eval, y_eval, bs_integrals_title):
    fig, axes = plt.subplots(
        len(specs), 3,
        figsize=(15, 4.0 * len(specs)),
        squeeze=False,
    )

    components = [
        ('B_x [T]', 0),
        ('B_y [T]', 1),
        ('B_s [T]', 2),
    ]

    for row, (spec, line) in enumerate(zip(specs, lines)):
        s_model, bx_model, by_model, bs_model = sample_splineboris_line(
            line, s0=spec.s_axis[0], x=x_eval, y=y_eval)
        x_arr = np.full_like(s_model, x_eval)
        y_arr = np.full_like(s_model, y_eval)
        bx_map, by_map, bs_map = spec.field_model.get_field(
            x_arr, y_arr, s_model)

        map_values = [
            spec.scale_b * bx_map,
            spec.scale_b * by_map,
            spec.scale_b * bs_map,
        ]
        model_values = [bx_model, by_model, bs_model]

        for col, (ylabel, _) in enumerate(components):
            ax = axes[row, col]
            ax.plot(s_model, map_values[col], '-', label='field map')
            ax.plot(s_model, model_values[col], '--', label='SplineBoris')
            ax.plot(
                s_model,
                model_values[col] - map_values[col],
                ':',
                label='difference',
            )
            ax.set_ylabel(f'{spec.name}\n{ylabel}')
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(loc='best')

    for ax in axes[-1, :]:
        ax.set_xlabel('s [m]')

    fig.suptitle(
        'Field-map and isolated SplineBoris-line comparison '
        f'at x={x_eval:g} m, y={y_eval:g} m\n'
        f'{bs_integrals_title}'
    )
    fig.tight_layout()
    return fig, axes


def compute_derivative_comparison_data(
        specs, lines, max_derivative_order, x_eval, y_eval):
    data = []

    for spec, line in zip(specs, lines):
        splineboris_derivatives = sample_splineboris_derivatives_up_to_order(
            line,
            s0=spec.s_axis[0],
            x_eval=x_eval,
            y_eval=y_eval,
            max_derivative_order=max_derivative_order,
        )
        s_model = splineboris_derivatives[0][0]
        field_map_derivatives = compute_field_map_derivatives_up_to_order(
            spec.field_model,
            s_model,
            x_eval=x_eval,
            y_eval=y_eval,
            max_derivative_order=max_derivative_order,
        )
        data.append({
            'spec': spec,
            's': s_model,
            'splineboris_derivatives': splineboris_derivatives,
            'field_map_derivatives': field_map_derivatives,
        })

    return data


def plot_transverse_derivative_comparison(
        data, derivative_order, x_eval, y_eval, bs_integrals_title):
    fig, axes = plt.subplots(
        len(data), 2,
        figsize=(12, 4.0 * len(data)),
        squeeze=False,
    )

    for row, item in enumerate(data):
        spec = item['spec']
        s_model = item['s']
        _, bx_model, by_model, _ = item[
            'splineboris_derivatives'][derivative_order]
        bx_map, by_map, _ = item[
            'field_map_derivatives'][derivative_order]

        map_values = [
            spec.scale_b * bx_map,
            spec.scale_b * by_map,
        ]
        model_values = [bx_model, by_model]

        for col, component in enumerate(('x', 'y')):
            ax = axes[row, col]
            ax.plot(s_model, map_values[col], '-', label='field map')
            ax.plot(s_model, model_values[col], '--', label='SplineBoris')
            ax.plot(
                s_model,
                model_values[col] - map_values[col],
                ':',
                label='difference',
            )
            ax.set_ylabel(
                f'{spec.name}\n{_derivative_label(component, derivative_order)}')
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(loc='best')

    for ax in axes[-1, :]:
        ax.set_xlabel('s [m]')

    fig.suptitle(
        f'Transverse derivative comparison, order {derivative_order} '
        f'at x={x_eval:g} m, y={y_eval:g} m\n'
        f'{bs_integrals_title}'
    )
    fig.tight_layout()
    return fig, axes


def make_solenoid_specs():
    main = SolenoidSpec(
        name='main_solenoid',
        field_model=TiltedSolenoid(L=1.23 * 2, a=0.13, B0=2.0, theta=THETA),
        s_axis=np.unique(np.r_[
            np.linspace(-2.399, 2.399, N_SLICES_MAIN_SOLENOID),
            -SOL_ORBIT_CORRECTOR_DS,
            SOL_ORBIT_CORRECTOR_DS,
        ]),
        scale_b=1.0,
    )

    comp = SolenoidSpec(
        name='compensation_solenoid',
        field_model=SolenoidField(L=1.5, a=0.03, B0=1.0, z0=0.0),
        s_axis=np.linspace(-1.0, 1.0, N_SLICES_COMP_SOLENOID),
        scale_b=1.0,
    )

    _, _, bz_main = main.field_model.get_field(
        np.zeros_like(main.s_axis), np.zeros_like(main.s_axis), main.s_axis)
    _, _, bz_comp = comp.field_model.get_field(
        np.zeros_like(comp.s_axis), np.zeros_like(comp.s_axis), comp.s_axis)
    comp.scale_b = (
        -np.trapezoid(bz_main, main.s_axis)
        / np.trapezoid(bz_comp, comp.s_axis)
        / 2.0
    )

    return [main, comp]


max_supported_order = xt.SplineBoris._SB_MAX_MULTIPOLE_ORDER - 1
if MAX_MULTIPOLE_ORDER < 0:
    raise ValueError('MAX_MULTIPOLE_ORDER must be non-negative')
if MAX_MULTIPOLE_ORDER > max_supported_order:
    raise ValueError(
        f'MAX_MULTIPOLE_ORDER={MAX_MULTIPOLE_ORDER} is too high; '
        f'this SplineBoris supports at most {max_supported_order}')

specs = make_solenoid_specs()

extracted_fields = [
    extract_required_field_data(spec, MAX_MULTIPOLE_ORDER)
    for spec in specs
]
bs_integrals = compute_bs_integrals(extracted_fields)
bs_integrals_title = format_bs_integrals_title(bs_integrals)
plt.close('all')

fig_extracted_fields, axes_extracted_fields = plot_extracted_fields(
    extracted_fields,
    MAX_MULTIPOLE_ORDER,
    bs_integrals_title,
)

lines = [
    build_splineboris_line(spec, extracted, MAX_MULTIPOLE_ORDER)
    for spec, extracted in zip(specs, extracted_fields)
]
fig_field_comparison, axes_field_comparison = plot_field_comparison(
    specs,
    lines,
    x_eval=X_FIELD_COMPARISON,
    y_eval=Y_FIELD_COMPARISON,
    bs_integrals_title=bs_integrals_title,
)

derivative_comparison_data = compute_derivative_comparison_data(
    specs,
    lines,
    max_derivative_order=MAX_MULTIPOLE_ORDER,
    x_eval=X_FIELD_COMPARISON,
    y_eval=Y_FIELD_COMPARISON,
)

derivative_comparison_figures = {}
derivative_comparison_axes = {}
for order in range(1, MAX_MULTIPOLE_ORDER + 1):
    fig, axes = plot_transverse_derivative_comparison(
        derivative_comparison_data,
        derivative_order=order,
        x_eval=X_FIELD_COMPARISON,
        y_eval=Y_FIELD_COMPARISON,
        bs_integrals_title=bs_integrals_title,
    )
    derivative_comparison_figures[order] = fig
    derivative_comparison_axes[order] = axes

print('Built isolated SplineBoris lines:')
for spec, line in zip(specs, lines):
    print(
        f'  {spec.name}: {len(line.elements)} elements, '
        f'scale_b={spec.scale_b:.16g}')

plt.show()
