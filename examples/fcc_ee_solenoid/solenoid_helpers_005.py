from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xtrack as xt

from tilted_solenoid import TiltedSolenoid
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField


HERE = Path(__file__).parent
FIELD_DATA_JSON = HERE / '005_solenoid_field_data.json'
LINES_JSON = HERE / '005_solenoid_lines.json'

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
    bx: dict[int, np.ndarray]
    by: dict[int, np.ndarray]
    scale_b: float
    bs_integral_average: np.ndarray
    bx_integral_average: dict[int, np.ndarray]
    by_integral_average: dict[int, np.ndarray]


def validate_max_multipole_order(max_multipole_order):
    max_supported_order = xt.SplineBoris._SB_MAX_MULTIPOLE_ORDER - 1
    if max_multipole_order < 0:
        raise ValueError('MAX_MULTIPOLE_ORDER must be non-negative')
    if max_multipole_order > max_supported_order:
        raise ValueError(
            f'MAX_MULTIPOLE_ORDER={max_multipole_order} is too high; '
            f'this SplineBoris supports at most {max_supported_order}')


def zero_negligible_central_derivatives(values_by_order, s_axis):
    if ZERO_CENTRAL_DERIVATIVES_HALF_LENGTH <= 0:
        return values_by_order

    values_by_order = {
        order: np.array(values, copy=True)
        for order, values in values_by_order.items()
    }
    mask_center = np.abs(s_axis) <= ZERO_CENTRAL_DERIVATIVES_HALF_LENGTH

    for order, values in values_by_order.items():
        if order >= ZERO_CENTRAL_DERIVATIVES_FROM_ORDER:
            values[mask_center] = 0.0

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

    values = {
        0: field_model.get_field(zero, zero, s_axis)[component_index],
    }
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
    values.update({
        order: derivatives[order] for order in range(1, max_order + 1)
    })
    return values


def _extract_values_at_s(field_model, s_axis, max_multipole_order):
    bx = _component_derivatives(field_model, s_axis, 'x', max_multipole_order)
    by = _component_derivatives(field_model, s_axis, 'y', max_multipole_order)

    bx = zero_negligible_central_derivatives(bx, s_axis)
    by = zero_negligible_central_derivatives(by, s_axis)

    return {
        'bs': _component_derivatives(field_model, s_axis, 'z', max_order=0)[0],
        'bx': bx,
        'by': by,
    }


def extract_required_field_data(spec, max_multipole_order):
    values = _extract_values_at_s(
        spec.field_model, spec.s_axis, max_multipole_order)

    s_axis = spec.s_axis
    n_intervals = len(s_axis) - 1
    s_integral = np.array([
        np.linspace(s_axis[ii], s_axis[ii + 1], SPLINE_INTEGRAL_POINTS)
        for ii in range(n_intervals)
    ])
    integral_values = _extract_values_at_s(
        spec.field_model, s_integral.ravel(), max_multipole_order)

    lengths = np.diff(s_axis)
    bs_integral = integral_values['bs'].reshape(n_intervals, -1)
    bx_integral = {
        order: vv.reshape(n_intervals, -1)
        for order, vv in integral_values['bx'].items()
    }
    by_integral = {
        order: vv.reshape(n_intervals, -1)
        for order, vv in integral_values['by'].items()
    }

    return ExtractedFieldData(
        name=spec.name,
        s_axis=s_axis,
        bs=values['bs'],
        bx=values['bx'],
        by=values['by'],
        scale_b=spec.scale_b,
        bs_integral_average=np.trapezoid(bs_integral, s_integral) / lengths,
        bx_integral_average={
            order: np.trapezoid(vv, s_integral) / lengths
            for order, vv in bx_integral.items()
        },
        by_integral_average={
            order: np.trapezoid(vv, s_integral) / lengths
            for order, vv in by_integral.items()
        },
    )


def extracted_field_to_dict(extracted):
    return {
        'name': extracted.name,
        's_axis': extracted.s_axis.tolist(),
        'bs': extracted.bs.tolist(),
        'bx': {
            str(order): vv.tolist()
            for order, vv in extracted.bx.items()
        },
        'by': {
            str(order): vv.tolist()
            for order, vv in extracted.by.items()
        },
        'scale_b': extracted.scale_b,
        'bs_integral_average': extracted.bs_integral_average.tolist(),
        'bx_integral_average': {
            str(order): vv.tolist()
            for order, vv in extracted.bx_integral_average.items()
        },
        'by_integral_average': {
            str(order): vv.tolist()
            for order, vv in extracted.by_integral_average.items()
        },
    }


def extracted_field_from_dict(data):
    bx_data = data['bx']
    by_data = data['by']
    bx_integral_data = data['bx_integral_average']
    by_integral_data = data['by_integral_average']

    if isinstance(bx_data, list):
        bx_data = {str(ii): vv for ii, vv in enumerate(bx_data)}
        by_data = {str(ii): vv for ii, vv in enumerate(by_data)}
        bx_integral_data = {
            str(ii): vv for ii, vv in enumerate(bx_integral_data)}
        by_integral_data = {
            str(ii): vv for ii, vv in enumerate(by_integral_data)}

    return ExtractedFieldData(
        name=data['name'],
        s_axis=np.array(data['s_axis'], dtype=float),
        bs=np.array(data['bs'], dtype=float),
        bx={
            int(order): np.array(vv, dtype=float)
            for order, vv in bx_data.items()
        },
        by={
            int(order): np.array(vv, dtype=float)
            for order, vv in by_data.items()
        },
        scale_b=float(data['scale_b']),
        bs_integral_average=np.array(data['bs_integral_average'], dtype=float),
        bx_integral_average={
            int(order): np.array(vv, dtype=float)
            for order, vv in bx_integral_data.items()
        },
        by_integral_average={
            int(order): np.array(vv, dtype=float)
            for order, vv in by_integral_data.items()
        },
    )


def field_data_metadata(max_multipole_order):
    return {
        'max_multipole_order': max_multipole_order,
        'derivative_step': DERIVATIVE_STEP,
        'zero_central_derivatives_from_order': (
            ZERO_CENTRAL_DERIVATIVES_FROM_ORDER),
        'zero_central_derivatives_half_length': (
            ZERO_CENTRAL_DERIVATIVES_HALF_LENGTH),
        'spline_integral_points': SPLINE_INTEGRAL_POINTS,
        'spline_steps_per_point': SPLINE_STEPS_PER_POINT,
    }


def save_extracted_fields_json(
        extracted_fields, file=FIELD_DATA_JSON,
        max_multipole_order=MAX_MULTIPOLE_ORDER):
    validate_max_multipole_order(max_multipole_order)
    data = {
        'metadata': field_data_metadata(max_multipole_order),
        'fields': {
            extracted.name: extracted_field_to_dict(extracted)
            for extracted in _iter_extracted_fields(extracted_fields)
        },
    }
    xt.json.dump(data, file, indent=1)


def load_extracted_fields_json(file=FIELD_DATA_JSON):
    data = xt.json.load(file)
    fields = data['fields']
    if isinstance(fields, list):
        fields = {
            item['name']: item for item in fields
        }
    return (
        data['metadata'],
        {
            name: extracted_field_from_dict(item)
            for name, item in fields.items()
        },
    )


def _make_spline4(values, s_derivatives, integral_average, ii):
    return xt.Spline4(
        val_start=values[ii],
        der_start=s_derivatives[ii],
        val_end=values[ii + 1],
        der_end=s_derivatives[ii + 1],
        integral=integral_average[ii],
    )


def build_splineboris_line(extracted, max_multipole_order):
    s_axis = extracted.s_axis
    n_intervals = len(s_axis) - 1

    bs_s_derivative = np.gradient(extracted.bs, s_axis, edge_order=2)
    bx_s_derivatives = {
        order: np.gradient(values, s_axis, edge_order=2)
        for order, values in extracted.bx.items()
    }
    by_s_derivatives = {
        order: np.gradient(values, s_axis, edge_order=2)
        for order, values in extracted.by.items()
    }

    elements = []
    names = []
    name_width = len(str(max(0, n_intervals - 1)))
    for ii in range(n_intervals):
        length = s_axis[ii + 1] - s_axis[ii]

        bs = _make_spline4(
            extracted.bs, bs_s_derivative,
            extracted.bs_integral_average, ii)

        bx = []
        by = []
        for order in range(max_multipole_order + 1):
            bx.append(_make_spline4(
                extracted.bx[order], bx_s_derivatives[order],
                extracted.bx_integral_average[order], ii))
            by.append(_make_spline4(
                extracted.by[order], by_s_derivatives[order],
                extracted.by_integral_average[order], ii))

        elements.append(xt.SplineBoris(
            bs=bs,
            bx=tuple(bx),
            by=tuple(by),
            length=length,
            n_steps=SPLINE_STEPS_PER_POINT,
            scale_b=extracted.scale_b,
        ))
        names.append(f'{extracted.name}_splineboris_{ii:0{name_width}d}')

    return xt.Line(elements=elements, element_names=names)


def save_lines_json(lines_by_name, file=LINES_JSON):
    data = {
        'lines': {
            name: line.to_dict()
            for name, line in lines_by_name.items()
        },
    }
    xt.json.dump(data, file, indent=1)


def load_lines_json(file=LINES_JSON):
    data = xt.json.load(file)
    return {
        name: xt.Line.from_dict(line_data)
        for name, line_data in data['lines'].items()
    }


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


def _iter_extracted_fields(extracted_fields):
    if isinstance(extracted_fields, dict):
        return extracted_fields.values()
    return extracted_fields


def compute_bs_integrals(extracted_fields):
    return {
        extracted.name: np.trapezoid(
            extracted.scale_b * extracted.bs, extracted.s_axis)
        for extracted in _iter_extracted_fields(extracted_fields)
    }


def format_bs_integrals_title(bs_integrals):
    entries = [
        f'{name}={value:.6g}' for name, value in bs_integrals.items()]
    entries.append(f'sum={sum(bs_integrals.values()):.6g}')
    return 'int Bs ds [T m]: ' + ', '.join(entries)


def _derivative_label(component, derivative_order):
    if derivative_order == 0:
        return f'B_{component} [T]'
    return f'd^{derivative_order} B_{component} / dx^{derivative_order}'


def plot_extracted_fields(
        extracted_fields, max_multipole_order, bs_integrals_title):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(extracted_fields), 3,
        figsize=(15, 4.0 * len(extracted_fields)),
        squeeze=False,
    )

    for row, extracted in enumerate(_iter_extracted_fields(extracted_fields)):
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


def plot_field_comparison(
        extracted_fields, lines_by_name, x_eval, y_eval, bs_integrals_title):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(extracted_fields), 3,
        figsize=(15, 4.0 * len(extracted_fields)),
        squeeze=False,
    )

    components = [
        ('B_x [T]', 'bx', 0),
        ('B_y [T]', 'by', 0),
        ('B_s [T]', 'bs', None),
    ]

    for row, extracted in enumerate(_iter_extracted_fields(extracted_fields)):
        line = lines_by_name[extracted.name]
        s_model, bx_model, by_model, bs_model = sample_splineboris_line(
            line, s0=extracted.s_axis[0], x=x_eval, y=y_eval)

        field_values = {
            'bx': extracted.scale_b * extracted.bx[0],
            'by': extracted.scale_b * extracted.by[0],
            'bs': extracted.scale_b * extracted.bs,
        }
        model_values = {
            'bx': bx_model,
            'by': by_model,
            'bs': bs_model,
        }

        for col, (ylabel, component, _) in enumerate(components):
            ax = axes[row, col]
            field_interp = np.interp(
                s_model, extracted.s_axis, field_values[component])
            ax.plot(
                extracted.s_axis, field_values[component], '.',
                label='field-map data')
            ax.plot(s_model, model_values[component], '--', label='SplineBoris')
            ax.plot(
                s_model,
                model_values[component] - field_interp,
                ':',
                label='difference',
            )
            ax.set_ylabel(f'{extracted.name}\n{ylabel}')
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(loc='best')

    for ax in axes[-1, :]:
        ax.set_xlabel('s [m]')

    fig.suptitle(
        'Saved field-map data and isolated SplineBoris-line comparison '
        f'at x={x_eval:g} m, y={y_eval:g} m\n'
        f'{bs_integrals_title}'
    )
    fig.tight_layout()
    return fig, axes


def compute_derivative_comparison_data(
        extracted_fields, lines_by_name, max_derivative_order, x_eval, y_eval):
    data = []

    for extracted in _iter_extracted_fields(extracted_fields):
        line = lines_by_name[extracted.name]
        splineboris_derivatives = sample_splineboris_derivatives_up_to_order(
            line,
            s0=extracted.s_axis[0],
            x_eval=x_eval,
            y_eval=y_eval,
            max_derivative_order=max_derivative_order,
        )
        data.append({
            'extracted': extracted,
            'splineboris_derivatives': splineboris_derivatives,
        })

    return data


def plot_transverse_derivative_comparison(
        data, derivative_order, x_eval, y_eval, bs_integrals_title):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(data), 2,
        figsize=(12, 4.0 * len(data)),
        squeeze=False,
    )

    for row, item in enumerate(data):
        extracted = item['extracted']
        s_model, bx_model, by_model, _ = item[
            'splineboris_derivatives'][derivative_order]
        field_values = [
            extracted.scale_b * extracted.bx[derivative_order],
            extracted.scale_b * extracted.by[derivative_order],
        ]
        model_values = [bx_model, by_model]

        for col, component in enumerate(('x', 'y')):
            ax = axes[row, col]
            field_interp = np.interp(
                s_model, extracted.s_axis, field_values[col])
            ax.plot(extracted.s_axis, field_values[col], '.', label='field-map data')
            ax.plot(s_model, model_values[col], '--', label='SplineBoris')
            ax.plot(
                s_model,
                model_values[col] - field_interp,
                ':',
                label='difference',
            )
            ax.set_ylabel(
                f'{extracted.name}\n'
                f'{_derivative_label(component, derivative_order)}')
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

    return {
        main.name: main,
        comp.name: comp,
    }
