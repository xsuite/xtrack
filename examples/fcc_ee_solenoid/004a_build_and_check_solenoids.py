from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
from scipy.interpolate import LSQUnivariateSpline

from tilted_solenoid import TiltedSolenoid
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField


HERE = Path(__file__).parent
OUTPUT_LINES_JSON = HERE / '004_solenoid_lines.json'

THETA = -0.015
PARTICLE = 'positron'
ENERGY0 = 45.6e9

MAX_TRANSVERSE_DERIVATIVE_ORDER = 4
MAX_TRANSVERSE_DERIVATIVE_ORDER_FOR_SPLINE = 1
DERIVATIVE_STEP = 5e-4
SPLINE_INTEGRAL_POINTS = 10
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

BETX = 0.09
BETY = 0.0007

MAIN_SOLENOID_S_AXIS = np.linspace(-2.399, 2.399, 401)
COMP_SOLENOID_S_AXIS = np.linspace(-1.0, 1.0, 401)
COMP_SOLENOID_DISTANCE_FROM_IP = 12.0


def smooth_edge_taper(s_axis, taper_length):
    taper = np.ones_like(s_axis)
    s_min = np.min(s_axis)
    s_max = np.max(s_axis)

    left = s_axis < s_min + taper_length
    u_left = (s_axis[left] - s_min) / taper_length
    taper[left] = u_left**3 * (10.0 - 15.0 * u_left + 6.0 * u_left**2)

    right = s_axis > s_max - taper_length
    u_right = (s_max - s_axis[right]) / taper_length
    taper[right] = u_right**3 * (10.0 - 15.0 * u_right + 6.0 * u_right**2)

    taper[s_axis == s_min] = 0.0
    taper[s_axis == s_max] = 0.0
    return taper


def fit_slice_boundary_spline(s_unique, values_unique, s_axis):
    return LSQUnivariateSpline(
        s_unique, values_unique,
        t=s_axis[1:-1],
        k=S_DERIVATIVE_SPLINE_ORDER,
    )


def extract_tapered_field_data(name, field_model, s_axis):
    zero = np.zeros_like(s_axis)
    bx0, by0, bs_raw = field_model.get_field(zero, zero, s_axis)

    bx = {0: np.array(bx0, copy=True)}
    by = {0: np.array(by0, copy=True)}
    bs_x_pure = {0: np.array(bs_raw, copy=True)}
    bs_y_pure = {0: np.array(bs_raw, copy=True)}
    bx_y_pure = {0: np.array(bx0, copy=True)}
    by_y_pure = {0: np.array(by0, copy=True)}

    if MAX_TRANSVERSE_DERIVATIVE_ORDER > 0:
        bx_derivatives = field_model.compute_pure_field_derivatives(
            s=s_axis,
            direction='x',
            step=DERIVATIVE_STEP,
            component='x',
            max_order=MAX_TRANSVERSE_DERIVATIVE_ORDER,
            min_order=1,
        )
        by_derivatives = field_model.compute_pure_field_derivatives(
            s=s_axis,
            direction='x',
            step=DERIVATIVE_STEP,
            component='y',
            max_order=MAX_TRANSVERSE_DERIVATIVE_ORDER,
            min_order=1,
        )
        bs_x_derivatives = field_model.compute_pure_field_derivatives(
            s=s_axis,
            direction='x',
            step=DERIVATIVE_STEP,
            component='z',
            max_order=MAX_TRANSVERSE_DERIVATIVE_ORDER,
            min_order=1,
        )
        bx_y_derivatives = field_model.compute_pure_field_derivatives(
            s=s_axis,
            direction='y',
            step=DERIVATIVE_STEP,
            component='x',
            max_order=MAX_TRANSVERSE_DERIVATIVE_ORDER,
            min_order=1,
        )
        by_y_derivatives = field_model.compute_pure_field_derivatives(
            s=s_axis,
            direction='y',
            step=DERIVATIVE_STEP,
            component='y',
            max_order=MAX_TRANSVERSE_DERIVATIVE_ORDER,
            min_order=1,
        )
        bs_y_derivatives = field_model.compute_pure_field_derivatives(
            s=s_axis,
            direction='y',
            step=DERIVATIVE_STEP,
            component='z',
            max_order=MAX_TRANSVERSE_DERIVATIVE_ORDER,
            min_order=1,
        )
        for order in range(1, MAX_TRANSVERSE_DERIVATIVE_ORDER + 1):
            bx[order] = bx_derivatives[order]
            by[order] = by_derivatives[order]
            bs_x_pure[order] = bs_x_derivatives[order]
            bs_y_pure[order] = bs_y_derivatives[order]
            bx_y_pure[order] = bx_y_derivatives[order]
            by_y_pure[order] = by_y_derivatives[order]

    taper = smooth_edge_taper(s_axis, TAPER_LENGTH)
    bs = np.array(bs_raw, copy=True) * taper
    for order in range(MAX_TRANSVERSE_DERIVATIVE_ORDER + 1):
        bx[order] = np.array(bx[order], copy=True) * taper
        by[order] = np.array(by[order], copy=True) * taper
        bs_x_pure[order] = np.array(bs_x_pure[order], copy=True) * taper
        bs_y_pure[order] = np.array(bs_y_pure[order], copy=True) * taper
        bx_y_pure[order] = np.array(bx_y_pure[order], copy=True) * taper
        by_y_pure[order] = np.array(by_y_pure[order], copy=True) * taper

    n_intervals = len(s_axis) - 1
    s_integral = np.array([
        np.linspace(s_axis[ii], s_axis[ii + 1], SPLINE_INTEGRAL_POINTS)
        for ii in range(n_intervals)
    ])
    zero_integral = np.zeros_like(s_integral)
    taper_integral = smooth_edge_taper(s_integral, TAPER_LENGTH)

    bx0_integral, by0_integral, bs_raw_integral = field_model.get_field(
        zero_integral, zero_integral, s_integral)

    bs_integral_values = bs_raw_integral * taper_integral
    bx_integral_values = {0: bx0_integral * taper_integral}
    by_integral_values = {0: by0_integral * taper_integral}

    if MAX_TRANSVERSE_DERIVATIVE_ORDER > 0:
        bx_integral_derivatives = field_model.compute_pure_field_derivatives(
            s=s_integral,
            direction='x',
            step=DERIVATIVE_STEP,
            component='x',
            max_order=MAX_TRANSVERSE_DERIVATIVE_ORDER,
            min_order=1,
        )
        by_integral_derivatives = field_model.compute_pure_field_derivatives(
            s=s_integral,
            direction='x',
            step=DERIVATIVE_STEP,
            component='y',
            max_order=MAX_TRANSVERSE_DERIVATIVE_ORDER,
            min_order=1,
        )
        for order in range(1, MAX_TRANSVERSE_DERIVATIVE_ORDER + 1):
            bx_integral_values[order] = (
                bx_integral_derivatives[order] * taper_integral)
            by_integral_values[order] = (
                by_integral_derivatives[order] * taper_integral)

    length = np.diff(s_axis)
    max_s_derivative_plot_order = MAX_S_DERIVATIVE_PLOT_ORDER

    s_flat = s_integral.ravel()
    order_sort = np.argsort(s_flat)
    s_sorted = s_flat[order_sort]
    s_unique, s_unique_start = np.unique(s_sorted, return_index=True)
    s_unique_count = np.diff(np.r_[s_unique_start, len(s_sorted)])

    s_fit_splines = {}
    values_sorted = bs_integral_values.ravel()[order_sort]
    values_unique = np.add.reduceat(
        values_sorted, s_unique_start) / s_unique_count
    s_fit_splines['bs'] = fit_slice_boundary_spline(
        s_unique, values_unique, s_axis)

    s_fit_splines['bx'] = {}
    s_fit_splines['by'] = {}
    for order in range(MAX_TRANSVERSE_DERIVATIVE_ORDER + 1):
        for component, values, destination in (
                ('bx', bx_integral_values[order], s_fit_splines['bx']),
                ('by', by_integral_values[order], s_fit_splines['by'])):
            values_sorted = values.ravel()[order_sort]
            values_unique = np.add.reduceat(
                values_sorted, s_unique_start) / s_unique_count
            destination[order] = fit_slice_boundary_spline(
                s_unique, values_unique, s_axis)

    bs = s_fit_splines['bs'](s_axis)
    bx = {
        order: s_fit_splines['bx'][order](s_axis)
        for order in range(MAX_TRANSVERSE_DERIVATIVE_ORDER + 1)
    }
    by = {
        order: s_fit_splines['by'][order](s_axis)
        for order in range(MAX_TRANSVERSE_DERIVATIVE_ORDER + 1)
    }
    bx_y_pure[0] = np.array(bx[0], copy=True)
    by_y_pure[0] = np.array(by[0], copy=True)
    bs_x_pure[0] = np.array(bs, copy=True)
    bs_y_pure[0] = np.array(bs, copy=True)

    bs[0] = 0.0
    bs[-1] = 0.0
    for order in range(MAX_TRANSVERSE_DERIVATIVE_ORDER + 1):
        bx[order][0] = 0.0
        bx[order][-1] = 0.0
        by[order][0] = 0.0
        by[order][-1] = 0.0

    bs_s_derivative_start = s_fit_splines['bs'].derivative(1)(s_axis[:-1])
    bs_s_derivative_end = s_fit_splines['bs'].derivative(1)(s_axis[1:])

    bx_s_derivatives_start = {}
    bx_s_derivatives_end = {}
    by_s_derivatives_start = {}
    by_s_derivatives_end = {}
    for order in range(MAX_TRANSVERSE_DERIVATIVE_ORDER + 1):
        bx_s_derivatives_start[order] = (
            s_fit_splines['bx'][order].derivative(1)(s_axis[:-1]))
        bx_s_derivatives_end[order] = (
            s_fit_splines['bx'][order].derivative(1)(s_axis[1:]))
        by_s_derivatives_start[order] = (
            s_fit_splines['by'][order].derivative(1)(s_axis[:-1]))
        by_s_derivatives_end[order] = (
            s_fit_splines['by'][order].derivative(1)(s_axis[1:]))

    bs_s_derivative_start[0] = 0.0
    bs_s_derivative_end[-1] = 0.0
    for order in range(MAX_TRANSVERSE_DERIVATIVE_ORDER + 1):
        bx_s_derivatives_start[order][0] = 0.0
        bx_s_derivatives_end[order][-1] = 0.0
        by_s_derivatives_start[order][0] = 0.0
        by_s_derivatives_end[order][-1] = 0.0

    s_derivative_plot_data = {}
    raw_derivative_values = {
        'bx': bx0_integral * taper_integral,
        'by': by0_integral * taper_integral,
        'bs': bs_integral_values,
    }
    for derivative_order in range(max_s_derivative_plot_order + 1):
        s_derivative_plot_data[derivative_order] = {
            's': s_unique,
        }

        for component, values in raw_derivative_values.items():
            values_sorted = values.ravel()[order_sort]
            derivative_values = (
                np.add.reduceat(values_sorted, s_unique_start)
                / s_unique_count
            )
            for _ in range(derivative_order):
                derivative_values = np.gradient(
                    derivative_values,
                    s_unique,
                    edge_order=2,
                )
            s_derivative_plot_data[derivative_order][component] = (
                derivative_values)

    bs_integral_average = (
        np.array([
            s_fit_splines['bs'].integral(s_axis[ii], s_axis[ii + 1])
            for ii in range(n_intervals)
        ])
        / length
    )
    bx_integral_average = {
        order: (
            np.array([
                s_fit_splines['bx'][order].integral(
                    s_axis[ii], s_axis[ii + 1])
                for ii in range(n_intervals)
            ])
            / length
        )
        for order in range(MAX_TRANSVERSE_DERIVATIVE_ORDER + 1)
    }
    by_integral_average = {
        order: (
            np.array([
                s_fit_splines['by'][order].integral(
                    s_axis[ii], s_axis[ii + 1])
                for ii in range(n_intervals)
            ])
            / length
        )
        for order in range(MAX_TRANSVERSE_DERIVATIVE_ORDER + 1)
    }

    return {
        'name': name,
        's_axis': s_axis,
        'taper': taper,
        'bs_raw': np.array(bs_raw, copy=True),
        'bs': bs,
        'bx': bx,
        'by': by,
        'bs_x_pure': bs_x_pure,
        'bs_y_pure': bs_y_pure,
        'bx_y_pure': bx_y_pure,
        'by_y_pure': by_y_pure,
        'bs_s_derivative_start': bs_s_derivative_start,
        'bs_s_derivative_end': bs_s_derivative_end,
        'bx_s_derivatives_start': bx_s_derivatives_start,
        'bx_s_derivatives_end': bx_s_derivatives_end,
        'by_s_derivatives_start': by_s_derivatives_start,
        'by_s_derivatives_end': by_s_derivatives_end,
        's_derivative_plot_data': s_derivative_plot_data,
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

        if USE_NEAR_AXIS_SIMPLIFIED_MODEL:
            bs_derivative = (
                (field_data['bs'][ii + 1] - field_data['bs'][ii])
                / length
            )
            bs_der_start = bs_derivative
            bs_der_end = bs_derivative
            bs_integral_average = 0.5 * (
                field_data['bs'][ii] + field_data['bs'][ii + 1])
        else:
            bs_derivative = None
            bs_der_start = field_data['bs_s_derivative_start'][ii]
            bs_der_end = field_data['bs_s_derivative_end'][ii]
            bs_integral_average = field_data['bs_integral_average'][ii]

        bs = xt.Spline4(
            val_start=field_data['bs'][ii],
            der_start=bs_der_start,
            val_end=field_data['bs'][ii + 1],
            der_end=bs_der_end,
            integral=bs_integral_average,
        )

        bx = []
        by = []
        for order in range(MAX_TRANSVERSE_DERIVATIVE_ORDER_FOR_SPLINE + 1):
            if USE_NEAR_AXIS_SIMPLIFIED_MODEL and order == 0:
                bx_val_start = field_data['bx'][order][ii]
                bx_val_end = field_data['bx'][order][ii + 1]
                bx_der_start = (
                    field_data['bx_s_derivatives_start'][order][ii])
                bx_der_end = (
                    field_data['bx_s_derivatives_end'][order][ii])
                bx_integral_average = (
                    field_data['bx_integral_average'][order][ii])

                by_val_start = field_data['by'][order][ii]
                by_val_end = field_data['by'][order][ii + 1]
                by_der_start = (
                    field_data['by_s_derivatives_start'][order][ii])
                by_der_end = (
                    field_data['by_s_derivatives_end'][order][ii])
                by_integral_average = (
                    field_data['by_integral_average'][order][ii])

            elif USE_NEAR_AXIS_SIMPLIFIED_MODEL and order == 1:
                bx_integral_average = -0.5 * bs_derivative
                bx_val_start = bx_integral_average
                bx_val_end = bx_integral_average
                bx_der_start = 0.0
                bx_der_end = 0.0

                by_val_start = 0.0
                by_val_end = 0.0
                by_der_start = 0.0
                by_der_end = 0.0
                by_integral_average = 0.0

            elif USE_NEAR_AXIS_SIMPLIFIED_MODEL and order > 1:
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

            else:
                bx_val_start = field_data['bx'][order][ii]
                bx_val_end = field_data['bx'][order][ii + 1]
                bx_der_start = (
                    field_data['bx_s_derivatives_start'][order][ii])
                bx_der_end = (
                    field_data['bx_s_derivatives_end'][order][ii])
                bx_integral_average = (
                    field_data['bx_integral_average'][order][ii])

                by_val_start = field_data['by'][order][ii]
                by_val_end = field_data['by'][order][ii + 1]
                by_der_start = (
                    field_data['by_s_derivatives_start'][order][ii])
                by_der_end = (
                    field_data['by_s_derivatives_end'][order][ii])
                by_integral_average = (
                    field_data['by_integral_average'][order][ii])

            bx.append(xt.Spline4(
                val_start=bx_val_start,
                der_start=bx_der_start,
                val_end=bx_val_end,
                der_end=bx_der_end,
                integral=bx_integral_average,
            ))
            by.append(xt.Spline4(
                val_start=by_val_start,
                der_start=by_der_start,
                val_end=by_val_end,
                der_end=by_der_end,
                integral=by_integral_average,
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


def symplectic_error(line, particle_ref):
    s_matrix = xt.linear_normal_form.S
    r_matrix = line.get_R_matrix(
        particle_on_co=particle_ref.copy())['R_matrix']
    return np.linalg.norm(r_matrix.T @ s_matrix @ r_matrix - s_matrix, ord=2)


def sample_splineboris_line(line, s0, x=0.0, y=0.0):
    s_out = []
    bx_out = []
    by_out = []
    bs_out = []

    s_start = s0
    for element in line.elements:
        s_local = np.linspace(0.0, element.length, SPLINE_STEPS_PER_POINT + 1)
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


assert MAX_TRANSVERSE_DERIVATIVE_ORDER_FOR_SPLINE <= xt.SplineBoris._SB_MAX_MULTIPOLE_ORDER - 1
assert MAX_TRANSVERSE_DERIVATIVE_ORDER <= 5
assert S_DERIVATIVE_SPLINE_ORDER == 4
assert MAX_S_DERIVATIVE_PLOT_ORDER <= 5


# Build the two physical solenoid models and extract the tapered field data.
main_field_model = TiltedSolenoid(L=1.23 * 2, a=0.13, B0=2.0, theta=THETA)
comp_field_model = SolenoidField(L=1.5, a=0.03, B0=1.0, z0=0.0)

main_field_data = extract_tapered_field_data(
    'main_solenoid', main_field_model, MAIN_SOLENOID_S_AXIS)
comp_field_data = extract_tapered_field_data(
    'compensation_solenoid', comp_field_model, COMP_SOLENOID_S_AXIS)

main_bs_integral = np.trapezoid(
    main_field_data['bs'], main_field_data['s_axis'])
comp_bs_integral_unscaled = np.trapezoid(
    comp_field_data['bs'], comp_field_data['s_axis'])
comp_scale_b = -main_bs_integral / comp_bs_integral_unscaled / 2.0

particle_ref = xt.Particles(PARTICLE, energy0=ENERGY0)
rigidity0 = particle_ref.rigidity0[0]


# Build the isolated SplineBoris templates that will be installed later.
line_main_solenoid = build_splineboris_line(
    'main_solenoid', main_field_data, 1.0)
line_compensation_solenoid = build_splineboris_line(
    'compensation_solenoid', comp_field_data, comp_scale_b)
line_main_solenoid.particle_ref = particle_ref.copy()
line_compensation_solenoid.particle_ref = particle_ref.copy()

if SAVE_SOLENOID_LINES_JSON:
    output_data = {
        'metadata': {
            'max_transverse_derivative_order': (
                MAX_TRANSVERSE_DERIVATIVE_ORDER),
            'max_transverse_derivative_order_for_spline': (
                MAX_TRANSVERSE_DERIVATIVE_ORDER_FOR_SPLINE),
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

##########
# Checks #
##########

# Build a complete local three-solenoid system for checks before FCC install.
main_half_length = (
    MAIN_SOLENOID_S_AXIS[-1] - MAIN_SOLENOID_S_AXIS[0]) / 2.0
drift_between_comp_and_main = (
    COMP_SOLENOID_DISTANCE_FROM_IP - main_half_length)

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


# Twiss checks for the local three-solenoid systems.
twiss_results = {}
twiss_results['SplineBoris'] = line_systems['SplineBoris'].twiss(
    init_at='ip', betx=BETX, bety=BETY)
twiss_results['VariableSolenoid'] = line_systems['VariableSolenoid'].twiss(
    init_at='ip', betx=BETX, bety=BETY)

main_symplectic_error = symplectic_error(line_main_solenoid, particle_ref)
comp_symplectic_error = symplectic_error(
    line_compensation_solenoid, particle_ref)


# Plots: compare field-map extraction against the built SplineBoris lines.
plt.close('all')

comparison_fields = {
}
if PLOT_MAIN_SOLENOID:
    comparison_fields['main_solenoid'] = {
        'field_data': main_field_data,
        'line': line_main_solenoid,
        'scale_b': 1.0,
    }
if PLOT_COMPENSATION_SOLENOID:
    comparison_fields['compensation_solenoid'] = {
        'field_data': comp_field_data,
        'line': line_compensation_solenoid,
        'scale_b': comp_scale_b,
    }

sampled_lines = {}
for name, item in comparison_fields.items():
    field_data = item['field_data']
    s_model, bx_model, by_model, bs_model = sample_splineboris_line(
        item['line'],
        s0=field_data['s_axis'][0],
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
            2, 3, figsize=(15, 8), sharex=True,
            num=figure_number_offset + 10 * component_index)
        axes_s_derivatives_flat = axes_s_derivatives.ravel()

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
            item['line'],
            s0=field_data['s_axis'][0],
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
            item['line'],
            s0=field_data['s_axis'][0],
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
            2, 3, figsize=(15, 8), sharex=True,
            num=figure_number_offset + plot_index)
        axes_derivatives_flat = axes_derivatives.ravel()

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
