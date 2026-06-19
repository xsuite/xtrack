import numpy as np
import xtrack as xt
from scipy.interpolate import LSQUnivariateSpline


def smooth_edge_taper(*, s_axis, taper_length):
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


def fit_slice_boundary_spline(
        *, s_unique, values_unique, s_axis, s_derivative_spline_order):
    return LSQUnivariateSpline(
        s_unique, values_unique,
        t=s_axis[1:-1],
        k=s_derivative_spline_order,
    )


def extract_tapered_field_data(
        *, name, field_model, s_axis,
        max_transverse_derivative_order,
        derivative_step,
        spline_integral_points,
        taper_length,
        s_derivative_spline_order,
        max_s_derivative_plot_order,
        decrease_s_poly_order_with_transverse_order):
    zero = np.zeros_like(s_axis)
    bx0, by0, bs_raw = field_model.get_field(zero, zero, s_axis)

    bx = {0: np.array(bx0, copy=True)}
    by = {0: np.array(by0, copy=True)}
    bs_x_pure = {0: np.array(bs_raw, copy=True)}
    bs_y_pure = {0: np.array(bs_raw, copy=True)}
    bx_y_pure = {0: np.array(bx0, copy=True)}
    by_y_pure = {0: np.array(by0, copy=True)}

    if max_transverse_derivative_order > 0:
        bx_derivatives = field_model.compute_pure_field_derivatives(
            s=s_axis,
            direction='x',
            step=derivative_step,
            component='x',
            max_order=max_transverse_derivative_order,
            min_order=1,
        )
        by_derivatives = field_model.compute_pure_field_derivatives(
            s=s_axis,
            direction='x',
            step=derivative_step,
            component='y',
            max_order=max_transverse_derivative_order,
            min_order=1,
        )
        bs_x_derivatives = field_model.compute_pure_field_derivatives(
            s=s_axis,
            direction='x',
            step=derivative_step,
            component='z',
            max_order=max_transverse_derivative_order,
            min_order=1,
        )
        bx_y_derivatives = field_model.compute_pure_field_derivatives(
            s=s_axis,
            direction='y',
            step=derivative_step,
            component='x',
            max_order=max_transverse_derivative_order,
            min_order=1,
        )
        by_y_derivatives = field_model.compute_pure_field_derivatives(
            s=s_axis,
            direction='y',
            step=derivative_step,
            component='y',
            max_order=max_transverse_derivative_order,
            min_order=1,
        )
        bs_y_derivatives = field_model.compute_pure_field_derivatives(
            s=s_axis,
            direction='y',
            step=derivative_step,
            component='z',
            max_order=max_transverse_derivative_order,
            min_order=1,
        )
        for order in range(1, max_transverse_derivative_order + 1):
            bx[order] = bx_derivatives[order]
            by[order] = by_derivatives[order]
            bs_x_pure[order] = bs_x_derivatives[order]
            bs_y_pure[order] = bs_y_derivatives[order]
            bx_y_pure[order] = bx_y_derivatives[order]
            by_y_pure[order] = by_y_derivatives[order]

    taper = smooth_edge_taper(s_axis=s_axis, taper_length=taper_length)
    bs = np.array(bs_raw, copy=True) * taper
    for order in range(max_transverse_derivative_order + 1):
        bx[order] = np.array(bx[order], copy=True) * taper
        by[order] = np.array(by[order], copy=True) * taper
        bs_x_pure[order] = np.array(bs_x_pure[order], copy=True) * taper
        bs_y_pure[order] = np.array(bs_y_pure[order], copy=True) * taper
        bx_y_pure[order] = np.array(bx_y_pure[order], copy=True) * taper
        by_y_pure[order] = np.array(by_y_pure[order], copy=True) * taper

    n_intervals = len(s_axis) - 1
    s_integral = np.array([
        np.linspace(s_axis[ii], s_axis[ii + 1], spline_integral_points)
        for ii in range(n_intervals)
    ])
    zero_integral = np.zeros_like(s_integral)
    taper_integral = smooth_edge_taper(
        s_axis=s_integral, taper_length=taper_length)

    bx0_integral, by0_integral, bs_raw_integral = field_model.get_field(
        zero_integral, zero_integral, s_integral)

    bs_integral_values = bs_raw_integral * taper_integral
    bx_integral_values = {0: bx0_integral * taper_integral}
    by_integral_values = {0: by0_integral * taper_integral}

    if max_transverse_derivative_order > 0:
        bx_integral_derivatives = field_model.compute_pure_field_derivatives(
            s=s_integral,
            direction='x',
            step=derivative_step,
            component='x',
            max_order=max_transverse_derivative_order,
            min_order=1,
        )
        by_integral_derivatives = field_model.compute_pure_field_derivatives(
            s=s_integral,
            direction='x',
            step=derivative_step,
            component='y',
            max_order=max_transverse_derivative_order,
            min_order=1,
        )
        for order in range(1, max_transverse_derivative_order + 1):
            bx_integral_values[order] = (
                bx_integral_derivatives[order] * taper_integral)
            by_integral_values[order] = (
                by_integral_derivatives[order] * taper_integral)

    length = np.diff(s_axis)
    max_s_derivative_plot_order = max_s_derivative_plot_order

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
        s_unique=s_unique,
        values_unique=values_unique,
        s_axis=s_axis,
        s_derivative_spline_order=s_derivative_spline_order)

    s_fit_splines['bx'] = {}
    s_fit_splines['by'] = {}
    for order in range(max_transverse_derivative_order + 1):
        for values, destination in (
                (bx_integral_values[order], s_fit_splines['bx']),
                (by_integral_values[order], s_fit_splines['by'])):
            values_sorted = values.ravel()[order_sort]
            values_unique = np.add.reduceat(
                values_sorted, s_unique_start) / s_unique_count
            destination[order] = fit_slice_boundary_spline(
                s_unique=s_unique,
                values_unique=values_unique,
                s_axis=s_axis,
                s_derivative_spline_order=s_derivative_spline_order)

    bs = s_fit_splines['bs'](s_axis)
    bx = {
        order: s_fit_splines['bx'][order](s_axis)
        for order in range(max_transverse_derivative_order + 1)
    }
    by = {
        order: s_fit_splines['by'][order](s_axis)
        for order in range(max_transverse_derivative_order + 1)
    }
    bx_y_pure[0] = np.array(bx[0], copy=True)
    by_y_pure[0] = np.array(by[0], copy=True)
    bs_x_pure[0] = np.array(bs, copy=True)
    bs_y_pure[0] = np.array(bs, copy=True)

    bs[0] = 0.0
    bs[-1] = 0.0
    for order in range(max_transverse_derivative_order + 1):
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
    for order in range(max_transverse_derivative_order + 1):
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
    for order in range(max_transverse_derivative_order + 1):
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
        for order in range(max_transverse_derivative_order + 1)
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
        for order in range(max_transverse_derivative_order + 1)
    }

    u_integral = np.linspace(-1.0, 1.0, spline_integral_points)
    bx_spline_data = {}
    by_spline_data = {}
    for order in range(max_transverse_derivative_order + 1):
        if decrease_s_poly_order_with_transverse_order:
            degree = max(0, 4 - order)
        else:
            degree = 4
        bx_spline_data[order] = {
            'degree': degree,
            'val_start': np.empty(n_intervals),
            'val_end': np.empty(n_intervals),
            'der_start': np.empty(n_intervals),
            'der_end': np.empty(n_intervals),
            'integral_average': np.empty(n_intervals),
        }
        by_spline_data[order] = {
            'degree': degree,
            'val_start': np.empty(n_intervals),
            'val_end': np.empty(n_intervals),
            'der_start': np.empty(n_intervals),
            'der_end': np.empty(n_intervals),
            'integral_average': np.empty(n_intervals),
        }

        for ii in range(n_intervals):
            for values_by_order, spline_data in (
                    (bx_integral_values, bx_spline_data),
                    (by_integral_values, by_spline_data)):
                values = values_by_order[order][ii]
                target_mean = (
                    np.trapezoid(values, s_integral[ii]) / length[ii])

                coefficients = np.polyfit(u_integral, values, degree)
                integral_coefficients = np.polyint(coefficients)
                fitted_mean = 0.5 * (
                    np.polyval(integral_coefficients, 1.0)
                    - np.polyval(integral_coefficients, -1.0)
                )
                coefficients[-1] += target_mean - fitted_mean

                derivative_coefficients = np.polyder(coefficients)
                spline_data[order]['val_start'][ii] = np.polyval(
                    coefficients, -1.0)
                spline_data[order]['val_end'][ii] = np.polyval(
                    coefficients, 1.0)
                spline_data[order]['der_start'][ii] = (
                    np.polyval(derivative_coefficients, -1.0)
                    * 2.0 / length[ii])
                spline_data[order]['der_end'][ii] = (
                    np.polyval(derivative_coefficients, 1.0)
                    * 2.0 / length[ii])
                spline_data[order]['integral_average'][ii] = target_mean

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
        'bx_spline_data': bx_spline_data,
        'by_spline_data': by_spline_data,
        'bs_integral_average': bs_integral_average,
        'bx_integral_average': bx_integral_average,
        'by_integral_average': by_integral_average,
    }


def build_splineboris_line(
        *, name, field_data, scale_b,
        max_transverse_derivative_order_for_spline,
        spline_steps_per_point,
        use_near_axis_simplified_model):
    s_axis = field_data['s_axis']
    elements = []
    element_names = []
    name_width = len(str(len(s_axis) - 2))

    for ii in range(len(s_axis) - 1):
        length = s_axis[ii + 1] - s_axis[ii]

        if use_near_axis_simplified_model:
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
            mean=bs_integral_average,
        )

        bx = []
        by = []
        for order in range(max_transverse_derivative_order_for_spline + 1):
            if use_near_axis_simplified_model and order == 0:
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

            elif use_near_axis_simplified_model and order == 1:
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

            elif use_near_axis_simplified_model and order > 1:
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
                bx_spline_data = field_data['bx_spline_data'][order]
                bx_val_start = bx_spline_data['val_start'][ii]
                bx_val_end = bx_spline_data['val_end'][ii]
                bx_der_start = bx_spline_data['der_start'][ii]
                bx_der_end = bx_spline_data['der_end'][ii]
                bx_integral_average = (
                    bx_spline_data['integral_average'][ii])

                by_spline_data = field_data['by_spline_data'][order]
                by_val_start = by_spline_data['val_start'][ii]
                by_val_end = by_spline_data['val_end'][ii]
                by_der_start = by_spline_data['der_start'][ii]
                by_der_end = by_spline_data['der_end'][ii]
                by_integral_average = (
                    by_spline_data['integral_average'][ii])

            bx.append(xt.Spline4(
                val_start=bx_val_start,
                der_start=bx_der_start,
                val_end=bx_val_end,
                der_end=bx_der_end,
                mean=bx_integral_average,
            ))
            by.append(xt.Spline4(
                val_start=by_val_start,
                der_start=by_der_start,
                val_end=by_val_end,
                der_end=by_der_end,
                mean=by_integral_average,
            ))

        elements.append(xt.SplineBoris(
            bs=bs,
            bx=tuple(bx),
            by=tuple(by),
            length=length,
            n_steps=spline_steps_per_point,
            scale_b=scale_b,
        ))
        element_names.append(f'{name}_{ii:0{name_width}d}')

    return xt.Line(elements=elements, element_names=element_names)


def build_variable_solenoid_line(*, name, field_data, scale_b, rigidity0):
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


def assemble_three_solenoid_system(
        *, line_comp_left, line_main, line_comp_right,
        drift_between_comp_and_main):
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


def symplectic_error(*, line, particle_ref):
    s_matrix = xt.linear_normal_form.S
    r_matrix = line.get_R_matrix(
        particle_on_co=particle_ref.copy())['R_matrix']
    return np.linalg.norm(r_matrix.T @ s_matrix @ r_matrix - s_matrix, ord=2)


def sample_splineboris_line(
        *, line, s0, spline_steps_per_point, x=0.0, y=0.0):
    s_out = []
    bx_out = []
    by_out = []
    bs_out = []

    s_start = s0
    for element in line.elements:
        s_local = np.linspace(0.0, element.length, spline_steps_per_point + 1)
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


def sample_splineboris_line_on_s(*, line, s_axis, s_eval, x=0.0, y=0.0):
    bx_out = np.empty_like(s_eval)
    by_out = np.empty_like(s_eval)
    bs_out = np.empty_like(s_eval)

    for ii, element in enumerate(line.elements):
        if ii == len(line.elements) - 1:
            mask = (s_eval >= s_axis[ii]) & (s_eval <= s_axis[ii + 1])
        else:
            mask = (s_eval >= s_axis[ii]) & (s_eval < s_axis[ii + 1])
        if not np.any(mask):
            continue

        s_local = s_eval[mask] - s_axis[ii]
        bx_out[mask], by_out[mask], bs_out[mask] = element.get_field(
            np.full_like(s_local, x),
            np.full_like(s_local, y),
            s_local,
        )

    return bx_out, by_out, bs_out
