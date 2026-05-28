
# Based on E. Boscolo, A. Ciarma, E. Burkhardt, https://cds.cern.ch/record/2948247
# Nuclear Instruments and Methods in Physics Research A 1083 (2026) 171135

import xtrack as xt

from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField
import numpy as np
from tilted_solenoid import TiltedSolenoid

env = xt.load('fccee_z_lcc.json')
line = env.fccee_p_ring

ip_names = ['ipa', 'ipd', 'ipg', 'ipj']

# Tilt with respect to the beam axis
theta = -0.015

SPLINE_MULTIPOLE_ORDER = 2
SPLINE_DERIVATIVE_STEP = 1e-5
SPLINE_STEPS_PER_POINT = 1
SPLINE_INTEGRAL_POINTS = 10


def compute_field_derivative(field_model, s_axis, component, derivative_order):
    if derivative_order == 0:
        zero = np.zeros_like(s_axis)
        return field_model.get_field(zero, zero, s_axis)[component]

    component_name = ('x', 'y', 'z')[component]
    return field_model.compute_pure_field_derivatives(
        s=s_axis,
        direction='x',
        step=SPLINE_DERIVATIVE_STEP,
        component=component_name,
        max_order=derivative_order,
        min_order=derivative_order,
    )[derivative_order]


def build_splineboris_line(
        env, name_prefix, field_model, s_axis, scale_b, corrector_specs=None):
    bs_values = compute_field_derivative(field_model, s_axis, component=2, derivative_order=0)
    bx_values = [
        compute_field_derivative(field_model, s_axis, component=0, derivative_order=order)
        for order in range(SPLINE_MULTIPOLE_ORDER)
    ]
    by_values = [
        compute_field_derivative(field_model, s_axis, component=1, derivative_order=order)
        for order in range(SPLINE_MULTIPOLE_ORDER)
    ]

    bs_s_derivative = np.gradient(bs_values, s_axis, edge_order=2)
    bx_s_derivatives = [
        np.gradient(values, s_axis, edge_order=2) for values in bx_values]
    by_s_derivatives = [
        np.gradient(values, s_axis, edge_order=2) for values in by_values]

    elements = []
    s_starts = []
    s_ends = []
    for ii in range(len(s_axis) - 1):
        length = s_axis[ii + 1] - s_axis[ii]
        s_integral = np.linspace(
            s_axis[ii], s_axis[ii + 1], SPLINE_INTEGRAL_POINTS)

        bs_integral_values = compute_field_derivative(
            field_model, s_integral, component=2, derivative_order=0)
        bs = xt.Spline4(
            val_start=bs_values[ii],
            der_start=bs_s_derivative[ii],
            val_end=bs_values[ii + 1],
            der_end=bs_s_derivative[ii + 1],
            integral=np.trapezoid(bs_integral_values, s_integral) / length,
        )

        bx = []
        by = []
        for order in range(SPLINE_MULTIPOLE_ORDER):
            bx_integral_values = compute_field_derivative(
                field_model, s_integral, component=0, derivative_order=order)
            by_integral_values = compute_field_derivative(
                field_model, s_integral, component=1, derivative_order=order)
            bx.append(xt.Spline4(
                val_start=bx_values[order][ii],
                der_start=bx_s_derivatives[order][ii],
                val_end=bx_values[order][ii + 1],
                der_end=bx_s_derivatives[order][ii + 1],
                integral=(
                    np.trapezoid(bx_integral_values, s_integral) / length),
            ))
            by.append(xt.Spline4(
                val_start=by_values[order][ii],
                der_start=by_s_derivatives[order][ii],
                val_end=by_values[order][ii + 1],
                der_end=by_s_derivatives[order][ii + 1],
                integral=(
                    np.trapezoid(by_integral_values, s_integral) / length),
            ))

        elements.append(xt.SplineBoris(
            bs=bs,
            bx=tuple(bx),
            by=tuple(by),
            length=length,
            n_steps=SPLINE_STEPS_PER_POINT,
        ))
        s_starts.append(float(s_axis[ii]))
        s_ends.append(float(s_axis[ii + 1]))

    corrector_specs = corrector_specs or []
    correctors_by_element = {}
    for side, ds_start, ds_end, ip_name in corrector_specs:
        knob_h = f'acbh1_sol_{side}_{ip_name}'
        knob_v = f'acbv1_sol_{side}_{ip_name}'
        env[knob_h] = 0
        env[knob_v] = 0

        s_min = min(ds_start, ds_end)
        s_max = max(ds_start, ds_end)
        selected = [
            ii for ii, (s_start, s_end) in enumerate(
                zip(s_starts, s_ends))
            if s_min <= 0.5 * (s_start + s_end) <= s_max
        ]
        if not selected:
            raise ValueError(
                f'No SplineBoris pieces selected for {side} corrector '
                f'in range [{ds_start}, {ds_end}]')

        l_tot = sum(elements[ii].length for ii in selected)
        for ii in selected:
            correctors_by_element[ii] = (side, knob_h, knob_v, l_tot)

    name_width = len(str(max(0, len(elements) - 1)))
    ele_names = []
    for ii, elem in enumerate(elements):
        name = f'{name_prefix}_{ii:0{name_width}d}'
        env.elements[name] = elem
        env.ref[name].scale_b = scale_b

        if ii in correctors_by_element:
            side, knob_h, knob_v, l_tot = correctors_by_element[ii]
            half_knl = 0.5 * env.ref[knob_h] / l_tot * elem.length
            half_ksl = 0.5 * env.ref[knob_v] / l_tot * elem.length

            corr_entry = f'{name_prefix}_corr_{side}_{ii:0{name_width}d}_entry'
            corr_exit = f'{name_prefix}_corr_{side}_{ii:0{name_width}d}_exit'
            env.new(corr_entry, xt.Multipole, knl=[half_knl], ksl=[half_ksl])
            env.new(corr_exit, xt.Multipole, knl=[half_knl], ksl=[half_ksl])
            ele_names.append(corr_entry)

        ele_names.append(name)

        if ii in correctors_by_element:
            ele_names.append(corr_exit)

    return env.new_line(components=ele_names)


def clone_splineboris_line(env, line_in, suffix, scale_b):
    ele_names = []
    for nn in line_in.element_names:
        name = f'{nn}{suffix}'
        env.elements[name] = env.get(nn).copy()
        env.ref[name].scale_b = scale_b
        ele_names.append(name)

    return env.new_line(components=ele_names)


for ip_name in ip_names:

    line.cycle(f'end_ds_start_straight_{ip_name}')
    tt = line.get_table()

    print(f'IP {ip_name}:')

    # Analytic field map
    sf = TiltedSolenoid(L=1.23*2, a=0.13, B0=2., theta=theta)

    # s coordinate along the beam axis
    s = np.linspace(-2.399, 2.399, 201)

    # Compute field on the beam reference trajectory in the beam frame
    bx, by, bz = sf.get_field(0 * s, 0 * s, s)

    rigidity0 = line.particle_ref.rigidity0[0]

    # Build solenoid slices
    env[f'on_sol_{ip_name}'] = 1
    line_solenoid = build_splineboris_line(
        env=env,
        name_prefix=f'sol_slice_{ip_name}',
        field_model=sf,
        s_axis=s,
        scale_b=env.ref[f'on_sol_{ip_name}'],
        corrector_specs=[
            ('right', 1.23, 2.29, ip_name),
            ('left', -2.29, -1.23, ip_name),
        ],
    )

    # Measure integrated field of the main solenoid
    ksol_l_main_solenoid = np.trapezoid(bz, s) / rigidity0

    # Make compensation solenoid
    sfc = SolenoidField(L=1.5, a=0.03, B0=1., z0=0)
    s_comp = np.linspace(-1, 1., 51)
    _, _, bzc = sfc.get_field(0*s_comp, 0*s_comp, s_comp)
    env[f'on_comp_sol_{ip_name}'] = 1
    env[f'field_comp_sol_{ip_name}'] = 1.
    comp_scale_b = (
        env.ref[f'on_comp_sol_{ip_name}']
        * env.ref[f'field_comp_sol_{ip_name}']
    )
    line_comp_solenoid = build_splineboris_line(
        env=env,
        name_prefix=f'comp_sol_slice_{ip_name}',
        field_model=sfc,
        s_axis=s_comp,
        scale_b=comp_scale_b,
    )

    # Measure integrated field compensation solenoid
    ksol_l_comp_solenoid = np.trapezoid(bzc, s_comp) / rigidity0

    # Scale to have zero integrated field (main + compensation)
    env[f'field_comp_sol_{ip_name}'] = -ksol_l_main_solenoid / ksol_l_comp_solenoid / 2

    # Put the solenoids in the fcc lattice
    line_comp_solenoid_left = clone_splineboris_line(
        env, line_comp_solenoid, suffix=f'left_{ip_name}',
        scale_b=comp_scale_b)
    line_comp_solenoid_right = clone_splineboris_line(
        env, line_comp_solenoid, suffix=f'right_{ip_name}',
        scale_b=comp_scale_b)
    s_ip = tt['s', ip_name]
    line.remove(ip_name)
    line.insert([
        env.place(line_solenoid, anchor='center', at=s_ip),
        env.place(ip_name, at=s_ip), # Put back the ip
        env.place(line_comp_solenoid_left, anchor='end', at=-12, from_=ip_name),
        env.place(line_comp_solenoid_right, anchor='start', at=12, from_=ip_name)
    ], s_tol=1e-9)

    # Insert markers and dedicated correctors for sol compensation
    line.insert([
        env.new('dy_match_r_'+ip_name, xt.Marker, at=11.95, from_=ip_name),
        env.new('dy_match_l_'+ip_name, xt.Marker, at=-11.95, from_=ip_name),
        env.new(f'corr_sol_right_{ip_name}', xt.Multipole, length=1., isthick=False,
                anchor='end', at=0, from_=f'dy_match_r_{ip_name}@start'),
        env.new(f'corr_sol_left_{ip_name}', xt.Multipole, length=1., isthick=False,
                anchor='start', at=0, from_=f'dy_match_l_{ip_name}@end'),
    ])

env.to_json('temp_fcc_ee_lcc_non_local_boris_solenoid.json')
