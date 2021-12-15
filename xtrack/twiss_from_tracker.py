import numpy as np

import xobjects as xo
import xpart as xp


def twiss_from_tracker(tracker, particle_ref, r_sigma=0.01,
        nemitt_x=1e-6, nemitt_y=2.5e-6,
        n_theta=1000, delta_disp=1e-5, delta_chrom = 1e-4):

    context = tracker._buffer.context

    part_on_co = tracker.find_closed_orbit(particle_ref)
    RR = tracker.compute_one_turn_matrix_finite_differences(
                                                     particle_on_co=part_on_co)
    W, Winv, Rot = xp.compute_linear_normal_form(RR)
    part_x = xp.build_particles(
                _context=context,
                x_norm=r_sigma*np.cos(np.linspace(0, 2*np.pi, n_theta)),
                px_norm=r_sigma*np.sin(np.linspace(0, 2*np.pi, n_theta)),
                zeta=part_on_co.zeta[0], delta=part_on_co.delta[0],
                particle_on_co=part_on_co,
                scale_with_transverse_norm_emitt=(nemitt_x, nemitt_y),
                R_matrix=RR)
    part_y = xp.build_particles(
                _context=context,
                y_norm=r_sigma*np.cos(np.linspace(0, 2*np.pi, n_theta)),
                py_norm=r_sigma*np.sin(np.linspace(0, 2*np.pi, n_theta)),
                zeta=part_on_co.zeta[0], delta=part_on_co.delta[0],
                particle_on_co=part_on_co,
                scale_with_transverse_norm_emitt=(nemitt_x, nemitt_y),
                R_matrix=RR)
    part_disp = xp.build_particles(
                _context=context,
                x_norm=0,
                zeta=part_on_co.zeta[0], delta=[delta_disp, -delta_disp],
                particle_on_co=part_on_co,
                scale_with_transverse_norm_emitt=(nemitt_x, nemitt_y),
                R_matrix=RR)


    num_elements = len(tracker.line.elements)
    max_x = np.zeros(num_elements, dtype=np.float64)
    max_y = np.zeros(num_elements, dtype=np.float64)
    min_x = np.zeros(num_elements, dtype=np.float64)
    min_y = np.zeros(num_elements, dtype=np.float64)
    x_co = np.zeros(num_elements, dtype=np.float64)
    y_co = np.zeros(num_elements, dtype=np.float64)
    px_co = np.zeros(num_elements, dtype=np.float64)
    py_co = np.zeros(num_elements, dtype=np.float64)
    x_disp_plus = np.zeros(num_elements, dtype=np.float64)
    x_disp_minus = np.zeros(num_elements, dtype=np.float64)
    y_disp_plus = np.zeros(num_elements, dtype=np.float64)
    y_disp_minus = np.zeros(num_elements, dtype=np.float64)
    px_disp_plus = np.zeros(num_elements, dtype=np.float64)
    px_disp_minus = np.zeros(num_elements, dtype=np.float64)
    py_disp_plus = np.zeros(num_elements, dtype=np.float64)
    py_disp_minus = np.zeros(num_elements, dtype=np.float64)

    ctx2np = context.nparray_from_context_array
    for ii, ee in enumerate(tracker.line.elements):
        print(f'{ii}/{len(tracker.line.elements)}        ',
              end='\r', flush=True)
        max_x[ii] = np.max(ctx2np(part_x.x))
        max_y[ii] = np.max(ctx2np(part_y.y))

        min_x[ii] = np.min(ctx2np(part_x.x))
        min_y[ii] = np.min(ctx2np(part_y.y))

        x_co[ii] = part_on_co._xobject.x[0]
        y_co[ii] = part_on_co._xobject.y[0]

        px_co[ii] = part_on_co._xobject.px[0]
        py_co[ii] = part_on_co._xobject.py[0]

        x_disp_plus[ii] = part_disp._xobject.x[0]
        x_disp_minus[ii] = part_disp._xobject.x[1]
        y_disp_plus[ii] = part_disp._xobject.y[0]
        y_disp_minus[ii] = part_disp._xobject.y[1]

        px_disp_plus[ii] = part_disp._xobject.px[0]
        px_disp_minus[ii] = part_disp._xobject.px[1]
        py_disp_plus[ii] = part_disp._xobject.py[0]
        py_disp_minus[ii] = part_disp._xobject.py[1]

        tracker.track(part_on_co, ele_start=ii, num_elements=1)
        tracker.track(part_x, ele_start=ii, num_elements=1)
        tracker.track(part_y, ele_start=ii, num_elements=1)
        tracker.track(part_disp, ele_start=ii, num_elements=1)

    s = np.array(tracker.line.get_s_elements())

    sigx_max = (max_x - x_co)/r_sigma
    sigy_max = (max_y - y_co)/r_sigma
    sigx_min = (x_co - min_x)/r_sigma
    sigy_min = (y_co - min_y)/r_sigma
    sigx = (sigx_max + sigx_min)/2
    sigy = (sigy_max + sigy_min)/2

    betx = (sigx**2*particle_ref._xobject.gamma0[0]
            * particle_ref._xobject.beta0[0]/nemitt_x)
    bety = (sigy**2*particle_ref._xobject.gamma0[0]
            * particle_ref._xobject.beta0[0]/nemitt_y)

    dx = (x_disp_plus-x_disp_minus)/delta_disp/2
    dy = (y_disp_plus-y_disp_minus)/delta_disp/2
    dpx = (px_disp_plus-px_disp_minus)/delta_disp/2
    dpy = (py_disp_plus-py_disp_minus)/delta_disp/2

    qx = np.angle(np.linalg.eig(Rot)[0][0])/(2*np.pi)
    qy = np.angle(np.linalg.eig(Rot)[0][2])/(2*np.pi)

    part_chrom_plus = xp.build_particles(
                _context=context,
                x_norm=0,
                zeta=part_on_co.zeta[0], delta=delta_chrom,
                particle_on_co=part_on_co,
                scale_with_transverse_norm_emitt=(nemitt_x, nemitt_y),
                R_matrix=RR)
    RR_chrom_plus = tracker.compute_one_turn_matrix_finite_differences(
                                         particle_on_co=part_chrom_plus.copy())
    (WW_chrom_plus, WWinv_chrom_plus, Rot_chrom_plus
        ) = xp.compute_linear_normal_form(RR_chrom_plus)
    qx_chrom_plus = np.angle(np.linalg.eig(Rot_chrom_plus)[0][0])/(2*np.pi)
    qy_chrom_plus = np.angle(np.linalg.eig(Rot_chrom_plus)[0][2])/(2*np.pi)

    part_chrom_minus = xp.build_particles(
                _context=context,
                x_norm=0,
                zeta=part_on_co.zeta[0], delta=-delta_chrom,
                particle_on_co=part_on_co,
                scale_with_transverse_norm_emitt=(nemitt_x, nemitt_y),
                R_matrix=RR)
    RR_chrom_minus = tracker.compute_one_turn_matrix_finite_differences(
                                        particle_on_co=part_chrom_minus.copy())
    (WW_chrom_minus, WWinv_chrom_minus, Rot_chrom_minus
        ) = xp.compute_linear_normal_form(RR_chrom_minus)
    qx_chrom_minus = np.angle(np.linalg.eig(Rot_chrom_minus)[0][0])/(2*np.pi)
    qy_chrom_minus = np.angle(np.linalg.eig(Rot_chrom_minus)[0][2])/(2*np.pi)

    dqx = (qx_chrom_plus - qx_chrom_minus)/delta_chrom/2
    dqy = (qy_chrom_plus - qy_chrom_minus)/delta_chrom/2

    twiss_res = {
        's': s,
        'x': x_co,
        'px': px_co,
        'y': y_co,
        'py': py_co,
        'betx': betx,
        'bety': bety,
        'sigx': sigx,
        'sigy': sigy,
        'dx': dx,
        'dpx': dpx,
        'dy': dy,
        'dpy': dpy,
        'qx': qx,
        'qy': qy,
        'dqx': dqx,
        'dqy': dqy}

    return twiss_res

