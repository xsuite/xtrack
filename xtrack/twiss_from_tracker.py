import numpy as np

import xobjects as xo
import xpart as xp

from scipy.optimize import fsolve

DEFAULT_STEPS_R_MATRIX = {
    'dx':1e-7, 'dpx':1e-10,
    'dy':1e-7, 'dpy':1e-10,
    'dzeta':1e-6, 'ddelta':1e-7
}

def find_closed_orbit(tracker, particle_co_guess=None, particle_ref=None,
                      co_search_settings=None):

    if particle_co_guess is None:
        particle_co_guess = particle_ref.copy()
        particle_co_guess.x = 0
        particle_co_guess.px = 0
        particle_co_guess.y = 0
        particle_co_guess.py = 0
        particle_co_guess.zeta = 0
        particle_co_guess.delta = 0
    else:
        assert particle_ref is None
        particle_ref = particle_co_guess

    if co_search_settings is None:
        co_search_settings = {}

    particle_co_guess = particle_co_guess.copy(
                        _context=tracker._buffer.context)

    res = fsolve(lambda p: p - _one_turn_map(p, particle_co_guess, tracker),
          x0=np.array([particle_co_guess._xobject.x[0],
                       particle_co_guess._xobject.px[0],
                       particle_co_guess._xobject.y[0],
                       particle_co_guess._xobject.py[0],
                       particle_co_guess._xobject.zeta[0],
                       particle_co_guess._xobject.delta[0]]),
          **co_search_settings)

    particle_on_co = particle_co_guess.copy()
    particle_on_co.x = res[0]
    particle_on_co.px = res[1]
    particle_on_co.y = res[2]
    particle_on_co.py = res[3]
    particle_on_co.zeta = res[4]
    particle_on_co.delta = res[5]

    return particle_on_co

def compute_one_turn_matrix_finite_differences(
        tracker, particle_on_co,
        steps_r_matrix=None):

    if steps_r_matrix is not None:
        steps_in = steps_r_matrix.copy()
        for nn in steps_in.keys():
            assert nn in DEFAULT_STEPS_R_MATRIX.keys(), (
                '`steps_r_matrix` can contain only ' +
                ' '.join(DEFAULT_STEPS_R_MATRIX.keys())
            )
        steps_r_matrix = DEFAULT_STEPS_R_MATRIX.copy()
        steps_r_matrix.update(steps_in)
    else:
        steps_r_matrix = DEFAULT_STEPS_R_MATRIX.copy()


    context = tracker._buffer.context

    particle_on_co = particle_on_co.copy(
                        _context=context)

    dx = steps_r_matrix["dx"]
    dpx = steps_r_matrix["dpx"]
    dy = steps_r_matrix["dy"]
    dpy = steps_r_matrix["dpy"]
    dzeta = steps_r_matrix["dzeta"]
    ddelta = steps_r_matrix["ddelta"]
    part_temp = xp.build_particles(_context=context,
            particle_ref=particle_on_co, mode='shift',
            x  =    [dx,  0., 0.,  0.,    0.,     0., -dx,   0.,  0.,   0.,     0.,      0.],
            px =    [0., dpx, 0.,  0.,    0.,     0.,  0., -dpx,  0.,   0.,     0.,      0.],
            y  =    [0.,  0., dy,  0.,    0.,     0.,  0.,   0., -dy,   0.,     0.,      0.],
            py =    [0.,  0., 0., dpy,    0.,     0.,  0.,   0.,  0., -dpy,     0.,      0.],
            zeta =  [0.,  0., 0.,  0., dzeta,     0.,  0.,   0.,  0.,   0., -dzeta,      0.],
            delta = [0.,  0., 0.,  0.,    0., ddelta,  0.,   0.,  0.,   0.,     0., -ddelta],)

    tracker.track(part_temp)

    temp_mat = np.zeros(shape=(6, 12), dtype=np.float64)
    temp_mat[0, :] = context.nparray_from_context_array(part_temp.x)
    temp_mat[1, :] = context.nparray_from_context_array(part_temp.px)
    temp_mat[2, :] = context.nparray_from_context_array(part_temp.y)
    temp_mat[3, :] = context.nparray_from_context_array(part_temp.py)
    temp_mat[4, :] = context.nparray_from_context_array(part_temp.zeta)
    temp_mat[5, :] = context.nparray_from_context_array(part_temp.delta)

    RR = np.zeros(shape=(6, 6), dtype=np.float64)

    for jj, dd in enumerate([dx, dpx, dy, dpy, dzeta, ddelta]):
        RR[:, jj] = (temp_mat[:, jj] - temp_mat[:, jj+6])/(2*dd)

    return RR

def twiss_from_tracker(tracker, particle_ref, r_sigma=0.01,
        nemitt_x=1e-6, nemitt_y=2.5e-6,
        n_theta=1000, delta_disp=1e-5, delta_chrom = 1e-4,
        particle_co_guess=None, steps_r_matrix=None,
        co_search_settings=None, at_elements=None):


    context = tracker._buffer.context

    part_on_co = tracker.find_closed_orbit(particle_co_guess=particle_co_guess,
                                        particle_ref=particle_ref,
                                        co_search_settings=co_search_settings)
    RR = tracker.compute_one_turn_matrix_finite_differences(
                                                steps_r_matrix=steps_r_matrix,
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

    enames = tracker.line.element_names
    if at_elements is not None:
        indx_twiss = []
        for nn in at_elements:
            if isinstance(nn, (int, np.integer)):
                indx_twiss.append(int(nn))
            else:
                assert nn in tracker.line.element_names
                indx_twiss.append(enames.index(nn))
        indx_twiss = sorted(indx_twiss)
    else:
        indx_twiss = list(range(len(enames)))

    n_twiss = len(indx_twiss)

    max_x = np.zeros(n_twiss, dtype=np.float64)
    max_y = np.zeros(n_twiss, dtype=np.float64)
    min_x = np.zeros(n_twiss, dtype=np.float64)
    min_y = np.zeros(n_twiss, dtype=np.float64)
    max_px = np.zeros(n_twiss, dtype=np.float64)
    max_py = np.zeros(n_twiss, dtype=np.float64)
    min_px = np.zeros(n_twiss, dtype=np.float64)
    min_py = np.zeros(n_twiss, dtype=np.float64)
    sign_alfx = np.zeros(n_twiss, dtype=np.float64)
    sign_alfy = np.zeros(n_twiss, dtype=np.float64)
    x_co = np.zeros(n_twiss, dtype=np.float64)
    y_co = np.zeros(n_twiss, dtype=np.float64)
    px_co = np.zeros(n_twiss, dtype=np.float64)
    py_co = np.zeros(n_twiss, dtype=np.float64)
    x_disp_plus = np.zeros(n_twiss, dtype=np.float64)
    x_disp_minus = np.zeros(n_twiss, dtype=np.float64)
    y_disp_plus = np.zeros(n_twiss, dtype=np.float64)
    y_disp_minus = np.zeros(n_twiss, dtype=np.float64)
    px_disp_plus = np.zeros(n_twiss, dtype=np.float64)
    px_disp_minus = np.zeros(n_twiss, dtype=np.float64)
    py_disp_plus = np.zeros(n_twiss, dtype=np.float64)
    py_disp_minus = np.zeros(n_twiss, dtype=np.float64)

    ctx2np = context.nparray_from_context_array

    tracker.track(part_on_co, ele_start=0, num_elements=indx_twiss[0])
    tracker.track(part_x, ele_start=0, num_elements=indx_twiss[0])
    tracker.track(part_y, ele_start=0, num_elements=indx_twiss[0])
    tracker.track(part_disp, ele_start=0, num_elements=indx_twiss[0])

    for ii, indx in enumerate(indx_twiss):

        print(f'{ii}/{len(indx_twiss)}        ',
              end='\r', flush=True)
        max_x[ii] = np.max(ctx2np(part_x.x))
        max_y[ii] = np.max(ctx2np(part_y.y))

        min_x[ii] = np.min(ctx2np(part_x.x))
        min_y[ii] = np.min(ctx2np(part_y.y))

        max_px[ii] = np.max(ctx2np(part_x.px))
        max_py[ii] = np.max(ctx2np(part_y.py))

        min_px[ii] = np.min(ctx2np(part_x.px))
        min_py[ii] = np.min(ctx2np(part_y.py))

        x_co[ii] = part_on_co._xobject.x[0]
        y_co[ii] = part_on_co._xobject.y[0]

        px_co[ii] = part_on_co._xobject.px[0]
        py_co[ii] = part_on_co._xobject.py[0]

        sign_alfx[ii] = -np.sign(np.sum(ctx2np(
            (part_x.x - x_co[ii]) * (part_x.px - px_co[ii]))))
        sign_alfy[ii] = -np.sign(np.sum(ctx2np(
            (part_y.y - y_co[ii]) * (part_y.py - py_co[ii]))))

        x_disp_plus[ii] = part_disp._xobject.x[0]
        x_disp_minus[ii] = part_disp._xobject.x[1]
        y_disp_plus[ii] = part_disp._xobject.y[0]
        y_disp_minus[ii] = part_disp._xobject.y[1]

        px_disp_plus[ii] = part_disp._xobject.px[0]
        px_disp_minus[ii] = part_disp._xobject.px[1]
        py_disp_plus[ii] = part_disp._xobject.py[0]
        py_disp_minus[ii] = part_disp._xobject.py[1]

        if ii == len(indx_twiss)-1:
            n_next_track = len(tracker.line.elements) - indx
        else:
            n_next_track = indx_twiss[ii+1] - indx
        tracker.track(part_on_co, ele_start=indx, num_elements=n_next_track)
        tracker.track(part_x, ele_start=indx, num_elements=n_next_track)
        tracker.track(part_y, ele_start=indx, num_elements=n_next_track)
        tracker.track(part_disp, ele_start=indx, num_elements=n_next_track)

    eta = -((part_disp._xobject.zeta[0] - part_disp._xobject.zeta[1])
             /(2*delta_disp)/tracker.line.get_length())
    alpha = eta + 1/particle_ref.gamma0[0]**2

    s = np.array(tracker.line.get_s_elements())[indx_twiss]

    sigx_max = (max_x - x_co)/r_sigma
    sigy_max = (max_y - y_co)/r_sigma
    sigx_min = (x_co - min_x)/r_sigma
    sigy_min = (y_co - min_y)/r_sigma
    sigx = (sigx_max + sigx_min)/2
    sigy = (sigy_max + sigy_min)/2

    sigpx_max = (max_px - px_co)/r_sigma
    sigpy_max = (max_py - py_co)/r_sigma
    sigpx_min = (px_co - min_px)/r_sigma
    sigpy_min = (py_co - min_py)/r_sigma
    sigpx = (sigpx_max + sigpx_min)/2
    sigpy = (sigpy_max + sigpy_min)/2

    betx = (sigx**2*particle_ref._xobject.gamma0[0]
            * particle_ref._xobject.beta0[0]/nemitt_x)
    bety = (sigy**2*particle_ref._xobject.gamma0[0]
            * particle_ref._xobject.beta0[0]/nemitt_y)

    gamx = (sigpx**2*particle_ref._xobject.gamma0[0]
            * particle_ref._xobject.beta0[0]/nemitt_x)
    gamy = (sigpy**2*particle_ref._xobject.gamma0[0]
            * particle_ref._xobject.beta0[0]/nemitt_y)

    mask_alfx_zero = np.abs(betx*gamx - 1) < 1e-4
    mask_alfx_neg = (betx*gamx - 1) < 0
    assert np.all(np.abs(betx*gamx - 1)[mask_alfx_neg] < 1e-2) # value is sufficiently small
    mask_alfx_zero[mask_alfx_neg] = True
    alfx = 0*betx
    alfx[~mask_alfx_zero] = np.sqrt(
            betx[~mask_alfx_zero]*gamx[~mask_alfx_zero] - 1)
    alfx*=sign_alfx

    mask_alfy_zero = np.abs(bety*gamy - 1) < 1e-4
    mask_alfy_neg = (bety*gamy - 1) < 0
    assert np.all(np.abs(bety*gamy - 1)[mask_alfy_neg] < 1e-2) # value is sufficiently small
    mask_alfy_zero[mask_alfy_neg] = True
    alfy = 0*bety
    alfy[~mask_alfy_zero] = np.sqrt(
            bety[~mask_alfy_zero]*gamy[~mask_alfy_zero] - 1)
    alfy*=sign_alfy

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
                                         particle_on_co=part_chrom_plus.copy(),
                                         steps_r_matrix=steps_r_matrix)
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
                                        particle_on_co=part_chrom_minus.copy(),
                                        steps_r_matrix=steps_r_matrix)
    (WW_chrom_minus, WWinv_chrom_minus, Rot_chrom_minus
        ) = xp.compute_linear_normal_form(RR_chrom_minus)
    qx_chrom_minus = np.angle(np.linalg.eig(Rot_chrom_minus)[0][0])/(2*np.pi)
    qy_chrom_minus = np.angle(np.linalg.eig(Rot_chrom_minus)[0][2])/(2*np.pi)

    dqx = (qx_chrom_plus - qx_chrom_minus)/delta_chrom/2
    dqy = (qy_chrom_plus - qy_chrom_minus)/delta_chrom/2

    twiss_res = {
        'name': [tracker.line.element_names[indx] for indx in indx_twiss],
        's': s,
        'x': x_co,
        'px': px_co,
        'y': y_co,
        'py': py_co,
        'betx': betx,
        'bety': bety,
        'alfx': alfx,
        'alfy': alfy,
        'gamx': gamx,
        'gamy': gamy,
        'sigx': sigx,
        'sigy': sigy,
        'dx': dx,
        'dpx': dpx,
        'dy': dy,
        'dpy': dpy,
        'qx': qx,
        'qy': qy,
        'dqx': dqx,
        'dqy': dqy,
        'slip_factor': eta,
        'momentum_compaction_factor': alpha,
        'R_matrix': RR,
        'particle_on_co':part_on_co.copy(_context=xo.context_default)
        }

    return twiss_res

def _one_turn_map(p, particle_ref, tracker):
    part = particle_ref.copy()
    part.x = p[0]
    part.px = p[1]
    part.y = p[2]
    part.py = p[3]
    part.zeta = p[4]
    part.delta = p[5]

    tracker.track(part)
    p_res = np.array([
           part._xobject.x[0],
           part._xobject.px[0],
           part._xobject.y[0],
           part._xobject.py[0],
           part._xobject.zeta[0],
           part._xobject.delta[0]])
    return p_res

