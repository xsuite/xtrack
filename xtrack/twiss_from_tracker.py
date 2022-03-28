import logging
import numpy as np

import xobjects as xo
import xpart as xp

from scipy.optimize import fsolve
from scipy.constants import c as clight

from . import linear_normal_form as lnf

import xtrack as xt # To avoid circular imports

DEFAULT_STEPS_R_MATRIX = {
    'dx':1e-7, 'dpx':1e-10,
    'dy':1e-7, 'dpy':1e-10,
    'dzeta':1e-6, 'ddelta':1e-7
}

log = logging.getLogger(__name__)

class ClosedOrbitSearchError(Exception):
    pass

def find_closed_orbit(tracker, particle_co_guess=None, particle_ref=None,
                      co_search_settings=None, delta_zeta=0):

    if particle_co_guess is None:
        if particle_ref is None:
            if tracker.particle_ref is not None:
                particle_ref = tracker.particle_ref
            else:
                raise ValueError(
                    "Either `particle_co_guess` or `particle_ref` must be provided")

        particle_co_guess = particle_ref.copy()
        particle_co_guess.x = 0
        particle_co_guess.px = 0
        particle_co_guess.y = 0
        particle_co_guess.py = 0
        particle_co_guess.zeta = 0
        particle_co_guess.delta = 0
        particle_co_guess.s = 0
        particle_co_guess.at_element = 0
        particle_co_guess.at_turn = 0
    else:
        particle_ref = particle_co_guess

    if co_search_settings is None:
        co_search_settings = {}

    co_search_settings = co_search_settings.copy()
    if 'xtol' not in co_search_settings.keys():
        co_search_settings['xtol'] = 1e-6 # Relative error between calls

    particle_co_guess = particle_co_guess.copy(
                        _context=tracker._buffer.context)

    for shift_factor in [0, 1.]: # if not found at first attempt we shift slightly the starting point
        if shift_factor>0:
            log.warning('Need second attempt on closed orbit search')
        (res, infodict, ier, mesg
            ) = fsolve(lambda p: p - _one_turn_map(p, particle_co_guess, tracker, delta_zeta),
                x0=np.array([particle_co_guess._xobject.x[0] + shift_factor * 1e-5,
                            particle_co_guess._xobject.px[0] + shift_factor * 1e-7,
                            particle_co_guess._xobject.y[0] + shift_factor * 1e-5,
                            particle_co_guess._xobject.py[0] + shift_factor * 1e-7,
                            particle_co_guess._xobject.zeta[0] + shift_factor * 1e-4,
                            particle_co_guess._xobject.delta[0] + shift_factor * 1e-5]),
                full_output=True,
                **co_search_settings)
        fsolve_info = {
            'res': res, 'info': infodict, 'ier': ier, 'mesg': mesg}
        if ier == 1:
            break

    if ier != 1:
        raise ClosedOrbitSearchError

    particle_on_co = particle_co_guess.copy()
    particle_on_co.x = res[0]
    particle_on_co.px = res[1]
    particle_on_co.y = res[2]
    particle_on_co.py = res[3]
    particle_on_co.zeta = res[4]
    particle_on_co.delta = res[5]

    particle_on_co._fsolve_info = fsolve_info

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
            delta = [0.,  0., 0.,  0.,    0., ddelta,  0.,   0.,  0.,   0.,     0., -ddelta],
            )
    if particle_on_co._xobject.at_element[0]>0:
        part_temp.s[:] = particle_on_co._xobject.s[0]
        part_temp.at_element[:] = particle_on_co._xobject.at_element[0]

    if particle_on_co._xobject.at_element[0]>0:
        i_start = particle_on_co._xobject.at_element[0]
        tracker.track(part_temp, ele_start=i_start)
        tracker.track(part_temp, num_elements=i_start)
    else:
        assert particle_on_co._xobject.at_element[0] == 0
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

def _behaves_like_drift(ee):
    return (hasattr(ee, 'behaves_like_drift') and ee.behaves_like_drift)


def _build_auxiliary_tracker_with_extra_markers(tracker, at_s, marker_prefix,
                                                algorithm='auto'):

    assert algorithm in ['auto', 'insert', 'regen_all_drift']
    if algorithm == 'auto':
        if len(at_s)<10:
            algorithm = 'insert'
        else:
            algorithm = 'regen_all_drifts'

    auxline = xt.Line(elements=list(tracker.line.elements).copy(),
                      element_names=list(tracker.line.element_names).copy())

    names_inserted_markers = []
    markers = []
    for ii, ss in enumerate(at_s):
        nn = marker_prefix + f'{ii}'
        names_inserted_markers.append(nn)
        markers.append(xt.Drift(length=0))

    if algorithm == 'insert':
        for nn, mm, ss in zip(names_inserted_markers, markers, at_s):
            auxline.insert_element(element=mm, name=nn, at_s=ss)
    elif algorithm == 'regen_all_drifts':
        s_elems = auxline.get_s_elements()
        s_keep = []
        enames_keep = []
        for ss, nn in zip(s_elems, auxline.element_names):
            if not (_behaves_like_drift(auxline[nn]) and np.abs(auxline[nn].length)>0):
                s_keep.append(ss)
                enames_keep.append(nn)
                assert not xt.line._is_thick(auxline[nn]) or auxline[nn].length == 0

        s_keep.extend(list(at_s))
        enames_keep.extend(names_inserted_markers)

        ind_sorted = np.argsort(s_keep)
        s_keep = np.take(s_keep, ind_sorted)
        enames_keep = np.take(enames_keep, ind_sorted)

        i_new_drift = 0
        new_enames = []
        new_ele_dict = auxline.element_dict.copy()
        new_ele_dict.update({nn: ee for nn, ee in zip(names_inserted_markers, markers)})
        s_curr = 0
        for ss, nn in zip(s_keep, enames_keep):
            if ss > s_curr + 1e-6:
                new_drift = xt.Drift(length=ss-s_curr)
                new_dname = f'_auxrift_{i_new_drift}'
                new_ele_dict[new_dname] = new_drift
                new_enames.append(new_dname)
                i_new_drift += 1
                s_curr = ss
            new_enames.append(nn)
        auxline = xt.Line(elements=new_ele_dict, element_names=new_enames)

    auxtracker = xt.Tracker(
        _buffer=tracker._buffer,
        line=auxline,
        track_kernel=tracker.track_kernel,
        element_classes=tracker.element_classes,
        particles_class=tracker.particles_class,
        skip_end_turn_actions=tracker.skip_end_turn_actions,
        reset_s_at_end_turn=tracker.reset_s_at_end_turn,
        particles_monitor_class=None,
        global_xy_limit=tracker.global_xy_limit,
        local_particle_src=tracker.local_particle_src
    )

    return auxtracker, names_inserted_markers

def twiss_from_tracker(tracker, particle_ref, r_sigma=0.01,
        nemitt_x=1e-6, nemitt_y=2.5e-6,
        n_theta=1000, delta_disp=1e-5, delta_chrom = 1e-4,
        particle_co_guess=None, steps_r_matrix=None,
        co_search_settings=None, at_elements=None, at_s=None,
        eneloss_and_damping=False,
        matrix_responsiveness_tol=lnf.DEFAULT_MATRIX_RESPONSIVENESS_TOL,
        matrix_stability_tol=lnf.DEFAULT_MATRIX_STABILITY_TOL,
        symplectify=False):

    if at_s is not None:

        if np.isscalar(at_s):
            at_s = [at_s]

        assert at_elements is None
        (auxtracker, names_inserted_markers
            ) = _build_auxiliary_tracker_with_extra_markers(
            tracker=tracker, at_s=at_s, marker_prefix='inserted_twiss_marker')

        twres = twiss_from_tracker(
            tracker=auxtracker,
            particle_ref=particle_ref,
            r_sigma=r_sigma,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            n_theta=n_theta,
            delta_disp=delta_disp,
            delta_chrom=delta_chrom,
            particle_co_guess=particle_co_guess,
            steps_r_matrix=steps_r_matrix,
            co_search_settings=co_search_settings,
            at_elements=names_inserted_markers,
            at_s=None,
            eneloss_and_damping=eneloss_and_damping,
            symplectify=symplectify)
        return twres

    context = tracker._buffer.context

    part_on_co = tracker.find_closed_orbit(
                                particle_co_guess=particle_co_guess,
                                particle_ref=particle_ref,
                                co_search_settings=co_search_settings)

    RR = tracker.compute_one_turn_matrix_finite_differences(
                                                steps_r_matrix=steps_r_matrix,
                                                particle_on_co=part_on_co)

    gemitt_x = nemitt_x/part_on_co._xobject.beta0[0]/part_on_co._xobject.gamma0[0]
    gemitt_y = nemitt_y/part_on_co._xobject.beta0[0]/part_on_co._xobject.gamma0[0]

    W, Winv, Rot = lnf.compute_linear_normal_form(RR, symplectify=symplectify,
                                responsiveness_tol=matrix_responsiveness_tol,
                                stability_tol=matrix_stability_tol)

    s = np.array(tracker.line.get_s_elements())

    scale_transverse_x = np.sqrt(gemitt_x)*r_sigma
    scale_transverse_y = np.sqrt(gemitt_y)*r_sigma
    part_for_twiss = xp.build_particles(_context=context,
                        particle_ref=part_on_co, mode='shift',
                        x=  list(W[0, :4] * scale_transverse_x) + [0],
                        px= list(W[1, :4] * scale_transverse_x) + [0],
                        y=  list(W[2, :4] * scale_transverse_y) + [0],
                        py= list(W[3, :4] * scale_transverse_y) + [0],
                        zeta = 0,
                        delta = 0,
                        )

    part_disp = xp.build_particles(
        _context=context,
        x_norm=0,
        zeta=part_on_co._xobject.zeta[0],
        delta=np.array([-delta_disp, +delta_disp])+part_on_co._xobject.delta[0],
        particle_on_co=part_on_co,
        scale_with_transverse_norm_emitt=(nemitt_x, nemitt_y),
        R_matrix=RR,
        matrix_responsiveness_tol=matrix_responsiveness_tol,
        matrix_stability_tol=matrix_stability_tol,
        symplectify=symplectify)

    part_for_twiss = xp.Particles.merge([part_for_twiss, part_disp])

    tracker.track(part_for_twiss, turn_by_turn_monitor='ONE_TURN_EBE')

    x_co = tracker.record_last_track.x[4, :].copy()
    y_co = tracker.record_last_track.y[4, :].copy()
    px_co = tracker.record_last_track.px[4, :].copy()
    py_co = tracker.record_last_track.py[4, :].copy()
    zeta_co = tracker.record_last_track.zeta[4, :].copy()
    delta_co = tracker.record_last_track.delta[4, :].copy()
    ptau_co = tracker.record_last_track.ptau[4, :].copy()

    x_disp_minus = tracker.record_last_track.x[5, :].copy()
    y_disp_minus = tracker.record_last_track.y[5, :].copy()
    px_disp_minus = tracker.record_last_track.px[5, :].copy()
    py_disp_minus = tracker.record_last_track.py[5, :].copy()
    delta_disp_minus = tracker.record_last_track.delta[5, :].copy()

    x_disp_plus = tracker.record_last_track.x[6, :].copy()
    y_disp_plus = tracker.record_last_track.y[6, :].copy()
    px_disp_plus = tracker.record_last_track.px[6, :].copy()
    py_disp_plus = tracker.record_last_track.py[6, :].copy()
    delta_disp_plus = tracker.record_last_track.delta[6, :].copy()

    dx = (x_disp_plus-x_disp_minus)/(delta_disp_plus - delta_disp_minus)
    dy = (y_disp_plus-y_disp_minus)/(delta_disp_plus - delta_disp_minus)
    dpx = (px_disp_plus-px_disp_minus)/(delta_disp_plus - delta_disp_minus)
    dpy = (py_disp_plus-py_disp_minus)/(delta_disp_plus - delta_disp_minus)

    W4 = np.zeros(shape=(4,4,len(s)), dtype=np.float64)
    W4[0, :, :] = (tracker.record_last_track.x[:4, :] - x_co) / scale_transverse_x
    W4[1, :, :] = (tracker.record_last_track.px[:4, :] - px_co) / scale_transverse_x
    W4[2, :, :] = (tracker.record_last_track.y[:4, :]  - y_co) / scale_transverse_y
    W4[3, :, :] = (tracker.record_last_track.py[:4, :] - py_co) / scale_transverse_y

    betx = W4[0, 0, :]**2 + W4[0, 1, :]**2
    bety = W4[2, 2, :]**2 + W4[2, 3, :]**2

    gamx = W4[1, 0, :]**2 + W4[1, 1, :]**2
    gamy = W4[3, 2, :]**2 + W4[3, 3, :]**2

    alfx = - W4[0, 0, :] * W4[1, 0, :] - W4[0, 1, :] * W4[1, 1, :]
    alfy = - W4[2, 2, :] * W4[3, 2, :] - W4[2, 3, :] * W4[3, 3, :]

    betz0 = W[4, 4]**2 + W[4, 5]**2

    mux = np.unwrap(np.arctan2(W4[0, 1, :], W4[0, 0, :]))/2/np.pi
    muy = np.unwrap(np.arctan2(W4[2, 3, :], W4[2, 2, :]))/2/np.pi

    eta = -((part_for_twiss._xobject.zeta[6] - part_for_twiss._xobject.zeta[5])
                /(2*delta_disp)/tracker.line.get_length())
    alpha = eta + 1/part_on_co._xobject.gamma0[0]**2

    part_chrom_plus = xp.build_particles(
                _context=context,
                x_norm=0,
                zeta=part_on_co._xobject.zeta[0], delta=delta_chrom,
                particle_on_co=part_on_co,
                scale_with_transverse_norm_emitt=(nemitt_x, nemitt_y),
                R_matrix=RR,
                matrix_stability_tol=matrix_stability_tol,
                matrix_responsiveness_tol=matrix_responsiveness_tol,
                symplectify=symplectify)
    RR_chrom_plus = tracker.compute_one_turn_matrix_finite_differences(
                                            particle_on_co=part_chrom_plus.copy(),
                                            steps_r_matrix=steps_r_matrix)
    (WW_chrom_plus, WWinv_chrom_plus, Rot_chrom_plus
        ) = lnf.compute_linear_normal_form(RR_chrom_plus,
                                        responsiveness_tol=matrix_responsiveness_tol,
                                        stability_tol=matrix_stability_tol,
                                        symplectify=symplectify)
    qx_chrom_plus = np.angle(np.linalg.eig(Rot_chrom_plus)[0][0])/(2*np.pi)
    qy_chrom_plus = np.angle(np.linalg.eig(Rot_chrom_plus)[0][2])/(2*np.pi)

    part_chrom_minus = xp.build_particles(
                _context=context,
                x_norm=0,
                zeta=part_on_co._xobject.zeta[0], delta=-delta_chrom,
                particle_on_co=part_on_co,
                scale_with_transverse_norm_emitt=(nemitt_x, nemitt_y),
                R_matrix=RR,
                matrix_responsiveness_tol=matrix_responsiveness_tol,
                matrix_stability_tol=matrix_stability_tol,
                symplectify=symplectify)
    RR_chrom_minus = tracker.compute_one_turn_matrix_finite_differences(
                                        particle_on_co=part_chrom_minus.copy(),
                                        steps_r_matrix=steps_r_matrix)
    (WW_chrom_minus, WWinv_chrom_minus, Rot_chrom_minus
        ) = lnf.compute_linear_normal_form(RR_chrom_minus,
                                          symplectify=symplectify,
                                          stability_tol=matrix_stability_tol,
                                          responsiveness_tol=matrix_responsiveness_tol)
    qx_chrom_minus = np.angle(np.linalg.eig(Rot_chrom_minus)[0][0])/(2*np.pi)
    qy_chrom_minus = np.angle(np.linalg.eig(Rot_chrom_minus)[0][2])/(2*np.pi)

    dist_from_half_integer_x = np.modf(mux[-1])[0] - 0.5
    dist_from_half_integer_y = np.modf(muy[-1])[0] - 0.5

    if np.abs(qx_chrom_plus - qx_chrom_minus) > np.abs(dist_from_half_integer_x):
        raise NotImplementedError(
                "Qx too close to half integer, impossible to evaluate Q'x")
    if np.abs(qy_chrom_plus - qy_chrom_minus) > np.abs(dist_from_half_integer_y):
        raise NotImplementedError(
                "Qy too close to half integer, impossible to evaluate Q'y")

    dqx = (qx_chrom_plus - qx_chrom_minus)/delta_chrom/2
    dqy = (qy_chrom_plus - qy_chrom_minus)/delta_chrom/2

    if dist_from_half_integer_x > 0:
        dqx = -dqx

    if dist_from_half_integer_y > 0:
        dqy = -dqy

    qs = np.angle(np.linalg.eig(Rot)[0][4])/(2*np.pi)

    beta0 = part_on_co._xobject.beta0[0]
    circumference = tracker.line.get_length()
    T_rev = circumference/clight/beta0

    if eneloss_and_damping:
        diff_ptau = np.diff(ptau_co)
        eloss_turn = -sum(diff_ptau[diff_ptau<0]) * part_on_co._xobject.p0c[0]

        # Get eigenvalues
        w0, v0 = np.linalg.eig(RR)

        # Sort eigenvalues
        indx = [
            int(np.floor(np.argmax(np.abs(v0[:, 2*ii]))/2)) for ii in range(3)]
        eigenvals = np.array([w0[ii*2] for ii in indx])

        # Damping constants and partition numbers
        energy0 = part_on_co.mass0 * part_on_co._xobject.gamma0[0]
        damping_constants_turns = -np.log(np.abs(eigenvals))
        damping_constants_s = damping_constants_turns / T_rev
        partition_numbers = (
            damping_constants_turns* 2 * energy0/eloss_turn)

        eneloss_damp_res = {
            'eneloss_turn': eloss_turn,
            'damping_constants_turns': damping_constants_turns,
            'damping_constants_s':damping_constants_s,
            'partition_numbers': partition_numbers
        }

    twiss_res = {
        'name': tracker.line.element_names,
        's': s,
        'x': x_co,
        'px': px_co,
        'y': y_co,
        'py': py_co,
        'zeta': zeta_co,
        'delta': delta_co,
        'ptau': ptau_co,
        'betx': betx,
        'bety': bety,
        'alfx': alfx,
        'alfy': alfy,
        'gamx': gamx,
        'gamy': gamy,
        'betz0': betz0,
        'dx': dx,
        'dpx': dpx,
        'dy': dy,
        'dpy': dpy,
        'mux': mux,
        'muy': muy,
        'qx': mux[-1],
        'qy': muy[-1],
        'qs': qs,
        'dqx': dqx,
        'dqy': dqy,
        'slip_factor': eta,
        'momentum_compaction_factor': alpha,
        'circumference': circumference,
        'T_rev': T_rev,
        'R_matrix': RR,
        'particle_on_co':part_on_co.copy(_context=xo.context_default)
    }
    twiss_res['particle_on_co']._fsolve_info = part_on_co._fsolve_info

    if eneloss_and_damping:
        twiss_res.update(eneloss_damp_res)

    # Downselect based on at_element
    enames = tracker.line.element_names
    if at_elements is not None:
        indx_twiss = []
        for nn in at_elements:
            if isinstance(nn, (int, np.integer)):
                indx_twiss.append(int(nn))
            else:
                assert nn in tracker.line.element_names
                indx_twiss.append(enames.index(nn))

        for kk, vv in twiss_res.items():
            if eneloss_and_damping and kk in eneloss_damp_res.keys():
                continue
            if hasattr(vv, '__len__') and len(vv) == len(s):
                if isinstance(vv, np.ndarray):
                    twiss_res[kk] = vv[indx_twiss]
                else:
                    twiss_res[kk] = [vv[ii] for ii in indx_twiss]

    return twiss_res

def _one_turn_map(p, particle_ref, tracker, delta_zeta):
    part = particle_ref.copy()
    part.x = p[0]
    part.px = p[1]
    part.y = p[2]
    part.py = p[3]
    part.zeta = p[4] + delta_zeta
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

