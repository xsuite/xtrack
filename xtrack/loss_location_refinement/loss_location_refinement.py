# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
from scipy.spatial import ConvexHull

import xobjects as xo
import xtrack as xt

from ..beam_elements import LimitPolygon, XYShift, SRotation, Drift, Marker
from ..line import (Line, _is_thick, _behaves_like_drift, _allow_loss_refinement,
                    _has_backtrack, _is_aperture)

from ..general import _print

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


def _skip_in_loss_location_refinement(element, line):
    if isinstance(element, xt.Replica):
        return _skip_in_loss_location_refinement(element.resolve(line), line)
    return (hasattr(element, 'skip_in_loss_location_refinement')
            and element.skip_in_loss_location_refinement)

class LossLocationRefinement:

    '''
    Class to refine the location of the lost particles within a line.

    Parameters
    ----------
    line : xtrack.Line
        Line for which the loss location refinement is performed.
    backtrack_line : xtrack.Line (optional)
        Line used to backtrack the lost particles. If None, the backtracking
        line is automatically generated from the line.
    n_theta : int
        Number of angles used to generate the interpolating aperture model.
        If None, the number of angles is automatically determined.
    r_max : float
        Radius larger than the largest aperture radius.
    dr : float
        Radius step used to generate the interpolating aperture model.
    ds : float
        Step in the `s` direction used to generate the interpolating aperture
        model.
    save_refine_lines : bool
        If True, the lines used to refine the loss location are saved.
    allowed_backtrack_types : list
        List of element types through which the backtracking is allowed.
        Elements exposing the attribute `allow_loss_refinement` are automatically
        added to the list.

    '''

    def __init__(self, line, backtrack_line=None,
                 n_theta=None, r_max=None, dr=None, ds=None,
                 save_refine_lines=False,
                 allowed_backtrack_types=[]):

        if backtrack_line is not None:
            raise ValueError('Backtracking line not supported anymore!')

        if line.iscollective:
            self._original_line = line
            self.line = line._get_non_collective_line()
        else:
            self._original_line = line
            self.line = line

        self._context = self.line._context
        assert isinstance(self._context, xo.ContextCpu), (
                "Other contexts are not supported!")

        # Build a polygon and compile the kernel
        temp_poly = LimitPolygon(_buffer=self.line._buffer,
                x_vertices=[1,-1, -1, 1], y_vertices=[1,1,-1,-1])
        na = lambda a : np.array(a, dtype=np.float64)
        temp_poly.impact_point_and_normal(x_in=na([0]), y_in=na([0]),
                                          z_in=na([0]), x_out=na([2]),
                                          y_out=na([2]), z_out=na([0]))

        # Build track kernel with all elements + polygon
        elm_gen = self.line.element_dict.copy()
        elm_gen['_xtrack_temp_poly_'] = temp_poly
        ln_gen = Line(elements=elm_gen,
                      element_names=list(line.element_names) + ['_xtrack_temp_poly_'])
        ln_gen.build_tracker(_buffer=self.line._buffer)
        ln_gen.config.XTRACK_GLOBAL_XY_LIMIT = line.config.XTRACK_GLOBAL_XY_LIMIT
        self._ln_gen = ln_gen

        self.i_apertures, self.apertures = find_apertures(self.line)

        self.save_refine_lines = save_refine_lines
        if save_refine_lines:
            self.refine_lines = {}

        self.n_theta = n_theta
        self.r_max = r_max
        self.dr = dr
        self.ds = ds
        self.allowed_backtrack_types = allowed_backtrack_types

    def refine_loss_location(self, particles, i_apertures=None):

        '''
        Refine the location of the lost particles within the line.

        Parameters
        ----------
        particles : xt.Particles
            Particles for which the loss location is refined.
        i_apertures : list (optional)
            List of indices of the apertures for which the loss location
            is refined. If None, the loss location is refined for all
            apertures.

        '''

        if i_apertures is None:
            i_apertures = self.i_apertures

        for i_ap in i_apertures:
            if np.any((particles.at_element==i_ap) & (particles.state==0)):

                if self.i_apertures.index(i_ap) == 0:
                    logger.warning(
                            'Unable to handle the first aperture in the line')
                    continue

                i_aper_1 = i_ap
                i_aper_0 = self.i_apertures[self.i_apertures.index(i_ap) - 1]
                logger.debug(f'i_aper_1={i_aper_1}, i_aper_0={i_aper_0}')

                s0, s1, _ = generate_interp_aperture_locations(self.line,
                                                   i_aper_0, i_aper_1, self.ds)
                assert s1 >= s0
                if s1 - s0 <= self.ds:
                    logger.debug('s1-s0 < ds: nothing to do')
                    continue

                presence_shifts_rotations = check_for_active_shifts_and_rotations(
                                                    self.line, i_aper_0, i_aper_1)
                logger.debug(f'presence_shifts_rotations={presence_shifts_rotations}')

                if (not(presence_shifts_rotations) and
                   apertures_are_identical(self.line.elements[i_aper_0],
                                           self.line.elements[i_aper_1], self.line)):

                    logger.debug('Replicate mode')
                    (interp_line, i_end_thin_0, i_start_thin_1, s0, s1
                            ) = interp_aperture_replicate(self._context,
                                      self.line,
                                      i_aper_0, i_aper_1,
                                      self.ds,
                                      _ln_gen=self._ln_gen)

                else:

                    logger.debug('Polygon interpolation mode')
                    (interp_line, i_end_thin_0, i_start_thin_1, s0, s1
                            ) = interp_aperture_using_polygons(self._context,
                                      self.line,
                                      i_aper_0, i_aper_1,
                                      self.n_theta, self.r_max, self.dr, self.ds,
                                      _ln_gen=self._ln_gen)

                interp_line._original_line = self._original_line
                part_refine = refine_loss_location_single_aperture(
                        particles, i_aper_1, i_end_thin_0,
                        self.line, interp_line, inplace=True,
                        allowed_backtrack_types=self.allowed_backtrack_types)

                if self.save_refine_lines:
                    interp_line.i_start_thin_0 = i_end_thin_0
                    interp_line.i_start_thin_1 = i_start_thin_1
                    interp_line.s0 = s0
                    interp_line.s1 = s1
                    self.refine_lines[i_ap] = interp_line


def check_for_active_shifts_and_rotations(line, i_aper_0, i_aper_1):

    presence_shifts_rotations = False
    for ii in range(i_aper_0, i_aper_1):
        ee = line.elements[ii]
        if ee.__class__ is SRotation:
            if not np.isclose(ee.angle, 0, rtol=0, atol=1e-15):
                presence_shifts_rotations = True
                break
        if ee.__class__ is XYShift:
            if not np.allclose([ee.dx, ee.dy], 0, rtol=0, atol=1e-15):
                presence_shifts_rotations = True
                break
    return presence_shifts_rotations


def apertures_are_identical(aper1, aper2, line):

    if isinstance(aper1, xt.Replica):
        aper1 = aper1.resolve(line)

    if isinstance(aper2, xt.Replica):
        aper2 = aper2.resolve(line)

    if aper1.__class__ != aper2.__class__:
        return False

    identical = True
    for ff in aper1._fields:
        tt = np.allclose(getattr(aper1, ff), getattr(aper2, ff),
                        rtol=0, atol=1e-15)
        if not tt:
            identical = False
            break
    return identical


def find_apertures(ln_gen):
    i_apertures = []
    apertures = []
    for ii, ee in enumerate(ln_gen.elements):
        if _is_aperture(ee, ln_gen):
            i_apertures.append(ii)
            apertures.append(ee)

    return i_apertures, apertures


def refine_loss_location_single_aperture(particles, i_aper_1, i_end_thin_0,
                    line, interp_line,
                    inplace=True,
                    allowed_backtrack_types=[]):

    mask_part = (particles.state == 0) & (particles.at_element == i_aper_1)

    part_refine = xt.Particles(
                    p0c=particles.p0c[mask_part],
                    mass0=particles.mass0,
                    q0=particles.q0,
                    x=particles.x[mask_part],
                    px=particles.px[mask_part],
                    y=particles.y[mask_part],
                    py=particles.py[mask_part],
                    zeta=particles.zeta[mask_part],
                    delta=particles.delta[mask_part],
                    s=particles.s[mask_part],
                    chi=particles.chi[mask_part],
                    charge_ratio=particles.charge_ratio[mask_part])

    i_start = i_end_thin_0 + 1
    i_stop = i_aper_1

    # Check that we are not backtracking through element types that are not allowed
    for nn in interp_line._original_line.element_names[i_start : i_stop]:
        ee = interp_line._original_line.element_dict[nn]

        can_backtrack = True
        if not _has_backtrack(ee, line):
            can_backtrack = False
        elif not _allow_loss_refinement(ee, line):
            can_backtrack = False

            # Check for override
            if isinstance(ee, xt.Replica):
                ee = ee.resolve(line)
            if isinstance(ee, tuple(allowed_backtrack_types)):
                can_backtrack = True

        if not can_backtrack:
            if _skip_in_loss_location_refinement(ee, line):
                return 'skipped'
            raise TypeError(
                f'Cannot backtrack through element {nn} of type '
                f'{ee.__class__.__name__}')

    with xt.line._preserve_config(line):
        line.config.XTRACK_GLOBAL_XY_LIMIT = None
        line.track(part_refine, ele_start=i_start, ele_stop=i_stop,
                    backtrack='force')

    # Track with extra apertures
    interp_line.track(part_refine)
    # There is a small fraction of particles that are not lost.
    # We verified that they are really at the edge. Their coordinates
    # correspond to the end fo the short line, which is correct

    if inplace:
        indx_sorted = np.argsort(part_refine.particle_id)
        with particles._bypass_linked_vars():
            particles.x[mask_part] = part_refine.x[indx_sorted]
            particles.px[mask_part] = part_refine.px[indx_sorted]
            particles.y[mask_part] = part_refine.y[indx_sorted]
            particles.py[mask_part] = part_refine.py[indx_sorted]
            particles.zeta[mask_part] = part_refine.zeta[indx_sorted]
            particles.s[mask_part] = part_refine.s[indx_sorted]
            particles.delta[mask_part] = part_refine.delta[indx_sorted]
            particles.ptau[mask_part] = part_refine.ptau[indx_sorted]
            particles.rvv[mask_part] = part_refine.rvv[indx_sorted]
            particles.rpp[mask_part] = part_refine.rpp[indx_sorted]
            particles.p0c[mask_part] = part_refine.p0c[indx_sorted]
            particles.gamma0[mask_part] = part_refine.gamma0[indx_sorted]
            particles.beta0[mask_part] = part_refine.beta0[indx_sorted]

    return part_refine

def interp_aperture_replicate(context, line,
                              i_aper_0, i_aper_1,
                              ds, _ln_gen, mode='end',):

    temp_buf = context.new_buffer()

    i_start_thin_1 = find_adjacent_drift(line, i_aper_1, direction='upstream') + 1
    i_end_thin_0 = find_adjacent_drift(line, i_aper_0, direction='downstream') - 1

    s0, s1, s_vect = generate_interp_aperture_locations(line,
                                                   i_aper_0, i_aper_1, ds)

    if mode=='end':
        aper_to_copy = line.elements[i_aper_1]
    elif mode=='start':
        aper_to_copy = line.elements[i_aper_0]
    else:
        raise ValueError(f'Invalid mode: {mode}')
    interp_apertures = []
    for ss in s_vect:
        interp_apertures.append(aper_to_copy.copy(_buffer=temp_buf))

    interp_line = build_interp_line(
            _buffer=temp_buf,
            s0=s0, s1=s1, s_interp=s_vect,
            aper_0=aper_to_copy.copy(_buffer=temp_buf),
            aper_1=aper_to_copy.copy(_buffer=temp_buf),
            aper_interp=interp_apertures,
            line=line, i_start_thin_0=i_end_thin_0,
            i_start_thin_1=i_start_thin_1,
            _ln_gen=_ln_gen)

    return interp_line, i_end_thin_0, i_start_thin_1, s0, s1

def interp_aperture_using_polygons(context, line,
                       i_aper_0, i_aper_1,
                       n_theta, r_max, dr, ds, _ln_gen):

    temp_buf = context.new_buffer()

    polygon_1, i_start_thin_1 = characterize_aperture(line,
                                 i_aper_1, n_theta, r_max, dr,
                                 buffer_for_poly=temp_buf,
                                 coming_from='upstream')

    polygon_0, i_end_thin_0 = characterize_aperture(line, i_aper_0,
                                 n_theta, r_max, dr,
                                 buffer_for_poly=temp_buf,
                                 coming_from='downstream')

    s0, s1, s_vect = generate_interp_aperture_locations(line,
                                                   i_aper_0, i_aper_1, ds)

    Delta_s = s1 - s0
    interp_polygons = []
    for ss in s_vect:
        x_non_convex=(polygon_1.x_vertices*(ss - s0) / Delta_s
                  + polygon_0.x_vertices*(s1 - ss) / Delta_s)
        y_non_convex=(polygon_1.y_vertices*(ss - s0) / Delta_s
                  + polygon_0.y_vertices*(s1 - ss) / Delta_s)
        hull = ConvexHull(np.array([x_non_convex, y_non_convex]).T)
        i_hull = np.sort(hull.vertices)
        x_hull = x_non_convex[i_hull]
        y_hull = y_non_convex[i_hull]
        interp_polygons.append(LimitPolygon(
            _buffer=temp_buf,
            x_vertices=x_hull,
            y_vertices=y_hull))

    interp_line = build_interp_line(
            _buffer=temp_buf,
            s0=s0, s1=s1, s_interp=s_vect,
            aper_0=polygon_0, aper_1=polygon_1,
            aper_interp=interp_polygons,
            line=line, i_start_thin_0=i_end_thin_0,
            i_start_thin_1=i_start_thin_1,
            _ln_gen=_ln_gen)

    return interp_line, i_end_thin_0, i_start_thin_1, s0, s1

def generate_interp_aperture_locations(line, i_aper_0, i_aper_1, ds):

    s0 = line.tracker._tracker_data_base.element_s_locations[i_aper_0]
    s1 = line.tracker._tracker_data_base.element_s_locations[i_aper_1]
    assert s1>=s0
    n_segments = int(np.ceil((s1-s0)/ds))
    if n_segments <= 1:
        s_vect = np.array([])
    else:
        s_vect = np.linspace(s0, s1, n_segments+1)[1:-1]

    return s0, s1, s_vect

def build_interp_line(_buffer, s0, s1, s_interp, aper_0, aper_1, aper_interp,
                         line, i_start_thin_0, i_start_thin_1, _ln_gen):

    # Build interp line
    s_elements = [s0] + list(s_interp) +[s1]
    elements = [aper_0] + aper_interp + [aper_1]

    for i_ele in range(i_start_thin_0+1, i_start_thin_1):
        ee = line.elements[i_ele]
        if not _behaves_like_drift(ee, line):
            assert not _is_thick(ee, line)
            ss_ee = line.tracker._tracker_data_base.element_s_locations[i_ele]
            elements.append(ee.copy(_buffer=_buffer))
            s_elements.append(ss_ee)
    i_sorted = np.argsort(s_elements)
    s_sorted = list(np.take(s_elements, i_sorted))
    ele_sorted = list(np.take(elements, i_sorted))

    s_all = [s_sorted[0]]
    ele_all = [ele_sorted[0]]

    for ii in range(1, len(s_sorted)):
        ss = s_sorted[ii]

        if ss-s_all[-1]>1e-14:
            ele_all.append(Drift(_buffer=_buffer, length=ss-s_all[-1]))
            s_all.append(ss)
        ele_all.append(ele_sorted[ii])
        s_all.append(s_sorted[ii])

    interp_line = Line(elements=ele_all)

    interp_line.build_tracker(_buffer=_buffer,
                              track_kernel=_ln_gen.tracker.track_kernel)
    interp_line.reset_s_at_end_turn = False
    interp_line.config.XTRACK_GLOBAL_XY_LIMIT = _ln_gen.config.XTRACK_GLOBAL_XY_LIMIT


    return interp_line

def find_adjacent_drift(line, i_element, direction):

    ii=i_element
    found = False
    assert direction in ['upstream', 'downstream']
    if direction == 'upstream':
        increment = -1
    else:
        increment = 1
    while not(found):
        ee = line.element_dict[line.element_names[ii]]
        if isinstance(ee, xt.Replica):
            ee = ee.resolve(line)
        ccnn = ee.__class__.__name__
        #_print(ccnn)
        if ccnn.startswith('Drift'):
            found = True
        elif _behaves_like_drift(ee, line):
            found = True
        else:
            ii += increment

    return ii

def find_previous_drift(line, i_aperture):

    ii=i_aperture
    found = False
    while not(found):
        ee = line.element_dict[line.element_names[ii]]
        if isinstance(ee, xt.Replica):
            ee = ee.resolve(line)
        ccnn = ee.__class__.__name__
        if ccnn == 'Drift':
            found = True
        elif _behaves_like_drift(ee, line):
            found = True
        else:
            ii -= 1
    i_start = ii + 1

    return i_start

def index_in_reversed_line(num_elements, ii):
    return num_elements - ii - 1


def characterize_aperture(line, i_aperture, n_theta, r_max, dr,
                          buffer_for_poly, coming_from='upstream'):

    assert coming_from in ['upstream', 'downstream']

    # find previous drift
    if coming_from == 'upstream':
        i_start = find_adjacent_drift(line, i_aperture, 'upstream') + 1
        i_stop = i_aperture + 1
        backtrack = False
        index_start_thin = i_start
    elif coming_from == 'downstream':
        i_stop = find_adjacent_drift(line, i_aperture, 'downstream')
        i_start = i_aperture
        backtrack = 'force'
        assert np.all([_has_backtrack(ee, line) for ee in
                line.tracker._tracker_data_base.elements[i_start:i_stop+1]])
        index_start_thin = i_stop - 1

    # Get polygon
    theta_vect = np.linspace(0, 2*np.pi, n_theta+1)[:-1]

    this_rmin = 0
    this_rmax = r_max
    this_dr = (this_rmax-this_rmin)/100.
    rmin_theta = 0*theta_vect
    t_iter = []
    for iteration in range(2):

        r_vect = np.arange(this_rmin, this_rmax, this_dr)

        RR, TT = np.meshgrid(r_vect, theta_vect)
        RR += np.atleast_2d(rmin_theta).T

        x_test = RR.flatten()*np.cos(TT.flatten())
        y_test = RR.flatten()*np.sin(TT.flatten())

        logger.info(f'iteration={iteration} num_part={x_test.shape[0]}')

        ptest = xt.Particles(p0c=1,
                x = x_test.copy(),
                y = y_test.copy())
        with xt.line._preserve_config(line):
            line.config.XTRACK_GLOBAL_XY_LIMIT = None
            line.track(ptest, ele_start=i_start, ele_stop=i_stop,
                       backtrack=backtrack)

        indx_sorted = np.argsort(ptest.particle_id)
        state_sorted = np.take(ptest.state, indx_sorted)

        state_mat = state_sorted.reshape(RR.shape)

        i_r_aper = np.argmin(state_mat>0, axis=1)

        rmin_theta = r_vect[i_r_aper-1]
        this_rmin=0
        this_rmax=2*this_dr
        this_dr = dr

    x_mat = x_test.reshape(RR.shape)
    y_mat = y_test.reshape(RR.shape)
    x_non_convex = np.array(
            [x_mat[itt, i_r_aper[itt]] for itt in range(n_theta)])
    y_non_convex = np.array(
            [y_mat[itt, i_r_aper[itt]] for itt in range(n_theta)])

    hull = ConvexHull(np.array([x_non_convex, y_non_convex]).T)
    i_hull = np.sort(hull.vertices)
    x_hull = x_non_convex[i_hull]
    y_hull = y_non_convex[i_hull]

    # Get a convex polygon that does not have points for all angles
    temp_poly = LimitPolygon(x_vertices=x_hull, y_vertices=y_hull)

    # Get a convex polygon that has vertices at all requested angles
    r_out = 1. # m
    res = temp_poly.impact_point_and_normal(
            x_in=0*theta_vect, y_in=0*theta_vect, z_in=0*theta_vect,
            x_out=r_out*np.cos(theta_vect),
            y_out=r_out*np.sin(theta_vect),
            z_out=0*theta_vect)

    polygon = LimitPolygon(x_vertices=res[0], y_vertices=res[1],
                              _buffer=buffer_for_poly)

    return polygon, index_start_thin

