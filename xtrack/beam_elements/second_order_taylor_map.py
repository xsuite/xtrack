# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement
import numpy as np
import xobjects as xo

class SecondOrderTaylorMap(BeamElement):

    '''
    Implements the second order Taylor map:

       z_out[i] = k[i] + sum_j (R[i,j]*z_in[j]) + sum_jk (T[i,j,k]*z_in[j]*z_in[k])

       where z = (x, px, y, py, zeta, pzeta)

    Parameters
    ----------
    length : float
        length of the element in meters.
    k : array_like
        6x1 array of the zero order Taylor map coefficients. Default is 0.
    R : array_like
        6x6 array of the first order Taylor map coefficients. Default is
        the identity matrix, so the element is an identity map by default.
    T : array_like
        6x6x6 array of the second order Taylor map coefficients. Default is 0.

    '''

    isthick = True

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/second_order_taylor_map.h"',
    ]

    _xofields={
        'k': xo.Field(xo.Float64[6], default=np.zeros(6, dtype=np.float64)),
        'R': xo.Field(xo.Float64[6, 6], default=np.eye(6, dtype=np.float64)),
        'T': xo.Field(xo.Float64[6, 6, 6], default=np.zeros((6, 6, 6), dtype=np.float64)),
        'length': xo.Float64,
    }

    @classmethod
    def from_line(cls, line, start, end, twiss_table=None,
                  **kwargs):

        '''
        Generate a `SecondOrderTaylorMap` from a `Line` object.
        The coefficients are computed with finite differences around the closed
        orbit.

        Parameters
        ----------
        line : Line
            A `Line` object.
        start : str
            Name of the element where the map starts.
        end : str
            Name of the element where the map stops.
        twiss_table : TwissTable, optional
            A `TwissTable` object. If not given, it will be computed.

        Returns
        -------
        SecondOrderTaylorMap
            A `SecondOrderTaylorMap` object.

        '''
        if start == end:
            # start == end will lead to get_R_matrix() computing a
            # full one-turn response matrix (but here we would rather expect identity)
            raise NotImplementedError('end element must be after start element')

        if twiss_table is None:
            tw = line.twiss(reverse=False)
        else:
            tw = twiss_table

        twinit = tw.get_twiss_init(start)
        twinit_out = tw.get_twiss_init(end)

        RR = line.get_R_matrix(
            start=start, end=end, particle_on_co=twinit.particle_on_co
            )['R_matrix']
        TT = line.get_T_matrix(start=start, end=end,
                                    particle_on_co=twinit.particle_on_co)

        x_co_in = np.array([
            twinit.particle_on_co.x[0],
            twinit.particle_on_co.px[0],
            twinit.particle_on_co.y[0],
            twinit.particle_on_co.py[0],
            twinit.particle_on_co.zeta[0],
            twinit.particle_on_co.pzeta[0],
        ])

        x_co_out = np.array([
            twinit_out.particle_on_co.x[0],
            twinit_out.particle_on_co.px[0],
            twinit_out.particle_on_co.y[0],
            twinit_out.particle_on_co.py[0],
            twinit_out.particle_on_co.zeta[0],
            twinit_out.particle_on_co.pzeta[0],
        ])

        # Handle feeddown (express the expansion in z instead of z - z_co)
        R_T_fd = np.einsum('ijk,k->ij', TT, x_co_in)
        K_T_fd = R_T_fd @ x_co_in

        K_hat = x_co_out - RR @ x_co_in + K_T_fd
        RR_hat = RR - 2 * R_T_fd

        smap = cls(R=RR_hat, T=TT, k=K_hat,
                   length=tw['s', end] - tw['s', start],
                   **kwargs)

        return smap

    def scale_coordinates(self, scale_x=1, scale_px=1, scale_y=1, scale_py=1,
                          scale_zeta=1, scale_pzeta=1):

        '''
        Generate a new `SecondOrderTaylorMap` with scaled coordinates.

        Parameters
        ----------
        scale_x : float
            Scaling factor for x.
        scale_px : float
            Scaling factor for px.
        scale_y : float
            Scaling factor for y.
        scale_py : float
            Scaling factor for py.
        scale_zeta : float
            Scaling factor for zeta.
        scale_pzeta : float
            Scaling factor for pzeta.

        Returns
        -------
        SecondOrderTaylorMap
            A new `SecondOrderTaylorMap` with scaled coordinates.

        '''

        out = self.copy()

        scale_factors = np.array(
            [scale_x, scale_px, scale_y, scale_py, scale_zeta, scale_pzeta])

        for ii in range(6):
            out.T[ii, :, :] *= scale_factors[ii]
            out.R[ii, :] *= scale_factors[ii]
            out.k[ii] *= scale_factors[ii]

        for jj in range(6):
            out.T[:, jj, :] *= scale_factors[jj]
            out.R[:, jj] *= scale_factors[jj]

        for kk in range(6):
            out.T[:, :, kk] *= scale_factors[kk]

        return out
