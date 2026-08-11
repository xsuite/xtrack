# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ...base_element import BeamElement
from scipy.special import factorial
import numpy as np
from warnings import warn
import xobjects as xo
from ...random import RandomNormal

class LineSegmentMap(BeamElement):

    _xofields={
        'length': xo.Float64,

        'qx': xo.Float64,
        'qy': xo.Float64,

        'coeffs_dqx': xo.Float64[:],
        'coeffs_dqy': xo.Float64[:],
        'det_xx': xo.Float64,
        'det_xy': xo.Float64,
        'det_yy': xo.Float64,
        'det_yx': xo.Float64,

        'betx': xo.Float64[2],
        'bety': xo.Float64[2],
        'alfx': xo.Float64[2],
        'alfy': xo.Float64[2],

        'dx': xo.Float64[2],
        'dpx': xo.Float64[2],
        'dy': xo.Float64[2],
        'dpy': xo.Float64[2],

        'x_ref': xo.Float64[2],
        'px_ref': xo.Float64[2],
        'y_ref': xo.Float64[2],
        'py_ref': xo.Float64[2],

        'energy_ref_increment': xo.Float64,
        'energy_increment': xo.Float64,
        'uncorrelated_rad_damping': xo.Int64,
        'correlated_rad_damping': xo.Int64,
        'damping_factors':xo.Float64[6,6],
        'uncorrelated_gauss_noise': xo.Int64,
        'correlated_gauss_noise': xo.Int64,
        'gauss_noise_matrix':xo.Float64[6,6],

        'longitudinal_mode_flag': xo.Int64,
        'qs': xo.Float64,
        'bets': xo.Float64,
        'bucket_length': xo.Float64,
        'momentum_compaction_factor': xo.Float64,
        'slippage_length': xo.Float64,
        'voltage_rf': xo.Float64[:],
        'frequency_rf': xo.Float64[:],
        'lag_rf': xo.Float64[:],
        'phase_rf': xo.Float64[:],
    }

    _depends_on = [RandomNormal]
    isthick = True

    _rename = {
        'lag_rf': '_lag_rf',
    }

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/linesegmentmap.h"',
    ]

    def __init__(self, length=0., qx=0, qy=0,
            betx=1., bety=1., alfx=0., alfy=0.,
            dx=0., dpx=0., dy=0., dpy=0.,
            x_ref=0.0, px_ref=0.0, y_ref=0.0, py_ref=0.0,
            longitudinal_mode=None,
            qs=None, bets=None,bucket_length=None,
            momentum_compaction_factor=None,
            slippage_length=None,
            voltage_rf=None, frequency_rf=None, lag_rf=None, phase_rf=None,
            dqx=0.0, dqy=0.0, ddqx=0.0, ddqy=0.0, dnqx=None, dnqy=None,
            det_xx=0.0, det_xy=0.0, det_yy=0.0, det_yx=0.0,
            energy_increment=0.0, energy_ref_increment=0.0,
            damping_rate_x = 0.0, damping_rate_px = 0.0,
            damping_rate_y = 0.0, damping_rate_py = 0.0,
            damping_rate_zeta = 0.0, damping_rate_pzeta = 0.0,
            gauss_noise_ampl_x=0.0,gauss_noise_ampl_px=0.0,
            gauss_noise_ampl_y=0.0,gauss_noise_ampl_py=0.0,
            gauss_noise_ampl_zeta=0.0,gauss_noise_ampl_pzeta=0.0,
            damping_matrix=None,gauss_noise_matrix=None,
            **nargs):

        '''
        Map representing a simplified segment of a beamline.

        Parameters
        ----------
        length : float
            Length of the segment in meters.
        qx : float
            Horizontal tune or phase advance of the segment.
        qy : float
            Vertical tune or phase advance of the segment.
        betx : tuple of length 2 or float
            Horizontal beta function at the entrance and exit of the segment.
            If a float is given, the same value is used for both entrance and exit.
        bety : tuple of length 2 or float
            Vertical beta function at the entrance and exit of the segment.
            If a float is given, the same value is used for both entrance and exit.
        alfx : tuple of length 2 or float
            Horizontal alpha function at the entrance and exit of the segment.
            If a float is given, the same value is used for both entrance and exit.
        alfy : tuple of length 2 or float
            Vertical alpha function at the entrance and exit of the segment.
            If a float is given, the same value is used for both entrance and exit.
        dx : tuple of length 2 or float
            Horizontal dispersion at the entrance and exit of the segment.
            If a float is given, the same value is used for both entrance and exit.
        dpx : tuple of length 2 or float
            Px dispersion at the entrance and exit of the segment.
            If a float is given, the same value is used for both entrance and exit.
        dy : tuple of length 2 or float
            Vertical dispersion at the entrance and exit of the segment.
            If a float is given, the same value is used for both entrance and exit.
        dpy : tuple of length 2 or float
            Py dispersion at the entrance and exit of the segment.
            If a float is given, the same value is used for both entrance and exit.
        x_ref : tuple of length 2 or float
            Horizontal position of the reference position at the entrance and
            exit of the segment (it is the closed orbit no other effects are
            present that perturb the closed orbit).
            If a float is given, the same value is used for both entrance and exit.
        px_ref : tuple of length 2 or float
            Px coordinate of the reference position at the entrance and
            exit of the segment (it is the closed orbit no other effects are
            present that perturb the closed orbit).
            If a float is given, the same value is used for both entrance and exit.
        y_ref : tuple of length 2 or float
            Vertical position of the reference position at the entrance and
            exit of the segment (it is the closed orbit no other effects are
            present that perturb the closed orbit).
            If a float is given, the same value is used for both entrance and exit.
        py_ref : tuple of length 2 or float
            Py coordinate of the reference position at the entrance and
            exit of the segment (it is the closed orbit no other effects are
            present that perturb the closed orbit).
            If a float is given, the same value is used for both entrance and exit.
        longitudinal_mode : str
            Longitudinal mode of the segment. Can be one of ``'linear_fixed_qs'``,
            ``'nonlinear'``, ``'linear_fixed_rf'`` or ``'frozen'``.
        qs : float
            Synchrotron tune of the segment. Only used if ``longitudinal_mode``
            is ``'linear_fixed_qs'``.
        bets : float
            Synchrotron beta function of the segment (positive above transition,
            negative below transition). Only used if ``longitudinal_mode``
            is ``'linear_fixed_qs'``.
        bucket_length : float
            The linear RF force becomes a sawtooth with a fixed point every
            bucket_length [full length in seconds]. Only used if
            ``longitudinal_mode`` is ``'linear_fixed_qs'``.
        momentum_compaction_factor : float
            Momentum compaction factor of the segment. Only used if
            ``longitudinal_mode`` is ``'nonlinear'`` or ``'linear_fixed_rf'``.
        slippage_length : float
            Slippage length of the segment. Only used if ``longitudinal_mode``
            is ``'nonlinear'`` or ``'linear_fixed_rf'``. If not given, the
            ``length`` of the segment is used.
        voltage_rf : list of float
            List of voltages of the RF kicks in the segment. Only used if
            ``longitudinal_mode`` is ``'nonlinear'`` or ``'linear_fixed_rf'``.
        frequency_rf : list of float
            List of frequencies of the RF kicks in the segment. Only used if
            ``longitudinal_mode`` is ``'nonlinear'`` or ``'linear_fixed_rf'``.
        lag_rf : list of float
            List of lags in degrees of the RF kicks in the segment. Only used if
            ``longitudinal_mode`` is ``'nonlinear'`` or ``'linear_fixed_rf'``.
        dqx : float or list of float
            Horizontal linear chromaticity of the segment.
        dqy : float or list of float
            Vertical linear chromaticity of the segment.
        ddqx: float
            Horizontal second order chromaticity of the segment
        ddqy: float
            Vertical second order chromaticity of the segment
        dnqx: list of float
            List of horizontal chromaticities up to any order. The first element
            of the list is the horizontal tune, the second element is the
            horizontal linear chromaticity, the third element the horizontal
            second order chromaticity and so on. It can be specified only if the
            horizontal tune, and chromaticities are not specified.
        dnqy: list of float
            List of vertical chromaticities up to any order. The first element
            of the list is the vertical tune, the second element is the
            vertical linear chromaticity, the third element the vertical
            second order chromaticity and so on. It can be specified only if the
            vertical tune, and chromaticities are not specified.
        det_xx : float
            Anharmonicity xx coefficient (i.e. dqx / dJx, where Jx is the horizontal
            action). Optional, default is ``0``.
        det_xy : float
            Anharmonicity xy coefficient (i.e. dqx / dJy, where Jy is the vertical
            action). Optional, default is ``0``.
        det_yx : float
            Anharmonicity yx coefficient (i.e. dqy / dJx, where Jx is the horizontal
            action). Optional, default is ``0``.
        det_yy : float
            Anharmonicity yy coefficient (i.e. dqy / dJy, where Jy is the vertical
            action). Optional, default is ``0``.
        energy_increment : float
            Energy increment of the segment in eV.
        energy_ref_increment : float
            Increment of the reference energy in eV.
        damping_rate_x : float
            Damping rate of the horizontal position
            x_n+1 = (1-damping_rate_x)*x_n. Optional, default is ``0``.
        damping_rate_px : float
            Damping rate of the horizontal momentum
            px_n+1 = (1-damping_rate_px)*px_n. Optional, default is ``0``.
        damping_rate_y : float
            Damping rate of the vertical position
            y_n+1 = (1-damping_rate_y)*y_n. Optional, default is ``0``.
        damping_rate_py : float
            Damping rate of the vertical momentum
            px_n+1 = (1-damping_rate_x)*py_n. Optional, default is ``0``.
        damping_rate_z : float
            Damping rate of the longitudinal position
            z_n+1 = (1-damping_rate_z)*z_n. Optional, default is ``0``.
        damping_rate_pzeta : float
            Damping rate on the momentum
            pzeta_n+1 = (1-damping_rate_pzeta)*pzeta_n. Optional, default is ``0``.
        gauss_noise_ampl_x : float
            Amplitude of Gaussian noise on the horizontal position. Optional, default is ``0``.
        gauss_noise_ampl_px : float
            Amplitude of Gaussian noise on the horizontal momentum. Optional, default is ``0``.
        gauss_noise_ampl_y : float
            Amplitude of Gaussian noise on the vertical position. Optional, default is ``0``.
        gauss_noise_ampl_py : float
            Amplitude of Gaussian noise on the vertical momentum. Optional, default is ``0``.
        gauss_noise_ampl_zeta : float
            Amplitude of Gaussian noise on the longitudinal position. Optional, default is ``0``.
        gauss_noise_ampl_pzeta : float
            Amplitude of Gaussian noise on the longitudinal momentum. Optional, default is ``0``.
        damping_matrix : float[6,6]
            Matrix of damping: Each paticles coordinate vector (x,px,y,py,zeta,pzeta) is multiplied
            by the identity + the damping matrix. Incompatible with inputs damping_rate_*.
            Optional, default is ``None``
        gauss_noise_matrix : float[6,6]
            Covariance matrix of the Gaussian noise applied in (x,px,y,py,zeta,pzeta).
            Incompatible with inputs gauss_noise_ampl_*. Optional, default is ``None``
        '''

        if '_xobject' in nargs.keys() and nargs['_xobject'] is not None:
            self._xobject = nargs['_xobject']
            return

        assert longitudinal_mode in [
            'linear_fixed_qs', 'nonlinear', 'linear_fixed_rf', 'frozen', None]

        if dnqx is not None:
            assert qx == 0 and dqx == 0 and ddqx == 0
            qx = dnqx[0]
        else:
            dnqx = [qx]
            if dqx != 0:
                dnqx.append(dqx)
            if ddqx != 0:
                dnqx.append(ddqx)

        if dnqy is not None:
            assert qy == 0 and dqy == 0 and ddqy == 0
            qy = dnqy[0]
        else:
            dnqy = [qy]
            if dqy != 0:
                dnqy.append(dqy)
            if ddqy != 0:
                dnqy.append(ddqy)

        coeffs_dqx = [dnqx[i] / float(factorial(i)) for i in range(len(dnqx))]
        coeffs_dqy = [dnqy[i] / float(factorial(i)) for i in range(len(dnqy))]

        nargs['qx'] = qx
        nargs['qy'] = qy
        nargs['coeffs_dqx'] = coeffs_dqx
        nargs['coeffs_dqy'] = coeffs_dqy
        nargs['det_xx'] = det_xx
        nargs['det_xy'] = det_xy
        nargs['det_yy'] = det_yy
        nargs['det_yx'] = det_yx
        nargs['length'] = length

        if longitudinal_mode is None:
            if qs is not None:
                longitudinal_mode = 'linear_fixed_qs'
            elif voltage_rf is not None:
                longitudinal_mode = 'nonlinear'
            else:
                longitudinal_mode = 'frozen'

        if longitudinal_mode == 'linear_fixed_qs':
            assert qs is not None
            assert bets is not None
            assert momentum_compaction_factor is None
            assert voltage_rf is None
            assert frequency_rf is None
            assert lag_rf is None
            assert phase_rf is None
            if bucket_length == None:
                bucket_length = -1.0
            nargs['longitudinal_mode_flag'] = 1
            nargs['qs'] = qs
            nargs['bets'] = bets
            nargs['bucket_length'] = bucket_length
            nargs['voltage_rf'] = [0]
            nargs['frequency_rf'] = [0]
            nargs['lag_rf'] = [0]
            nargs['phase_rf'] = [0]
        elif longitudinal_mode == 'nonlinear' or longitudinal_mode == 'linear_fixed_rf':
            assert voltage_rf is not None
            assert frequency_rf is not None
            assert momentum_compaction_factor is not None
            assert qs is None
            assert bets is None
            assert bucket_length is None

            if lag_rf is None:
                try:
                    lag_rf = [0]*len(frequency_rf)
                except TypeError:
                    lag_rf = [0]
            if phase_rf is None:
                try:
                    phase_rf = [0]*len(frequency_rf)
                except TypeError:
                    phase_rf = [0]

            if slippage_length is None:
                nargs['slippage_length'] = length
            else:
                nargs['slippage_length'] = slippage_length

            if longitudinal_mode == 'nonlinear':
                nargs['longitudinal_mode_flag'] = 2
            elif longitudinal_mode == 'linear_fixed_rf':
                nargs['longitudinal_mode_flag'] = 3

            nargs['voltage_rf'] = voltage_rf
            nargs['frequency_rf'] = frequency_rf
            nargs['phase_rf'] = phase_rf
            nargs['lag_rf'] = lag_rf
            nargs['momentum_compaction_factor'] = momentum_compaction_factor
            for nn in ['frequency_rf', 'lag_rf', 'voltage_rf', 'phase_rf']:
                if np.isscalar(nargs[nn]):
                    nargs[nn] = [nargs[nn]]

            assert (len(nargs['frequency_rf'])
                    == len(nargs['lag_rf'])
                    == len(nargs['phase_rf'])
                    == len(nargs['voltage_rf']))

            if longitudinal_mode == 'linear_fixed_rf':
                assert len(nargs['frequency_rf']) == 1

        elif longitudinal_mode == 'frozen':
            nargs['longitudinal_mode_flag'] = 0
            nargs['voltage_rf'] = [0]
            nargs['frequency_rf'] = [0]
            nargs['lag_rf'] = [0]
            nargs['phase_rf'] = [0]
        else:
            raise ValueError('longitudinal_mode must be one of "linear_fixed_qs", "nonlinear" or "frozen"')

        if np.isscalar(betx): betx = [betx, betx]
        else: assert len(betx) == 2

        if np.isscalar(bety): bety = [bety, bety]
        else: assert len(bety) == 2

        if np.isscalar(alfx): alfx = [alfx, alfx]
        else: assert len(alfx) == 2

        if np.isscalar(alfy): alfy = [alfy, alfy]
        else: assert len(alfy) == 2

        if np.isscalar(dx): dx = [dx, dx]
        else: assert len(dx) == 2

        if np.isscalar(dpx): dpx = [dpx, dpx]
        else: assert len(dpx) == 2

        if np.isscalar(dy): dy = [dy, dy]
        else: assert len(dy) == 2

        if np.isscalar(dpy): dpy = [dpy, dpy]
        else: assert len(dpy) == 2

        if np.isscalar(x_ref): x_ref = [x_ref, x_ref]
        else: assert len(x_ref) == 2

        if np.isscalar(px_ref): px_ref = [px_ref, px_ref]
        else: assert len(px_ref) == 2

        if np.isscalar(y_ref): y_ref = [y_ref, y_ref]
        else: assert len(y_ref) == 2

        if np.isscalar(py_ref): py_ref = [py_ref, py_ref]
        else: assert len(py_ref) == 2

        nargs['betx'] = betx
        nargs['bety'] = bety
        nargs['alfx'] = alfx
        nargs['alfy'] = alfy
        nargs['dx'] = dx
        nargs['dpx'] = dpx
        nargs['dy'] = dy
        nargs['dpy'] = dpy
        nargs['x_ref'] = x_ref
        nargs['px_ref'] = px_ref
        nargs['y_ref'] = y_ref
        nargs['py_ref'] = py_ref

        # acceleration with change of reference momentum
        nargs['energy_ref_increment'] = energy_ref_increment
        # acceleration without change of reference momentum
        nargs['energy_increment'] = energy_increment

        assert damping_rate_x >= 0.0
        assert damping_rate_px >= 0.0
        assert damping_rate_y >= 0.0
        assert damping_rate_py >= 0.0
        assert damping_rate_zeta >= 0.0
        assert damping_rate_pzeta >= 0.0

        if (damping_rate_x > 0.0 or damping_rate_px > 0.0
                or damping_rate_y > 0.0 or damping_rate_py > 0.0
                or damping_rate_zeta > 0.0 or damping_rate_pzeta > 0.0):
            assert damping_matrix is None
            nargs['uncorrelated_rad_damping'] = True
            nargs['correlated_rad_damping'] = False
            nargs['damping_factors'] = np.identity(6,dtype=float)
            nargs['damping_factors'][0,0] -= damping_rate_x
            nargs['damping_factors'][1,1] -= damping_rate_px
            nargs['damping_factors'][2,2] -= damping_rate_y
            nargs['damping_factors'][3,3] -= damping_rate_py
            nargs['damping_factors'][4,4] -= damping_rate_zeta
            nargs['damping_factors'][5,5] -= damping_rate_pzeta
        elif damping_matrix is not None:
            assert np.shape(damping_matrix) == (6,6)
            nargs['correlated_rad_damping'] = True
            nargs['uncorrelated_rad_damping'] = False
            nargs['damping_factors'] = np.identity(6,dtype=float)+damping_matrix
        else:
            nargs['uncorrelated_rad_damping'] = False
            nargs['correlated_rad_damping'] = False

        assert gauss_noise_ampl_x >= 0.0
        assert gauss_noise_ampl_px >= 0.0
        assert gauss_noise_ampl_y >= 0.0
        assert gauss_noise_ampl_py >= 0.0
        assert gauss_noise_ampl_zeta >= 0.0
        assert gauss_noise_ampl_pzeta >= 0.0
        if (gauss_noise_ampl_x > 0 or gauss_noise_ampl_px > 0 or
                gauss_noise_ampl_y > 0 or gauss_noise_ampl_py > 0 or
                gauss_noise_ampl_zeta > 0 or gauss_noise_ampl_pzeta > 0):
            assert gauss_noise_matrix is None
            nargs['uncorrelated_gauss_noise'] = True
            nargs['correlated_gauss_noise'] = False
            nargs['gauss_noise_matrix'] = np.zeros((6,6),dtype=float)
            nargs['gauss_noise_matrix'][0,0] = gauss_noise_ampl_x
            nargs['gauss_noise_matrix'][1,1] = gauss_noise_ampl_px
            nargs['gauss_noise_matrix'][2,2] = gauss_noise_ampl_y
            nargs['gauss_noise_matrix'][3,3] = gauss_noise_ampl_py
            nargs['gauss_noise_matrix'][4,4] = gauss_noise_ampl_zeta
            nargs['gauss_noise_matrix'][5,5] = gauss_noise_ampl_pzeta
        elif gauss_noise_matrix is not None:
            nargs['correlated_gauss_noise'] = True
            nargs['uncorrelated_gauss_noise'] = False
            assert np.shape(gauss_noise_matrix) == (6,6)
            (u, s, vh) = np.linalg.svd(gauss_noise_matrix)
            nargs['gauss_noise_matrix'] = u*np.sqrt(s)
        else:
            nargs['uncorrelated_gauss_noise'] = False
            nargs['correlated_gauss_noise'] = False

        # Warn if lag_rf (deprecation)
        for vv in nargs['lag_rf']:
            if vv != 0:
                warn("`lag_rf` (in degrees) is deprecated and will be removed in a future version. "
                     "Please use `phase_rf` (in radians) instead. "
                     "Note that if both `lag_rf` and `phase_rf` are set, the effect is the sum of the two "
                     "with `lag_rf` converted to radians.",
                     FutureWarning, stacklevel=2)
                break

        super().__init__(**nargs)

    @property
    def longitudinal_mode(self):
        ret = {
            0: 'frozen',
            1: 'linear_fixed_qs',
            2: 'nonlinear',
            3: 'linear_fixed_rf'
        }[self.longitudinal_mode_flag]
        return ret

    @property
    def lag_rf(self):
        return self._buffer.context.linked_array_type.from_array(
            self._lag_rf,
            mode='setitem_from_container',
            container=self,
            container_setitem_name='_lag_rf_setitem')

    @lag_rf.setter
    def lag_rf(self, value):
        self.lag_rf[:] = value

    def _lag_rf_setitem(self, index, value):

        need_warn = False
        if np.isscalar(value) and value != 0:
            need_warn = True
        elif not np.isscalar(value):
            for v in value:
                if v != 0:
                    need_warn = True
                    break

        if need_warn:
            warn('`lag_rf` (in degrees) is deprecated and will be removed in a future version.'
            'Please use `phase_rf` (in radians) instead. '
            'Note that if both `lag_rf` and `phase_rf` are set, the effect is the sum of the two '
            'with `lag_rf` converted to radians.',
            FutureWarning, stacklevel=2)

        self._lag_rf[index] = value
