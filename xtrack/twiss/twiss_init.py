# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import io
import json

import numpy as np
import xobjects as xo

from .. import json as json_utils
from ..table import Table
from .constants import VARS_FOR_TWISS_INIT_GENERATION
from .element_indexing import _str_to_index

import xtrack as xt  # To avoid circular imports


class TwissInit:

    def __init__(self, particle_on_co=None, W_matrix=None, element_name=None,
                line=None, particle_ref=None,
                x=None, px=None, y=None, py=None, zeta=None, delta=None,
                betx=None, alfx=None, bety=None, alfy=None, bets=None,
                dx=None, dpx=None, dy=None, dpy=None, dzeta=None,
                mux=None, muy=None, muzeta=None,
                ddx=None, ddpx=None, ddy=None, ddpy=None, ddzeta=None,
                spin_x=None, spin_y=None, spin_z=None,
                ax_chrom=None, bx_chrom=None, ay_chrom=None, by_chrom=None,
                reference_frame=None):

        # Custom setattr needs to be bypassed for creation of attributes
        object.__setattr__(self, 'particle_on_co', None)
        self._temp_co_data = None
        self._temp_optics_data = None

        if particle_on_co is None:
            self._temp_co_data = dict(
                x=(x or 0.),
                px=(px or 0.),
                y=(y or 0.),
                py=(py or 0.),
                zeta=(zeta or 0.),
                delta=(delta or 0.),
                spin_x=(spin_x or 0.),
                spin_y=(spin_y or 0.),
                spin_z=(spin_z or 0.)
            )
        else:
            assert x is None, "``x`` must be None if ``particle_on_co`` is provided"
            assert px is None, "``px`` must be None if ``particle_on_co`` is provided"
            assert y is None, "``y`` must be None if ``particle_on_co`` is provided"
            assert py is None, "``py`` must be None if ``particle_on_co`` is provided"
            assert zeta is None, "``zeta`` must be None if ``particle_on_co`` is provided"
            assert delta is None, "``delta`` must be None if ``particle_on_co`` is provided"
            assert spin_x is None, "``spin_x`` must be None if ``particle_on_co`` is provided"
            assert spin_y is None, "``spin_y`` must be None if ``particle_on_co`` is provided"
            assert spin_z is None, "``spin_z`` must be None if ``particle_on_co`` is provided"
            assert particle_ref is None, (
                "``particle_ref`` must be None if ``particle_on_co`` is provided")
            self.__dict__['particle_on_co'] = particle_on_co

        if W_matrix is None:
            alfx = alfx or 0
            alfy = alfy or 0
            betx = betx or 1
            bety = bety or 1
            bets = bets or 1
            dx = dx or 0
            dpx = dpx or 0
            dy = dy or 0
            dpy = dpy or 0

            self._temp_optics_data = dict(
                betx=betx, alfx=alfx, bety=bety, alfy=alfy, bets=bets,
                dx=dx, dpx=dpx, dy=dy, dpy=dpy)
        else:
            assert betx is None, "``betx`` must be None if ``W_matrix`` is provided"
            assert alfx is None, "``alfx`` must be None if ``W_matrix`` is provided"
            assert bety is None, "``bety`` must be None if ``W_matrix`` is provided"
            assert alfy is None, "``alfy`` must be None if ``W_matrix`` is provided"
            assert bets is None, "``bets`` must be None if ``W_matrix`` is provided"
            self._temp_co_data = None

        self.element_name = element_name
        self.W_matrix = W_matrix
        self.mux = (mux or 0.)
        self.muy = (muy or 0.)
        self.muzeta = (muzeta or 0.)
        self.dzeta = (dzeta or 0.)
        self.ax_chrom = (ax_chrom or 0.)
        self.bx_chrom = (bx_chrom or 0.)
        self.ay_chrom = (ay_chrom or 0.)
        self.by_chrom = (by_chrom or 0.)
        self.ddx = (ddx or 0.)
        self.ddpx = (ddpx or 0.)
        self.ddy = (ddy or 0.)
        self.ddpy = (ddpy or 0.)
        self.reference_frame = reference_frame

        if line is not None and element_name is not None:
            self._finish_initialization(line, element_name)

    def to_dict(self):
        '''
        Convert to dictionary representation.
        '''

        out = self.__dict__.copy()
        out['particle_on_co'] = out['particle_on_co'].to_dict()
        return out

    def to_json(self, file, indent=1, **kwargs):

        '''
        Convert to JSON representation.

        Parameters
        ----------
        file : str or file-like

        '''

        json_utils.dump(self.to_dict(**kwargs), file, indent=indent)

    @classmethod
    def from_dict(cls, dct):
        '''
        Convert from dictionary representation.

        Parameters
        ----------
        dct : dict
            Dictionary representation.

        Returns
        -------
        out : TwissInit
            TwissInit instance.
        '''

        # Need the values as numpy types, in particular arrays
        numpy_dct = {}
        for key, value in dct.items():
            if key == 'particle_on_co':
                continue
            if isinstance(value, int):
                numpy_dct[key] = np.int64(value)
            elif isinstance(value, float):
                numpy_dct[key] = np.float64(value)
            elif isinstance(value, str):
                numpy_dct[key] = np.str_(value)
            elif isinstance(value, list):
                numpy_dct[key] = np.array(value)
            else:
                numpy_dct[key] = value

        numpy_dct['particle_on_co'] = xt.Particles.from_dict(dct['particle_on_co'])

        out = cls()
        out.__dict__.update(numpy_dct)
        return out

    @classmethod
    def from_json(cls, file):

        '''
        Convert from JSON representation.

        Parameters
        ----------
        file : str or file-like
            File name or file-like object.

        Returns
        -------
        out : TwissInit
            TwissInit instance.

        '''

        if isinstance(file, io.IOBase):
            dct = json.load(file)
        else:
            with open(file, 'r') as fid:
                dct = json.load(fid)

        return cls.from_dict(dct)

    def _finish_initialization(self, line, element_name):

        if (line is not None and 'reverse' in line.twiss_default
            and line.twiss_default['reverse']):
            input_reversed = True
            assert self.reference_frame is None, ("``reference_frame`` must be None "
                "if ``twiss_default['reverse']`` is True")
        else:
            input_reversed = False

        if self._temp_co_data is not None:
            import xpart
            assert line is not None, (
                "``line`` must be provided if ``particle_on_co`` is None")

            i_ele_in_line = _str_to_index(line, element_name, allow_end_point=False)
            s_ele_in_line = line.tracker._tracker_data_base.element_s_locations[i_ele_in_line]

            if input_reversed:
                s_ele_twiss = line.tracker._tracker_data_base.line_length - s_ele_in_line
                first_ele = line[line._element_names_unique[i_ele_in_line]]
                if hasattr(first_ele, 'isthick') and first_ele.isthick:
                    s_ele_twiss -= first_ele.length
            else:
                s_ele_twiss = s_ele_in_line

            particle_on_co = xpart.build_particles(
                x=self._temp_co_data['x'], px=self._temp_co_data['px'],
                y=self._temp_co_data['y'], py=self._temp_co_data['py'],
                spin_x=self._temp_co_data.get('spin_x', 0),
                spin_y=self._temp_co_data.get('spin_y', 0),
                spin_z=self._temp_co_data.get('spin_z', 0),
                delta=self._temp_co_data['delta'], zeta=self._temp_co_data['zeta'],
                line=line,
                include_collective=True, # In fact it does not matter
            )
            particle_on_co.s = s_ele_twiss
            self.__dict__['particle_on_co'] = particle_on_co
            self._temp_co_data = None
        else:
            particle_on_co = self.particle_on_co

        if self._temp_optics_data is not None:

            # aux_segment = xt.LineSegmentMap(
            #     length=1., # dummy
            #     qx=0.55, # dummy
            #     qy=0.57, # dummy
            #     qs=0.0000001, # dummy
            #     bets=self._temp_optics_data['bets'],
            #     betx=self._temp_optics_data['betx'],
            #     bety=self._temp_optics_data['bety'],
            #     alfx=self._temp_optics_data['alfx'] * (-1 if input_reversed else 1),
            #     alfy=self._temp_optics_data['alfy'] * (-1 if input_reversed else 1),
            #     dx=self._temp_optics_data['dx'] * (-1 if input_reversed else 1),
            #     dy=self._temp_optics_data['dy'],
            #     dpx=self._temp_optics_data['dpx'],
            #     dpy=self._temp_optics_data['dpy'] * (-1 if input_reversed else 1),
            #     )
            # aux_line = xt.Line(elements=[aux_segment])
            # aux_line.particle_ref = particle_on_co.copy(
            #                             _context=xo.context_default)
            # aux_line.particle_ref.reorganize()
            # aux_line.build_tracker()
            # aux_tw = aux_line.twiss()
            # W_matrix = aux_tw.W_matrix[0]

            W_matrix = _6d_w_matrix(
                bets=self._temp_optics_data['bets'],
                betx=self._temp_optics_data['betx'],
                bety=self._temp_optics_data['bety'],
                alfx=self._temp_optics_data['alfx'] * (-1 if input_reversed else 1),
                alfy=self._temp_optics_data['alfy'] * (-1 if input_reversed else 1),
                dx=self._temp_optics_data['dx'] * (-1 if input_reversed else 1),
                dy=self._temp_optics_data['dy'],
                dpx=self._temp_optics_data['dpx'],
                dpy=self._temp_optics_data['dpy'] * (-1 if input_reversed else 1),
            )

            if input_reversed:
                W_matrix[0, :] = -W_matrix[0, :]
                W_matrix[1, :] = W_matrix[1, :]
                W_matrix[2, :] = W_matrix[2, :]
                W_matrix[3, :] = -W_matrix[3, :]
                W_matrix[4, :] = -W_matrix[4, :]
                W_matrix[5, :] = W_matrix[5, :]
                self.reference_frame = 'reverse'

            self.W_matrix = W_matrix
            self._temp_optics_data = None

        self.element_name = element_name

    def _has_deferred_inputs(self):
        return self._temp_co_data is not None or self._temp_optics_data is not None

    def copy(self):
        if self.particle_on_co is not None:
            pco = self.particle_on_co.copy()
        else:
            pco = None

        if self.W_matrix is not None:
            wmat = self.W_matrix.copy()
        else:
            wmat = None

        out =  TwissInit(
            particle_on_co=pco,
            W_matrix=wmat,
            element_name=self.element_name,
            mux=self.mux,
            muy=self.muy,
            muzeta=self.muzeta,
            dzeta=self.dzeta,
            ax_chrom=self.ax_chrom,
            bx_chrom=self.bx_chrom,
            ay_chrom=self.ay_chrom,
            by_chrom=self.by_chrom,
            ddx=self.ddx,
            ddpx=self.ddpx,
            ddy=self.ddy,
            ddpy=self.ddpy,
            reference_frame=self.reference_frame)

        if self._temp_co_data is not None:
            out._temp_co_data = self._temp_co_data.copy()

        if self._temp_optics_data is not None:
            out._temp_optics_data = self._temp_optics_data.copy()

        return out

    def reverse(self):
        out = TwissInit(
            particle_on_co=self.particle_on_co.copy(),
            W_matrix=self.W_matrix.copy(),
            ax_chrom=(-self.ax_chrom if self.ax_chrom is not None else None),
            ay_chrom=(-self.ay_chrom if self.ay_chrom is not None else None),
            bx_chrom=self.bx_chrom,
            by_chrom=self.by_chrom,
            ddx=(-self.ddx if self.ddx is not None else None),
            ddpx=(self.ddpx if self.ddpx is not None else None),
            ddy=(self.ddy if self.ddy is not None else None),
            ddpy=(-self.ddpy if self.ddpy is not None else None),
        )
        out.particle_on_co.x = -out.particle_on_co.x
        out.particle_on_co.py = -out.particle_on_co.py
        out.particle_on_co.zeta = -out.particle_on_co.zeta
        out.particle_on_co.spin_x *= -1
        out.particle_on_co.spin_z *= -1

        out.W_matrix[0, :] = -out.W_matrix[0, :]
        out.W_matrix[1, :] = out.W_matrix[1, :]
        out.W_matrix[2, :] = out.W_matrix[2, :]
        out.W_matrix[3, :] = -out.W_matrix[3, :]
        out.W_matrix[4, :] = -out.W_matrix[4, :]
        out.W_matrix[5, :] = out.W_matrix[5, :]

        out.mux = 0
        out.muy = 0
        out.muzeta = 0
        out.dzeta = 0

        out.element_name = self.element_name
        out.reference_frame = {'proper': 'reverse', 'reverse': 'proper'}[self.reference_frame]

        return out

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif hasattr(self.__dict__['particle_on_co'], name):
            # e.g. tw_init['x'] returns tw_init.particle_on_co.x
            out = getattr(self.__dict__['particle_on_co'], name)
            #always cpu
            if hasattr(out, 'get'):
                out = out.get()
            if hasattr(out, '__iter__'):
                out = out [0]
            return out
        else:
            raise AttributeError(f'No attribute {name} found in TwissInit')

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        elif hasattr(self.particle_on_co, name):
            setattr(self.particle_on_co, name, value)
        else:
            self.__dict__[name] = value

    def get_normalized_coordinates(self, particles, nemitt_x=None, nemitt_y=None,
                                   nemitt_zeta=None):

        ctx2np = particles._context.nparray_from_context_array

        part_id = ctx2np(particles.particle_id).copy()
        at_element = ctx2np(particles.at_element).copy()
        at_turn = ctx2np(particles.at_turn).copy()
        x_norm = ctx2np(particles.x).copy()
        px_norm = x_norm.copy()
        y_norm = x_norm.copy()
        py_norm = x_norm.copy()
        zeta_norm = x_norm.copy()
        pzeta_norm = x_norm.copy()

        XX_norm  = _W_phys2norm(x = ctx2np(particles.x),
                                px = ctx2np(particles.px),
                                y = ctx2np(particles.y),
                                py = ctx2np(particles.py),
                                zeta = ctx2np(particles.zeta),
                                pzeta = ctx2np(particles.ptau)/ctx2np(particles.beta0),
                                W_matrix = self.W_matrix,
                                co_dict = self.particle_on_co.copy(_context=xo.context_default).to_dict(),
                                nemitt_x = nemitt_x,
                                nemitt_y = nemitt_y,
                                nemitt_zeta = nemitt_zeta)

        if XX_norm.ndim == 2:
            x_norm = XX_norm[0, :]
            px_norm = XX_norm[1, :]
            y_norm = XX_norm[2, :]
            py_norm = XX_norm[3, :]
            zeta_norm = XX_norm[4, :]
            pzeta_norm = XX_norm[5, :]

        elif XX_norm.ndim == 3:
            x_norm = XX_norm[0, :, :].flatten()
            px_norm = XX_norm[1, :, :].flatten()
            y_norm = XX_norm[2, :, :].flatten()
            py_norm = XX_norm[3, :, :].flatten()
            zeta_norm = XX_norm[4, :, :].flatten()
            pzeta_norm = XX_norm[5, :, :].flatten()
            part_id = part_id.flatten()
            at_element = at_element.flatten()
            at_turn = at_turn.flatten()

        return Table({'particle_id': part_id, 'at_element': at_element,'at_turn':at_turn,
                      'x_norm': x_norm, 'px_norm': px_norm, 'y_norm': y_norm,
                      'py_norm': py_norm, 'zeta_norm': zeta_norm,
                      'pzeta_norm': pzeta_norm}, index='particle_id')

    @property
    def betx(self):
        WW = self.W_matrix
        return WW[0, 0]**2 + WW[0, 1]**2

    @property
    def bety(self):
        WW = self.W_matrix
        return WW[2, 2]**2 + WW[2, 3]**2

    @property
    def betzeta(self):
        WW = self.W_matrix
        return WW[4, 4]**2 + WW[4, 5]**2

    @property
    def alfx(self):
        WW = self.W_matrix
        return -WW[0, 0] * WW[1, 0] - WW[0, 1] * WW[1, 1]

    @property
    def alfy(self):
        WW = self.W_matrix
        return -WW[2, 2] * WW[3, 2] - WW[2, 3] * WW[3, 3]

    @property
    def alfzeta(self):
        WW = self.W_matrix
        return -WW[4, 4] * WW[5, 4] - WW[4, 5] * WW[5, 5]

    @property
    def dx(self):
        WW = self.W_matrix
        return (WW[0, 5] - WW[0, 4] * WW[4, 5] / WW[4, 4]) / (
                WW[5, 5] - WW[5, 4] * WW[4, 5] / WW[4, 4])

    @property
    def dpx(self):
        WW = self.W_matrix
        return (WW[1, 5] - WW[1, 4] * WW[4, 5] / WW[4, 4]) / (
                WW[5, 5] - WW[5, 4] * WW[4, 5] / WW[4, 4])

    @property
    def dy(self):
        WW = self.W_matrix
        return (WW[2, 5] - WW[2, 4] * WW[4, 5] / WW[4, 4]) / (
                WW[5, 5] - WW[5, 4] * WW[4, 5] / WW[4, 4])

    @property
    def dpy(self):
        WW = self.W_matrix
        return (WW[3, 5] - WW[3, 4] * WW[4, 5] / WW[4, 4]) / (
                WW[5, 5] - WW[5, 4] * WW[4, 5] / WW[4, 4])


def _W_phys2norm(x, px, y, py, zeta, pzeta, W_matrix, co_dict, nemitt_x=None, nemitt_y=None, nemitt_zeta=None):

    # Compute geometric emittances if normalized emittances are provided
    gemitt_x = np.ones(shape=np.shape(co_dict['beta0'])) if nemitt_x is None else (
        nemitt_x / co_dict['beta0'] / co_dict['gamma0'])
    gemitt_y = np.ones(shape=np.shape(co_dict['beta0'])) if nemitt_y is None else (
        nemitt_y / co_dict['beta0'] / co_dict['gamma0'])
    gemitt_zeta = np.ones(shape=np.shape(co_dict['beta0'])) if nemitt_zeta is None else (
        nemitt_zeta / co_dict['beta0'] / co_dict['gamma0'])

    # Prepaing co arrray and gemitt array:
    co = np.array([co_dict['x'], co_dict['px'], co_dict['y'], co_dict['py'],
                  co_dict['zeta'], co_dict['ptau'] / co_dict['beta0']])
    gemitt_values = np.array(
        [gemitt_x, gemitt_x, gemitt_y, gemitt_y, gemitt_zeta, gemitt_zeta])

    # Ensuring consistent dimensions
    for add_axis in range(-1, len(np.shape(x))-len(np.shape(co))):
        co = co[:, np.newaxis]
    for add_axis in range(-1, len(np.shape(x))-len(np.shape(gemitt_values))):
        gemitt_values = gemitt_values[:, np.newaxis]

    # substracting closed orbit
    XX = np.array([x, px, y, py, zeta, pzeta])
    XX -= co

    # Apply the inverse transformation matrix
    W_inv = np.linalg.inv(W_matrix)

    if len(np.shape(XX)) == 3:
        XX_norm = np.dot(W_inv, XX.reshape(6, x.shape[0]*x.shape[1]))
        XX_norm = XX_norm.reshape(6, x.shape[0], x.shape[1])
    else:
        XX_norm = np.dot(W_inv, XX)

    # Normalize the coordinates with the geometric emittances
    XX_norm /= np.sqrt(gemitt_values)

    return XX_norm


def _2d_w_matrix(bet, alf):
    sqrt_bet = np.sqrt(bet)
    return np.array([
        [sqrt_bet,      0.],
        [-alf/sqrt_bet, 1/sqrt_bet]
    ])

def _6d_w_matrix(betx, bety, alfx, alfy, bets, dx, dpx, dy, dpy):

    out = np.eye(6)
    out[0:2, 0:2] = _2d_w_matrix(betx, alfx)
    out[2:4, 2:4] = _2d_w_matrix(bety, alfy)
    out[4:6, 4:6] = _2d_w_matrix(bets, 0)
    out[0, 5] = dx
    out[1, 5] = dpx
    out[2, 5] = dy
    out[3, 5] = dpy
    return out


# Request-level init handling

_PERIODIC_INIT_ARGUMENTS_FROM_BASE_DATA = (
    'line',
    'particle_on_co',
    'particle_ref',
    'method',
    'co_search_settings',
    'continue_on_closed_orbit_error',
    'delta0',
    'zeta0',
    'zeta_shift',
    'steps_R_matrix',
    'W_matrix',
    'R_matrix',
    'co_guess',
    'delta_disp',
    'symplectify',
    'matrix_responsiveness_tol',
    'matrix_stability_tol',
    'num_turns',
    'co_search_at',
    'search_for_t_rev',
    'spin',
    'num_turns_search_t_rev',
    'nemitt_x',
    'nemitt_y',
    'step_W_sigma',
    'compute_R_element_by_element',
    'only_markers',
    'only_orbit',
    'periodic_mode',
    'include_collective',
)


def _build_twiss_init_from_inputs(twiss_config):

    init = twiss_config['init']
    if isinstance(init, TwissInit) and twiss_config['init_at'] is not None:
        init.element_name = twiss_config['init_at']

    if twiss_config['start'] is not None or twiss_config['end'] is not None:
        assert twiss_config['start'] is not None and twiss_config['end'] is not None, (
            'start and end must be provided together')

        if init is None:
            assert twiss_config['betx'] is not None and twiss_config['bety'] is not None, (
                'betx and bety or init must be provided when start '
                'and end are used')
            init_kwargs = {
                name: twiss_config[name]
                for name in VARS_FOR_TWISS_INIT_GENERATION
            }
            init_kwargs.update(
                spin_x=twiss_config['spin_x'],
                spin_y=twiss_config['spin_y'],
                spin_z=twiss_config['spin_z'],
            )
            init = TwissInit(
                element_name=twiss_config['init_at'], **init_kwargs)
        else:
            assert all(
                twiss_config[name] is None
                for name in VARS_FOR_TWISS_INIT_GENERATION)

    if init is not None and not isinstance(init, str):
        assert isinstance(init, TwissInit)
        init = init.copy()  # Do not change the supplied init while completing it.
        if init._has_deferred_inputs():
            assert isinstance(twiss_config['start'], str), (
                'start must be provided as name when an incomplete '
                'init is provided')
            init._finish_initialization(
                line=twiss_config['line'],
                element_name=(init.element_name or twiss_config['start']))

        if init.reference_frame is None:
            init.reference_frame = {
                True: 'reverse', False: 'proper', None: None,
            }[twiss_config['reverse']]

        if twiss_config['reverse'] is not None:
            if init.reference_frame == 'proper':
                assert not twiss_config['reverse'], (
                    '``init`` needs to be given in the proper reference '
                    'frame when ``reverse`` is False')
            elif init.reference_frame == 'reverse':
                assert twiss_config['reverse'] is True, (
                    '``init`` needs to be given in the reverse reference '
                    'frame when ``reverse`` is True')

    completed_init = (init.copy() if hasattr(init, 'copy') else init)
    return init, completed_init


def _clear_twiss_init_input_fields(twiss_config):

    twiss_config['init_at'] = None
    for field_name in (
            *VARS_FOR_TWISS_INIT_GENERATION,
            'spin_x', 'spin_y', 'spin_z'):
        twiss_config[field_name] = None


def _compute_periodic_twiss_init(twiss_config):

    # Local imports avoid a module cycle: periodic_solution imports TwissInit.
    from .periodic_solution import _find_periodic_solution
    from .transfer_matrices import _complete_steps_r_matrix_with_default

    assert twiss_config['periodic']
    if twiss_config['start'] is None and twiss_config['end'] is None:
        periodic_start = periodic_end = None
    else:
        assert twiss_config['start'] is not None and twiss_config['end'] is not None
        if twiss_config['reverse']:
            # Periodic solutions are computed in forward physical order.
            periodic_start, periodic_end = twiss_config['end'], twiss_config['start']
        else:
            periodic_start, periodic_end = twiss_config['start'], twiss_config['end']

    periodic_init_kwargs = {
        name: twiss_config[name]
        for name in _PERIODIC_INIT_ARGUMENTS_FROM_BASE_DATA
    }
    periodic_init_kwargs.update(
        start=periodic_start,
        end=periodic_end,
    )
    assert not twiss_config['_initial_particles']
    periodic_init_kwargs['steps_R_matrix'] = (
        _complete_steps_r_matrix_with_default(
            periodic_init_kwargs['steps_R_matrix']))

    (init, R_matrix, steps_R_matrix, eigenvalues, Rot, RR_ebe
     ) = _find_periodic_solution(**periodic_init_kwargs)

    return {
        'init': init,
        'R_matrix': R_matrix,
        'steps_R_matrix': steps_R_matrix,
        'eigenvalues': eigenvalues,
        'Rot': Rot,
        'RR_ebe': RR_ebe,
    }
