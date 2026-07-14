import numpy as np
import xobjects as xo
from ..base_element import BeamElement

class CompressedProfile(BeamElement):
    """
    An object holding a compressed version of the beam data. This allows to
    store the moments of the beam in a compressed way, i.e. avoiding to store
    the moments in the empty slices between bunches.
    The CompressedProfile is used for example in the _ConvData class for the
    computation of the wake kicks.
    The way this is handled is based on the algorithm devised by J. Komppula
    (https://indico.cern.ch/event/735184/contributions/3032237/attachments/1668727/2676169/Multibunch_pyheadtail_algorithms.pdf)
    and N. Mounet (https://indico.cern.ch/event/735184/contributions/3032242/attachments/1668613/2676354/20180615_PyHEADTAIL_convolution_algorithm.pdf).

    Parameters
    ----------.
    moments: List
        Stored moments
    zeta_range : Tuple
        Zeta range for each bunch.
    num_slices : int
        Number of slices per bunch.
    bunch_spacing_zeta : float
        Bunch spacing in meters.
    num_periods: int
        Number of periods in the compressed profile. Concretely, this is the
        number of bunches in the beam.
    num_turns: int
        Number of turns for which the moments are recorded.
    num_targets: int
        Number of target bunches. If it is not specified it is just the same
        as `num_periods`.
    num_slices_target: int
        Number of slices per target bunch. If it is not specified it is just
        the same as `num_bunches`.
    circumference: float
        Machine length in meters.
    """

    _xofields = {
        '_N_aux': xo.Int64,
        '_N_T': xo.Int64,
        '_first_target_slot': xo.Int64,
        'num_turns': xo.Int64,
        'data': xo.Float64[:,:,:],
    }

    _extra_c_sources = ['#include "xtrack/slicers/slicers_src/compressed_profile.h"']

    _per_particle_kernels = {
        '_interp_result': xo.Kernel(
            c_name='CompressedProfile_interp_result',
            args=[
                xo.Arg(xo.Int64, name='data_shape_0'),
                xo.Arg(xo.Int64, name='data_shape_1'),
                xo.Arg(xo.Int64, name='data_shape_2'),
                xo.Arg(xo.Float64, pointer=True, name='data'),
                xo.Arg(xo.Int64, pointer=True, name='i_slot_particles'),
                xo.Arg(xo.Int64, pointer=True, name='i_slice_particles'),
                xo.Arg(xo.Float64, pointer=True, name='out'),
            ]),
        }

    def __init__(self,
                 moments,
                 zeta_range=None,  # These are [a, b] in the paper
                 num_slices=None,  # Per bunch, this is N_1 in the paper
                 bunch_spacing_zeta=None,  # This is P in the paper
                 num_periods=None,
                 num_turns=1,
                 first_target_slot = 0,
                 num_targets=None,
                 num_slices_target=None,
                 circumference=None,
                 **kwargs,
                 ):

        if '_xobject' in kwargs.keys():
            self.xoinitialize(**kwargs)
            return

        if num_turns > 1:
            assert circumference is not None, (
                'circumference must be specified if num_turns > 1')

        self.circumference = circumference
        # the following needs to be generalized when the first bucket is not filled
        self.dz = (np.atleast_2d(zeta_range)[0, 1] -
                   np.atleast_2d(zeta_range)[0, 0]) / num_slices  # h in the paper
        self._z_a = np.atleast_2d(zeta_range)[0, 0]
        self._z_b = np.atleast_2d(zeta_range)[0, -1]

        self._N_1 = num_slices  # N_1 in the
        self._z_P = bunch_spacing_zeta  # P in the paper
        self._N_S = num_periods  # N_S in the paper

        if num_slices_target is not None:
            self._N_2 = num_slices_target
        else:
            self._N_2 = self._N_1

        if num_targets is not None:
            _N_T = num_targets
        else:
            _N_T = self._N_S

        self._BB = 1  # B in the paper
        # (for now we assume that B=0 is the first bunch in time
        # and the last one in zeta)
        self._AA = self._BB - self._N_S

        _N_aux = self._N_1 + self._N_2  # n_aux in the paper

        # Compute m_aux
        self._M_aux = (self._N_S +
                       _N_T - 1) * _N_aux  # m_aux in the paper

        self.moments_names = moments

        self.xoinitialize(_N_T=int(_N_T),_first_target_slot=int(first_target_slot), _N_aux=int(_N_aux), num_turns=int(num_turns), 
                          data=(len(moments), int(num_turns), int(self._M_aux)),
                          **kwargs)


    def __getitem__(self, key):
        assert isinstance(key, str), 'other modes not supported yet'
        assert key in self.moments_names, (
            f'Moment {key} not in defined moments_names')
        i_moment = self.moments_names.index(key)
        return self.data[i_moment]

    def __setitem__(self, key, value):
        self[key][:] = value

    @property
    def num_slices(self):
        return self._N_1

    @property
    def num_periods(self):
        return self._N_S

    @property
    def z_period(self):
        return self._z_P

    def set_all_beam_moments(self,moment_name,i_turn,all_beam_moments):
        """
        Set the moments for a all sources at a given turn.

        Parameters
        ----------
        moment_name : str
            item in self.moments_names
        i_turn : int
            The turn index, 0 <= i_turn < self.num_turns
        all_beam_moments : array
            Array with with slice moments with shape (n_sources, n_slices)
        """
        assert self._context.nplike_lib.shape(all_beam_moments)[0] == self._N_S
        assert self._context.nplike_lib.shape(all_beam_moments)[1] == self._N_1
        assert moment_name in self.moments_names
        assert np.isscalar(i_turn)
        i_moment = self.moments_names.index(moment_name)
        all_beam_moments = self._context.nplike_lib.hstack([all_beam_moments,self._context.nplike_lib.zeros((self._context.nplike_lib.shape(all_beam_moments)[0],self._N_aux-self._N_1))])
        all_beam_moments = self._context.nplike_lib.flip(all_beam_moments,axis=0)
        all_beam_moments = self._context.nplike_lib.ravel(all_beam_moments)
        self.data[i_moment, i_turn, :] = all_beam_moments

    def set_moments(self, i_source, i_turn, moments):
        """
        Set the moments for a given source and turn.

        Parameters
        ----------
        i_source : int
            The source index, 0 <= i_source < self.num_periods
        i_turn : int
            The turn index, 0 <= i_turn < self.num_turns
        moments : dict
            A dictionary of the form {moment_name: moment_value}

        """
        assert np.isscalar(i_source)
        assert np.isscalar(i_turn)

        assert i_source < self._N_S
        assert i_source >= 0

        assert i_turn < self.num_turns
        assert i_turn >= 0

        for nn, vv in moments.items():
            assert nn in self.moments_names, (
                f'Moment {nn} not in defined moments_names')
            assert len(vv) == self._N_1, (
                f'Length of moment {nn} is not equal to num_slices')
            i_moment = self.moments_names.index(nn)
            i_start_in_moments_data = (self._N_S - i_source - 1) * self._N_aux
            i_end_in_moments_data = i_start_in_moments_data + self._N_1
            self.data[i_moment, i_turn,
                    i_start_in_moments_data:i_end_in_moments_data] = vv

    def get_moment_profile(self, moment_name, i_turn):
        """
        Get the moment profile for a given turn.

        Parameters
        ----------
        moment_name : str
            The name of the moment to get
        i_turn : int
            The turn index, 0 <= i_turn < self.num_turns

        Returns
        -------
        z_out : np.ndarray
            The z positions within the moment profile
        moment_out : np.ndarray
            The moment profile
        """

        z_out = self._context.nplike_lib.zeros(self._N_S * self._N_1)
        moment_out = self._context.nplike_lib.zeros(self._N_S * self._N_1)
        i_moment = self.moments_names.index(moment_name)
        _z_P = self._z_P or 0
        for i_source in range(self._N_S):
            i_start_out = (self._N_S - (i_source + 1)) * self._N_1
            i_end_out = i_start_out + self._N_1
            z_out[i_start_out:i_end_out] = (
                self._z_a + self.dz / 2
                - i_source * _z_P + self.dz * self._context.nplike_lib.arange(self._N_1))

            i_start_in_moments_data = (self._N_S - i_source - 1) * self._N_aux
            i_end_in_moments_data = i_start_in_moments_data + self._N_1
            moment_out[i_start_out:i_end_out] = (
                self.data[i_moment, i_turn,
                          i_start_in_moments_data:i_end_in_moments_data])

        return z_out, moment_out

    def get_source_moment_profile(self, moment_name, i_turn,i_source):
        """
        Get the moment profile for a given turn.

        Parameters
        ----------
        moment_name : str
            The name of the moment to get
        i_turn : int
            The turn index, 0 <= i_turn < self.num_turns

        Returns
        -------
        z_out : np.ndarray
            The z positions within the moment profile
        moment_out : np.ndarray
            The moment profile
        """

        z_out = self._context.nplike_lib.zeros(self._N_1)
        moment_out = self._context.nplike_lib.zeros(self._N_1)
        i_moment = self.moments_names.index(moment_name)
        _z_P = self._z_P or 0

        z_out = (
            self._z_a + self.dz / 2
            - i_source * _z_P + self.dz * self._context.nplike_lib.arange(self._N_1))

        i_start_in_moments_data = (self._N_S - i_source - 1) * self._N_aux
        i_end_in_moments_data = i_start_in_moments_data + self._N_1
        moment_out = (
            self.data[i_moment, i_turn,
                      i_start_in_moments_data:i_end_in_moments_data])

        return z_out, moment_out
