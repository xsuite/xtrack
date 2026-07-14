import numpy as np

from ..base_element import BeamElement
from ..pipeline import PipelineStatus
from .compressed_profile import CompressedProfile
from .uniform import UniformBinSlicer


class ElementWithSlicer(BeamElement):
    """
    Base class for elements with a slicer.

    Parameters
    ----------
    slicer_moments: List
        Moments for the slicer
    zeta_range : Tuple
        Zeta range for each bunch used in the underlying slicer.
    num_slices : int
        Number of slices per bunch used in the underlying slicer.
    bunch_spacing_zeta : float
        Bunch spacing in meters.
    filling_scheme: np.ndarray
        List of zeros and ones representing the filling scheme. The length
        of the array is equal to the number of slots in the machine and each
        element of the array holds a one if the slot is filled or a zero
        otherwise.
    filled_slots: np.ndarray
        Explicit physical slot numbers which are filled.
    bunch_selection: np.ndarray
        List of the bunches indicating which slots from the filling scheme are
        used (not all the bunches are used when using multi-processing)
    num_turns : int
        Number of turns which are consiered for the multi-turn wake.
    circumference: float
        Machine length in meters.
    log_moments: list
        List of moments logged in the slicer.
    _flatten: bool
        Use flattened wakes
    """

    def __init__(self,
                 slicer_moments=None,
                 log_moments=None,
                 zeta_range=None,  # These are [a, b] in the paper
                 num_slices=None,  # Per bunch, this is N_1 in the paper
                 bunch_spacing_zeta=None,  # This is P in the paper
                 filling_scheme=None,
                 filled_slots=None,
                 bunch_selection=None,
                 num_turns=1,
                 circumference=None,
                 with_compressed_profile=False,
                 **kwargs):

        self.xoinitialize(**kwargs)

        self.iscollective = True

        self.with_compressed_profile = with_compressed_profile
        self.pipeline_manager = None

        if slicer_moments is None:
            slicer_moments = ['num_particles']

        self.source_moments = slicer_moments.copy()

        if log_moments is not None:
            slicer_moments += log_moments
        slicer_moments = list(set(slicer_moments))

        self.init_slicer(zeta_range=zeta_range,
                         num_slices=num_slices,
                         filling_scheme=filling_scheme,
                         filled_slots=filled_slots,
                         bunch_selection=bunch_selection,
                         bunch_spacing_zeta=bunch_spacing_zeta,
                         slicer_moments=slicer_moments)

        if with_compressed_profile:
            self._initialize_moments(
                zeta_range=zeta_range,  # These are [a, b] in the paper
                num_slices=num_slices,  # Per bunch, this is N_1 in the paper
                bunch_spacing_zeta=bunch_spacing_zeta,  # This is P in the paper
                filling_scheme=filling_scheme,
                bunch_selection=bunch_selection,
                num_turns=num_turns,
                circumference=circumference)

    @staticmethod
    def _check_filling_scheme_info(filling_scheme, bunch_numbers, num_slots):
        if filling_scheme is None and bunch_numbers is None:
            if num_slots is None:
                num_slots = 1
            filling_scheme = np.ones(num_slots, dtype=np.int64)
            bunch_numbers = np.arange(num_slots, dtype=np.int64)
        else:
            assert (num_slots is None and filling_scheme is not None and
                    bunch_numbers is not None)

        return filling_scheme, bunch_numbers

    def init_slicer(self, zeta_range, num_slices, filling_scheme, filled_slots,
                    bunch_selection, bunch_spacing_zeta, slicer_moments):
        if zeta_range is not None:
            if 'num_particles' in slicer_moments:
                slicer_moments.remove('num_particles')
            self.slicer = UniformBinSlicer(
                zeta_range=zeta_range,
                num_slices=num_slices,
                filling_scheme=filling_scheme,
                filled_slots=filled_slots,
                bunch_selection=bunch_selection,
                bunch_spacing_zeta=bunch_spacing_zeta,
                moments=slicer_moments,
                _context=self._context
            )
        else:
            self.zeta_range = None
            self.slicer = None

    def _initialize_moments(
            self,
            zeta_range=None,  # These are [a, b] in the paper
            num_slices=None,  # Per bunch, this is N_1 in the paper
            bunch_spacing_zeta=None,  # This is P in the paper
            filling_scheme=None,
            bunch_selection=None,
            num_turns=1,
            circumference=None):


        if filling_scheme is not None:
            i_last_bunch = np.where(filling_scheme)[0][-1]
            num_periods = i_last_bunch + 1
        else:
            num_periods = 1
            
        if bunch_selection is None:
            num_targets = num_periods
            first_target_slot = 0
        else:
            slots_in_slicer = self.slicer.filled_slots[bunch_selection]
            first_target_slot = np.min(slots_in_slicer)
            num_targets = 1+ np.max(slots_in_slicer)-first_target_slot

            
        self.moments_data = CompressedProfile(
                moments=self.source_moments + ['result'],
                zeta_range=zeta_range,
                num_slices=num_slices,
                bunch_spacing_zeta=bunch_spacing_zeta,
                first_target_slot = first_target_slot,
                num_targets = num_targets,
                num_periods=num_periods,
                num_turns=num_turns,
                circumference=circumference,
                _context=self.context)

    def init_pipeline(self, pipeline_manager, element_name, partner_names):
        self.pipeline_manager = pipeline_manager
        self.partner_names = partner_names
        self.name = element_name

        self._send_buffer = self.slicer._to_npbuffer()
        self._send_buffer_length = np.zeros(1, dtype=int)
        self._send_buffer_length[0] = len(self._send_buffer)

        self._recv_buffer = np.zeros_like(self._send_buffer)
        self._recv_buffer_length_buffer = np.zeros(1, dtype=int)

    def _slicer_to_buffer(self, slicer):
        self._send_buffer = slicer._to_npbuffer()
        self._send_buffer_length[0] = len(slicer._to_npbuffer())

    def _ensure_recv_buffer_size(self):
        if self._recv_buffer_length_buffer[0] != len(self._recv_buffer):
            self._recv_buffer = np.zeros(self._recv_buffer_length_buffer[0],
                                         dtype=self._recv_buffer.dtype)

    def _slice_set_from_buffer(self):
        return UniformBinSlicer._from_npbuffer(self._recv_buffer)

    def _slice_and_store(self, particles, _slice_result=None):
        if _slice_result is not None:
            self.i_slice_particles = _slice_result['i_slice_particles']
            self.i_slot_particles = _slice_result['i_slot_particles']
            self.slicer = _slice_result['slicer']
        else:
            # Measure slice moments and get slice indeces
            self.i_slice_particles = particles.particle_id * 0 + -999
            self.i_slot_particles = particles.particle_id * 0 + -9999
            self.slicer.slice(particles,
                              i_slice_particles=self.i_slice_particles,
                              i_slot_particles=self.i_slot_particles)

    def _add_slicer_moments_to_moments_data(self, slicer):
        if not self.with_compressed_profile:
            raise RuntimeError('_initialize_conv_data can be called only if'
                               'the sliced element has a CompressedProfile')
        # Set slice moments for fast convolution
        means = {}
        for mm in self.moments_data.moments_names:
            if mm == 'num_particles' or mm == 'result':
                continue
            means[mm] = slicer.mean(mm)

        for i_bunch_in_slicer, bunch_number in enumerate(slicer._xobject.bunch_selection):
            moments_bunch = {}
            for nn in means.keys():
                moments_bunch[nn] = self._context.nplike_lib.atleast_2d(means[nn])[i_bunch_in_slicer, :]
            moments_bunch['num_particles'] = self._context.nplike_lib.atleast_2d(slicer.num_particles)[i_bunch_in_slicer, :]
            self.moments_data.set_moments(
                moments=moments_bunch,
                i_turn=0,
                i_source=slicer._xobject.filled_slots[bunch_number])

    def _update_moments_for_new_turn(self, particles, _slice_result=None):
        if self.with_compressed_profile:
            # Trash oldest turn
            self.moments_data.data[:, 1:, :] = self.moments_data.data[:, :-1, :]
            self.moments_data.data[:, 0, :] = 0

        self._slice_and_store(particles, _slice_result)

        if self.with_compressed_profile:
            self._add_slicer_moments_to_moments_data(self.slicer)

    def track(self, particles, _slice_result=None, _other_bunch_slicers=None):
        if not self.with_compressed_profile:
            self._slice_result = None

        if self.pipeline_manager is None:
            self.other_bunch_slicers = None
            self._update_moments_for_new_turn(particles,
                                              _slice_result=_slice_result)
        else:
            self.other_bunch_slicers = []
            is_ready_to_send = True

            for partner_name in self.partner_names:
                if not self.pipeline_manager.is_ready_to_send(
                        self.name,
                        particles.name,
                        partner_name,
                        particles.at_turn[0],
                        internal_tag=0
                ):
                    is_ready_to_send = False
                    break

            if is_ready_to_send:
                self._update_moments_for_new_turn(particles,
                                                  _slice_result=_slice_result)
                self._slicer_to_buffer(self.slicer)
                for partner_name in self.partner_names:
                    self.pipeline_manager.send_message(
                        self._send_buffer_length,
                        element_name=self.name,
                        sender_name=particles.name,
                        receiver_name=partner_name,
                        turn=particles.at_turn[0],
                        internal_tag=0
                    )
                    self.pipeline_manager.send_message(
                        self._send_buffer,
                        element_name=self.name,
                        sender_name=particles.name,
                        receiver_name=partner_name,
                        turn=particles.at_turn[0],
                        internal_tag=1)

            for partner_name in self.partner_names:
                if not self.pipeline_manager.is_ready_to_receive(
                                        self.name, partner_name,
                                        particles.name, internal_tag=0):
                    return PipelineStatus(on_hold=True)

        if self.pipeline_manager is not None:
            for i_partner, partner_name in enumerate(self.partner_names):
                self.pipeline_manager.receive_message(self._recv_buffer_length_buffer, self.name, partner_name,
                                                      particles.name, internal_tag=0)
                self._ensure_recv_buffer_size()
                self.pipeline_manager.receive_message(self._recv_buffer, self.name, partner_name, particles.name,

                                                      internal_tag=1)
                other_bunch_slicer = self._slice_set_from_buffer()
                if not self.with_compressed_profile:
                    self.other_bunch_slicers.append(other_bunch_slicer)
                else:
                    other_bunch_slicer = self._slice_set_from_buffer()
                    self._add_slicer_moments_to_moments_data(other_bunch_slicer)

        if _other_bunch_slicers is not None and self.with_compressed_profile:
            for other_bunch_slicer in _other_bunch_slicers:
                self._add_slicer_moments_to_moments_data(other_bunch_slicer)

        if not self.with_compressed_profile:
            self._slice_result = {'i_slice_particles': self.i_slice_particles,
                                  'i_slot_particles': self.i_slot_particles,
                                  'slicer': self.slicer}
