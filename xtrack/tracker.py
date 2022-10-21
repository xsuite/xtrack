# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import logging
from functools import partial

from .general import _pkg_root
from .line_frozen import LineFrozen
from .base_element import _handle_per_particle_blocks
from .twiss import (twiss_from_tracker,
                                 compute_one_turn_matrix_finite_differences,
                                 find_closed_orbit, match_tracker
                                )
from .survey import survey_from_tracker
from .internal_record import (new_io_buffer,
                             start_internal_logging_for_elements_of_type,
                             stop_internal_logging_for_elements_of_type)
from .pipeline import PipelineStatus

import xobjects as xo
import xpart as xp

from .beam_elements import Drift
from .line import Line

from . import linear_normal_form as lnf

logger = logging.getLogger(__name__)

def _check_is_collective(ele):
    iscoll = not hasattr(ele, 'iscollective') or ele.iscollective
    return iscoll

class Tracker:

    def __init__(
        self,
        _context=None,
        _buffer=None,
        _offset=None,
        line=None,
        sequence=None,
        track_kernel=None,
        element_classes=None,
        particles_class=None,
        skip_end_turn_actions=False,
        reset_s_at_end_turn=True,
        particles_monitor_class=None,
        global_xy_limit=1.0,
        extra_headers=(),
        local_particle_src=None,
        save_source_as=None,
        io_buffer=None,
        compile=True,
        enable_pipeline_hold=False,
    ):

        if sequence is not None:
            raise ValueError(
                    "`Tracker(... sequence=... ) is deprecated use `line=`)")

        # Check if there are collective elements
        self.iscollective = False
        for ee in line.elements:
            if _check_is_collective(ee):
                self.iscollective = True
                break

        if self.iscollective:
            self._init_track_with_collective(
                _context=_context,
                _buffer=_buffer,
                _offset=_offset,
                line=line,
                track_kernel=track_kernel,
                element_classes=element_classes,
                particles_class=particles_class,
                skip_end_turn_actions=skip_end_turn_actions,
                reset_s_at_end_turn=reset_s_at_end_turn,
                particles_monitor_class=particles_monitor_class,
                global_xy_limit=global_xy_limit,
                extra_headers=extra_headers,
                local_particle_src=local_particle_src,
                save_source_as=save_source_as,
                io_buffer=io_buffer,
                compile=compile,
                enable_pipeline_hold=enable_pipeline_hold)
        else:
            self._init_track_no_collective(
                _context=_context,
                _buffer=_buffer,
                _offset=_offset,
                line=line,
                track_kernel=track_kernel,
                element_classes=element_classes,
                particles_class=particles_class,
                skip_end_turn_actions=skip_end_turn_actions,
                reset_s_at_end_turn=reset_s_at_end_turn,
                particles_monitor_class=particles_monitor_class,
                global_xy_limit=global_xy_limit,
                extra_headers=extra_headers,
                local_particle_src=local_particle_src,
                save_source_as=save_source_as,
                io_buffer=io_buffer,
                compile=compile,
                enable_pipeline_hold=enable_pipeline_hold)

        self.matrix_responsiveness_tol = lnf.DEFAULT_MATRIX_RESPONSIVENESS_TOL
        self.matrix_stability_tol = lnf.DEFAULT_MATRIX_STABILITY_TOL

    def _init_track_with_collective(
        self,
        _context=None,
        _buffer=None,
        _offset=None,
        line=None,
        track_kernel=None,
        element_classes=None,
        particles_class=None,
        skip_end_turn_actions=False,
        reset_s_at_end_turn=True,
        particles_monitor_class=None,
        global_xy_limit=1.0,
        extra_headers=(),
        local_particle_src=None,
        save_source_as=None,
        io_buffer=None,
        compile=True,
        enable_pipeline_hold=False
    ):

        assert _offset is None

        if not compile:
            raise NotImplementedError("Skip compilation is not implemented in "
                                      "collective mode")

        self.skip_end_turn_actions = skip_end_turn_actions
        self.particles_class = particles_class
        self.global_xy_limit = global_xy_limit
        self.extra_headers = extra_headers
        self.local_particle_src = local_particle_src
        self.save_source_as = save_source_as
        self._enable_pipeline_hold = enable_pipeline_hold

        if _buffer is None:
            if _context is None:
                _context = xo.context_default
            _buffer = _context.new_buffer()
        self._buffer = _buffer

        if io_buffer is None:
            io_buffer = new_io_buffer(_context=_buffer.context)
        self.io_buffer = io_buffer

        # Split the sequence
        parts = []
        part_names = []
        _element_part = []
        _element_index_in_part=[]
        this_part = Line(elements=[], element_names=[])
        ii_in_part = 0
        i_part = 0
        for nn, ee in zip(line.element_names, line.elements):
            if not _check_is_collective(ee):
                this_part.append_element(ee, nn)
                _element_part.append(i_part)
                _element_index_in_part.append(ii_in_part)
                ii_in_part += 1
            else:
                if len(this_part.elements)>0:
                    this_part.iscollective=False
                    parts.append(this_part)
                    part_names.append(f'part_{i_part}_non_collective')
                    i_part += 1
                parts.append(ee)
                part_names.append(nn)
                _element_part.append(i_part)
                _element_index_in_part.append(None)
                i_part += 1
                this_part = Line(elements=[], element_names=[])
                ii_in_part = 0
        if len(this_part.elements)>0:
            this_part.iscollective=False
            parts.append(this_part)
            part_names.append(f'part_{i_part}_non_collective')

        # Transform non collective elements into xtrack elements
        noncollective_xelements = []
        for ii, pp in enumerate(parts):
            if not _check_is_collective(pp):
                tempxtline = LineFrozen(_buffer=_buffer,
                                   line=pp)
                pp.element_dict = dict(zip(
                    tempxtline.element_names, tempxtline.elements))
                pp.element_names = tempxtline.element_names
                noncollective_xelements += pp.elements
            else:
                if hasattr(pp, 'isthick') and pp.isthick:
                    ldrift = pp.length
                else:
                    ldrift = 0.

                noncollective_xelements.append(
                    Drift(_buffer=_buffer, length=ldrift))

        # Build tracker for all non collective elements
        # (with collective elements replaced by Drifts)
        supertracker = Tracker(_buffer=_buffer,
                line=Line(elements=noncollective_xelements,
                          element_names=line.element_names),
                track_kernel=track_kernel,
                element_classes=element_classes,
                particles_class=particles_class,
                particles_monitor_class=particles_monitor_class,
                global_xy_limit=global_xy_limit,
                extra_headers=extra_headers,
                reset_s_at_end_turn=reset_s_at_end_turn,
                local_particle_src=local_particle_src,
                save_source_as=save_source_as,
                io_buffer=self.io_buffer
                )

        # Build trackers for non collective parts
        for ii, pp in enumerate(parts):
            if not _check_is_collective(pp):
                parts[ii] = Tracker(_buffer=_buffer,
                                line=pp,
                                element_classes=supertracker.element_classes,
                                track_kernel=supertracker.track_kernel,
                                particles_class=particles_class,
                                particles_monitor_class=particles_monitor_class,
                                global_xy_limit=global_xy_limit,
                                extra_headers=extra_headers,
                                local_particle_src=local_particle_src,
                                skip_end_turn_actions=True,
                                io_buffer=self.io_buffer)

        # Make a "marker" element to increase at_element
        self._zerodrift = Drift(_context=_buffer.context, length=0)

        assert len(line.element_names) == len(supertracker.line.element_names)
        assert len(line.element_names) == len(_element_index_in_part)
        assert len(line.element_names) == len(_element_part)
        assert _element_part[-1] == len(parts) - 1

        self.line = line
        self.num_elements = len(line.element_names)
        self._supertracker = supertracker
        self._parts = parts
        self._part_names = part_names
        self.track = self._track_with_collective
        self.particles_class = supertracker.particles_class
        self.particles_monitor_class = supertracker.particles_monitor_class
        self._element_part = _element_part
        self._element_index_in_part = _element_index_in_part


    def _init_track_no_collective(
        self,
        _context=None,
        _buffer=None,
        _offset=None,
        line=None,
        track_kernel=None,
        element_classes=None,
        particles_class=None,
        skip_end_turn_actions=False,
        reset_s_at_end_turn=True,
        particles_monitor_class=None,
        global_xy_limit=1.0,
        extra_headers=(),
        local_particle_src=None,
        save_source_as=None,
        io_buffer=None,
        compile=True,
        enable_pipeline_hold=False
    ):

        assert not(enable_pipeline_hold), (
            "enable_pipeline_hold is not implemented in non collective mode")
        self._enable_pipeline_hold = False

        if particles_class is None:
            particles_class = xp.Particles

        if particles_monitor_class is None:
            import xtrack as xt  # I have to do it like this
                                 # to avoid circular import #TODO to be solved
            particles_monitor_class = xt.ParticlesMonitor

        if local_particle_src is None:
            local_particle_src = xp.gen_local_particle_api()

        self.global_xy_limit = global_xy_limit
        self.extra_headers = extra_headers

        frozenline = LineFrozen(
                    _context=_context, _buffer=_buffer, _offset=_offset,
                    line=line)

        context = frozenline._buffer.context

        if io_buffer is None:
            io_buffer = new_io_buffer(_context=context)
        self.io_buffer = io_buffer

        if track_kernel is None:
            # Kernel relies on element_classes ordering
            assert element_classes is None

        if element_classes is None:
            # Kernel relies on element_classes ordering
            assert track_kernel=='skip' or track_kernel is None
            element_classes = frozenline._ElementRefClass._reftypes + [
                particles_monitor_class._XoStruct,
            ]

        line._freeze()
        self.line = line
        self.line.tracker = self
        self._line_frozen = frozenline
        ele_offsets = np.array(
            [ee._offset for ee in frozenline.elements], dtype=np.int64)
        ele_typeids = np.array(
            [element_classes.index(ee._xobject.__class__)
                for ee in frozenline.elements], dtype=np.int64)
        ele_offsets_dev = context.nparray_to_context_array(ele_offsets)
        ele_typeids_dev = context.nparray_to_context_array(ele_typeids)

        self.particles_class = particles_class
        self.particles_monitor_class = particles_monitor_class
        self.ele_offsets_dev = ele_offsets_dev
        self.ele_typeids_dev = ele_typeids_dev
        self.num_elements = len(frozenline.elements)
        self.global_xy_limit = global_xy_limit
        self.extra_headers = extra_headers
        self.skip_end_turn_actions = skip_end_turn_actions
        self.reset_s_at_end_turn = reset_s_at_end_turn
        self.local_particle_src = local_particle_src
        self.element_classes = element_classes
        self._buffer = frozenline._buffer

        if track_kernel == 'skip':
            self.track_kernel = None
        elif track_kernel is None:
            self._build_kernel(save_source_as, compile=compile)
        else:
            self.track_kernel = track_kernel

        self.track=self._track_no_collective

    def _invalidate(self):
        if self.iscollective:
            self._invalidated_parts  = self._parts
            self._parts = None
        else:
            self._invalidated_line_frozen = self._line_frozen
            self._line_frozen = None
        self._is_invalidated = True

    def _check_invalidated(self):
        if hasattr(self, '_is_invalidated') and self._is_invalidated:
            raise RuntimeError(
                "This tracker is not anymore valid, most probably because the corresponding line has been unfrozen. "
                "Please rebuild the tracker, for example using `line.build_tracker(...)`.")

    def find_closed_orbit(self, particle_co_guess=None, particle_ref=None,
                          co_search_settings={}, delta_zeta=0, delta0=None,
                          continue_on_closed_orbit_error=False):

        self._check_invalidated()

        if particle_ref is None and particle_co_guess is None:
            particle_ref = self.particle_ref

        if self.iscollective:
            logger.warning(
                'The tracker has collective elements.\n'
                'In the twiss computation collective elements are'
                ' replaced by drifts')
            tracker = self._supertracker
        else:
            tracker = self

        return find_closed_orbit(tracker, particle_co_guess=particle_co_guess,
                                 particle_ref=particle_ref, delta0=delta0,
                                 co_search_settings=co_search_settings, delta_zeta=delta_zeta,
                                 continue_on_closed_orbit_error=continue_on_closed_orbit_error)

    def compute_one_turn_matrix_finite_differences(
            self, particle_on_co,
            steps_r_matrix=None):

        self._check_invalidated()

        if self.iscollective:
            logger.warning(
                'The tracker has collective elements.\n'
                'In the twiss computation collective elements are'
                ' replaced by drifts')
            tracker = self._supertracker
        else:
            tracker = self
        return compute_one_turn_matrix_finite_differences(tracker, particle_on_co,
                                                   steps_r_matrix)

    def twiss(self, particle_ref=None, delta0=None, method='6d',
        r_sigma=0.01, nemitt_x=1e-6, nemitt_y=1e-6,
        delta_disp=1e-5, delta_chrom=1e-4,
        particle_co_guess=None, R_matrix=None, W_matrix=None,
        steps_r_matrix=None, co_search_settings=None, at_elements=None, at_s=None,
        values_at_element_exit=False,
        continue_on_closed_orbit_error=False,
        eneloss_and_damping=False,
        ele_start=0, ele_stop=None, twiss_init=None,
        particle_on_co=None,
        matrix_responsiveness_tol=None,
        matrix_stability_tol=None,
        symplectify=False
        ):

        self._check_invalidated()

        kwargs = locals().copy()
        kwargs.pop('self')

        return twiss_from_tracker(self, **kwargs)

    def survey(self,X0=0,Y0=0,Z0=0,theta0=0,phi0=0,psi0=0):
        return survey_from_tracker(self,X0=X0,Y0=Y0,Z0=Z0,theta0=theta0,phi0=phi0,psi0=psi0)

    def match(self, vary, targets, **kwargs):
        return match_tracker(self, vary, targets, **kwargs)

    def filter_elements(self, mask=None, exclude_types_starting_with=None):

        self._check_invalidated()

        return self.__class__(
                 _buffer=self._buffer,
                 line=self.line.filter_elements(mask=mask,
                     exclude_types_starting_with=exclude_types_starting_with),
                 track_kernel=(self.track_kernel if not self.iscollective
                                    else self._supertracker.track_kernel),
                 element_classes=(self.element_classes if not self.iscollective
                                    else self._supertracker.element_classes))

    def cycle(self, index_first_element=None, name_first_element=None,
              _buffer=None, _context=None):

        self._check_invalidated()

        cline = self.line.cycle(index_first_element=index_first_element,
                                name_first_element=name_first_element)

        if _buffer is None:
            if _context is None:
                _buffer = self._buffer
            else:
                _buffer = _context.new_buffer()

        return self.__class__(
                _buffer=_buffer,
                line=cline,
                track_kernel=(self.track_kernel if not self.iscollective
                                    else self._supertracker.track_kernel),
                element_classes=(self.element_classes if not self.iscollective
                                    else self._supertracker.element_classes),
                particles_class=self.particles_class,
                skip_end_turn_actions=self.skip_end_turn_actions,
                particles_monitor_class=self.particles_monitor_class,
                global_xy_limit=self.global_xy_limit,
                extra_headers=self.extra_headers,
                local_particle_src=self.local_particle_src,
            )

    def get_backtracker(self, _context=None, _buffer=None,
                        global_xy_limit='from_tracker'):

        self._check_invalidated()

        assert not self.iscollective

        if _buffer is None:
            if _context is None:
                _context = self._buffer.context
            _buffer = _context.new_buffer()

        line = Line(elements=[], element_names=[])
        for nn, ee in zip(self.line.element_names[::-1],
                          self.line.elements[::-1]):
            line.append_element(
                    ee.get_backtrack_element(_buffer=_buffer), nn)

        if global_xy_limit == 'from_tracker':
            global_xy_limit = self.global_xy_limit
            track_kernel = self.track_kernel
            element_classes = self.element_classes
        else:
            track_kernel = None
            element_classes = None

        return self.__class__(
                    _buffer=_buffer,
                    line=line,
                    track_kernel=track_kernel,
                    element_classes=element_classes,
                    particles_class=self.particles_class,
                    skip_end_turn_actions=self.skip_end_turn_actions,
                    particles_monitor_class=self.particles_monitor_class,
                    global_xy_limit=global_xy_limit,
                    extra_headers=self.extra_headers,
                    local_particle_src=self.local_particle_src,
                )

    @property
    def particle_ref(self):
        self._check_invalidated()
        return self.line.particle_ref

    @property
    def vars(self):
        self._check_invalidated()
        return self.line.vars

    @property
    def element_refs(self):
        self._check_invalidated()
        return self.line.element_refs

    @property
    def enable_pipeline_hold(self):
        return self._enable_pipeline_hold

    @enable_pipeline_hold.setter
    def enable_pipeline_hold(self, value):
        if not self.iscollective:
            raise ValueError(
                'enable_pipeline_hold is not supported non collective trackers')
        else:
            self._enable_pipeline_hold = value

    @property
    def _context(self):
        return self._buffer.context

    def configure_radiation(self, mode=None):

        self._check_invalidated()

        self.line.configure_radiation(mode=mode)

    def _build_kernel(self, save_source_as, compile):

        context = self._line_frozen._buffer.context

        kernels = {}
        headers = []

        headers.extend(self.extra_headers)

        if self.global_xy_limit is not None:
            headers.append(
                f"#define XTRACK_GLOBAL_POSLIMIT ({self.global_xy_limit})")
        headers.append(_pkg_root.joinpath("headers/constants.h"))

        src_lines = []
        src_lines.append(
            r"""
            /*gpukern*/
            void track_line(
                /*gpuglmem*/ int8_t* buffer,
                /*gpuglmem*/ int64_t* ele_offsets,
                /*gpuglmem*/ int64_t* ele_typeids,
                             ParticlesData particles,
                             int num_turns,
                             int ele_start,
                             int num_ele_track,
                             int flag_end_turn_actions,
                             int flag_reset_s_at_end_turn,
                             int flag_monitor,
                /*gpuglmem*/ int8_t* buffer_tbt_monitor,
                             int64_t offset_tbt_monitor,
                /*gpuglmem*/ int8_t* io_buffer){


            LocalParticle lpart;
            lpart.io_buffer = io_buffer;

            int64_t part_id = 0;                    //only_for_context cpu_serial cpu_openmp
            int64_t part_id = blockDim.x * blockIdx.x + threadIdx.x; //only_for_context cuda
            int64_t part_id = get_global_id(0);                    //only_for_context opencl


            /*gpuglmem*/ int8_t* tbt_mon_pointer =
                            buffer_tbt_monitor + offset_tbt_monitor;
            ParticlesMonitorData tbt_monitor =
                            (ParticlesMonitorData) tbt_mon_pointer;

            int64_t part_capacity = ParticlesData_get__capacity(particles);
            if (part_id<part_capacity){
            Particles_to_LocalParticle(particles, &lpart, part_id);

            int64_t isactive = check_is_active(&lpart);

            for (int64_t iturn=0; iturn<num_turns; iturn++){

                if (!isactive){
                    break;
                }

                if (flag_monitor==1){
                    ParticlesMonitor_track_local_particle(tbt_monitor, &lpart);
                }

                for (int64_t ee=ele_start; ee<ele_start+num_ele_track; ee++){

                        if (flag_monitor==2){
                            ParticlesMonitor_track_local_particle(tbt_monitor, &lpart);
                        }

                        /*gpuglmem*/ int8_t* el = buffer + ele_offsets[ee];
                        int64_t ee_type = ele_typeids[ee];

                        switch(ee_type){
        """
        )

        for ii, cc in enumerate(self.element_classes):
            ccnn = cc.__name__.replace("Data", "")
            src_lines.append(
                f"""
                        case {ii}:
"""
            )
            if ccnn == "Drift":
                src_lines.append(
                    """
                            #ifdef XTRACK_GLOBAL_POSLIMIT
                            global_aperture_check(&lpart);
                            #endif

                            """
                )
            src_lines.append(
                f"""
                            {ccnn}_track_local_particle(({ccnn}Data) el, &lpart);
                            break;"""
            )

        src_lines.append(
            """
                        } //switch
                    isactive = check_is_active(&lpart);
                    if (!isactive){
                        break;
                    }
                    increment_at_element(&lpart);
                } // for elements
                if (flag_monitor==2){
                    // End of turn (element-by-element mode)
                    ParticlesMonitor_track_local_particle(tbt_monitor, &lpart);
                }
                if (flag_end_turn_actions>0){
                    if (isactive){
                        increment_at_turn(&lpart, flag_reset_s_at_end_turn);
                    }
                }
            } // for turns

            LocalParticle_to_Particles(&lpart, particles, part_id, 1);

            }// if partid
        }//kernel
        """
        )

        source_track = "\n".join(src_lines)

        kernel_descriptions = {
            "track_line": xo.Kernel(
                args=[
                    xo.Arg(xo.Int8, pointer=True, name="buffer"),
                    xo.Arg(xo.Int64, pointer=True, name="ele_offsets"),
                    xo.Arg(xo.Int64, pointer=True, name="ele_typeids"),
                    xo.Arg(self.particles_class._XoStruct, name="particles"),
                    xo.Arg(xo.Int32, name="num_turns"),
                    xo.Arg(xo.Int32, name="ele_start"),
                    xo.Arg(xo.Int32, name="num_ele_track"),
                    xo.Arg(xo.Int32, name="flag_end_turn_actions"),
                    xo.Arg(xo.Int32, name="flag_reset_s_at_end_turn"),
                    xo.Arg(xo.Int32, name="flag_monitor"),
                    xo.Arg(xo.Int8, pointer=True, name="buffer_tbt_monitor"),
                    xo.Arg(xo.Int64, name="offset_tbt_monitor"),
                    xo.Arg(xo.Int8, pointer=True, name="io_buffer"),
                ],
            )
        }

        # Internal API can be exposed only on CPU
        if not isinstance(context, xo.ContextCpu):
            kernels = {}
        kernels.update(kernel_descriptions)

        # Random number generator init kernel
        kernels.update(self.particles_class._kernels)

        # Compile!
        context.add_kernels(
            [source_track],
            kernels,
            extra_headers=headers,
            extra_classes=self.element_classes,
            apply_to_source=[
                partial(_handle_per_particle_blocks,
                        local_particle_src=self.local_particle_src)],
            save_source_as=save_source_as,
            specialize=True,
            compile=compile
        )

        self.track_kernel = context.kernels.track_line

    def _prepare_collective_track_session(self, particles, ele_start, ele_stop,
                                       num_elements, num_turns, turn_by_turn_monitor):

        # Start position
        if particles.start_tracking_at_element >= 0:
            if ele_start != 0:
                raise ValueError("The argument ele_start is used, but particles.start_tracking_at_element is set as well. "
                                + "Please use only one of those methods.")
            ele_start = particles.start_tracking_at_element
            particles.start_tracking_at_element = -1
        if isinstance(ele_start,str):
            ele_start = self.line.element_names.index(ele_start)

        # ele_start can only have values of existing element id's,
        # but also allowed: all elements+1 (to perform end-turn actions)
        assert ele_start >= 0
        assert ele_start < self.num_elements

        # Stop position
        if num_elements is not None:
            # We are using ele_start and num_elements
            if ele_stop is not None:
                raise ValueError("Cannot use both num_elements and ele_stop!")
            if num_turns is not None:
                raise ValueError("Cannot use both num_elements and num_turns!")
            num_turns, ele_stop = np.divmod(ele_start + num_elements, self.num_elements)
            if ele_stop == 0:
                ele_stop = None
            else:
                num_turns += 1
        else:
            # We are using ele_start, ele_stop, and num_turns
            if num_turns is None:
                num_turns = 1
            else:
                assert num_turns > 0
            if isinstance(ele_stop,str):
                ele_stop = self.line.element_names.index(ele_stop)

            # If ele_stop comes before ele_start, we need to add an additional turn to
            # reach the required ele_stop
            if ele_stop == 0:
                ele_stop = None

            if ele_stop is not None and ele_stop <= ele_start:
                num_turns += 1

        if ele_stop is not None:
            assert ele_stop >= 0
            assert ele_stop < self.num_elements

        assert num_turns >= 1

        assert turn_by_turn_monitor != 'ONE_TURN_EBE', (
            "Element-by-element monitor not available in collective mode")

        (flag_monitor, monitor, buffer_monitor, offset_monitor
            ) = self._get_monitor(particles, turn_by_turn_monitor, num_turns)

        if particles._num_active_particles < 0:
            _context_needs_clean_active_lost_state = True
        else:
            _context_needs_clean_active_lost_state = False

        if self.line._needs_rng and not particles._has_valid_rng_state():
            particles._init_random_number_generator()

        return (ele_start, ele_stop, num_turns, flag_monitor, monitor,
                buffer_monitor, offset_monitor,
                _context_needs_clean_active_lost_state)

    def _prepare_particles_for_part(self, particles, pp,
                                    moveback_to_buffer, moveback_to_offset,
                                    _context_needs_clean_active_lost_state):
        if hasattr(self, '_slice_sets'):
            # If pyheadtail object, remove any stored slice sets
            # (they are made invalid by the xtrack elements changing zeta)
            self._slice_sets = {}

        if (hasattr(pp, 'needs_cpu') and pp.needs_cpu):
            # Move to CPU if not already there
            if (moveback_to_buffer is None
                and not isinstance(particles._buffer.context, xo.ContextCpu)):
                moveback_to_buffer = particles._buffer
                moveback_to_offset = particles._offset
                particles.move(_context=xo.ContextCpu())
                particles.reorganize()
        else:
            # Move to GPU if not already there
            if moveback_to_buffer is not None:
                particles.move(_buffer=moveback_to_buffer, _offset=moveback_to_offset)
                moveback_to_buffer = None
                moveback_to_offset = None
                if _context_needs_clean_active_lost_state:
                    particles._num_active_particles = -1
                    particles._num_lost_particles = -1

        # Hide lost particles if required by element
        _need_unhide_lost_particles = False
        if (hasattr(pp, 'needs_hidden_lost_particles')
            and pp.needs_hidden_lost_particles):
            if not particles.lost_particles_are_hidden:
                _need_unhide_lost_particles = True
            particles.hide_lost_particles()

        return _need_unhide_lost_particles, moveback_to_buffer, moveback_to_offset

    def _track_part(self, particles, pp, tt, ipp, ele_start, ele_stop, num_turns):

        ret = None
        skip = False
        stop_tracking = False
        if tt == 0 and ipp < self._element_part[ele_start]:
            # Do not track before ele_start in the first turn
            skip = True

        elif tt == 0 and self._element_part[ele_start] == ipp:
            # We are in the part that contains the start element
            i_start_in_part = self._element_index_in_part[ele_start]
            if i_start_in_part is None:
                # The start part is collective
                ret = pp.track(particles)
            else:
                # The start part is a non-collective tracker
                if (ele_stop is not None
                    and tt == num_turns - 1 and self._element_part[ele_stop] == ipp):
                    # The stop element is also in this part, so track until ele_stop
                    i_stop_in_part = self._element_index_in_part[ele_stop]
                    ret = pp.track(particles, ele_start=i_start_in_part, ele_stop=i_stop_in_part)
                    stop_tracking = True
                else:
                    # Track until end of part
                    ret = pp.track(particles, ele_start=i_start_in_part)

        elif (ele_stop is not None
                and tt == num_turns-1 and self._element_part[ele_stop] == ipp):
            # We are in the part that contains the stop element
            i_stop_in_part = self._element_index_in_part[ele_stop]
            if i_stop_in_part is not None:
                # If not collective, track until ele_stop
                ret = pp.track(particles, num_elements=i_stop_in_part)
            stop_tracking = True

        else:
            # We are in between the part that contains the start element,
            # and the one that contains the stop element, so track normally
            ret = pp.track(particles)

        return stop_tracking, skip, ret

    def resume(self, session):
        return self._track_with_collective(particles=None, _session_to_resume=session)


    def _track_with_collective(
        self,
        particles,
        ele_start=0,
        ele_stop=None,     # defaults to full lattice
        num_elements=None, # defaults to full lattice
        num_turns=None,    # defaults to 1
        turn_by_turn_monitor=None,
        _session_to_resume=None
    ):

        self._check_invalidated()

        if (isinstance(self._buffer.context, xo.ContextCpu)
            and _session_to_resume is None):
            assert (particles._num_active_particles >= 0 and
                    particles._num_lost_particles >= 0), (
                        "Particles state is not valid to run on CPU, please "
                        "call `particles.reorganize()` first."
                    )

        if _session_to_resume is not None:
            if isinstance(_session_to_resume, PipelineStatus):
                _session_to_resume = _session_to_resume.data

            assert not(_session_to_resume['resumed']), (
                "This session hase been already resumed")

            assert _session_to_resume['tracker'] is self, (
                "This session was not created by this tracker")

            ele_start = _session_to_resume['ele_start']
            ele_stop = _session_to_resume['ele_stop']
            num_turns = _session_to_resume['num_turns']
            flag_monitor = _session_to_resume['flag_monitor']
            monitor = _session_to_resume['monitor']
            _context_needs_clean_active_lost_state = _session_to_resume[
                                    '_context_needs_clean_active_lost_state']
            tt_resume = _session_to_resume['tt']
            ipp_resume = _session_to_resume['ipp']
            _session_to_resume['resumed'] = True
        else:
            (ele_start, ele_stop, num_turns, flag_monitor, monitor,
                buffer_monitor, offset_monitor,
                _context_needs_clean_active_lost_state
                ) = self._prepare_collective_track_session(
                                particles, ele_start, ele_stop,
                                num_elements, num_turns, turn_by_turn_monitor)
            tt_resume = None
            ipp_resume = None

        for tt in range(num_turns):
            if tt_resume is not None and tt < tt_resume:
                continue

            if (flag_monitor and (ele_start == 0 or tt>0)): # second condition is for delayed start
                if not(tt_resume is not None and tt == tt_resume):
                    monitor.track(particles)

            moveback_to_buffer = None
            moveback_to_offset = None
            for ipp, pp in enumerate(self._parts):
                if (ipp_resume is not None and ipp < ipp_resume):
                    continue
                elif (ipp_resume is not None and ipp == ipp_resume):
                    assert particles is None
                    particles = _session_to_resume['particles']
                    pp = self._parts[ipp]
                    moveback_to_buffer = _session_to_resume['moveback_to_buffer']
                    moveback_to_offset = _session_to_resume['moveback_to_offset']
                    _context_needs_clean_active_lost_state = _session_to_resume[
                                    '_context_needs_clean_active_lost_state']
                    _need_unhide_lost_particles = _session_to_resume[
                                    '_need_unhide_lost_particles']
                    # Clear
                    tt_resume = None
                    ipp_resume = None
                else:
                    (_need_unhide_lost_particles, moveback_to_buffer,
                        moveback_to_offset) = self._prepare_particles_for_part(
                                            particles, pp,
                                            moveback_to_buffer, moveback_to_offset,
                                            _context_needs_clean_active_lost_state)

                # Track!
                stop_tracking, skip, returned_by_track = self._track_part(
                        particles, pp, tt, ipp, ele_start, ele_stop, num_turns)

                if returned_by_track is not None:
                    if returned_by_track.on_hold:

                        assert self.enable_pipeline_hold, (
                            "Hold session not enabled for this tracker.")

                        session_on_hold = {
                            'particles': particles,
                            'tracker': self,
                            'status_from_element': returned_by_track,
                            'ele_start': ele_start,
                            'ele_stop': ele_stop,
                            'num_elements': num_elements,
                            'num_turns': num_turns,
                            'flag_monitor': flag_monitor,
                            'monitor': monitor,
                            '_context_needs_clean_active_lost_state':
                                        _context_needs_clean_active_lost_state,
                            '_need_unhide_lost_particles':
                                        _need_unhide_lost_particles,
                            'moveback_to_buffer': moveback_to_buffer,
                            'moveback_to_offset': moveback_to_offset,
                            'ipp': ipp,
                            'tt': tt,
                            'resumed': False
                        }
                    return PipelineStatus(on_hold=True, data=session_on_hold)

                # Do nothing before ele_start in the first turn
                if skip:
                    continue

                # For collective parts increment at_element
                if not isinstance(pp, Tracker) and not stop_tracking:
                    if moveback_to_buffer is not None: # The particles object is temporarily on CPU
                        if not hasattr(self, '_zerodrift_cpu'):
                            self._zerodrift_cpu = self._zerodrift.copy(particles._buffer.context)
                        self._zerodrift_cpu.track(particles, increment_at_element=True)
                    else:
                        self._zerodrift.track(particles, increment_at_element=True)

                if _need_unhide_lost_particles:
                    particles.unhide_lost_particles()

                # Break from loop over parts if stop element reached
                if stop_tracking:
                    break

            ## Break from loop over turns if stop element reached
            if stop_tracking:
                break

            if moveback_to_buffer is not None:
                particles.move(
                        _buffer=moveback_to_buffer, _offset=moveback_to_offset)
                moveback_to_buffer = None
                moveback_to_offset = None
                if _context_needs_clean_active_lost_state:
                    particles._num_active_particles = -1
                    particles._num_lost_particles = -1

            # Increment at_turn and reset at_element
            # (use the supertracker to perform only end-turn actions)
            self._supertracker.track(particles,
                               ele_start=self._supertracker.num_elements,
                               num_elements=0)

        self.record_last_track = monitor


    def _track_no_collective(
        self,
        particles,
        ele_start=0,
        ele_stop=None,     # defaults to full lattice
        num_elements=None, # defaults to full lattice
        num_turns=None,    # defaults to 1
        turn_by_turn_monitor=None
    ):

        self._check_invalidated()

        if isinstance(self._buffer.context, xo.ContextCpu):
            assert (particles._num_active_particles >= 0 and
                    particles._num_lost_particles >= 0), (
                        "Particles state is not valid to run on CPU, please "
                        "call `particles.reorganize()` first."
                    )

        # Start position
        if particles.start_tracking_at_element >= 0:
            if ele_start != 0:
                raise ValueError("The argument ele_start is used, but particles.start_tracking_at_element is set as well. "
                                + "Please use only one of those methods.")
            ele_start = particles.start_tracking_at_element
            particles.start_tracking_at_element = -1
        if isinstance(ele_start,str):
            ele_start = self.line.element_names.index(ele_start)

        assert ele_start >= 0
        assert ele_start <= self.num_elements

        # Logic to split the tracking turns:
        # Case 1: 0 <= start < stop <= L
        #      Track first turn from start until stop (with num_elements_first_turn=stop-start)
        # Case 2: 0 <= start < L < stop < 2L
        #      Track first turn from start until L    (with num_elements_first_turn=L-start)
        #      Track last turn from 0 until stop      (with num_elements_last_turn=stop)
        # Case 3: 0 <= start < L < stop=nL
        #      Track first turn from start until L    (with num_elements_first_turn=L-start)
        #      Track middle turns from 0 until (n-1)L (with num_middle_turns=n-1)
        # Case 4: 0 <= start < L < nL < stop
        #      Track first turn from start until L    (with num_elements_first_turn=L-start)
        #      Track middle turns from 0 until (n-1)L (with num_middle_turns=n-1)
        #      Track last turn from 0 until stop      (with num_elements_last_turn=stop)

        num_middle_turns = 0
        num_elements_last_turn = 0

        if num_elements is not None:
            # We are using ele_start and num_elements
            assert num_elements >= 0
            if ele_stop is not None:
                raise ValueError("Cannot use both num_elements and ele_stop!")
            if num_turns is not None:
                raise ValueError("Cannot use both num_elements and num_turns!")
            if num_elements + ele_start <= self.num_elements:
                # Track only the first (potentially partial) turn
                num_elements_first_turn = num_elements
            else:
                # Track the first turn until the end of the lattice
                num_elements_first_turn = self.num_elements - ele_start
                # Middle turns and potential last turn
                num_middle_turns, ele_stop = np.divmod(ele_start + num_elements, self.num_elements)
                num_elements_last_turn = ele_stop
                num_middle_turns -= 1

        else:
            # We are using ele_start, ele_stop, and num_turns
            if num_turns is None:
                num_turns = 1
            else:
                assert num_turns > 0
            if ele_stop is None:
                # Track the first turn until the end of the lattice
                # (last turn is also a full cycle, so will be treated as a middle turn)
                num_elements_first_turn = self.num_elements - ele_start
                num_middle_turns = num_turns - 1
            else:
                if isinstance(ele_stop, str):
                    ele_stop = self.line.element_names.index(ele_stop)
                assert ele_stop >= 0
                assert ele_stop < self.num_elements
                if ele_stop <= ele_start:
                    # Correct for overflow:
                    num_turns += 1
                if num_turns == 1:
                    # Track only the first partial turn
                    num_elements_first_turn = ele_stop - ele_start
                else:
                    # Track the first turn until the end of the lattice
                    num_elements_first_turn = self.num_elements - ele_start
                    # Track the middle turns
                    num_middle_turns = num_turns - 2
                    # Track the last turn until ele_stop
                    num_elements_last_turn = ele_stop

        if self.skip_end_turn_actions:
            flag_end_first_turn_actions = False
            flag_end_middle_turn_actions = False
        else:
            flag_end_first_turn_actions = (
                    num_elements_first_turn + ele_start == self.num_elements)
            flag_end_middle_turn_actions = True

        if num_elements_last_turn > 0:
            # One monitor record for the initial turn, num_middle_turns records for the middle turns,
            # and one for the last turn
            monitor_turns = num_middle_turns + 2
        else:
            # One monitor record for the initial turn, and num_middle_turns record for the middle turns
            monitor_turns = num_middle_turns + 1

        (flag_monitor, monitor, buffer_monitor, offset_monitor
            ) = self._get_monitor(particles, turn_by_turn_monitor, monitor_turns)

        if self.line._needs_rng and not particles._has_valid_rng_state():
            particles._init_random_number_generator()

        self.track_kernel.description.n_threads = particles._capacity

        # First turn
        self.track_kernel(
            buffer=self._line_frozen._buffer.buffer,
            ele_offsets=self.ele_offsets_dev,
            ele_typeids=self.ele_typeids_dev,
            particles=particles._xobject,
            num_turns=1,
            ele_start=ele_start,
            num_ele_track=num_elements_first_turn,
            flag_end_turn_actions=flag_end_first_turn_actions,
            flag_reset_s_at_end_turn=self.reset_s_at_end_turn,
            flag_monitor=flag_monitor,
            buffer_tbt_monitor=buffer_monitor,
            offset_tbt_monitor=offset_monitor,
            io_buffer=self.io_buffer.buffer,
        )

        # Middle turns
        if num_middle_turns > 0:
            self.track_kernel(
                buffer=self._line_frozen._buffer.buffer,
                ele_offsets=self.ele_offsets_dev,
                ele_typeids=self.ele_typeids_dev,
                particles=particles._xobject,
                num_turns=num_middle_turns,
                ele_start=0, # always full turn
                num_ele_track=self.num_elements, # always full turn
                flag_end_turn_actions=flag_end_middle_turn_actions,
                flag_reset_s_at_end_turn=self.reset_s_at_end_turn,
                flag_monitor=flag_monitor,
                buffer_tbt_monitor=buffer_monitor,
                offset_tbt_monitor=offset_monitor,
                io_buffer=self.io_buffer.buffer,
            )

        # Last turn, only if incomplete
        if num_elements_last_turn > 0:
            self.track_kernel(
                buffer=self._line_frozen._buffer.buffer,
                ele_offsets=self.ele_offsets_dev,
                ele_typeids=self.ele_typeids_dev,
                particles=particles._xobject,
                num_turns=1,
                ele_start=0,
                num_ele_track=num_elements_last_turn,
                flag_end_turn_actions=False,
                flag_reset_s_at_end_turn=self.reset_s_at_end_turn,
                flag_monitor=flag_monitor,
                buffer_tbt_monitor=buffer_monitor,
                offset_tbt_monitor=offset_monitor,
                io_buffer=self.io_buffer.buffer,
            )

        self.record_last_track = monitor

    def _get_monitor(self, particles, turn_by_turn_monitor, num_turns):

        if turn_by_turn_monitor is None or turn_by_turn_monitor is False:
            flag_monitor = 0
            monitor = None
            buffer_monitor = particles._buffer.buffer  # I just need a valid buffer
            offset_monitor = 0
        elif turn_by_turn_monitor is True:
            flag_monitor = 1
            # TODO Assumes at_turn starts from zero, to be generalized
            monitor = self.particles_monitor_class(
                _context=particles._buffer.context,
                start_at_turn=0,
                stop_at_turn=num_turns,
                particle_id_range=particles.get_active_particle_id_range()
            )
            buffer_monitor = monitor._buffer.buffer
            offset_monitor = monitor._offset
        elif turn_by_turn_monitor == 'ONE_TURN_EBE':
            (_, monitor, buffer_monitor, offset_monitor
                ) = self._get_monitor(particles, turn_by_turn_monitor=True,
                                      num_turns=len(self.line.elements)+1)
            monitor.ebe_mode = 1
            flag_monitor = 2
        elif isinstance(turn_by_turn_monitor, self.particles_monitor_class):
            flag_monitor = 1
            monitor = turn_by_turn_monitor
            buffer_monitor = monitor._buffer.buffer
            offset_monitor = monitor._offset
        else:
            raise ValueError('Please provide a valid monitor object')

        return flag_monitor, monitor, buffer_monitor, offset_monitor

    def start_internal_logging_for_elements_of_type(self,
                                                    element_type, capacity):
        return start_internal_logging_for_elements_of_type(self,
                                                    element_type, capacity)

    def stop_internal_logging_for_elements_of_type(self, element_type):
        self._check_invalidated()
        stop_internal_logging_for_elements_of_type(self, element_type)
