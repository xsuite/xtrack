from pathlib import Path
import numpy as np
import logging

from .general import _pkg_root
from .line_frozen import LineFrozen
from .base_element import _handle_per_particle_blocks
from .twiss_from_tracker import (twiss_from_tracker,
                                 compute_one_turn_matrix_finite_differences,
                                 find_closed_orbit,
                                )

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
        local_particle_src=None,
        save_source_as=None,
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
                local_particle_src=local_particle_src,
                save_source_as=save_source_as)
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
                local_particle_src=local_particle_src,
                save_source_as=save_source_as)

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
        local_particle_src=None,
        save_source_as=None,
    ):

        assert _offset is None

        self.skip_end_turn_actions = skip_end_turn_actions
        self.particles_class = particles_class
        self.global_xy_limit = global_xy_limit
        self.local_particle_src = local_particle_src
        self.save_source_as = save_source_as

        if _buffer is None:
            if _context is None:
                _context = xo.context_default
            _buffer = _context.new_buffer()
        self._buffer = _buffer

        # Split the sequence
        parts = []
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
                    i_part += 1
                parts.append(ee)
                _element_part.append(i_part)
                _element_index_in_part.append(None)
                i_part += 1
                this_part = Line(elements=[], element_names=[])
                ii_in_part = 0
        if len(this_part.elements)>0:
            this_part.iscollective=False
            parts.append(this_part)

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
        supertracker = Tracker(_buffer=_buffer,
                line=Line(elements=noncollective_xelements,
                          element_names=line.element_names),
                track_kernel=track_kernel,
                element_classes=element_classes,
                particles_class=particles_class,
                particles_monitor_class=particles_monitor_class,
                global_xy_limit=global_xy_limit,
                reset_s_at_end_turn=reset_s_at_end_turn,
                local_particle_src=local_particle_src,
                save_source_as=save_source_as
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
                                local_particle_src=local_particle_src,
                                skip_end_turn_actions=True)

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
        local_particle_src=None,
        save_source_as=None,
    ):
        if particles_class is None:
            particles_class = xp.Particles

        if particles_monitor_class is None:
            import xtrack as xt  # I have to do it like this
                                 # to avoid circular import #TODO to be solved
            particles_monitor_class = xt.ParticlesMonitor

        if local_particle_src is None:
            local_particle_src = xp.gen_local_particle_api()

        self.global_xy_limit = global_xy_limit

        frozenline = LineFrozen(
                    _context=_context, _buffer=_buffer, _offset=_offset,
                    line=line)

        context = frozenline._buffer.context

        if track_kernel is None:
            # Kernel relies on element_classes ordering
            assert element_classes is None

        if element_classes is None:
            # Kernel relies on element_classes ordering
            assert track_kernel=='skip' or track_kernel is None
            element_classes = frozenline._ElementRefClass._reftypes + [
                particles_monitor_class.XoStruct,
            ]

        line._freeze()
        self.line = line
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
        self.skip_end_turn_actions = skip_end_turn_actions
        self.reset_s_at_end_turn = reset_s_at_end_turn
        self.local_particle_src = local_particle_src
        self.element_classes = element_classes
        self._buffer = frozenline._buffer

        if track_kernel == 'skip':
            self.track_kernel = None
        elif track_kernel is None:
            self._build_kernel(save_source_as)
        else:
            self.track_kernel = track_kernel

        self.track=self._track_no_collective

    def find_closed_orbit(self, particle_co_guess=None, particle_ref=None,
                          co_search_settings={}, delta_zeta=0):

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
                                 particle_ref=particle_ref,
                                 co_search_settings=co_search_settings, delta_zeta=delta_zeta)

    def compute_one_turn_matrix_finite_differences(
            self, particle_on_co,
            steps_r_matrix=None):
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

    def twiss(self, particle_ref=None, r_sigma=0.01,
        nemitt_x=1e-6, nemitt_y=1e-6,
        n_theta=1000, delta_disp=1e-5, delta_chrom=1e-4,
        particle_co_guess=None, steps_r_matrix=None,
        co_search_settings=None, at_elements=None, at_s=None,
        eneloss_and_damping=False,
        matrix_responsiveness_tol=None,
        matrix_stability_tol=None,
        symplectify=False
        ):

        if matrix_responsiveness_tol is None:
            matrix_responsiveness_tol = self.matrix_responsiveness_tol
        if matrix_stability_tol is None:
            matrix_stability_tol = self.matrix_stability_tol

        if self.iscollective:
            logger.warning(
                'The tracker has collective elements.\n'
                'In the twiss computation collective elements are'
                ' replaced by drifts')
            tracker = self._supertracker
        else:
            tracker = self

        if particle_ref is None:
            if particle_co_guess is None:
                particle_ref = self.particle_ref

        if particle_ref is None and particle_co_guess is None:
            raise ValueError(
                "Either `particle_ref` or `particle_co_guess` must be provided")

        return twiss_from_tracker(tracker, particle_ref, r_sigma=r_sigma,
            nemitt_x=nemitt_x, nemitt_y=nemitt_y,
            n_theta=n_theta, delta_disp=delta_disp, delta_chrom=delta_chrom,
            particle_co_guess=particle_co_guess,
            steps_r_matrix=steps_r_matrix,
            co_search_settings=co_search_settings,
            at_elements=at_elements, at_s=at_s,
            eneloss_and_damping=eneloss_and_damping,
            matrix_responsiveness_tol=matrix_responsiveness_tol,
            matrix_stability_tol=matrix_stability_tol,
            symplectify=symplectify)


    def filter_elements(self, mask=None, exclude_types_starting_with=None):
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
                local_particle_src=self.local_particle_src,
            )

    def get_backtracker(self, _context=None, _buffer=None,
                        global_xy_limit='from_tracker'):

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
                    local_particle_src=self.local_particle_src,
                )

    @property
    def particle_ref(self):
        return self.line.particle_ref

    @property
    def vars(self):
        return self.line.vars

    @property
    def element_refs(self):
        return self.line.element_refs

    def configure_radiation(self, mode=None):
        self.line.configure_radiation(mode=mode)

    def _build_kernel(self, save_source_as):

        context = self._line_frozen._buffer.context

        sources = []
        kernels = {}
        cdefs = []

        if self.global_xy_limit is not None:
            sources.append(
                f"#define XTRACK_GLOBAL_POSLIMIT ({self.global_xy_limit})")
        sources.append(_pkg_root.joinpath("headers/constants.h"))


        # Local particles
        sources.append(self.local_particle_src)

        # Elements
        sources.append(_pkg_root.joinpath("tracker_src/tracker.h"))


        for ee in self.element_classes:
            for ss in ee.extra_sources:
                sources.append(ss)

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
                             int64_t offset_tbt_monitor){


            LocalParticle lpart;

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
                if (flag_end_turn_actions>0){
                    if (isactive){
                        increment_at_turn(&lpart, flag_reset_s_at_end_turn);
                    }
                }
            } // for turns
            }// if partid
        }//kernel
        """
        )

        source_track = "\n".join(src_lines)
        sources.append(source_track)

        kernel_descriptions = {
            "track_line": xo.Kernel(
                args=[
                    xo.Arg(xo.Int8, pointer=True, name="buffer"),
                    xo.Arg(xo.Int64, pointer=True, name="ele_offsets"),
                    xo.Arg(xo.Int64, pointer=True, name="ele_typeids"),
                    xo.Arg(self.particles_class.XoStruct, name="particles"),
                    xo.Arg(xo.Int32, name="num_turns"),
                    xo.Arg(xo.Int32, name="ele_start"),
                    xo.Arg(xo.Int32, name="num_ele_track"),
                    xo.Arg(xo.Int32, name="flag_end_turn_actions"),
                    xo.Arg(xo.Int32, name="flag_reset_s_at_end_turn"),
                    xo.Arg(xo.Int32, name="flag_monitor"),
                    xo.Arg(xo.Int8, pointer=True, name="buffer_tbt_monitor"),
                    xo.Arg(xo.Int64, name="offset_tbt_monitor"),
                ],
            )
        }

        # Internal API can be exposed only on CPU
        if not isinstance(context, xo.ContextCpu):
            kernels = {}
        kernels.update(kernel_descriptions)

        # Random number generator init kernel
        sources.extend(self.particles_class.XoStruct.extra_sources)
        kernels.update(self.particles_class.XoStruct.custom_kernels)

        sources = _handle_per_particle_blocks(sources)

        # Compile!
        context.add_kernels(
            sources,
            kernels,
            extra_classes=self.element_classes,
            save_source_as=save_source_as,
            specialize=True,
        )

        self.track_kernel = context.kernels.track_line


    def _track_with_collective(
        self,
        particles,
        ele_start=None,
        num_elements=None,
        num_turns=1,
        turn_by_turn_monitor=None,
    ):

        if particles.start_tracking_at_element >= 0:
            assert ele_start is None
            ele_start = particles.start_tracking_at_element
            particles.start_tracking_at_element = -1
        else:
            if ele_start is None:
                ele_start = 0

        assert ele_start >= 0
        assert ele_start <= self.num_elements

        assert num_elements is None
        assert turn_by_turn_monitor != 'ONE_TURN_EBE'


        (flag_monitor, monitor, buffer_monitor, offset_monitor
             ) = self._get_monitor(particles, turn_by_turn_monitor, num_turns)

        for tt in range(num_turns):

            if (flag_monitor and (ele_start == 0 or tt>0)): # second condition is for delayed start
                monitor.track(particles)

            moveback_to_buffer = None
            for ipp, pp in enumerate(self._parts):

                if hasattr(self, '_slice_sets'):
                    # If pyheadtail object, remove any stored slice sets
                    # (they are made invalid by the xtrack elements changing zeta)
                    self._slice_sets = {}

                # Move to CPU if needed
                if (hasattr(pp, 'needs_cpu') and pp.needs_cpu
                    and not isinstance(particles._buffer.context, xo.ContextCpu)):
                    if  moveback_to_buffer is None:
                        moveback_to_buffer = particles._buffer
                        moveback_to_offset = particles._offset
                        particles._move_to(_context=xo.ContextCpu())
                else:
                    if moveback_to_buffer is not None:
                        particles._move_to(_buffer=moveback_to_buffer, _offset=moveback_to_offset)
                        moveback_to_buffer = None
                        moveback_to_offset = None

                # Hide lost particles if required by element
                _need_clean_active_lost_state = False
                _need_unhide_lost_particles = False
                if (hasattr(pp, 'needs_hidden_lost_particles')
                    and pp.needs_hidden_lost_particles):
                    if particles._num_active_particles < 0:
                        _need_clean_active_lost_state = True
                    if not particles.lost_particles_are_hidden:
                        _need_unhide_lost_particles = True
                    particles.hide_lost_particles()

                # Track!
                if (tt == 0 and ele_start > 0): # handle delayed start
                    if ipp < self._element_part[ele_start]:
                        continue
                    if self._element_part[ele_start] == ipp:
                        ii_in_part = self._element_index_in_part[ele_start]
                        if ii_in_part is None:
                            pp.track(particles)
                        else:
                            pp.track(particles, ele_start=ii_in_part)
                    if ipp > self._element_part[ele_start]:
                        pp.track(particles)
                else: # not in first turn or no delayed start
                    pp.track(particles)

                if not isinstance(pp, Tracker):
                    if moveback_to_buffer is not None: # The particles object is temporarily on CPU
                        if not hasattr(self, '_zerodrift_cpu'):
                            self._zerodrift_cpu = self._zerodrift.copy(particles._buffer.context)
                        self._zerodrift_cpu.track(particles, increment_at_element=True)
                    else:
                        self._zerodrift.track(particles, increment_at_element=True)

                if _need_unhide_lost_particles:
                    particles.unhide_lost_particles()

                if _need_clean_active_lost_state:
                    particles._num_active_particle = -1
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
        ele_start=None,
        num_elements=None,
        num_turns=1,
        turn_by_turn_monitor=None,
    ):

        if particles.start_tracking_at_element >=0:
            assert ele_start is None
            ele_start = particles.start_tracking_at_element
            particles.start_tracking_at_element = -1
        else:
            if ele_start is None:
                ele_start = 0

        assert ele_start >= 0
        assert ele_start <= self.num_elements

        if num_turns > 1:
            assert num_elements is None

        if num_elements is None:
            # get to the end of the turn
            num_elements = self.num_elements - ele_start

        assert num_elements + ele_start <= self.num_elements

        if self.skip_end_turn_actions:
            flag_end_turn_actions=False
        else:
            flag_end_turn_actions = (
                    num_elements + ele_start == self.num_elements)

        (flag_monitor, monitor, buffer_monitor, offset_monitor
            ) = self._get_monitor(particles, turn_by_turn_monitor, num_turns)

        if self.line._needs_rng and not particles._has_valid_rng_state():
            particles._init_random_number_generator()

        self.track_kernel.description.n_threads = particles._capacity

        if ele_start > 0 or num_elements < self.num_elements: # Handle first partial turn
            self.track_kernel(
                buffer=self._line_frozen._buffer.buffer,
                ele_offsets=self.ele_offsets_dev,
                ele_typeids=self.ele_typeids_dev,
                particles=particles._xobject,
                num_turns=1,
                ele_start=ele_start,
                num_ele_track=num_elements,
                flag_end_turn_actions=flag_end_turn_actions,
                flag_reset_s_at_end_turn=self.reset_s_at_end_turn,
                flag_monitor=flag_monitor,
                buffer_tbt_monitor=buffer_monitor,
                offset_tbt_monitor=offset_monitor,
            )
            num_turns -= 1

        if num_turns > 0:
            self.track_kernel(
                buffer=self._line_frozen._buffer.buffer,
                ele_offsets=self.ele_offsets_dev,
                ele_typeids=self.ele_typeids_dev,
                particles=particles._xobject,
                num_turns=num_turns,
                ele_start=0, # always full turn
                num_ele_track=self.num_elements, # always full turn
                flag_end_turn_actions=flag_end_turn_actions,
                flag_reset_s_at_end_turn=self.reset_s_at_end_turn,
                flag_monitor=flag_monitor,
                buffer_tbt_monitor=buffer_monitor,
                offset_tbt_monitor=offset_monitor,
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
                                      num_turns=len(self.line.elements))
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
