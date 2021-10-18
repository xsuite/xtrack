from pathlib import Path
import numpy as np

from .particles import (
    Particles,
    ParticlesData,
    gen_local_particle_api,
    LocalParticleVar,
)
from .general import _pkg_root
from .line import Line as xtLine
from .base_element import _handle_per_particle_blocks

import xobjects as xo
import xline as xl


def _check_is_collective(ele):
    iscoll = not hasattr(ele, "iscollective") or ele.iscollective
    return iscoll


class Tracker:
    def __init__(
        self,
        _context=None,
        _buffer=None,
        _offset=None,
        sequence=None,
        track_kernel=None,
        element_classes=None,
        particles_class=None,
        skip_end_turn_actions=False,
        particles_monitor_class=None,
        global_xy_limit=1.0,
        local_particle_src=None,
        save_source_as=None,
        local_particle_mode=LocalParticleVar.ADAPTER,
    ):
        self._local_particle_mode = None
        # Check if there are collective elements
        self.iscollective = False
        for ee in sequence.elements:
            if _check_is_collective(ee):
                self.iscollective = True
                break

        if (local_particle_mode == LocalParticleVar.SHARED_COPY) and (
            _context is None
            or (
                not isinstance(_context, xo.ContextPyopencl)
                and not isinstance(_context, xo.ContextCupy)
            )
        ):
            local_particle_mode = LocalParticleVar.ADAPTER

        if self.iscollective:
            self._init_track_with_collective(
                _context=_context,
                _buffer=_buffer,
                _offset=_offset,
                sequence=sequence,
                track_kernel=track_kernel,
                element_classes=element_classes,
                particles_class=particles_class,
                skip_end_turn_actions=skip_end_turn_actions,
                particles_monitor_class=particles_monitor_class,
                global_xy_limit=global_xy_limit,
                local_particle_src=local_particle_src,
                save_source_as=save_source_as,
                local_particle_mode=local_particle_mode,
            )
        else:
            self._init_track_no_collective(
                _context=_context,
                _buffer=_buffer,
                _offset=_offset,
                sequence=sequence,
                track_kernel=track_kernel,
                element_classes=element_classes,
                particles_class=particles_class,
                skip_end_turn_actions=skip_end_turn_actions,
                particles_monitor_class=particles_monitor_class,
                global_xy_limit=global_xy_limit,
                local_particle_src=local_particle_src,
                save_source_as=save_source_as,
                local_particle_mode=local_particle_mode,
            )

    def _init_track_with_collective(
        self,
        _context=None,
        _buffer=None,
        _offset=None,
        sequence=None,
        track_kernel=None,
        element_classes=None,
        particles_class=None,
        skip_end_turn_actions=False,
        particles_monitor_class=None,
        global_xy_limit=1.0,
        local_particle_src=None,
        save_source_as=None,
        local_particle_mode=LocalParticleVar.ADAPTER,
    ):

        assert _offset is None
        assert track_kernel is None
        assert element_classes is None

        self.skip_end_turn_actions = skip_end_turn_actions
        self.particles_class = particles_class
        self.global_xy_limit = global_xy_limit
        self.local_particle_src = local_particle_src
        self.save_source_as = save_source_as
        self._local_particle_mode = local_particle_mode

        if _buffer is None:
            if _context is None:
                _context = xo.context.context_default
            _buffer = _context.new_buffer()
        self._buffer = _buffer

        # Split the sequence
        parts = []
        this_part = xl.Line(elements=[], element_names=[])
        for nn, ee in zip(sequence.element_names, sequence.elements):
            if not _check_is_collective(ee):
                this_part.append_element(ee, nn)
            else:
                if len(this_part.elements) > 0:
                    this_part.iscollective = False
                    parts.append(this_part)
                parts.append(ee)
                this_part = xl.Line(elements=[], element_names=[])
        if len(this_part.elements) > 0:
            this_part.iscollective = False
            parts.append(this_part)

        # Transform non collective elements into xtrack elements
        noncollective_xelements = []
        for ii, pp in enumerate(parts):
            if not _check_is_collective(pp):
                tempxtline = xtLine(_buffer=_buffer, sequence=pp)
                pp.elements = tempxtline.elements
                noncollective_xelements += pp.elements

        # Build tracker for all non collective elements
        supertracker = Tracker(
            _buffer=_buffer,
            sequence=xl.Line(
                elements=noncollective_xelements,
                element_names=[f"e{ii}" for ii in range(len(noncollective_xelements))],
            ),
            particles_class=particles_class,
            particles_monitor_class=particles_monitor_class,
            global_xy_limit=global_xy_limit,
            local_particle_src=local_particle_src,
            save_source_as=save_source_as,
            local_particle_mode=self._local_particle_mode,
        )

        # Build trackers for non collective parts
        for ii, pp in enumerate(parts):
            if not _check_is_collective(pp):
                parts[ii] = Tracker(
                    _buffer=_buffer,
                    sequence=pp,
                    element_classes=supertracker.element_classes,
                    track_kernel=supertracker.track_kernel,
                    particles_class=particles_class,
                    particles_monitor_class=particles_monitor_class,
                    global_xy_limit=global_xy_limit,
                    local_particle_src=local_particle_src,
                    skip_end_turn_actions=True,
                    local_particle_mode=self._local_particle_mode,
                )

        self._supertracker = supertracker
        self._parts = parts
        self.track = self._track_with_collective
        self.particles_class = supertracker.particles_class
        self.particles_monitor_class = supertracker.particles_monitor_class

    def _init_track_no_collective(
        self,
        _context=None,
        _buffer=None,
        _offset=None,
        sequence=None,
        track_kernel=None,
        element_classes=None,
        particles_class=None,
        skip_end_turn_actions=False,
        particles_monitor_class=None,
        global_xy_limit=1.0,
        local_particle_src=None,
        save_source_as=None,
        local_particle_mode=LocalParticleVar.ADAPTER,
    ):
        if particles_class is None:
            import xtrack as xt  # I have to do it like this

            # to avoid circular import
            particles_class = xt.Particles

        if particles_monitor_class is None:
            import xtrack as xt  # I have to do it like this

            # to avoid circular import
            particles_monitor_class = xt.ParticlesMonitor

        if local_particle_src is None:
            local_particle_src = gen_local_particle_api(local_particle_mode)

        self.global_xy_limit = global_xy_limit

        line = xtLine(
            _context=_context, _buffer=_buffer, _offset=_offset, sequence=sequence
        )

        context = line._buffer.context

        if track_kernel is None:
            # Kernel relies on element_classes ordering
            assert element_classes is None

        if element_classes is None:
            # Kernel relies on element_classes ordering
            assert track_kernel == "skip" or track_kernel is None
            element_classes = line._ElementRefClass._reftypes + [
                particles_monitor_class.XoStruct,
            ]

        self.line = line
        ele_offsets = np.array([ee._offset for ee in line.elements], dtype=np.int64)
        ele_typeids = np.array(
            [element_classes.index(ee._xobject.__class__) for ee in line.elements],
            dtype=np.int64,
        )
        ele_offsets_dev = context.nparray_to_context_array(ele_offsets)
        ele_typeids_dev = context.nparray_to_context_array(ele_typeids)

        self.particles_class = particles_class
        self.particles_monitor_class = particles_monitor_class
        self.ele_offsets_dev = ele_offsets_dev
        self.ele_typeids_dev = ele_typeids_dev
        self.num_elements = len(line.elements)
        self.global_xy_limit = global_xy_limit
        self.skip_end_turn_actions = skip_end_turn_actions
        self.local_particle_src = local_particle_src
        self._local_particle_mode = local_particle_mode
        self.element_classes = element_classes

        if track_kernel == "skip":
            self.track_kernel = None
        elif track_kernel is None:
            self._build_kernel(save_source_as)
        else:
            self.track_kernel = track_kernel

        self.track = self._track_no_collective


    def _build_kernel(self, save_source_as):
        context = self.line._buffer.context

        sources = []
        kernels = {}
        cdefs = []

        if self.global_xy_limit is not None:
            sources.append( r"""
        #if !defined( XTRACK_GLOBAL_POSLIMIT )
            #define XTRACK_GLOBAL_POSLIMIT """ +
            f"({self.global_xy_limit})" + r"""
        #endif /* !defined( XTRACK_GLOBAL_POSLIMIT ) */

        """ )


        sources.append(_pkg_root.joinpath("headers/constants.h"))

        # Local particles
        sources.append(self.local_particle_src)

        # Elements
        sources.append(_pkg_root.joinpath("tracker_src/tracker.h"))

        for ee in self.element_classes:
            for ss in ee.extra_sources:
                sources.append(ss)

        src_lines = [
            r"""
        /*gpukern*/ void track_line(
                /*gpuglmem*/ int8_t* buffer,
                /*gpuglmem*/ int64_t* ele_offsets,
                /*gpuglmem*/ int64_t* ele_typeids,
                             ParticlesData particles,
                             int num_turns,
                             int ele_start,
                             int num_ele_track,
                             int flag_end_turn_actions,
                             int flag_tbt_monitor,
                /*gpuglmem*/ int8_t* buffer_tbt_monitor,
                             int64_t offset_tbt_monitor""",
        ]

        if self._local_particle_mode == LocalParticleVar.SHARED_COPY and (
            context is None
            or (
                not isinstance(context, xo.ContextPyopencl)
                and not isinstance(context, xo.ContextCupy)
            )
        ):
            raise ValueError(
                "shared copy local_particle_mode not "
                + "available for provided context"
            )

        use_shared_copy = False
        if self._local_particle_mode == LocalParticleVar.SHARED_COPY and isinstance(
            context, xo.ContextPyopencl
        ):
            src_lines[
                -1
            ] += r""",
                /*gpusharedmem*/ char* local_fields"""
            use_shared_copy = True
        src_lines[
            -1
        ] += r""" )
        {
            """

        if self._local_particle_mode == LocalParticleVar.SHARED_COPY and isinstance(
            context, xo.ContextCupy
        ):
            src_lines[-1] += "extern __shared__ char local_fields[];\r\n"
            use_shared_copy = True

        if use_shared_copy:
            shared_num_bytes_per_thread = 136
            shared_num_bytes_common = 16

        break_on_lost_particles = not use_shared_copy
        if break_on_lost_particles:
            for ee in self.element_classes:
                if ee.requires_sync(mode="local"):
                    break_on_lost_particles = False
                    break

        if self._local_particle_mode == LocalParticleVar.ADAPTER:
            loc_particle_mode_str = "/* local_particle_mode = ADAPTER */"
        elif self._local_particle_mode == LocalParticleVar.THREAD_LOCAL_COPY:
            loc_particle_mode_str = "/* local_particle_mode = THREAD_LOCAL_COPY */"
        elif self._local_particle_mode == LocalParticleVar.SHARED_COPY:
            loc_particle_mode_str = "/* local_particle_mode = SHARED_COPY */"

        src = (
            r"""
            LocalParticle lpart; """
            + loc_particle_mode_str
            + r"""
            """
        )
        if break_on_lost_particles:
            src += r"""bool is_active  = false;

            """

        src += r"""int64_t part_idx = 0; //only_for_context cpu_serial cpu_openmp
            int64_t part_idx = blockDim.x * blockIdx.x + threadIdx.x; //only_for_context cuda
            int64_t part_idx = get_global_id(0); //only_for_context opencl
            int64_t iturn   = 0;

            /*gpuglmem*/ int8_t* tbt_mon_pointer = buffer_tbt_monitor + offset_tbt_monitor;
            ParticlesMonitorData tbt_monitor = ( ParticlesMonitorData )tbt_mon_pointer;

            int64_t const ele_stop = ele_start + num_ele_track;
            int64_t const n_part = ParticlesData_get__capacity(particles);

            if( part_idx >= n_part ) part_idx = -1;

            """

        if not use_shared_copy:
            src += r"""LocalParticle_init_from_particles_data(
                particles, &lpart, part_idx );

            """

        else:
            src += r"""LocalParticle_init_from_particles_data(
                particles, &lpart, part_idx, local_fields );
            sync_locally();

            """

        src += r"""if( part_idx < n_part ) {
                """

        if break_on_lost_particles:
            src += r"""is_active = check_is_active( &lpart );

                """

        src += r"""for( ; iturn < num_turns ; ++iturn ){
                    int64_t ee = ele_start;
                    """

        if break_on_lost_particles:
            src += r"""if( !is_active ) break;
                    """

        src += r"""if( flag_tbt_monitor ) ParticlesMonitor_track_local_particle(
                        tbt_monitor, &lpart );

                    for( ; ee < ele_stop; ++ee ){
                        /*gpuglmem*/ int8_t* el = buffer + ele_offsets[ ee ];
                        int64_t const ee_type = ele_typeids[ ee ];

                        switch( ee_type ){
                            """

        for ii, cc in enumerate(self.element_classes):
            ccnn = cc.__name__.replace("Data", "")
            src += f"""case {ii}: {{
                                """

            if self.global_xy_limit is not None and cc.requires_global_aperture_check():
                src += r"""global_aperture_check( &lpart );
                                """
            track_cmd = f"{ccnn}_track_local_particle( ({ccnn}Data) el, &lpart );"
            case_cmt_str = f"/* case {ii}: {ccnn} */"

            if not break_on_lost_particles and not cc.requires_sync("local"):
                src += (
                    r"""if( LocalParticle_is_active( &lpart ) ) {
                                    """
                    + track_cmd
                    + r"""
                                } /* !( is_active ) */
                                break;
                            }"""
                    + case_cmt_str
                    + r"""

                            """
                )
            else:
                src += (
                    track_cmd
                    + r"""
                                break;
                            }"""
                    + case_cmt_str
                    + r"""

                            """
                )
        src += r"""
                        }; /* switch */

                        """

        if break_on_lost_particles:
            src += r"""is_active = check_is_active( &lpart );
                        if( !is_active ) break;

                        increment_at_element( &lpart );
                    """
        else:
            src += r"""if( LocalParticle_is_active( &lpart ) ) {
                            increment_at_element( &lpart ); }
                    """

        src += r"""} /* for all elements */

                    """

        if break_on_lost_particles:
            src += r"""if( ( flag_end_turn_actions ) && ( is_active ) ) increment_at_turn( &lpart );
                """
        else:
            src += r"""if( ( flag_end_turn_actions ) &&
                    ( LocalParticle_is_active( &lpart ) ) ) {
                        increment_at_turn( &lpart ); } /* if flags and is_active */
                """
        src += r"""} /* for all turns */

            } /* if ipart < n_part */
        """

        if use_shared_copy:
            src += r"""
            sync_locally();
        """

        if not (self._local_particle_mode == LocalParticleVar.ADAPTER):
            src += r"""
            LocalParticle_sync_to_particles_data( &lpart, particles, ( part_idx == 0 ) ); //only_for_context opencl cuda
            LocalParticle_sync_to_particles_data( &lpart, particles, true ); //only_for_context cpu_serial cpu_openmp
        """

        src += "} /* kernel */"
        src_lines.append(src)

        source_track = "\n".join(src_lines)
        sources.append(source_track)

        kernel_args = [
            xo.Arg(xo.Int8, pointer=True, name="buffer"),
            xo.Arg(xo.Int64, pointer=True, name="ele_offsets"),
            xo.Arg(xo.Int64, pointer=True, name="ele_typeids"),
            xo.Arg(self.particles_class.XoStruct, name="particles"),
            xo.Arg(xo.Int32, name="num_turns"),
            xo.Arg(xo.Int32, name="ele_start"),
            xo.Arg(xo.Int32, name="num_ele_track"),
            xo.Arg(xo.Int32, name="flag_end_turn_actions"),
            xo.Arg(xo.Int32, name="flag_tbt_monitor"),
            xo.Arg(xo.Int8, pointer=True, name="buffer_tbt_monitor"),
            xo.Arg(xo.Int64, name="offset_tbt_monitor"),
        ]

        if use_shared_copy and isinstance(context, xo.ContextPyopencl):
            kernel_args.append(
                xo.LocalMemPyopenclArg(
                    xo.Int8,
                    name="local_fields",
                    num_bytes_per_thread=shared_num_bytes_per_thread,
                    num_bytes_common=shared_num_bytes_common,
                )
            )

        # Internal API can be exposed only on CPU
        if not isinstance(context, xo.ContextCpu):
            kernels = {}
        kernels.update({"track_line": xo.Kernel(args=kernel_args)})

        sources = _handle_per_particle_blocks(sources)

        build_options = []
        if isinstance( context, xo.ContextPyopencl ):
            build_options.append( "-cl-kernel-arg-info" )

        # Compile!
        context.add_kernels(
            sources,
            kernels,
            extra_classes=self.element_classes,
            save_source_as=save_source_as,
            specialize=True,
            build_options=build_options
        )

        self.track_kernel = context.kernels.track_line
        if use_shared_copy:
            if isinstance(context, xo.ContextPyopencl):
                assert self.track_kernel.num_local_mem_args == 1
                idx = self.track_kernel.local_mem_arg_indices[ 0 ]
                assert idx == len( kernel_args ) - 1
                assert isinstance( kernel_args[ idx ], xo.LocalMemPyopenclArg )
                kernel_args[ idx ].assign_to_kernel( self.track_kernel, idx )
            elif isinstance( context, xo.ContextCupy ):
                self.track_kernel.update_num_bytes_shared_mem(
                    shared_num_bytes_per_thread, shared_num_bytes_common )


    def _track_with_collective(
        self,
        particles,
        ele_start=0,
        num_elements=None,
        num_turns=1,
        turn_by_turn_monitor=None,
    ):

        assert ele_start == 0
        assert num_elements is None

        (flag_tbt, monitor, buffer_monitor, offset_monitor) = self._get_monitor(
            particles, turn_by_turn_monitor, num_turns
        )

        for tt in range(num_turns):
            if flag_tbt:
                monitor.track(particles)
            for pp in self._parts:
                pp.track(particles)
            # Increment at_turn and reset at_element
            # (use the supertracker to perform only end-turn actions)
            self._supertracker.track(
                particles, ele_start=self._supertracker.num_elements, num_elements=0
            )

        self.record_last_track = monitor

    def _track_no_collective(
        self,
        particles,
        ele_start=0,
        num_elements=None,
        num_turns=1,
        turn_by_turn_monitor=None,
    ):

        if num_elements is None:
            # get to the end of the turn
            num_elements = self.num_elements - ele_start

        assert num_elements + ele_start <= self.num_elements

        if self.skip_end_turn_actions:
            flag_end_turn_actions = False
        else:
            flag_end_turn_actions = num_elements + ele_start == self.num_elements

        (flag_tbt, monitor, buffer_monitor, offset_monitor) = self._get_monitor(
            particles, turn_by_turn_monitor, num_turns
        )

        self.track_kernel.description.n_threads = particles._capacity
        self.track_kernel(
            buffer=self.line._buffer.buffer,
            ele_offsets=self.ele_offsets_dev,
            ele_typeids=self.ele_typeids_dev,
            particles=particles._xobject,
            num_turns=num_turns,
            ele_start=ele_start,
            num_ele_track=num_elements,
            flag_end_turn_actions=flag_end_turn_actions,
            flag_tbt_monitor=flag_tbt,
            buffer_tbt_monitor=buffer_monitor,
            offset_tbt_monitor=offset_monitor,
        )

        self.record_last_track = monitor

    def _get_monitor(self, particles, turn_by_turn_monitor, num_turns):

        if turn_by_turn_monitor is None or turn_by_turn_monitor is False:
            flag_tbt = 0
            monitor = None
            buffer_monitor = particles._buffer.buffer  # I just need a valid buffer
            offset_monitor = 0
        elif turn_by_turn_monitor is True:
            flag_tbt = 1
            # TODO Assumes at_turn starts from zero, to be generalized
            monitor = self.particles_monitor_class(
                _context=particles._buffer.context,
                start_at_turn=0,
                stop_at_turn=num_turns,
                particle_id_range=particles.get_active_particle_id_range(),
            )
            buffer_monitor = monitor._buffer.buffer
            offset_monitor = monitor._offset
        elif isinstance(turn_by_turn_monitor, self.particles_monitor_class):
            flag_tbt = 1
            monitor = turn_by_turn_monitor
            buffer_monitor = monitor._buffer.buffer
            offset_monitor = monitor._offset
        else:
            raise ValueError("Please provide a valid monitor object")

        return flag_tbt, monitor, buffer_monitor, offset_monitor
