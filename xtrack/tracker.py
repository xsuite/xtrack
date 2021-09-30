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

    @staticmethod
    def _indent_str(offset=0, **kwargs):
        indent_per_level = kwargs.get("indent_per_level", "    ")
        indent_level = kwargs.get("indent_level", 0)
        return indent_per_level * (indent_level + offset)

    @staticmethod
    def _inc_indent_level(kwargs):
        indent_level = kwargs.get("indent_level", 0)
        kwargs.update(
            {
                "indent_level": indent_level + 1,
            }
        )
        return indent_level

    @staticmethod
    def _set_indent_level(lvl, kwargs):
        cur_level = kwargs.get("indent_level", 0)
        kwargs.update(
            {
                "indent_level": lvl,
            }
        )
        return cur_level

    def _gen_track_kernel_preamble(self, context, **kwargs):
        sources = []
        indent_str = Tracker._indent_str(0, **kwargs)
        body_indent_str = Tracker._indent_str(1, **kwargs)

        if self.global_xy_limit is not None:
            src = f"{indent_str}#if !defined( XTRACK_GLOBAL_POSLIMIT )\r\n"
            src += f"{body_indent_str}#define XTRACK_GLOBAL_POSLIMIT ({self.global_xy_limit})\r\n"
            src += f"{indent_str}#endif  /* !defined( XTRACK_GLOBAL_POSLIMIT ) */\r\n"
            sources.append(src)

        sources.append(_pkg_root.joinpath("headers/constants.h"))

        # Local particles
        sources.append(self.local_particle_src)

        # Elements
        sources.append(_pkg_root.joinpath("tracker_src/tracker.h"))

        for ee in self.element_classes:
            for ss in ee.extra_sources:
                sources.append(ss)

        return sources

    def _gen_track_kernel_common_start(
        self,
        context,
        ind,
        indent_level,
        use_shared_copy,
        break_on_lost_particles,
        **kwargs,
    ):

        buffer_param = kwargs.get("buffer_param", "buffer")
        loc_particle_param = kwargs.get("loc_particle_param", "p")
        pdata_param = kwargs.get("pdata_param", "particles")
        n_part_param = kwargs.get("n_part_param", "n_part")
        part_idx_param = kwargs.get("part_idx_param", "part_idx")

        ele_start_param = kwargs.get("ele_start_param", "ele_start")
        n_ele_track_param = kwargs.get("n_ele_track_param", "num_ele_track")
        ele_stop_param = kwargs.get("ele_stop_param", "ele_stop")
        ele_cnt_param = kwargs.get("ele_cnt_param", "ele")
        ele_offsets_param = kwargs.get("ele_offsets_param", "ele_offsets")
        ele_typeids_param = kwargs.get("ele_typeids_param", "ele_typeids")
        el_ptr_param = kwargs.get("el_ptr_param", "elem")
        el_type_param = kwargs.get("el_type_param", "elem_type_id")

        n_turns_param = kwargs.get("n_turns_param", "num_turns")
        turn_cnt_param = kwargs.get("turn_cnt_param", "iturn")

        flag_tbt_monitor_param = kwargs.get(
            "flag_tbt_monitor_param", "flag_tbt_monitor"
        )
        tbt_mon_ptr_param = kwargs.get("tbt_mon_ptr_param", "tbt_mon_pointer")
        buffer_tbt_monitor_param = kwargs.get(
            "buffer_tbt_monitor_param", "buffer_tbt_monitor"
        )
        tbt_monitor_param = kwargs.get("tbt_monitor_param", "tbt_monitor")
        offset_tbt_monitor_param = kwargs.get(
            "offset_tbt_monitor_param", "offset_tbt_monitor"
        )

        elem_by_elem_particles_param = kwargs.get("elem_by_elem_particles_param", None)
        num_ebe_part_param = kwargs.get("num_ebe_part_param", None)
        num_ebe_part_per_turn_param = kwargs.get("num_ebe_part_per_turn_param", None)
        ebe_store_at_param = kwargs.get("ebe_store_at_param", "store_at")
        ebe_sync_common_fields_param = kwargs.get(
            "ebe_sync_common_fields_param", "sync_common_fields"
        )

        il = indent_level

        if self._local_particle_mode == LocalParticleVar.ADAPTER:
            loc_particle_mode_str = "/* local_particle_mode = ADAPTER */"
        elif self._local_particle_mode == LocalParticleVar.THREAD_LOCAL_COPY:
            loc_particle_mode_str = "/* local_particle_mode = THREAD_LOCAL_COPY */"
        elif self._local_particle_mode == LocalParticleVar.SHARED_COPY:
            loc_particle_mode_str = "/* local_particle_mode = SHARED_COPY */"

        src = f"{ind[ il ]}LocalParticle {loc_particle_param}; {loc_particle_mode_str}\r\n"
        if break_on_lost_particles:
            src += f"{ind[ il ]}bool is_active = false;\r\n"

        src += (
            f"{ind[ il ]}int64_t {part_idx_param} = 0; "
            + f"//only_for_context cpu_serial cpu_openmp\r\n"
        )
        src += (
            f"{ind[ il ]}int64_t {part_idx_param} = "
            + f"blockDim.x * blockIdx.x + threadIdx.x; "
            + f"//only_for_context cuda\r\n"
        )
        src += (
            f"{ind[ il ]}int64_t {part_idx_param} = "
            + f"get_global_id(0); //only_for_context opencl\r\n"
        )
        src += f"{ind[ il ]}int64_t {turn_cnt_param} = ( int64_t )0;\r\n\r\n"

        src += (
            f"{ind[ il ]}/*gpuglmem*/ int8_t* {tbt_mon_ptr_param} = "
            + f"{buffer_tbt_monitor_param} + {offset_tbt_monitor_param};\r\n"
        )

        src += (
            f"{ind[ il ]}ParticlesMonitorData {tbt_monitor_param} = "
            + f"( ParticlesMonitorData ){tbt_mon_ptr_param};\r\n\r\n"
        )

        src += (
            f"{ind[ il ]}int64_t const {ele_stop_param} = "
            + f"{ele_start_param} + {n_ele_track_param};\r\n"
        )
        src += (
            f"{ind[ il ]}int64_t {n_part_param} = "
            + f"ParticlesData_get__capacity( {pdata_param} );\r\n"
        )

        if (
            elem_by_elem_particles_param is not None
            and num_ebe_part_param is not None
            and num_ebe_part_per_turn_param is not None
        ):
            src += f"{ind[ il ]}int64_t const {num_ebe_part_param} = "
            src += "ParticlesData_get__capacity( \r\n"
            src += f"{ind[ il + 1 ]}{elem_by_elem_particles_param} );\r\n"
            src += f"{ind[ il ]}int64_t const {num_ebe_part_per_turn_param} = "
            src += f"{n_ele_track_param} * {n_part_param};\r\n"

        src += (
            f"{ind[ il ]}if( {part_idx_param} >= {n_part_param} ) "
            + f"{part_idx_param} = ( int64_t )-1;\r\n"
        )

        src += "\r\n"
        src += f"{ind[ il ]}LocalParticle_init_from_particles_data( \r\n"
        src += f"{ind[ il + 2 ]}{pdata_param}, &{loc_particle_param}, "
        src += f"{part_idx_param}"

        if not use_shared_copy:
            src += " );\r\n"
        else:
            src += f",{local_fields_param} );\r\n"
            src += f"{ind[ il ]}sync_locally();\r\n"

        src += "\r\n"
        src += f"{ind[ il ]}if( {part_idx_param} < {n_part_param} ) {{ \r\n"
        il += 1

        if break_on_lost_particles:
            src += (
                f"{ind[ il ]}is_active = "
                + f"check_is_active( &{loc_particle_param} );\r\n"
            )

        src += f"{ind[ il ]}for( ; {turn_cnt_param} < {n_turns_param}; "
        src += f"++{turn_cnt_param} ){{\r\n"
        il += 1

        src += f"{ind[ il ]}int64_t {ele_cnt_param} = {ele_start_param};\r\n"
        if break_on_lost_particles:
            src += f"{ind[ il ]}if( !is_active ) break;\r\n"

        if tbt_monitor_param is not None and flag_tbt_monitor_param is not None:
            src += "\r\n"
            src += f"{ind[ il ]}if( {flag_tbt_monitor_param} ) \r\n"
            src += f"{ind[ il + 1 ]}ParticlesMonitor_track_local_particle( "
            src += f"{tbt_monitor_param}, &{loc_particle_param} );\r\n\r\n"

        src += "\r\n"
        src += (
            f"{ind[ il ]}for( ; {ele_cnt_param} < {ele_stop_param} ; "
            + f"++{ele_cnt_param} ){{\r\n"
        )
        il += 1
        src += (
            f"{ind[ il ]}/*gpuglmem*/ int8_t* {el_ptr_param} = "
            + f"{buffer_param} + {ele_offsets_param}[ {ele_cnt_param} ];\r\n"
        )
        src += (
            f"{ind[ il ]}int64_t const {el_type_param} = "
            + f"{ele_typeids_param}[ {ele_cnt_param} ];\r\n"
        )

        if elem_by_elem_particles_param is not None and num_ebe_part_param is not None:
            src += "\r\n"
            if break_on_lost_particles:
                src += (
                    f"{ind[il]}if( {num_ebe_part_per_turn_param} <= "
                    + f"{num_ebe_part_param} ) {{ \r\n"
                )
            else:
                src += (
                    f"{ind[il]}if( ( {num_ebe_part_per_turn_param} <= "
                    + f"{num_ebe_part_param} ) && \r\n"
                )
                src += (
                    f"{ind[il]}    ( LocalParticle_is_active( "
                    + f"&{loc_particle_param} ) ) {{ \r\n"
                )

            src += (
                f"{ind[il+1]}bool const {ebe_sync_common_fields_param} = "
                + f"( {part_idx_param} == 0 );\r\n"
            )
            src += f"{ind[il+1]}int64_t {ebe_store_at_param} = \r\n"
            src += (
                f"{ind[il+2]}LocalParticle_get_at_turn( "
                + f"&{loc_particle_param} ) * {num_ebe_part_per_turn_param} +\r\n"
            )
            src += (
                f"{ind[il+2]}LocalParticle_get_at_element( "
                + f"&{loc_particle_param} ) * {n_part_param};\r\n"
            )
            src += (
                f"{ind[il+1]}{ebe_store_at_param} = {ebe_store_at_param} % "
                + f"{num_ebe_part_param};\r\n"
            )
            src += f"{ind[il+2]}if( {part_idx_param} == 0 ) "
            src += 'printf( "store_at = %ld\\n", ( long int )store_at );'
            src += f"\r\n"
            src += f"{ind[il+1]}LocalParticle_to_particles_data(\r\n"
            src += (
                f"{ind[il+2]}&{loc_particle_param}, {pdata_param}, "
                + f"{ebe_store_at_param}, {ebe_sync_common_fields_param} );\r\n"
            )
            src += f"{ind[il]}}} /* store elem - by - elem */\r\n"

        return src

    def _get_track_kernel_elem_dispatcher(
        self,
        context,
        ind,
        il,
        use_shared_copy,
        break_on_lost_particles,
        element_classes,
        **kwargs,
    ):

        loc_particle_param = kwargs.get("loc_particle_param", "p")
        part_idx_param = kwargs.get("part_idx_param", "part_idx")
        el_ptr_param = kwargs.get("el_ptr_param", "elem")
        el_type_param = kwargs.get("el_type_param", "elem_type_id")

        src = "\r\n"
        src += f"{ind[il]}switch( {el_type_param} ) {{\r\n"
        il += 1
        for ii, cc in enumerate(element_classes):
            ccnn = cc.__name__.replace("Data", "")
            src += f"{ind[il]}case {ii}: {{ \r\n"

            loc_il = il + 1
            if self.global_xy_limit is not None and cc.requires_global_aperture_check():
                src += (
                    f"{ind[loc_il]}global_aperture_check( &{loc_particle_param} );\r\n"
                )
                if break_on_lost_particles:
                    src += f"{ind[loc_il]}is_active = check_is_active( &{loc_particle_param} );\r\n"
                    src += f"{ind[loc_il]}if( !is_active ) break;\r\n"
                elif not cc.requires_sync("local"):
                    src += f"{ind[loc_il]}if( LocalParticle_is_active( &{loc_particle_param} ) ){{\r\n"
                    loc_il += 1

            src += f"{ind[loc_il]}{ccnn}_track_local_particle( \r\n"
            src += f"{ind[loc_il+1]}( {ccnn}Data ){el_ptr_param}, &{loc_particle_param} );\r\n"

            if self.global_xy_limit is not None and cc.requires_global_aperture_check():
                if not break_on_lost_particles and not cc.requires_sync("local"):
                    loc_il -= 1
                    src += f"{ind[loc_il]}}} /* is_active */\r\n"
                elif not break_on_lost_particles and cc.requires_sync("local"):
                    src += f"{ind[loc_il]}sync_locally();\r\n"

            src += f"{ind[loc_il]}break;\r\n"
            src += f"{ind[il]}}} /* case: {ii} :: {ccnn}Data */\r\n\r\n"

        il -= 1
        src += f"{ind[il]}}}; /* switch( {el_type_param} ) */\r\n"
        return src

    def _gen_track_kernel_common_end(
        self, context, ind, il, use_shared_copy, break_on_lost_particles, **kwargs
    ):

        loc_particle_param = kwargs.get("loc_particle_param", "p")
        pdata_param = kwargs.get("pdata_param", "particles")
        n_part_param = kwargs.get("n_part_param", "n_part")
        part_idx_param = kwargs.get("part_idx_param", "part_idx")
        flag_eot_actions_param = kwargs.get("flag_eot_actions_param", "flag_eot")

        il += 4
        src = "\r\n"

        if break_on_lost_particles:
            src += f"{ind[il]}is_active = check_is_active( &{loc_particle_param} );\r\n"
            src += f"{ind[il]}if( !is_active ) break;\r\n"
            src += f"{ind[il]}increment_at_element( &{loc_particle_param} );\r\n"
        else:
            src += (
                f"{ind[il]}if( LocalParticle_is_active( "
                + f"&{loc_particle_param} ) ) {{\r\n"
            )
            src += f"{ind[il+1]}increment_at_element( &{loc_particle_param} ); }}\r\n"

        il -= 1
        src += f"{ind[il]}}} /* for all elements */\r\n"
        src += "\r\n"

        il -= 1
        if break_on_lost_particles:
            src += (
                f"{ind[il]}if( ( {flag_eot_actions_param} ) && ( is_active ) ) {{\r\n"
            )
        else:
            src += f"{ind[il]}if( ( {flag_eot_actions_param} ) && \r\n"
            src += f"{ind[il]}    ( LocalParticle_is_active( &{loc_particle_param} ) ) ) {{ \r\n"
        src += f"{ind[il+1]}increment_at_turn( &{loc_particle_param} ); }} \r\n"
        src += "\r\n"

        il -= 1
        src += f"{ind[il]}}} /* for all turns */ \r\n"
        src += "\r\n"

        il -= 1
        src += f"{ind[il]}}} /* {part_idx_param} < {n_part_param} */ \r\n"

        if use_shared_copy:
            src += f"{ind[il]}sync_locally();\r\n"
        src += "\r\n"

        if not (self._local_particle_mode == LocalParticleVar.ADAPTER):
            src += (
                f"{ind[il]}LocalParticle_sync_to_particles_data( "
                + f"&{loc_particle_param}, {pdata_param}, "
                + f"( {part_idx_param} == 0 ) ); //only_for_context opencl cuda"
            )
            src += "\r\n"
            src += (
                f"{ind[il]}LocalParticle_sync_to_particles_data( "
                + f"&{loc_particle_param}, {pdata_param}, "
                + f", true ); //only_for_context cpu_serial cpu_openmp"
            )
            src += "\r\n"

        return src

    def _build_track_elem_by_elem_kernel(self, save_source_as, **kwargs):
        context = self.line._buffer.context
        sources = self._gen_track_kernel_preamble(context, **kwargs)
        kernels = {}
        cdefs = []

        indents = {}
        for offset in range(20):
            indents[offset] = Tracker._indent_str(offset, **kwargs)

        buffer_param = kwargs.get("buffer_param", "buffer")
        ele_offsets_param = kwargs.get("ele_offsets_param", "ele_offsets")
        ele_typeids_param = kwargs.get("ele_typeids_param", "ele_typeids")
        pdata_param = kwargs.get("pdata_param", "particles")
        n_turns_param = kwargs.get("n_turns_param", "num_turns")
        ele_start_param = kwargs.get("ele_start_param", "ele_start")
        n_ele_track_param = kwargs.get("n_ele_track_param", "num_ele_track")
        flag_eot_actions_param = kwargs.get("flag_eot_actions_param", "flag_eot")
        flag_tbt_monitor_param = kwargs.get(
            "flag_tbt_monitor_param", "flag_tbt_monitor"
        )
        buffer_tbt_monitor_param = kwargs.get(
            "buffer_tbt_monitor_param", "buffer_tbt_monitor"
        )
        offset_tbt_monitor_param = kwargs.get(
            "offset_tbt_monitor_param", "offset_tbt_monitor"
        )
        elem_by_elem_particles_param = kwargs.get(
            "elem_by_elem_particles_param", "elem_by_elem_particles"
        )
        num_ebe_part_param = kwargs.get("num_ebe_part_param", "n_ebe_part")
        num_ebe_part_per_turn_param = kwargs.get(
            "num_ebe_part_per_turn_param", "n_ebe_part_per_turn"
        )

        local_fields_param = kwargs.get("local_fields_param", "local_fields")

        src = "\r\n"
        src += f"{indents[ 0 ]}/*gpukern*/ void track_elem_by_elem(\r\n"
        src += f"{indents[ 2 ]}/*gpuglmem*/ int8_t* {buffer_param},\r\n"
        src += f"{indents[ 2 ]}/*gpuglmem*/ int64_t* {ele_offsets_param},\r\n"
        src += f"{indents[ 2 ]}/*gpuglmem*/ int64_t* {ele_typeids_param},\r\n"
        src += f"{indents[ 2 ]}ParticlesData {pdata_param},\r\n"
        src += f"{indents[ 2 ]}int {n_turns_param}, \r\n"
        src += f"{indents[ 2 ]}int {ele_start_param}, \r\n"
        src += f"{indents[ 2 ]}int {n_ele_track_param}, \r\n"
        src += f"{indents[ 2 ]}int {flag_eot_actions_param}, \r\n"
        src += f"{indents[ 2 ]}int {flag_tbt_monitor_param}, \r\n"
        src += f"{indents[ 2 ]}/*gpuglmem*/ int8_t* {buffer_tbt_monitor_param},\r\n"
        src += f"{indents[ 2 ]}int {offset_tbt_monitor_param}, \r\n"
        src += f"{indents[ 2 ]}ParticlesData {elem_by_elem_particles_param}"

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
            src += ", \r\n"
            src += f"{ind[ 2 ]}/*gpusharedmem*/ int8_t* {local_fields_param} "
            use_shared_copy = True

        src += f" )\r\n{indents[0]}{{\r\n"

        if self._local_particle_mode == LocalParticleVar.SHARED_COPY and isinstance(
            context, xo.ContextCupy
        ):
            src += f"{ind[ 2 ]}extern /*gpusharedmem*/ char "
            src += f"{local_fields_param}[];\r\n"
            use_shared_copy = True

        break_on_lost_particles = True  # not use_shared_copy
        if break_on_lost_particles:
            for ee in self.element_classes:
                if ee.requires_sync(mode="local"):
                    break_on_lost_particles = False
                    break

        indent_level = 2
        src += self._gen_track_kernel_common_start(
            context,
            indents,
            indent_level,
            use_shared_copy,
            break_on_lost_particles,
            elem_by_elem_particles_param=elem_by_elem_particles_param,
            num_ebe_part_param=num_ebe_part_param,
            num_ebe_part_per_turn_param=num_ebe_part_per_turn_param,
            **kwargs,
        )

        indent_level = 5
        src += self._get_track_kernel_elem_dispatcher(
            context,
            indents,
            indent_level,
            use_shared_copy,
            break_on_lost_particles,
            self.element_classes,
            **kwargs,
        )

        indent_level = 1
        src += self._gen_track_kernel_common_end(
            context,
            indents,
            indent_level,
            use_shared_copy,
            break_on_lost_particles,
            **kwargs,
        )

        src += f"{indents[0]}}} /* kernel */\r\n"
        sources.append(src)

        kernel_args = [
            xo.Arg(xo.Int8, pointer=True, name=f"{buffer_param}"),
            xo.Arg(xo.Int64, pointer=True, name=f"{ele_offsets_param}"),
            xo.Arg(xo.Int64, pointer=True, name=f"{ele_typeids_param}"),
            xo.Arg(self.particles_class.XoStruct, name=f"{pdata_param}"),
            xo.Arg(xo.Int32, name=f"{n_turns_param}"),
            xo.Arg(xo.Int32, name=f"{ele_start_param}"),
            xo.Arg(xo.Int32, name=f"{n_ele_track_param}"),
            xo.Arg(xo.Int32, name=f"{flag_eot_actions_param}"),
            xo.Arg(xo.Int32, name=f"{flag_tbt_monitor_param}"),
            xo.Arg(xo.Int8, pointer=True, name=f"{buffer_tbt_monitor_param}"),
            xo.Arg(xo.Int64, name=f"{offset_tbt_monitor_param}"),
            xo.Arg(
                self.particles_class.XoStruct, name=f"{elem_by_elem_particles_param}"
            ),
        ]

        if use_shared_copy and isinstance(context, xo.ContextPyopencl):
            kernel_args.append(
                xo.SharedMemPyopenclArg(
                    xo.Int8,
                    name=f"{local_fields_param}",
                    shmem_per_work_group=16,
                    shmem_per_work_item=136,
                )
            )

        kernel_descriptions = {"track_elem_by_elem": xo.Kernel(args=kernel_args)}

        # Internal API can be exposed only on CPU
        if not isinstance(context, xo.ContextCpu):
            kernels = {}
        kernels.update(kernel_descriptions)

        sources = _handle_per_particle_blocks(sources)

        # Compile!
        context.add_kernels(
            sources,
            kernels,
            extra_classes=self.element_classes,
            save_source_as=save_source_as,
            specialize=True,
        )

        if use_shared_copy and isinstance(context, xo.ContextPyopencl):
            kernel_args[-1].update(kernel=context.kernels.track_elem_by_elem)

        self.track_elem_by_elem_kernel = context.kernels.track_elem_by_elem

    def _build_kernel(self, save_source_as):
        context = self.line._buffer.context

        sources = []
        kernels = {}
        cdefs = []

        if self.global_xy_limit is not None:
            sources.append(f"#define XTRACK_GLOBAL_POSLIMIT ({self.global_xy_limit})")

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
                /*gpusharedmem*/ char* shared_special_fields"""
            use_shared_copy = True
        src_lines[
            -1
        ] += r""" )
        {
            """

        if self._local_particle_mode == LocalParticleVar.SHARED_COPY and isinstance(
            context, xo.ContextCupy
        ):
            src_lines[-1] += "extern /*gpusharedmem*/ char shared_special_fields[];\r\n"
            use_shared_copy = True

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
                particles, &lpart, part_idx, shared_special_fields );
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
            LocalParticle_sync_to_particles_data( &lpart, particles, ( part_idx == 0 ) ); //only_for_context opencl cuda"
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
                xo.SharedMemPyopenclArg(
                    xo.Int8,
                    name="shared_special_fields",
                    shmem_per_work_group=16,
                    shmem_per_work_item=136,
                )
            )

        kernel_descriptions = {"track_line": xo.Kernel(args=kernel_args)}

        # Internal API can be exposed only on CPU
        if not isinstance(context, xo.ContextCpu):
            kernels = {}
        kernels.update(kernel_descriptions)

        sources = _handle_per_particle_blocks(sources)

        # Compile!
        context.add_kernels(
            sources,
            kernels,
            extra_classes=self.element_classes,
            save_source_as=save_source_as,
            specialize=True,
        )

        if use_shared_copy and isinstance(context, xo.ContextPyopencl):
            kernel_args[-1].update(kernel=context.kernels.track_line)

        self.track_kernel = context.kernels.track_line

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

    def track_elem_by_elem(
        self,
        particles,
        elem_by_elem_particles,
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

        self.track_elem_by_elem_kernel.description.n_threads = particles._capacity
        self.track_elem_by_elem_kernel(
            buffer=self.line._buffer.buffer,
            ele_offsets=self.ele_offsets_dev,
            ele_typeids=self.ele_typeids_dev,
            particles=particles._xobject,
            num_turns=num_turns,
            ele_start=ele_start,
            num_ele_track=num_elements,
            flag_eot=flag_end_turn_actions,
            flag_tbt_monitor=flag_tbt,
            buffer_tbt_monitor=buffer_monitor,
            offset_tbt_monitor=offset_monitor,
            elem_by_elem_particles=elem_by_elem_particles._xobject,
        )

        self.record_last_track = monitor
