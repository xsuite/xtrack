# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from functools import partial
from pathlib import Path

import numpy as np

import xobjects as xo
import xtrack as xt
from xgtpsa import Tpsa, ffi
from xobjects.context import Source
from xobjects.general import Print
from xobjects.hybrid_class import _build_xofields_dict
from .general import _pkg_root
from .internal_record import RecordIdentifier, RecordIndex, generate_get_record
from .particles import Particles
from .track_flags import c_header_flag_mapping

def _float_to_uint64_bits(value):
    value = np.asarray(value).item()
    return np.array([float(value)], dtype=np.float64).view(np.uint64)[0]


def _uint64_bits_to_float(value):
    return np.array([np.uint64(value)], dtype=np.uint64).view(np.float64)[0]


class FloatOrTpsa(xo.RawUnion):
    """8-byte field storing either double bits or a ``tpsa_t*`` pointer."""

    scalar = xo.Float64
    tpsa = xo.UInt64
    _is_float_or_tpsa = True

    @classmethod
    def _from_buffer(cls, buffer, offset=0, container=None):
        data = buffer.to_bytearray(offset, cls._size)
        bits = np.frombuffer(data, dtype=np.uint64)[0]
        enabled = False
        if container is not None:
            enabled = getattr(container, "_tpsa_enabled", 0)
            if not enabled and hasattr(container, "_xobject"):
                enabled = getattr(container._xobject, "_tpsa_enabled", 0)
        if enabled:
            ptr = ffi().cast("void*", int(bits))
            descriptor = getattr(container, "_tpsa_descriptor", None)
            if descriptor is None and hasattr(container, "_DressingClass"):
                raise ValueError(
                    "Cannot decode a TPSA-enabled FloatOrTpsa field without "
                    "the owning beam element"
                )
            return Tpsa.from_ptr(ptr, descriptor=descriptor)
        return _uint64_bits_to_float(bits)

    @classmethod
    def _to_buffer(cls, buffer, offset, value, info=None, container=None):
        if isinstance(value, cls):
            buffer.update_from_xbuffer(offset, value._buffer, value._offset, cls._size)
            return
        if isinstance(value, Tpsa):
            if container is None:
                raise ValueError(
                    "FloatOrTpsa fields can only be initialized from scalars "
                    "without an owning container"
                )
            bits = int(ffi().cast("uintptr_t", value.ptr))
        else:
            bits = int(_float_to_uint64_bits(value))
        data = np.array([bits], dtype=np.uint64).tobytes()
        buffer.update_from_buffer(offset, data)


def _float_or_tpsa_accessor_paths(xostruct):
    paths = []
    seen = set()
    for path in xostruct._gen_data_paths():
        if not getattr(path[-1], "_is_float_or_tpsa", False):
            continue
        field_names = [part.name for part in path if hasattr(part, "name")]
        key = tuple(field_names)
        if key in seen:
            continue
        seen.add(key)
        flag_names = field_names[:-1] + ["_tpsa_enabled"]
        paths.append(("_".join(field_names), "_".join(flag_names)))
    return paths


def _generate_float_or_tpsa_accessors(xostruct):
    paths = _float_or_tpsa_accessor_paths(xostruct)
    if not paths:
        return None

    data_name = xostruct.__name__
    blocks = ['#include "xtrack/headers/track.h"']
    for field_name, flag_name in paths:
        getter = f"{data_name}_get_{field_name}"
        scalar_getter = f"{getter}_scalar"
        tpsa_getter = f"{getter}_tpsa"
        tpsa_enabled_getter = f"{data_name}_get_{flag_name}"
        blocks.append(f"""
#ifndef XTRACK_TPSA_TRACK
GPUFUN double {getter}({data_name} obj) {{
    if ({tpsa_enabled_getter}(obj)) {{
        return xt_float_or_tpsa_get_double({tpsa_getter}(obj), 1);
    }}
    return {scalar_getter}(obj);
}}
#else
GPUFUN xt_num_t {getter}({data_name} obj) {{
    uint64_t bits = {tpsa_getter}(obj);
    if ({tpsa_enabled_getter}(obj)) {{
        return xt_float_or_tpsa_get(&bits, 1);
    }}
    return {scalar_getter}(obj);
}}
#endif
""")

    source = Source(
        source="\n".join(blocks),
        name=f"{data_name}_float_or_tpsa_accessors",
    )
    return source


class _FieldOfBeamElement:
    def __init__(self, name, xo_field, base_descriptor):
        self.name = name
        self.xo_field = xo_field
        self.base_descriptor = base_descriptor

    def __get__(self, element, owner=None):
        if element is None:
            return self
        if getattr(element._xobject, "_tpsa_enabled", 0):
            handles = _get_tpsa_handles(element)
            try:
                return handles[self.name]
            except KeyError:
                value = self._read(element)
                handles[self.name] = value
                return value
        return self._read(element)

    def __set__(self, element, value):
        if isinstance(value, Tpsa):
            if not isinstance(element._buffer.context, xo.ContextCpu):
                raise NotImplementedError(
                    "TPSA-enabled beam elements are only supported on CPU contexts")
            if not getattr(element._xobject, "_tpsa_enabled", 0):
                element.enable_tpsa(value.descriptor)
            _set_tpsa_handle(element, self.name, value)
            self._write(element, value)
            element._xobject._tpsa_enabled = 1
            return

        if getattr(element._xobject, "_tpsa_enabled", 0):
            descriptor = _get_tpsa_descriptor(element)
            value = descriptor.constant(float(np.asarray(value).item()))
            _set_tpsa_handle(element, self.name, value)
            self._write(element, value)
            return

        _get_tpsa_handles(element).pop(self.name, None)
        self._write(element, float(np.asarray(value).item()))

    def _read(self, element):
        ftype, offset = self.xo_field.get_offset(element._xobject)
        return ftype._from_buffer(
            element._xobject._buffer, offset, container=element)

    def _write(self, element, value):
        ftype, offset = self.xo_field.get_offset(element._xobject)
        ftype._to_buffer(
            element._xobject._buffer, offset, value, container=element)


def _get_tpsa_handles(element):
    handles = element.__dict__.get("_tpsa_handles")
    if handles is None:
        handles = {}
        element.__dict__["_tpsa_handles"] = handles
    return handles


def _get_tpsa_descriptor(element):
    descriptor = element.__dict__.get("_tpsa_descriptor")
    if descriptor is None:
        raise RuntimeError(
            f"{element.__class__.__name__} is TPSA-enabled but has no descriptor"
        )
    return descriptor


def _set_tpsa_handle(element, name, value):
    descriptor = element.__dict__.get("_tpsa_descriptor")
    if descriptor is None:
        element.__dict__["_tpsa_descriptor"] = value.descriptor
    elif descriptor is not value.descriptor:
        raise ValueError("All TPSA fields on an element need to use the same descriptor")
    _get_tpsa_handles(element)[name] = value

def _handle_per_particle_blocks(sources):

    if isinstance(sources, str):
        sources = (sources, )
        wasstring = True
    else:
        wasstring = False

    out = []
    for ii, ss in enumerate(sources):
        if isinstance(ss, Path):
            with open(ss, 'r') as fid:
                strss = fid.read()
        else:
            strss = ss

        if '//start_per_particle_block' in strss:

            lines = strss.splitlines()
            for ill, ll in enumerate(lines):
                if '//start_per_particle_block' in ll:
                    indent = ll[:len(ll) - len(ll.lstrip())]
                    lines[ill] = f"{indent}START_PER_PARTICLE_BLOCK(part0, part);"
                if '//end_per_particle_block' in ll:
                    indent = ll[:len(ll) - len(ll.lstrip())]
                    lines[ill] = f"{indent}END_PER_PARTICLE_BLOCK;"

            # TODO: this is very dirty, just for check!!!!!
            out.append('\n'.join(lines))
        else:
            out.append(strss)


    if wasstring:
        out = out[0]

    return out


def _generate_track_local_particle_with_transformations(
    element_name,
    allow_rot_and_shift,
    rot_and_shift_from_parent,
    isthick,
    xofields,
    is_thin_slice,
):
    options = {
        'ELEMENT_NAME': element_name,
    }

    if allow_rot_and_shift:
        options['ALLOW_ROT_AND_SHIFT'] = 1

    if rot_and_shift_from_parent:
        options['IS_SLICE'] = 1
        curves_reference_frame = hasattr(xofields['_parent']._reftype, 'h')
    else:
        curves_reference_frame = 'h' in xofields

    if curves_reference_frame:
        options['CURVED'] = 1

    if 'isthick' in xofields:
        options['IS_THICK_DYNAMIC'] = 1
    elif isthick:
        options['IS_THICK'] = 1

    if is_thin_slice and curves_reference_frame:
        options['THIN_SLICE_OF_CURVED_ELEMENT'] = 1

    preamble_lines = []
    epilogue_lines = []

    for flag, value in options.items():
        preamble_lines.append(f'#define {flag} {value}')
        epilogue_lines.append(f'#undef {flag}')

    source_lines = [
        *preamble_lines,
        '#include "xtrack/headers/track_local_particle_with_transformations.h"',
        *epilogue_lines,
    ]
    return '\n'.join(source_lines)


def _generate_per_particle_kernel_from_local_particle_function(
                                                element_name, kernel_name,
                                                local_particle_function_name,
                                                additional_args=[]):

    if len(additional_args) > 0:
        add_to_signature = ", ".join([
            f"{' /*gpuglmem*/ ' if arg.pointer else ''} {arg.get_c_type()} {arg.name}"
                for arg in additional_args]) + ", "
        add_to_call = ", " + ", ".join(f"{arg.name}" for arg in additional_args)

    source = ('''
            #ifndef XTRACK_TPSA_TRACK
            /*
             * Scalar element kernels operate on ParticlesData and are not supported
             * when compiling a TPSA tracker, which uses TpsaParticleData instead.
             */
            /*gpukern*/
            '''
            f'void {kernel_name}(\n'
            f'               {element_name}Data el,\n'
'''
                             ParticlesData particles,
'''
            f'{(add_to_signature if len(additional_args) > 0 else "")}'
'''
                             int64_t flag_increment_at_element,
                /*gpuglmem*/ int8_t* io_buffer){

            #define CONTEXT_OPENMP  //only_for_context cpu_openmp
            #ifdef CONTEXT_OPENMP
                const int64_t capacity = ParticlesData_get__capacity(particles);
                const int num_threads = omp_get_max_threads();

                #ifndef XT_OMP_SKIP_REORGANIZE
                    const int64_t num_particles_to_track = ParticlesData_get__num_active_particles(particles);

                    {
                        LocalParticle lpart;
                        lpart.io_buffer = io_buffer;
                        Particles_to_LocalParticle(particles, &lpart, 0, capacity);
                        check_is_active(&lpart);
                        count_reorganized_particles(&lpart);
                        LocalParticle_to_Particles(&lpart, particles, 0, capacity);
                    }
                #else // When we skip reorganize, we cannot just batch active particles
                    const int64_t num_particles_to_track = capacity;
                #endif

                const int64_t chunk_size = (num_particles_to_track + num_threads - 1)/num_threads; // ceil division
            #endif // CONTEXT_OPENMP

            #pragma omp parallel for                                                           //only_for_context cpu_openmp
            for (int64_t batch_id = 0; batch_id < num_threads; batch_id++) {                   //only_for_context cpu_openmp
                LocalParticle lpart;
                lpart.io_buffer = io_buffer;
                lpart.track_flags = 0;
                int64_t part_id = batch_id * chunk_size;                                       //only_for_context cpu_openmp
                int64_t end_id = (batch_id + 1) * chunk_size;                                  //only_for_context cpu_openmp
                if (end_id > num_particles_to_track) end_id = num_particles_to_track;          //only_for_context cpu_openmp

                int64_t part_id = 0;                    //only_for_context cpu_serial
                int64_t part_id = blockDim.x * blockIdx.x + threadIdx.x; //only_for_context cuda
                int64_t part_id = get_global_id(0);                    //only_for_context opencl
                int64_t end_id = 0; // unused outside of openmp  //only_for_context cpu_serial cuda opencl

                int64_t part_capacity = ParticlesData_get__capacity(particles);
                if (part_id<part_capacity){
                    Particles_to_LocalParticle(particles, &lpart, part_id, end_id);
                    if (check_is_active(&lpart)>0){
    '''
            f'          {local_particle_function_name}(el, &lpart{(add_to_call if len(additional_args) > 0 else "")});\n'
    '''
                    }
                    if (check_is_active(&lpart)>0 && flag_increment_at_element){
                            increment_at_element(&lpart, 1);
                    }
                }
            } //only_for_context cpu_openmp

            // On OpenMP we want to additionally by default reorganize all
            // the particles.
            #ifndef XT_OMP_SKIP_REORGANIZE                             //only_for_context cpu_openmp
            LocalParticle lpart;                                       //only_for_context cpu_openmp
            lpart.io_buffer = io_buffer;                               //only_for_context cpu_openmp
            Particles_to_LocalParticle(particles, &lpart, 0, capacity);//only_for_context cpu_openmp
            check_is_active(&lpart);                                   //only_for_context cpu_openmp
            #endif                                                     //only_for_context cpu_openmp
        }
            #endif /* XTRACK_TPSA_TRACK */
''')
    return source


def _tranformations_active(beam_element):
    """This internal function is provided for backward compatibility but
    should not be used and will beb removed soon. Use the following instead:"""
    return beam_element.transformations_active



class MetaBeamElement(xo.MetaHybridClass):

    def __new__(cls, name, bases, data):

        _XoStruct_name = name+'Data'

        data_in = data.copy()
        data = {}

        for bb in bases:
            if bb.__name__ == 'HybridClass':
                continue
            if bb.__name__ == 'BeamElement':
                continue
            for kk, vv in bb.__dict__.items():
                if kk.startswith('__') or kk in data_in.keys():
                    continue
                data[kk] = vv

        # If inheriting _extra_c_sources, remove get_record function
        if '_extra_c_sources' in data:
            ii_remove = None
            for ii, ss in enumerate(data['_extra_c_sources']):
                if isinstance(ss, str) and '/*---GENERATED GET RECORD FUNCTION---*/' in ss:
                   ii_remove = ii
                   break
            if ii_remove is not None:
                data['_extra_c_sources'].pop(ii_remove)

        data.update(data_in)

        data['_isthick'] = False
        istk = data.pop('isthick', False)
        if istk in [True, False]:
            data['_isthick'] = istk
        else: # is property
            data['isthick'] = istk

        # Take xofields from data['_xofields'] or from bases
        xofields = _build_xofields_dict(bases, data)

        allow_rot_and_shift = data.get('allow_rot_and_shift', True)

        # For now assume that when there is a parent, the element inherits the parent's transformations
        rot_and_shift_from_parent = False
        if '_parent' in xofields.keys():
            assert 'rot_and_shift_from_parent' in data.keys()
            rot_and_shift_from_parent = data['rot_and_shift_from_parent']

        if allow_rot_and_shift and not rot_and_shift_from_parent:
            xofields['shift_x'] = xo.Field(xo.Float64, 0)
            xofields['shift_y'] = xo.Field(xo.Float64, 0)
            xofields['shift_s'] = xo.Field(xo.Float64, 0)
            xofields['rot_s_rad'] = xo.Field(xo.Float64)
            xofields['rot_x_rad'] = xo.Field(xo.Float64, 0)
            xofields['rot_y_rad'] = xo.Field(xo.Float64, 0)
            xofields['rot_s_rad_no_frame'] = xo.Field(xo.Float64, 0)
            xofields['rot_shift_anchor'] = xo.Field(xo.Float64, 0)

        data = data.copy()
        data['_xofields'] = xofields
        data['_float_or_tpsa_fields'] = tuple(
            nn for nn, tt in xofields.items()
            if getattr(tt, '_is_float_or_tpsa', False)
        )

        depends_on = []
        extra_c_source = [
            _pkg_root.joinpath('headers','constants.h'),
            _pkg_root.joinpath('headers','checks.h'),
            _pkg_root.joinpath('headers','particle_states.h'),
            _pkg_root.joinpath('beam_elements', 'elements_src', 'track_srotation.h'),
            _pkg_root.joinpath('beam_elements', 'elements_src', 'track_drift.h'),
            c_header_flag_mapping
        ]
        kernels = {}

        # Handle internal record
        if '_internal_record_class' in data.keys():
            data['_xofields']['_internal_record_id'] = RecordIdentifier
            if '_skip_in_to_dict' not in data.keys():
                data['_skip_in_to_dict'] = []
            data['_skip_in_to_dict'].append('_internal_record_id')

            depends_on.append(RecordIndex)
            depends_on.append(data['_internal_record_class']._XoStruct)
            extra_c_source.append(
                generate_get_record(ele_classname=_XoStruct_name,
                    record_classname=data['_internal_record_class']._XoStruct.__name__))

        # Get user-defined source, dependencies and kernels
        if '_extra_c_sources' in data.keys():
            extra_c_source.extend(data['_extra_c_sources'])

        if '_depends_on' in data.keys():
            depends_on.extend(data['_depends_on'])

        if '_kernels' in data.keys():
            kernels.update(data['_kernels'])

        # Add dependency on Particles class
        depends_on.append(Particles._XoStruct)

        track_kernel_name = None
        if ('allow_track' not in data.keys() or data['allow_track']):
            extra_c_source.append(
                _generate_track_local_particle_with_transformations(
                    element_name=name,
                    allow_rot_and_shift=(allow_rot_and_shift or rot_and_shift_from_parent),
                    rot_and_shift_from_parent=rot_and_shift_from_parent,
                    isthick=data['_isthick'],
                    xofields=xofields,
                    is_thin_slice=(
                        '_ThinSliceElementBase' in (base.__name__ for base in bases)),
                )
            )

            # Generate track kernel
            extra_c_source.append(
                _generate_per_particle_kernel_from_local_particle_function(
                    element_name=name, kernel_name=name+'_track_particles',
                    local_particle_function_name=name+'_track_local_particle_with_transformations'))

            # Define track kernel
            track_kernel_name = f'{name}_track_particles'
            kernels[track_kernel_name] = xo.Kernel(
                c_name=track_kernel_name,
                args=[xo.Arg(xo.ThisClass, name='el'),
                    xo.Arg(Particles._XoStruct, name='particles'),
                    xo.Arg(xo.Int64, name='flag_increment_at_element'),
                    xo.Arg(xo.Int8, pointer=True, name="io_buffer")]
            )

        # Generate per-particle kernels
        if '_per_particle_kernels' in data.keys():
            for nn, kk in data['_per_particle_kernels'].items():
                extra_c_source.append(
                    _generate_per_particle_kernel_from_local_particle_function(
                        element_name=name, kernel_name=nn,
                        local_particle_function_name=kk.c_name,
                        additional_args=kk.args))
                if Particles._XoStruct not in depends_on:
                    depends_on.append(Particles._XoStruct)

                kernels.update(
                    {nn: xo.Kernel(
                        c_name=nn,
                        args=[
                        xo.Arg(xo.ThisClass, name='el'),
                        xo.Arg(Particles._XoStruct, name='particles'),
                        *kk.args,
                        xo.Arg(xo.Int64, name='flag_increment_at_element'),
                        xo.Arg(xo.Int8, pointer=True, name="io_buffer"),
                    ])}
                )

        # Call HybridClass metaclass
        data['_depends_on'] = depends_on
        data['_extra_c_sources'] = extra_c_source
        og_kernels = data.get('_kernels', {}).copy()
        data['_kernels'] = kernels
        new_class = xo.MetaHybridClass.__new__(cls, name, bases, data)

        # Attach some information to the class
        new_class._track_kernel_name = track_kernel_name
        if '_internal_record_class' in data.keys():
            new_class._XoStruct._internal_record_class = data['_internal_record_class']
            new_class._internal_record_class = data['_internal_record_class']

        float_or_tpsa_accessors = _generate_float_or_tpsa_accessors(new_class._XoStruct)
        if float_or_tpsa_accessors is not None:
            new_class._XoStruct._extra_c_sources.insert(0, float_or_tpsa_accessors)

        for ff in new_class._XoStruct._fields:
            if ff.ftype is FloatOrTpsa:
                pyname = new_class._rename.get(ff.name, ff.name)
                setattr(new_class, pyname, _FieldOfBeamElement(
                    ff.name, ff, new_class.__dict__[pyname]))

        # Attach methods corresponding to per-particle kernels
        if '_per_particle_kernels' in data.keys():
            for nn, desc in data['_per_particle_kernels'].items():
                setattr(new_class, nn, PerParticlePyMethodDescriptor(
                    kernel_name=nn,
                    additional_arg_names=tuple(arg.name for arg in desc.args),
                ))

        # Attach methods corresponding to kernels
        for nn, desc in og_kernels.items():
            setattr(new_class, nn, PyMethodDescriptor(
                kernel_name=nn,
                additional_arg_names=tuple(arg.name for arg in desc.args),
            ))

        return new_class


class BeamElement(xo.HybridClass, metaclass=MetaBeamElement):

    iscollective = False
    behaves_like_drift = False
    allow_track = True
    has_backtrack = False
    allow_loss_refinement = False
    allow_rot_and_shift = True
    skip_in_loss_location_refinement = False
    needs_rng = False
    name_associated_aperture = None
    prototype = None

    def __init__(self, *args, **kwargs):
        xo.HybridClass.__init__(self, *args, **kwargs)
        if getattr(self, "_float_or_tpsa_fields", ()):
            self.__dict__.setdefault("_tpsa_handles", {})
            self.__dict__.setdefault("_tpsa_descriptor", None)

    def _field_raw_bits(self, name):
        xo_obj = self._xobject
        offset = xo_obj._get_offset(name)
        data = xo_obj._buffer.to_bytearray(offset, 8)
        return int(np.frombuffer(data, dtype=np.uint64)[0])

    def _field_raw_float(self, name):
        return float(_uint64_bits_to_float(self._field_raw_bits(name)))

    def _set_float_or_tpsa_field(self, name, value):
        getattr(type(self), name).__set__(self, value)

    def enable_tpsa(self, descriptor_or_proto):
        if not isinstance(self._buffer.context, xo.ContextCpu):
            raise NotImplementedError(
                "TPSA-enabled beam elements are only supported on CPU contexts")
        if isinstance(descriptor_or_proto, Tpsa):
            descriptor = descriptor_or_proto.descriptor
        else:
            descriptor = descriptor_or_proto
        self.__dict__["_tpsa_descriptor"] = descriptor
        handles = {}
        for name in self._float_or_tpsa_fields:
            if self._xobject._tpsa_enabled:
                value = getattr(self, name)
            else:
                value = descriptor.constant(self._field_raw_float(name))
            handles[name] = value
            setattr(self._xobject, name, value)
        self.__dict__["_tpsa_handles"] = handles
        self._xobject._tpsa_enabled = 1

    def disable_tpsa(self):
        if not getattr(self._xobject, "_tpsa_enabled", 0):
            return
        handles = _get_tpsa_handles(self)
        for name in self._float_or_tpsa_fields:
            value = handles.get(name)
            if value is None:
                value = self._field_raw_float(name)
            else:
                value = value.const_part
            setattr(self._xobject, name, float(value))
        self._xobject._tpsa_enabled = 0
        self.__dict__["_tpsa_handles"] = {}
        self.__dict__["_tpsa_descriptor"] = None

    @property
    def isthick(self):
        return self._isthick

    @isthick.setter
    def isthick(self, value):
        if value != self._isthick:
            raise AttributeError("The property 'isthick' cannot be changed dynamically for "
                             f"elements of type {self.__class__.__name__}")

    def init_pipeline(self, pipeline_manager, name, partners_names=[]):
        self._pipeline_manager = pipeline_manager
        self.name = name
        self.partners_names = partners_names

    def compile_kernels(self, *args, **kwargs):
        if 'apply_to_source' not in kwargs.keys():
            kwargs['apply_to_source'] = []
        kwargs['apply_to_source'].append(_handle_per_particle_blocks)

        only_if_needed = kwargs.pop('only_if_needed', True)

        xo.HybridClass.compile_kernels(
            self,
            extra_classes=[Particles._XoStruct],
            extra_compile_args=(),
            only_if_needed=only_if_needed,
            *args,
            **kwargs,
        )

    def track(self, particles=None, increment_at_element=False):
        if not self.allow_track:
            raise RuntimeError(f"BeamElement {self.__class__.__name__} "
                             + f"has no valid track method.")
        elif particles is None:
            raise RuntimeError("Please provide particles to track!")

        if not isinstance(particles, xt.Particles):
            from xtrack.tpsa import ParticlesTpsa
            if not isinstance(particles, ParticlesTpsa):
                raise TypeError(f"Cannot track particles of type {type(particles)}")
            line = xt.Line(elements=[self], element_names=["__element__"])
            line.particle_ref = particles._ref_particle.copy()
            line.build_tracker(use_prebuilt_kernels=False)
            return line.track(particles)

        if getattr(self, "_tpsa_enabled", 0):
            raise RuntimeError(
                "Cannot track normal Particles through a TPSA-enabled "
                f"{self.__class__.__name__}. Disable TPSA on the element first."
            )

        if self.needs_rng and not particles._has_valid_rng_state():
            particles._init_random_number_generator()

        context = self._buffer.context

        if self._track_kernel_name not in context.kernels:
            self.compile_kernels()

        _track_kernel = context.kernels[self._track_kernel_name]

        if hasattr(self, 'io_buffer') and self.io_buffer is not None:
            io_buffer_arr = self.io_buffer.buffer
        else:
            io_buffer_arr = context.zeros(1, dtype=np.int8)  # dummy

        _track_kernel.description.n_threads = particles._capacity
        _track_kernel(el=self._xobject, particles=particles,
                      flag_increment_at_element=increment_at_element,
                      io_buffer=io_buffer_arr)

    @property
    def context(self):
        return self._buffer.context

    def _arr2ctx(self, arr):
        ctx = self._buffer.context

        if isinstance(arr, list):
            arr = np.array(arr)

        if np.isscalar(arr):
            if hasattr(arr, 'item'):
                return arr.item()
            else:
                return arr
        elif isinstance(arr, ctx.nplike_array_type):
            return arr
        elif isinstance(arr, np.ndarray):
            return ctx.nparray_to_context_array(arr)
        else:
            raise ValueError("Invalid array type")

    def xoinitialize(self, **kwargs):
        rot_s_rad = kwargs.pop('rot_s_rad', kwargs.pop('_rot_s_rad', None))
        shift_x = kwargs.pop('shift_x', kwargs.pop('_shift_x', None))
        shift_y = kwargs.pop('shift_y', kwargs.pop('_shift_y', None))
        shift_s = kwargs.pop('shift_s', kwargs.pop('_shift_s', None))
        rot_x_rad = kwargs.pop('rot_x_rad', kwargs.pop('_rot_x_rad', None))
        rot_y_rad = kwargs.pop('rot_y_rad', kwargs.pop('_rot_y_rad', None))
        rot_s_rad_no_frame = kwargs.pop('rot_s_rad_no_frame', kwargs.pop('_rot_s_rad_no_frame', None))

        xo.HybridClass.xoinitialize(self, **kwargs)

        if rot_s_rad is not None:
            self.rot_s_rad = rot_s_rad

        rot_s_rad_legacy_from_trig = False
        sin_s_rad = 0
        cos_s_rad = 1

        if '_sin_rot_s' in kwargs or '_cos_rot_s' in kwargs:
            rot_s_rad_legacy_from_trig = True
            sin_s_rad = kwargs.pop('_sin_rot_s', 0)
            cos_s_rad = kwargs.pop('_cos_rot_s', 0)

        if rot_s_rad_legacy_from_trig:
            computed_rot_s_rad = np.arctan2(sin_s_rad, cos_s_rad)
            if rot_s_rad is not None:
                if not np.isclose(rot_s_rad, computed_rot_s_rad, atol=1e-14, rtol=1e-14):
                    raise ValueError(
                        f'{type(self).__name__} initialised with both `rot_s_rad` '
                        f'and `_sin_rot_s` or `_cos_rot_s` arguments, but they are '
                        f'inconsistent with each other.'
                    )
            else:
                self.rot_s_rad = computed_rot_s_rad

        if shift_x is not None:
            self.shift_x = shift_x

        if shift_y is not None:
            self.shift_y = shift_y

        if shift_s is not None:
            self.shift_s = shift_s

        if rot_x_rad is not None:
            self.rot_x_rad = rot_x_rad

        if rot_y_rad is not None:
            self.rot_y_rad = rot_y_rad

        if rot_s_rad_no_frame is not None:
            self.rot_s_rad_no_frame = rot_s_rad_no_frame

    def to_dict(self, **kwargs):
        if getattr(self._xobject, "_tpsa_enabled", 0):
            raise NotImplementedError(
                "Serializing TPSA-enabled beam elements is not supported")
        dct = xo.HybridClass.to_dict(self, **kwargs)
        if self.name_associated_aperture is not None:
            dct['name_associated_aperture'] = self.name_associated_aperture
        if hasattr(self, 'extra') and self.extra:
            dct['extra'] = self.extra.copy()
        if hasattr(self, 'prototype') and self.prototype is not None:
            dct['prototype'] = self.prototype
        return dct

    @classmethod
    def from_dict(cls, dct, **kwargs):
        if 'name_associated_aperture' in dct.keys():
            name_associated_aperture = dct.pop('name_associated_aperture')
        else:
            name_associated_aperture = None

        instance = xo.HybridClass._static_from_dict(cls, dct, **kwargs)
        instance.name_associated_aperture = name_associated_aperture

        if 'extra' in dct.keys():
            instance.extra = dct['extra'].copy()
        if 'prototype' in dct.keys():
            instance.prototype = dct['prototype']
        return instance

    def copy(self, **kwargs):
        out = super().copy(**kwargs)
        if hasattr(self, 'extra'):
            try:
                out.extra = self.extra.copy()
            except AttributeError:
                out.extra = self.extra
        if hasattr(self, 'prototype'):
            out.prototype = self.prototype
        return out

    @property
    def transformations_active(self):
        if not self.allow_rot_and_shift:
            return False
        if hasattr(self, '_parent') and self.rot_and_shift_from_parent:
            return self._parent.transformations_active()
        if np.any([
            self.shift_x,
            self.shift_y,
            self.shift_s,
            self.rot_s_rad,
            self.rot_x_rad,
            self.rot_y_rad,
            self.rot_s_rad_no_frame,
        ]):
            return True
        return False

    @property
    def _add_to_repr(self):
        out = []
        if hasattr(self, 'parent_name'):
            out.append('parent_name')
        return out

class Replica:
    def __init__(self, parent_name):
        self.parent_name = parent_name

    def __repr__(self):
        return f'Replica(parent_name="{self.parent_name}")'

    def to_dict(self):
        return {
            '__class__': 'Replica',
            'parent_name': self.parent_name}

    @classmethod
    def from_dict(cls, dct, **kwargs):
        return cls(parent_name=dct['parent_name'])

    def copy(self, **kwargs):
        return Replica(parent_name=self.parent_name)

    def resolve(self, element_container, get_name=False):
        if hasattr(element_container, '_element_dict'):
            element_container = element_container._element_dict
        target_name = self.parent_name
        visited = {target_name}
        while isinstance(element := element_container[target_name], Replica):
            target_name = element.parent_name
            if target_name in visited:
                raise RecursionError(
                    f"Resolving replica of `{self.parent_name}` leads to a "
                    "circular reference: check the correctness of your line."
                )
            visited.add(target_name)

        if get_name:
            return target_name

        return element_container[target_name]

class PerParticlePyMethod:

    def __init__(self, kernel_name, element, additional_arg_names):
        self.kernel_name = kernel_name
        self.element = element
        self.additional_arg_names = additional_arg_names

    def __call__(self, particles, increment_at_element=False, **kwargs):
        instance = self.element
        context = instance._context

        only_if_needed = kwargs.pop('only_if_needed', True)
        BeamElement.compile_kernels(instance, only_if_needed=only_if_needed)
        kernel = context.kernels[self.kernel_name]

        if hasattr(self.element, 'io_buffer') and self.element.io_buffer is not None:
            io_buffer_arr = self.element.io_buffer.buffer
        else:
            io_buffer_arr = context.zeros(1, dtype=np.int8)  # dummy

        kernel.description.n_threads = particles._capacity
        kernel(el=self.element._xobject,
               particles=particles,
               flag_increment_at_element=increment_at_element,
               io_buffer=io_buffer_arr,
               **kwargs)


class PerParticlePyMethodDescriptor:
    def __init__(self, kernel_name, additional_arg_names):
        self.kernel_name = kernel_name
        self.additional_arg_names = additional_arg_names

    def __get__(self, instance, owner):
        return PerParticlePyMethod(kernel_name=self.kernel_name,
                                   element=instance,
                                   additional_arg_names=self.additional_arg_names)


class PyMethod:

    def __init__(self, kernel_name, element, additional_arg_names):
        self.kernel_name = kernel_name
        self.element = element
        self.additional_arg_names = additional_arg_names

    def __call__(self, **kwargs):
        instance = self.element
        context = instance._context

        only_if_needed = kwargs.pop('only_if_needed', True)
        BeamElement.compile_kernels(instance, only_if_needed=only_if_needed)
        kernel = context.kernels[self.kernel_name]

        el_var_name = None
        for arg in instance._kernels[self.kernel_name].args:
            if arg.atype == instance.__class__._XoStruct:
                el_var_name = arg.name
        if el_var_name is None:
            raise ValueError(f"Kernel {self.kernel_name} does not depend "
                           + f"on element type {instance.__class__._XoStruct} "
                           + f"which it is attached to!")
        kwargs[el_var_name] = instance._xobject

        return kernel(**kwargs)


class PyMethodDescriptor:
    def __init__(self, kernel_name, additional_arg_names):
        self.kernel_name = kernel_name
        self.additional_arg_names = additional_arg_names

    def __get__(self, instance, owner):
        return PyMethod(kernel_name=self.kernel_name,
                        element=instance,
                        additional_arg_names=self.additional_arg_names)
