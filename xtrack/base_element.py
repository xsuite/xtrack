# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from pathlib import Path
import numpy as np
from functools import partial

import xobjects as xo
from xobjects.general import Print

from xobjects.hybrid_class import _build_xofields_dict

from .general import _pkg_root
from .internal_record import RecordIdentifier, RecordIndex, generate_get_record
from .particles import Particles

start_per_part_block = """
    {
    const int64_t XT_part_block_start_idx = part0->ipart; //only_for_context cpu_openmp
    const int64_t XT_part_block_end_idx = part0->endpart; //only_for_context cpu_openmp

    const int64_t XT_part_block_start_idx = 0;                                            //only_for_context cpu_serial
    const int64_t XT_part_block_end_idx = LocalParticle_get__num_active_particles(part0); //only_for_context cpu_serial

    //#pragma omp simd // TODO: currently does not work, needs investigating
    for (int64_t XT_part_block_ii = XT_part_block_start_idx; XT_part_block_ii<XT_part_block_end_idx; XT_part_block_ii++) { //only_for_context cpu_openmp cpu_serial

        LocalParticle lpart = *part0;    //only_for_context cpu_serial cpu_openmp
        LocalParticle* part = &lpart;    //only_for_context cpu_serial cpu_openmp
        part->ipart = XT_part_block_ii;  //only_for_context cpu_serial cpu_openmp

        LocalParticle* part = part0;     //only_for_context opencl cuda

        if (LocalParticle_get_state(part) > 0) {  //only_for_context cpu_openmp
"""

end_part_part_block = """
        }  //only_for_context cpu_openmp
    }  //only_for_context cpu_serial cpu_openmp
    }
"""

def _handle_per_particle_blocks(sources, local_particle_src):

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

        strss = strss.replace('/*placeholder_for_local_particle_src*/',
                                local_particle_src,
                                )
        if '//start_per_particle_block' in strss:

            lines = strss.splitlines()
            for ill, ll in enumerate(lines):
                if '//start_per_particle_block' in ll:
                    lines[ill] = start_per_part_block
                if '//end_per_particle_block' in ll:
                    lines[ill] = end_part_part_block

            # TODO: this is very dirty, just for check!!!!!
            out.append('\n'.join(lines))
        else:
            out.append(ss)


    if wasstring:
        out = out[0]

    return out

def _generate_track_local_particle_with_transformations(
                                                element_name,
                                                allow_rot_and_shift,
                                                rot_and_shift_from_parent,
                                                local_particle_function_name):

    source = ('''
            /*gpufun*/
            '''
            f'void {local_particle_function_name}_with_transformations({element_name}Data el, LocalParticle* part0)'
            '{\n')

    if rot_and_shift_from_parent:
        add_to_call = '__parent'
    else:
        add_to_call = ''

    if allow_rot_and_shift:

        source += (
            '    // Transform to local frame\n'
            # f'    printf("Transform to local frame {element_name}\\n");\n'
            f'double const _sin_rot_s = {element_name}Data_get{add_to_call}__sin_rot_s(el);\n'
            'if (_sin_rot_s > -2.) {\n'
            f'    double const _cos_rot_s = {element_name}Data_get{add_to_call}__cos_rot_s(el);\n'
            f'    double const shift_x = {element_name}Data_get{add_to_call}__shift_x(el);\n'
            f'    double const shift_y = {element_name}Data_get{add_to_call}__shift_y(el);\n'
            f'    double const shift_s = {element_name}Data_get{add_to_call}__shift_s(el);\n'
            '\n'
            '    if (shift_s != 0.) {\n'
            '        //start_per_particle_block (part0->part)\n'
            '            Drift_single_particle(part, shift_s);\n'
            '        //end_per_particle_block\n'
            '    }\n'
            '\n'
            '    //start_per_particle_block (part0->part)\n'
            '       LocalParticle_add_to_x(part, -shift_x);\n'
            '       LocalParticle_add_to_y(part, -shift_y);\n'
            '    //end_per_particle_block\n'
            '\n'
            '     //start_per_particle_block (part0->part)\n'
            '          SRotation_single_particle(part, _sin_rot_s, _cos_rot_s);\n'
            '     //end_per_particle_block\n'
            '}\n'
        )

    source += (
            f'    {local_particle_function_name}(el, part0);\n'
    )

    if allow_rot_and_shift:
        source += (
            '    // Transform back to global frame\n'
            #f'    printf("Transform to back to global frame {element_name}\\n");\n'
            'if (_sin_rot_s > -2.) {\n'
            f'    double const _cos_rot_s = {element_name}Data_get{add_to_call}__cos_rot_s(el);\n'
            f'    double const shift_x = {element_name}Data_get{add_to_call}__shift_x(el);\n'
            f'    double const shift_y = {element_name}Data_get{add_to_call}__shift_y(el);\n'
            f'    double const shift_s = {element_name}Data_get{add_to_call}__shift_s(el);\n'
            '\n'
            '    //start_per_particle_block (part0->part)\n'
            '       SRotation_single_particle(part, -_sin_rot_s, _cos_rot_s);\n'
            '    //end_per_particle_block\n'
            '\n'
            '    //start_per_particle_block (part0->part)\n'
            '       LocalParticle_add_to_x(part, shift_x);\n'
            '       LocalParticle_add_to_y(part, shift_y);\n'
            '    //end_per_particle_block\n'
            '\n'
            '    if (shift_s != 0.) {\n'
            '        //start_per_particle_block (part0->part)\n'
            '            Drift_single_particle(part, -shift_s);\n'
            '        //end_per_particle_block\n'
            '    }\n'
            '}\n'
        )
    source += '}\n'
    return source

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
''')
    return source

def _tranformations_active(self):

    if (self.shift_x == 0 and self.shift_y == 0 and self.shift_s == 0
        and self._sin_rot_s == 0 and self._cos_rot_s >= 0): # means angle is zero
        return False
    elif (self.shift_x == 0 and self.shift_y == 0 and self.shift_s == 0
          and self._sin_rot_s < -2.):
        return False
    else:
        return True

def _rot_s_property(self):
    if self._sin_rot_s < -2.:
        return 0.
    return np.arctan2(self._sin_rot_s, self._cos_rot_s)

def _set_rot_s_property_setter(self, value):
    self._sin_rot_s = np.sin(value)
    self._cos_rot_s = np.cos(value)
    if not _tranformations_active(self):
        self._sin_rot_s = -999.
    elif self._sin_rot_s < -2.:
        self._sin_rot_s = 0.
        self._cos_rot_s = 1.

def _shiftx_property(self):
    return self._shift_x

def _set_shiftx_property_setter(self, value):
    self._shift_x = value
    if not _tranformations_active(self):
        self._sin_rot_s = -999.
    elif self._sin_rot_s < -2.:
        self._sin_rot_s = 0.
        self._cos_rot_s = 1.

def _shifty_property(self):
    return self._shift_y

def _set_shifty_property_setter(self, value):
    self._shift_y = value
    if not _tranformations_active(self):
        self._sin_rot_s = -999.
    elif self._sin_rot_s < -2.:
        self._sin_rot_s = 0.
        self._cos_rot_s = 1.

def _shifts_property(self):
    return self._shift_s

def _set_shifts_property_setter(self, value):
    self._shift_s = value
    if not _tranformations_active(self):
        self._sin_rot_s = -999.
    elif self._sin_rot_s < -2.:
        self._sin_rot_s = 0.
        self._cos_rot_s = 1.

class MetaBeamElement(xo.MetaHybridClass):

    def __new__(cls, name, bases, data):
        _XoStruct_name = name+'Data'

        # Take xofields from data['_xofields'] or from bases
        xofields = _build_xofields_dict(bases, data)

        allow_rot_and_shift = data.get('allow_rot_and_shift', True)

        if allow_rot_and_shift:
            xofields['_sin_rot_s'] = xo.Field(xo.Float64, default=-999.)
            xofields['_cos_rot_s'] = xo.Field(xo.Float64, default=-999.)
            xofields['_shift_x'] = xo.Field(xo.Float64, 0)
            xofields['_shift_y'] = xo.Field(xo.Float64, 0)
            xofields['_shift_s'] = xo.Field(xo.Float64, 0)

        data = data.copy()
        data['_xofields'] = xofields

        depends_on = []
        extra_c_source = [
            _pkg_root.joinpath('headers','constants.h'),
            _pkg_root.joinpath('headers','checks.h'),
            _pkg_root.joinpath('headers','particle_states.h'),
            _pkg_root.joinpath('beam_elements', 'elements_src', 'track_srotation.h'),
            _pkg_root.joinpath('beam_elements', 'elements_src', 'drift.h'),
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

        # For now I assume that when there is a parent, the element inherits the parent's transformations
        rot_and_shift_from_parent = False
        if '_parent' in xofields.keys():
            assert 'rot_and_shift_from_parent' in data.keys()
            rot_and_shift_from_parent = data['rot_and_shift_from_parent']

        track_kernel_name = None
        if ('allow_track' not in data.keys() or data['allow_track']):

            extra_c_source.append(
                _generate_track_local_particle_with_transformations(
                    element_name=name,
                    allow_rot_and_shift=(allow_rot_and_shift or rot_and_shift_from_parent),
                    rot_and_shift_from_parent=rot_and_shift_from_parent,
                    local_particle_function_name=name+'_track_local_particle'))

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

        if allow_rot_and_shift:
            new_class.rot_s_rad = property(_rot_s_property, _set_rot_s_property_setter)
            new_class.shift_x = property(_shiftx_property, _set_shiftx_property_setter)
            new_class.shift_y = property(_shifty_property, _set_shifty_property_setter)
            new_class.shift_s = property(_shifts_property, _set_shifts_property_setter)

        return new_class


class BeamElement(xo.HybridClass, metaclass=MetaBeamElement):

    iscollective = False
    isthick = False
    behaves_like_drift = False
    allow_track = True
    has_backtrack = False
    allow_loss_refinement = False
    allow_rot_and_shift = True
    skip_in_loss_location_refinement = False
    needs_rng = False
    name_associated_aperture = None

    def __init__(self, *args, **kwargs):
        xo.HybridClass.__init__(self, *args, **kwargs)

    def init_pipeline(self, pipeline_manager, name, partners_names=[]):
        self._pipeline_manager = pipeline_manager
        self.name = name
        self.partners_names = partners_names

    def compile_kernels(self, extra_classes=(), *args, **kwargs):
        if 'apply_to_source' not in kwargs.keys():
            kwargs['apply_to_source'] = []
        kwargs['apply_to_source'].append(
            partial(_handle_per_particle_blocks,
                    local_particle_src=Particles.gen_local_particle_api()))
        context = self._context
        cls = self.__class__

        if context.allow_prebuilt_kernels:
            # Default config is empty (all flags default to not defined, which
            # enables most behaviours). In the future this has to be looked at
            # whenever a new flag is needed.
            _default_config = {}
            _print_state = Print.suppress
            Print.suppress = True
            classes = (cls._XoStruct,) + tuple(extra_classes)
            try:
                from xsuite import (
                    get_suitable_kernel,
                    XSK_PREBUILT_KERNELS_LOCATION,
                )
            except ImportError:
                kernel_info = None
            else:
                kernel_info = get_suitable_kernel(
                    _default_config, classes
                )

            Print.suppress = _print_state
            if kernel_info:
                module_name, _ = kernel_info
                kernels = context.kernels_from_file(
                    module_name=module_name,
                    containing_dir=XSK_PREBUILT_KERNELS_LOCATION,
                    kernel_descriptions=self._kernels,
                )
                context.kernels.update(kernels)
                return
        xo.HybridClass.compile_kernels(
            self,
            extra_classes=[Particles._XoStruct],
            *args,
            **kwargs,
        )

    def track(self, particles=None, increment_at_element=False):
        if not self.allow_track:
            raise RuntimeError(f"BeamElement {self.__class__.__name__} "
                             + f"has no valid track method.")
        elif particles is None:
            raise RuntimeError("Please provide particles to track!")

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
        rot_s_rad = kwargs.pop('rot_s_rad', None)
        shift_x = kwargs.pop('shift_x', None)
        shift_y = kwargs.pop('shift_y', None)
        shift_s = kwargs.pop('shift_s', None)

        xo.HybridClass.xoinitialize(self, **kwargs)

        if rot_s_rad is not None:
            self.rot_s_rad = rot_s_rad

        if shift_x is not None:
            self.shift_x = shift_x

        if shift_y is not None:
            self.shift_y = shift_y

        if shift_s is not None:
            self.shift_s = shift_s

    def to_dict(self, **kwargs):
        dct = xo.HybridClass.to_dict(self, **kwargs)
        if self.name_associated_aperture is not None:
            dct['name_associated_aperture'] = self.name_associated_aperture
        return dct

    @classmethod
    def from_dict(cls, dct, **kwargs):
        if 'name_associated_aperture' in dct.keys():
            name_associated_aperture = dct.pop('name_associated_aperture')
        else:
            name_associated_aperture = None

        instance = xo.HybridClass._static_from_dict(cls, dct, **kwargs)
        instance.name_associated_aperture = name_associated_aperture
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
        if hasattr(element_container, 'element_dict'):
            element_container = element_container.element_dict
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

