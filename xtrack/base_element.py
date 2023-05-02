# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from pathlib import Path
import numpy as np
from functools import partial

import xobjects as xo
import xpart as xp

from xobjects.hybrid_class import _build_xofields_dict

from .general import _pkg_root
from .internal_record import RecordIdentifier, RecordIndex, generate_get_record

start_per_part_block = """
    {
    const int64_t start_idx = part0->ipart; //only_for_context cpu_openmp
    const int64_t end_idx = part0->endpart; //only_for_context cpu_openmp
    
    const int64_t start_idx = 0;                                            //only_for_context cpu_serial
    const int64_t end_idx = LocalParticle_get__num_active_particles(part0); //only_for_context cpu_serial
    
    //#pragma omp simd // TODO: currently does not work, needs investigating
    for (int64_t ii=start_idx; ii<end_idx; ii++) { //only_for_context cpu_openmp cpu_serial

        LocalParticle lpart = *part0;  //only_for_context cpu_serial cpu_openmp
        LocalParticle* part = &lpart;  //only_for_context cpu_serial cpu_openmp
        part->ipart = ii;              //only_for_context cpu_serial cpu_openmp

        LocalParticle* part = part0;   //only_for_context opencl cuda
        
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
            const int num_threads = omp_get_max_threads();                                 //only_for_context cpu_openmp
            const int64_t capacity = ParticlesData_get__capacity(particles);               //only_for_context cpu_openmp
            const int64_t chunk_size = (capacity + num_threads - 1)/num_threads; // ceil division  //only_for_context cpu_openmp
            #pragma omp parallel for                                                       //only_for_context cpu_openmp
            for (int64_t batch_id = 0; batch_id < num_threads; batch_id++) {               //only_for_context cpu_openmp
                LocalParticle lpart;
                lpart.io_buffer = io_buffer;
                int64_t part_id = batch_id * chunk_size;                                       //only_for_context cpu_openmp
                int64_t end_id = (batch_id + 1) * chunk_size;                                  //only_for_context cpu_openmp
                if (end_id > capacity) end_id = capacity;                                      //only_for_context cpu_openmp
    
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
                            increment_at_element(&lpart);
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

class MetaBeamElement(xo.MetaHybridClass):

    def __new__(cls, name, bases, data):
        _XoStruct_name = name+'Data'

        # Take xofields from data['_xofields'] or from bases
        xofields = _build_xofields_dict(bases, data)
        data = data.copy()
        data['_xofields'] = xofields

        depends_on = []
        extra_c_source = [
            _pkg_root.joinpath('headers','constants.h'),
            _pkg_root.joinpath('headers','checks.h'),
            _pkg_root.joinpath('headers','particle_states.h')
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

        # Generate track kernel
        extra_c_source.append(
            _generate_per_particle_kernel_from_local_particle_function(
                element_name=name, kernel_name=name+'_track_particles',
                local_particle_function_name=name+'_track_local_particle'))

        # Add dependency on Particles class
        depends_on.append(xp.ParticlesBase._XoStruct)

        # Define track kernel
        track_kernel_name = f'{name}_track_particles'
        kernels[track_kernel_name] = xo.Kernel(
                    args=[xo.Arg(xo.ThisClass, name='el'),
                        xo.Arg(xp.ParticlesBase._XoStruct, name='particles'),
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
                if xp.ParticlesBase._XoStruct not in depends_on:
                    depends_on.append(xp.ParticlesBase._XoStruct)

                kernels.update(
                    {nn: xo.Kernel(args=[
                        xo.Arg(xo.ThisClass, name='el'),
                        xo.Arg(xp.ParticlesBase._XoStruct, name='particles'),
                        *kk.args,
                        xo.Arg(xo.Int64, name='flag_increment_at_element'),
                        xo.Arg(xo.Int8, pointer=True, name="io_buffer"),
                    ])}
                )


        # Call HybridClass metaclass
        data['_depends_on'] = depends_on
        data['_extra_c_sources'] = extra_c_source
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

        return new_class


class BeamElement(xo.HybridClass, metaclass=MetaBeamElement):

    iscollective = None
    isthick = False
    behaves_like_drift = False
    allow_backtrack = False
    skip_in_loss_location_refinement = False

    def __init__(self, *args, **kwargs):
        xo.HybridClass.__init__(self, *args, **kwargs)

    def init_pipeline(self, pipeline_manager, name, partners_names=[]):
        self._pipeline_manager = pipeline_manager
        self.name = name
        self.partners_names = partners_names

    def compile_kernels(self, particles_class, *args, **kwargs):
        if 'apply_to_source' not in kwargs.keys():
            kwargs['apply_to_source'] = []
        kwargs['apply_to_source'].append(
            partial(_handle_per_particle_blocks,
                    local_particle_src=particles_class.gen_local_particle_api()))
        xo.HybridClass.compile_kernels(self,
                                       extra_classes=[particles_class._XoStruct],
                                       *args, **kwargs)

    def track(self, particles, increment_at_element=False):
        context = self._buffer.context

        desired_classes = (
            self._XoStruct,  # el
            particles._XoStruct,  # particles
        )

        if (self._track_kernel_name, desired_classes) not in context.kernels:
            self.compile_kernels(particles_class=particles.__class__)

        _track_kernel = context.kernels[(self._track_kernel_name,
                                         desired_classes)]

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


class PerParticlePyMethod:

    def __init__(self, kernel_name, element, additional_arg_names):
        self.kernel_name = kernel_name
        self.element = element
        self.additional_arg_names = additional_arg_names

    def __call__(self, particles, increment_at_element=False, **kwargs):
        instance = self.element
        context = instance.context

        desired_classes = (self.element._XoStruct,  # el
                           particles._XoStruct)  # part0

        if (self.kernel_name, desired_classes) not in context.kernels:
            instance.compile_kernels(particles_class=particles.__class__)

        kernel = context.kernels[(self.kernel_name, desired_classes)]

        if hasattr(self.element, 'io_buffer') and self.element.io_buffer is not None:
            io_buffer_arr = self.element.io_buffer.buffer
        else:
            context = kernel.context
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
