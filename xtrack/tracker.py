from pathlib import Path
import numpy as np

from .particles import Particles, gen_local_particle_api
from .general import _pkg_root
from .line import Line

import xobjects as xo

api_conf = {'prepointer': ' /*gpuglmem*/ '}

class Tracker:

    def __init__(self, context, sequence,
            particles_class=Particles,
            local_particle_src=None,
            save_source_as=None):


        line = Line(_context=context, sequence=sequence)

        sources = []
        kernels = {}
        cdefs = []

        sources.append(_pkg_root.joinpath('headers/constants.h'))

        # Particles
        source_particles, kernels_particles, cdefs_particles = (
                                    particles_class.XoStruct._gen_c_api(conf=api_conf))
        sources.append(source_particles)
        kernels.update(kernels_particles)
        cdefs += cdefs_particles.split('\n')

        # Local particles
        if local_particle_src is None:
            local_particle_src = gen_local_particle_api()
        sources.append(local_particle_src)

        # Elements
        element_classes = line._ElementRefClass._rtypes
        for cc in element_classes:
            ss, kk, dd = cc._gen_c_api(conf=api_conf)
            sources.append(ss)
            kernels.update(kk)
            cdefs += dd.split('\n')

            sources += cc.extra_sources

        sources.append(_pkg_root.joinpath('tracker_src/tracker.h'))

        cdefs_norep=[] # TODO Check if this can be handled be the context
        for cc in cdefs:
            if cc not in cdefs_norep:
                cdefs_norep.append(cc)

        src_lines = []
        src_lines.append(r'''
            /*gpukern*/
            void track_line(
                /*gpuglmem*/ int8_t* buffer,
                /*gpuglmem*/ int64_t* ele_offsets,
                /*gpuglmem*/ int64_t* ele_typeids,
                             ParticlesData particles,
                             int num_turns,
                             int ele_start,
                             int num_ele_track){


            LocalParticle lpart;

            int64_t part_id = 0;                    //only_for_context cpu_serial cpu_openmp
            int64_t part_id = blockDim.x * blockIdx.x + threadIdx.x; //only_for_context cuda
            int64_t part_id = get_global_id(0);                    //only_for_context opencl

            int64_t n_part = ParticlesData_get_num_particles(particles);
            if (part_id<n_part){
            Particles_to_LocalParticle(particles, &lpart, part_id);

            for (int64_t iturn=0; iturn<num_turns; iturn++){
                for (int64_t ee=ele_start; ee<ele_start+num_ele_track; ee++){
                    /*gpuglmem*/ int8_t* el = buffer + ele_offsets[ee];
                    int64_t ee_type = ele_typeids[ee];

                    switch(ee_type){
        ''')

        for ii, cc in enumerate(element_classes):
            ccnn = cc.__name__.replace('Data', '')
            src_lines.append(f'''
                    case {ii}:
                        {ccnn}_track_local_particle(({ccnn}Data) el, &lpart);
                        break;''')

        src_lines.append('''
                    } //switch
                } //for elements
            } //for turns
            }//if
        }//kernel
        ''')

        source_track = '\n'.join(src_lines)
        sources.append(source_track)

        for ii, ss in enumerate(sources):
            if isinstance(ss, Path):
                with open(ss, 'r') as fid:
                    ss = fid.read()
            sources[ii] = ss.replace('/*gpufun*/', '/*gpufun*/ static inline')

        kernel_descriptions = {
            "track_line": xo.Kernel(
                args=[
                    xo.Arg(xo.Int8, pointer=True, name='buffer'),
                    xo.Arg(xo.Int64, pointer=True, name='ele_offsets'),
                    xo.Arg(xo.Int64, pointer=True, name='ele_typeids'),
                    xo.Arg(particles_class.XoStruct, name='particles'),
                    xo.Arg(xo.Int32, name='num_turns'),
                    xo.Arg(xo.Int32, name='ele_start'),
                    xo.Arg(xo.Int32, name='num_ele_track'),
                ],
            )
        }

        # Internal API can be exposed only on CPU
        if not isinstance(context, xo.ContextCpu):
            kernels = {}
        kernels.update(kernel_descriptions)

        # Compile!
        context.add_kernels(sources, kernels, extra_cdef='\n\n'.join(cdefs_norep),
                            save_source_as=save_source_as,
                            specialize=True)

        ele_offsets = np.array([ee._offset for ee in line.elements], dtype=np.int64)
        ele_typeids = np.array(
                [element_classes.index(ee._xobject.__class__) for ee in line.elements],
                dtype=np.int64)
        ele_offsets_dev = context.nparray_to_context_array(ele_offsets)
        ele_typeids_dev = context.nparray_to_context_array(ele_typeids)

        self.context = context
        self.particles_class = particles_class
        self.ele_offsets_dev = ele_offsets_dev
        self.ele_typeids_dev = ele_typeids_dev
        self.track_kernel = context.kernels.track_line
        self.num_elements = len(line.elements)
        self.line = line


    def track(self, particles, ele_start=0, num_elements=None, num_turns=1):

        if num_elements is None:
            num_elements = self.num_elements

        assert num_elements + ele_start <= self.num_elements

        self.track_kernel.description.n_threads = particles.num_particles
        self.track_kernel(
                buffer=self.line._buffer.buffer,
                ele_offsets=self.ele_offsets_dev,
                ele_typeids=self.ele_typeids_dev,
                particles=particles._xobject,
                num_turns=num_turns,
                ele_start=ele_start,
                num_ele_track=num_elements)

