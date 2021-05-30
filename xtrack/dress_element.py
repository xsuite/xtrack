import xobjects as xo
from .particles import ParticlesData, gen_local_particle_api
from .dress import dress

api_conf = {'prepointer': ' /*gpuglmem*/ '}

def dress_element(XoElementData):

    DressedElement = dress(XoElementData)
    assert XoElementData.__name__.endswith('Data')
    name = XoElementData.__name__[:-4]

    DressedElement.track_kernel_source = ('''
            /*gpukern*/
            '''
            f'void {name}_track_particles(\n'
            f'               {name}Data el,\n'
'''
                             ParticlesData particles){
            LocalParticle lpart;
            int64_t part_id = 0;                    //only_for_context cpu_serial cpu_openmp
            int64_t part_id = blockDim.x * blockIdx.x + threadIdx.x; //only_for_context cuda
            int64_t part_id = get_global_id(0);                    //only_for_context opencl

            int64_t n_part = ParticlesData_get_num_particles(particles);
            if (part_id<n_part){
                Particles_to_LocalParticle(particles, &lpart, part_id);
'''
            f'{name}_track_local_particle(el, &lpart);\n'
'''
                }
            }
''')
    DressedElement._track_kernel_name = f'{name}_track_particles'
    DressedElement.track_kernel_description = {DressedElement._track_kernel_name:
        xo.Kernel(args=[xo.Arg(XoElementData, name='el'),
                        xo.Arg(ParticlesData, name='particles')])}

    def compile_track_kernel(self, save_source_as=None):
        context = self._buffer.context

        src_part, _, cdefs_part= ParticlesData._gen_c_api(api_conf)
        src_ele, _, cdefs_ele = self.XoStruct._gen_c_api(api_conf)

        cdefs = '\n'.join([cdefs_part, cdefs_ele])
        cdefs_norep=[]
        for cc in cdefs.split('\n'):
            if cc not in cdefs_norep:
                cdefs_norep.append(cc)
        cdefs_norep = '\n'.join(cdefs_norep)

        context.add_kernels(sources=[
                src_part,
                gen_local_particle_api(),
                src_ele]
                + self.XoStruct.extra_sources
                + [self.track_kernel_source],
            kernels=self.track_kernel_description,
            extra_cdef=cdefs_norep,
            save_source_as=save_source_as)


    def track(self, particles):

        if not hasattr(self, '_track_kernel'):
            context = self._buffer.context
            if self._track_kernel_name not in context.kernels.keys():
                self.compile_track_kernel()
            self._track_kernel = context.kernels[self._track_kernel_name]

        self._track_kernel.description.n_threads = particles.num_particles
        self._track_kernel(el=self._xobject, particles=particles)

    DressedElement.compile_track_kernel = compile_track_kernel
    DressedElement.track = track

    return DressedElement
