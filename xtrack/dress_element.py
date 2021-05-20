import xobjects as xo
from .particles import ParticlesData, gen_local_particle_api
from .dress import dress

def dress_element(XoElementData):

    DressedElement = dress(XoElementData)
    assert XoElementData.__name__.endswith('Data')
    name = XoElementData.__name__[:-4]

    DressedElement.track_kernel_source = ('''
            /*gpukern*/'''
            f'void {name}_track_particles('
            f'               {name}Data el,'
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
            f'{name}_track_local_particle(el, &lpart);'
'''
                }
            }
''')
    DressedElement.track_kernel_description = {f'{name}_track_particles':
        xo.Kernel(args=[xo.Arg(XoElementData, name='el'),
                        xo.Arg(ParticlesData, name='particles')])}

    def compile_track_kernel(self):
        context = self._buffer.context

        context.add_kernels(sources=[
                ParticlesData._gen_c_api()[0],
                gen_local_particle_api(),
                self.XoStruct._gen_c_api()[0],
                self.XoStruct.track_function_source,
                self.track_kernel_source],
            kernels=self.track_kernel_description,
            extra_cdef='\n'.join([
                self.XoStruct._gen_c_api()[2],
                ParticlesData._gen_c_api()[2]]),
            save_source_as=None)

    DressedElement.compile_track_kernel = compile_track_kernel

    return DressedElement
