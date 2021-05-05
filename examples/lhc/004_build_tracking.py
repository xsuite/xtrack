from pathlib import Path
import numpy as np
from scipy.special import factorial

import xtrack as xt
import xobjects as xo
import sixtracktools
import pysixtrack

context = xo.ContextCpu()

six = sixtracktools.SixInput(".")
pyst_line = pysixtrack.Line.from_sixinput(six)
sixdump = sixtracktools.SixDump101("res/dump3.dat")

# TODO: The two particles look identical, to be checked
part0_pyst = pysixtrack.Particles(**sixdump[0::2][0].get_minimal_beam())
part1_pyst = pysixtrack.Particles(**sixdump[1::2][0].get_minimal_beam())
pysixtrack_particles = [part0_pyst, part1_pyst]

particles = xt.Particles(pysixtrack_particles=[part0_pyst, part1_pyst])

print('Creating line...')
xtline = xt.Line(_context=context, sequence=pyst_line)

print('Build capi')
sources = []
kernels = {}
cdefs = []

# Particles
source_particles, kernels_particles, cdefs_particles = (
                                xt.Particles.XoStruct._gen_c_api())
sources.append(source_particles)
kernels.update(kernels_particles)
cdefs += cdefs_particles.split('\n')

# Local particles
sources.append(xt.particles.gen_local_particle_api())

# Elements
element_classes = xtline._ElementRefClass._rtypes
for cc in element_classes:
    ss, kk, dd = cc._gen_c_api()
    sources.append(ss)
    kernels.update(kk)
    cdefs += dd.split('\n')

sources.append(Path('./constants.h'))
sources.append(Path('./drift.h'))
sources.append(Path('./multipole.h'))
sources.append(Path('./cavity.h'))
sources.append(Path('./xyshift.h'))
sources.append(Path('./srotation.h'))

cdefs_norep=[]
for cc in cdefs:
    if cc not in cdefs_norep:
        cdefs_norep.append(cc)

src_lines = []
src_lines.append(r'''
    void track_line(
        int8_t* buffer,
        int64_t* ele_offsets,
        int64_t* ele_types,
        ParticlesData particles,
        int64_t ele_start,
        int64_t num_ele_track){


    LocalParticle lpart;
    Particles_to_LocalParticle(particles, &lpart, 0);

    printf("Buffer = %p\n", buffer);

    for (int64_t ee=ele_start; ee<num_ele_track; ee++){
        int8_t* el = buffer + ele_offsets[ee];
        int64_t ee_type = ele_types[ee];

        printf("el = %p\n", el);

        switch(ee_type){
            case 0:
                printf("Element %ld is a Cavity having voltage %f\n", ee,
                    CavityData_get_voltage((CavityData) el));
                Cavity_track_local_particle((CavityData) el, &lpart);
                break;
            case 1:
                printf("Element %ld is a Drift having length %f\n", ee,
                    DriftData_get_length((DriftData) el));
                Drift_track_local_particle((DriftData) el, &lpart);
                break;
            case 2:
                printf("Element %ld is a Multipole having order %ld\n", ee,
                    MultipoleData_get_order((MultipoleData) el));
                Multipole_track_local_particle((MultipoleData) el, &lpart);
                break;
        }
    }
}
''')
source_track = '\n'.join(src_lines)
sources.append(source_track)

#    for (int ii=0; ii<npart; ii++){
#        lpart.ipart = ii;

kernel_descriptions = {
    "track_line": xo.Kernel(
        args=[
            xo.Arg(xo.Int8, pointer=True, name='buffer'),
            xo.Arg(xo.Int64, pointer=True, name='ele_offsets'),
            xo.Arg(xo.Int64, pointer=True, name='ele_types'),
            xo.Arg(xt.particles.ParticlesData, name='particles'),
            xo.Arg(xo.Int64, name='ele_start'),
            xo.Arg(xo.Int64, name='num_ele_track'),
        ],
    )
}
kernels.update(kernel_descriptions)

# Compile!
context.add_kernels(sources, kernels, extra_cdef='\n\n'.join(cdefs_norep))

ele_offsets = np.array([ee._offset for ee in xtline.elements], dtype=np.int64)
ele_types = np.array(
        [element_classes.index(ee._xobject.__class__) for ee in xtline.elements],
        dtype=np.int64)
context.kernels.track_line(buffer=xtline._buffer.buffer, ele_offsets=ele_offsets,
                           ele_types=ele_types, particles=particles,
                           ele_start=0, num_ele_track=5)
ip_check = 1
pyst_part = pysixtrack_particles[ip_check]
for ii, (eepyst, nn) in enumerate(zip(pyst_line.elements, pyst_line.element_names)):
    print(nn)

