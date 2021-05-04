import numpy as np

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

# Kick one particle
part1_pyst.x += 1e-3
part1_pyst.y += 2e-3

particles = xt.Particles(pysixtrack_particles=[part0_pyst, part1_pyst])

source_particles, kernels, cdefs_particles = xt.Particles.XoStruct._gen_c_api()

from xtrack.particles import scalar_vars, per_particle_vars

# Generate local particle CPU
source_local_part = xt.particles.gen_local_particle_api()

# Make a drift
drift = xt.Drift(length=2.)
source_drift, _, cdefs_drift = xt.Drift.XoStruct._gen_c_api()

source_custom = r'''

void Drift_track_local_particle(DriftData el, LocalParticle* part){

    double const length = DriftData_get_length(el);

    double const rpp    = LocalParticle_get_rpp(part);
    double const xp     = LocalParticle_get_px(part) * rpp;
    double const yp     = LocalParticle_get_py(part) * rpp;
    double const dzeta  = LocalParticle_get_rvv(part) -
                           ( 1. + ( xp*xp + yp*yp ) / 2. );

    LocalParticle_add_to_x(part, xp * length );
    LocalParticle_add_to_y(part, yp * length );
    LocalParticle_add_to_s(part, length);
    LocalParticle_add_to_zeta(part, length * dzeta );

}

void Drift_track_particles(DriftData el, ParticlesData particles){
    int64_t npart = ParticlesData_get_num_particles(particles);
    printf("Hello\n");
    printf("I got %ld particles\n", npart);

    double length = DriftData_get_length(el);
    printf("and I got a drift of length %f\n", length);

    LocalParticle lpart;
    Particles_to_LocalParticle(particles, &lpart, 0);

    for (int ii=0; ii<npart; ii++){
        lpart.ipart = ii;
        double x = LocalParticle_get_x(&lpart);
        printf("x[%d] = %f\n", ii, x);
        }
}

'''

kernel_descriptions = {
    "Drift_track_particles": xo.Kernel(
        args=[
            xo.Arg(xt.Drift.XoStruct, name="el"),
            xo.Arg(xt.Particles.XoStruct, name="particles"),
        ],
    )
}

context.add_kernels(
        sources=[source_particles, source_local_part, source_drift,
                 source_custom,],
        kernels=kernel_descriptions,
        extra_cdef='\n'.join([cdefs_drift, cdefs_particles]))

context.kernels.Drift_track_particles(particles=particles, el=drift)
