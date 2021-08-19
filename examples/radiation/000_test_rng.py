import numpy as np

import xobjects as xo
import xtrack as xt
from pathlib import Path

ctx = xo.ContextCpu()
ctx.add_kernels(
        sources=[
            xt._pkg_root.joinpath(
                'random_number_generator/rng_src/base_rng.h'),
            xt._pkg_root.joinpath(
                'random_number_generator/rng_src/particles_rng.h')],
        kernels={'Particles_initialize_rand_gen': xo.Kernel(
                     args=[
                         xo.Arg(xt.Particles.XoStruct, name='particles'),
                         xo.Arg(xo.UInt32, pointer=True, name='seeds'),
                         xo.Arg(xo.Int32, name='n_init')],
                     n_threads='n_init')})

part = xt.Particles(context=ctx, p0c=6.5e12, x=[1,2,3])
seeds = np.random.randint(low=1, high=4e9, size=part._capacity, dtype=np.uint32)

seeds_dev = ctx.nparray_to_context_array(seeds)
ctx.kernels.Particles_initialize_rand_gen(particles=part,
        seeds=seeds, n_init=part._capacity)

class TestElement(xt.BeamElement):
     _xofields={
        'dummy': xo.Float64,
        }

TestElement.XoStruct.extra_sources = [
    xt._pkg_root.joinpath('random_number_generator/rng_src/base_rng.h'),
    xt._pkg_root.joinpath('random_number_generator/rng_src/local_particle_rng.h'),
    Path('./test_elem.h')]

telem = TestElement()

telem.track(part)
