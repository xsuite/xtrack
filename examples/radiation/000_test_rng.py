import numpy as np

import xobjects as xo
import xtrack as xt
from pathlib import Path

ctx = xo.ContextCpu()
ctx.add_kernels(
        sources=[xt._pkg_root.joinpath('headers/rng.h'),
                 Path('./particles_rng.h')],
        kernels={'Particles_initialize_rand_gen': xo.Kernel(
                     args=[
                         xo.Arg(xt.Particles.XoStruct, name='particles'),
                         xo.Arg(xo.UInt32, pointer=True, name='seeds'),
                         xo.Arg(xo.Int32, name='n_init')],
                     n_threads='n_init')})

part = xt.Particles(context=ctx, p0c=6.5e12, x=[1,2,3])
seeds = ctx.nparray_to_context_array(
        np.array([3,4,7], dtype=np.uint32))
ctx.kernels.Particles_initialize_rand_gen(particles=part,
        seeds=seeds, n_init=part._capacity)
