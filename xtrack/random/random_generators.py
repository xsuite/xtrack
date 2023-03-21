# copyright ############################### //
# This file is part of the Xtrack Package.   //
# Copyright (c) CERN, 2023.                 //
# ######################################### //

import xobjects as xo
import xpart as xp

from ..base_element import BeamElement
from ..general import _pkg_root

import numpy as np


# Random generators need to be a BeamElement to get the LocalParticle API
class RandomUniform(BeamElement):
    _xofields = {
        '_dummy': xo.UInt8,  # TODO: a hack for allocating empty struct on OCL
    }

    iscollective = True

    _extra_c_sources = [
        # The base (bitwise) rng is in xpart, as this is where the
        # seeds are stored. This is needed to avoid circular imports
        # in xpart.Particles
        xp._pkg_root.joinpath('rng_src','base_rng.h'),
        _pkg_root.joinpath('random','random_src','uniform.h')
    ]

    _per_particle_kernels = {
        'sample_uniform': xo.Kernel(
                c_name='RandomUniform_sample',
                args=[
                    xo.Arg(xo.Float64, pointer=True, name='samples'),
                    xo.Arg(xo.Int64, name='n_samples_per_seed')
                ]
            )
        }

    def _sample(self, *args, **kwargs):
        self.sample_uniform(*args, **kwargs)

    def track(self, *args, **kwargs):
        raise RuntimeError("Random generators have no valid track method.")

    def generate(self, n_samples=1000, n_seeds=1000, particles=None):
        context = self._context
        n_seeds = int(n_seeds)
        if particles is None:
            particles = xp.Particles(state=np.ones(n_seeds),
                                     x=np.ones(n_seeds), _context=context)
        elif n_seeds != len(particles._rng_s1):
            print("Warning: both 'particles' and 'n_seeds' are given, but "
                  + "are not compatible. Ignoring 'n_seeds'...")
            n_seeds = len(particles._rng_s1)
        if not particles._has_valid_rng_state():
            particles._init_random_number_generator()

        n_samples_per_seed = int(np.floor(n_samples/n_seeds))
        if n_samples_per_seed < 1:
            raise ValueError("Not enough samples to accomodate all seeds!")

        samples = context.zeros(shape=(n_seeds*n_samples_per_seed,),
                                dtype=np.float64)
        self._sample(particles=particles, samples=samples,
                     n_samples_per_seed=n_samples_per_seed)

        return np.reshape(samples[:n_samples_per_seed*n_seeds],
                    (-1, n_samples_per_seed)
                ), particles


class RandomExponential(RandomUniform):
    _xofields = {
        '_dummy': xo.UInt8,  # TODO: a hack for allocating an empty struct on OCL
    }

    iscollective = True

    _depends_on = [RandomUniform]

    _extra_c_sources = [
        _pkg_root.joinpath('random','random_src','exponential.h')
    ]

    _per_particle_kernels = {
        'sample_exp': xo.Kernel(
                c_name='RandomExponential_sample',
                args=[
                    xo.Arg(xo.Float64, pointer=True, name='samples'),
                    xo.Arg(xo.Int64, name='n_samples_per_seed')
                ]
            )
        }

    def _sample(self, *args, **kwargs):
        self.sample_exp(*args, **kwargs)


class RandomNormal(RandomUniform):
    _xofields = {
        '_dummy': xo.UInt8,  # TODO: a hack for allocating an empty struct on OCL
    }

    iscollective = True

    _depends_on = [RandomUniform]

    _extra_c_sources = [
        _pkg_root.joinpath('random','random_src','normal.h')
    ]

    _per_particle_kernels = {
        'sample_gauss': xo.Kernel(
                c_name='RandomNormal_sample',
                args=[
                    xo.Arg(xo.Float64, pointer=True, name='samples'),
                    xo.Arg(xo.Int64, name='n_samples_per_seed')
                ]
            )
        }

    def _sample(self, *args, **kwargs):
        self.sample_gauss(*args, **kwargs)


class RandomRutherford(RandomUniform):
    _xofields = {
        'lower_val':         xo.Float64,
        'upper_val':         xo.Float64,
        'A':                 xo.Float64,
        'B':                 xo.Float64,
        'Newton_iterations': xo.Int8
    }

    iscollective = True

    _depends_on = [RandomUniform]

    _extra_c_sources = [
        _pkg_root.joinpath('random','random_src','exponential_integral_Ei.h'),
        _pkg_root.joinpath('random','random_src','rutherford.h')
    ]

    _per_particle_kernels = {
        'sample_ruth': xo.Kernel(
                c_name='RandomRutherford_sample',
                args=[
                    xo.Arg(xo.Float64, pointer=True, name='samples'),
                    xo.Arg(xo.Int64, name='n_samples_per_seed')
                ]
            ),
        }

    _kernels = {
        'set_rutherford': xo.Kernel(
                c_name='RandomRutherford_set',
                args=[
                    xo.Arg(xo.ThisClass, name='rng'),
                    xo.Arg(xo.Float64, name='A'),
                    xo.Arg(xo.Float64, name='B'),
                    xo.Arg(xo.Float64, name='lower_val'),
                    xo.Arg(xo.Float64, name='upper_val')
                ]
            )
        }

    def _sample(self, *args, **kwargs):
        self.sample_ruth(*args, **kwargs)

    def __init__(self, **kwargs):
        if '_xobject' not in kwargs:
            kwargs.setdefault('Newton_iterations', 7)
            kwargs.setdefault('lower_val', 1.)
            kwargs.setdefault('upper_val', 1.)
            kwargs.setdefault('A', 0.)
            kwargs.setdefault('B', 0.)

        super().__init__(**kwargs)

        if not isinstance(self._context, xo.ContextCpu):
            raise ValueError('Rutherford random generator is not currently supported on GPU.')

    def set_parameters(self, A, B, lower_val, upper_val):
        self.compile_kernels(only_if_needed=True)
        context = self._buffer.context
        context.kernels.set_rutherford(rng=self, A=A, B=B, lower_val=lower_val, upper_val=upper_val)

