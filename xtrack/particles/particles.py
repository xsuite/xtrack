import numpy as np
import xobjects as xo
from enum import Enum

from ._pyparticles import Pyparticles
from .local_particle import (
    gen_local_particle_common_src,
    gen_local_particle_adapter_src,
    gen_local_particle_local_copy_src,
    gen_local_particle_shared_copy_src,
)

from ..dress import dress
from ..general import _pkg_root

from scipy.constants import m_p
from scipy.constants import e as qe
from scipy.constants import c as clight

pmass = m_p * clight * clight / qe


LAST_INVALID_STATE = -999999999

size_vars = (
    (xo.Int64, "_capacity"),
    (xo.Int64, "_num_active_particles"),
    (xo.Int64, "_num_lost_particles"),
)
# Capacity is always kept up to date
# the other two are placeholders to be used if needed
# i.e. on ContextCpu

scalar_vars = (
    (xo.Float64, "q0"),
    (xo.Float64, "mass0"),
)

per_particle_vars = (
    (xo.Float64, "p0c"),
    (xo.Float64, "gamma0"),
    (xo.Float64, "beta0"),
    (xo.Float64, "s"),
    (xo.Float64, "x"),
    (xo.Float64, "y"),
    (xo.Float64, "px"),
    (xo.Float64, "py"),
    (xo.Float64, "zeta"),
    (xo.Float64, "psigma"),
    (xo.Float64, "delta"),
    (xo.Float64, "rpp"),
    (xo.Float64, "rvv"),
    (xo.Float64, "chi"),
    (xo.Float64, "charge_ratio"),
    (xo.Float64, "weight"),
    (xo.Int64, "particle_id"),
    (xo.Int64, "at_element"),
    (xo.Int64, "at_turn"),
    (xo.Int64, "state"),
    (xo.Int64, "parent_particle_id"),
    (xo.UInt32, "__rng_s1"),
    (xo.UInt32, "__rng_s2"),
    (xo.UInt32, "__rng_s3"),
    (xo.UInt32, "__rng_s4"),
)

fields = {}
for tt, nn in size_vars + scalar_vars:
    fields[nn] = tt

for tt, nn in per_particle_vars:
    fields[nn] = tt[:]

ParticlesData = type("ParticlesData", (xo.Struct,), fields)

ParticlesData.extra_sources = [
    _pkg_root.joinpath("random_number_generator/rng_src/base_rng.h"),
    _pkg_root.joinpath("random_number_generator/rng_src/particles_rng.h"),
]
ParticlesData.custom_kernels = {
    "Particles_initialize_rand_gen": xo.Kernel(
        args=[
            xo.Arg(ParticlesData, name="particles"),
            xo.Arg(xo.UInt32, pointer=True, name="seeds"),
            xo.Arg(xo.Int32, name="n_init"),
        ],
        n_threads="n_init",
    )
}

pysixtrack_naming = (
    ("qratio", "charge_ratio"),
    ("mratio", "mass_ratio"),
    ("partid", "particle_id"),
    ("turn", "at_turn"),
    ("elemid", "at_element"),
)


class Particles(dress(ParticlesData)):
    _structure = {
        "size_vars": size_vars,
        "scalar_vars": scalar_vars,
        "per_particle_vars": per_particle_vars,
    }

    def __init__(self, **kwargs):

        # Compatibility with old pysixtrack naming
        for old, new in pysixtrack_naming:
            if old in kwargs.keys():
                assert new not in kwargs.keys()
                kwargs[new] = kwargs[old]

        if "_xobject" in kwargs.keys():
            # Initialize xobject
            self.xoinitialize(**kwargs)
        else:

            if any([nn in kwargs.keys() for tt, nn in per_particle_vars]):
                # Needed to generate consistent longitudinal variables
                pyparticles = Pyparticles(**kwargs)

                part_dict = _pyparticles_to_xtrack_dict(pyparticles)
                if "_capacity" in kwargs.keys():
                    assert kwargs["_capacity"] >= part_dict["_capacity"]
                else:
                    kwargs["_capacity"] = part_dict["_capacity"]
            else:
                assert "_capacity" in kwargs.keys()
                pyparticles = None

            # Make sure _capacity is integer
            kwargs["_capacity"] = int(kwargs["_capacity"])

            # We just provide array sizes to xoinitialize (we will set values later)
            kwargs.update({kk: kwargs["_capacity"] for tt, kk in per_particle_vars})

            # Initialize xobject
            self.xoinitialize(**kwargs)

            # Initialize coordinates
            if pyparticles is not None:
                context = self._buffer.context
                for tt, kk in list(scalar_vars):
                    setattr(self, kk, part_dict[kk])
                for tt, kk in list(per_particle_vars):
                    if kk.startswith("__"):
                        continue
                    vv = getattr(self, kk)
                    vals = context.nparray_to_context_array(part_dict[kk])
                    ll = len(vals)
                    vv[:ll] = vals
                    vv[ll:] = LAST_INVALID_STATE
            else:
                for tt, kk in list(scalar_vars):
                    setattr(self, kk, 0.0)

                for tt, kk in list(per_particle_vars):
                    if kk == "chi" or kk == "charge_ratio" or kk == "state":
                        value = 1.0
                    elif kk == "particle_id":
                        value = np.arange(0, self._capacity, dtype=np.int64)
                    else:
                        value = 0.0
                    getattr(self, kk)[:] = value

        self._num_active_particles = -1  # To be filled in only on CPU
        self._num_lost_particles = -1  # To be filled in only on CPU

        if isinstance(self._buffer.context, xo.ContextCpu):
            # Particles always need to be organized to run on CPU
            self.reorganize()

    def _init_random_number_generator(self, seeds=None):

        self.compile_custom_kernels(only_if_needed=True)

        if seeds is None:
            seeds = np.random.randint(
                low=1, high=4e9, size=self._capacity, dtype=np.uint32
            )
        else:
            assert len(seeds) == particles._capacity
            if not hasattr(seeds, "dtype") or seeds.dtype != np.uint32:
                seeds = np.array(seeds, dtype=np.uint32)

        context = self._buffer.context
        seeds_dev = context.nparray_to_context_array(seeds)
        context.kernels.Particles_initialize_rand_gen(
            particles=self, seeds=seeds_dev, n_init=self._capacity
        )

    def reorganize(self):
        assert not isinstance(
            self._buffer.context, xo.ContextPyopencl
        ), "Masking does not work with pyopencl"
        mask_active = self.state > 0
        mask_lost = (self.state < 1) & (self.state > LAST_INVALID_STATE)

        n_active = np.sum(mask_active)
        n_lost = np.sum(mask_lost)

        for tt, nn in self._structure["per_particle_vars"]:
            vv = getattr(self, nn)
            vv_active = vv[mask_active]
            vv_lost = vv[mask_lost]

            vv[:n_active] = vv_active
            vv[n_active : n_active + n_lost] = vv_lost
            vv[n_active + n_lost :] = LAST_INVALID_STATE

        if isinstance(self._buffer.context, xo.ContextCpu):
            self._num_active_particles = n_active
            self._num_lost_particles = n_lost

        return n_active, n_lost

    def add_particles(self, part, keep_lost=False):

        if keep_lost:
            raise NotImplementedError
        assert not isinstance(
            self._buffer.context, xo.ContextPyopencl
        ), "Masking does not work with pyopencl"

        mask_copy = part.state > 0
        n_copy = np.sum(mask_copy)

        n_active, n_lost = self.reorganize()
        i_start_copy = n_active + n_lost
        n_free = self._capacity - n_active - n_lost

        max_id = np.max(self.particle_id[: n_active + n_lost])

        if n_copy > n_free:
            raise NotImplementedError("Out of space, need to regenerate xobject")

        for tt, nn in self._structure["scalar_vars"]:
            assert np.isclose(
                getattr(self, nn), getattr(part, nn), rtol=1e-14, atol=1e-14
            )

        for tt, nn in self._structure["per_particle_vars"]:
            vv = getattr(self, nn)
            vv_copy = getattr(part, nn)[mask_copy]
            vv[i_start_copy : i_start_copy + n_copy] = vv_copy

        self.particle_id[i_start_copy : i_start_copy + n_copy] = np.arange(
            max_id + 1, max_id + 1 + n_copy, dtype=np.int64
        )

        self.reorganize()

    def get_active_particle_id_range(self):
        ctx2np = self._buffer.context.nparray_from_context_array
        mask_active = ctx2np(self.state) > 0
        ids_active_particles = ctx2np(self.particle_id)[mask_active]
        # Behaves as python rante (+1)
        return np.min(ids_active_particles), np.max(ids_active_particles) + 1

    def _set_p0c(self):
        energy0 = np.sqrt(self.p0c ** 2 + self.mass0 ** 2)
        self.beta0 = self.p0c / energy0
        self.gamma0 = energy0 / self.mass0

    @property
    def ptau(self):
        return (
            np.sqrt(self.delta ** 2 + 2 * self.delta + 1 / self.beta0 ** 2)
            - 1 / self.beta0
        )

    def set_reference(self, p0c=7e12, mass0=pmass, q0=1):
        self.q0 = q0
        self.mass0 = mass0
        self.p0c = p0c
        return self

    def set_particle(
        self, index, set_scalar_vars=False, check_scalar_vars=True, **kwargs
    ):

        # Compatibility with old pysixtrack naming
        for old, new in pysixtrack_naming:
            if old in kwargs.keys():
                assert new not in kwargs.keys()
                kwargs[new] = kwargs[old]

        # Needed to generate consistent longitudinal variables
        pyparticles = Pyparticles(**kwargs)
        part_dict = _pyparticles_to_xtrack_dict(pyparticles)
        for tt, kk in list(scalar_vars):
            setattr(self, kk, part_dict[kk])
        for tt, kk in list(per_particle_vars):
            if kk.startswith("__") and kk not in part_dict.keys():
                continue
            getattr(self, kk)[index] = part_dict[kk][0]

    def _update_delta(self, new_delta_value):
        beta0 = self.beta0
        delta_beta0 = new_delta_value * beta0
        ptau_beta0 = (
            np.sqrt(delta_beta0 * delta_beta0 + 2.0 * delta_beta0 * beta0 + 1.0) - 1.0
        )
        one_plus_delta = 1.0 + new_delta_value
        rvv = (one_plus_delta) / (1.0 + ptau_beta0)
        rpp = 1.0 / one_plus_delta
        psigma = ptau_beta0 / (beta0 * beta0)

        self.delta[:] = new_delta_value
        self.rvv[:] = rvv
        self.rpp[:] = rpp
        self.psigma[:] = psigma


class LocalParticleVar(Enum):
    ADAPTER = 0
    THREAD_LOCAL_COPY = 1
    SHARED_COPY = 2


def gen_local_particle_api(mode=LocalParticleVar.ADAPTER):
    if mode is not None and isinstance(mode, str):
        if mode == "no_local_copy":
            mode = LocalParticleVar.ADAPTER
        elif mode == "local_copy":
            mode = LocalParticleVar.THREAD_LOCAL_COPY
        elif mode == "shared_copy":
            mode = LocalParticleVar.SHARED_COPY

    if mode == LocalParticleVar.ADAPTER:
        src = gen_local_particle_adapter_src()
    elif mode == LocalParticleVar.THREAD_LOCAL_COPY:
        src = gen_local_particle_local_copy_src()
    elif mode == LocalParticleVar.SHARED_COPY:
        src = gen_local_particle_shared_copy_src()
    else:
        raise NotImplementedError

    if len(src) > 0:
        src += "\r\n"
        src += gen_local_particle_common_src()
    return src


def _pyparticles_to_xtrack_dict(pyparticles):

    out = {}

    pyst_dict = pyparticles.to_dict()
    for old, new in pysixtrack_naming:
        if hasattr(pyparticles, old):
            assert new not in pyst_dict.keys()
            pyst_dict[new] = getattr(pyparticles, old)

    if hasattr(pyparticles, "weight"):
        pyst_dict["weight"] = getattr(pyparticles, "weight")
    else:
        pyst_dict["weight"] = 1.0

    for tt, kk in scalar_vars + per_particle_vars:
        if kk.startswith("__"):
            continue
        # Use properties
        pyst_dict[kk] = getattr(pyparticles, kk)

    for kk, vv in pyst_dict.items():
        pyst_dict[kk] = np.atleast_1d(vv)

    lll = [len(vv) for kk, vv in pyst_dict.items() if hasattr(vv, "__len__")]
    lll = list(set(lll))
    assert len(set(lll) - {1}) <= 1
    _capacity = max(lll)
    out["_capacity"] = _capacity

    for tt, kk in scalar_vars:
        val = pyst_dict[kk]
        assert np.allclose(val, val[0], rtol=1e-10, atol=1e-14)
        out[kk] = val[0]

    for tt, kk in per_particle_vars:
        if kk.startswith("__"):
            continue

        val_pyst = pyst_dict[kk]

        if _capacity > 1 and len(val_pyst) == 1:
            temp = np.zeros(int(_capacity), dtype=tt._dtype)
            temp += val_pyst[0]
            val_pyst = temp

        if type(val_pyst) != tt._dtype:
            val_pyst = np.array(val_pyst, dtype=tt._dtype)

        out[kk] = val_pyst

    # out['_num_active_particles'] = np.sum(out['state']>0)
    # out['_num_lost_particles'] = np.sum((out['state'] < 0) &
    #                                      (out['state'] > LAST_INVALID_STATE))

    return out
