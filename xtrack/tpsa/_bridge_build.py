"""Compile (and cache) the element-tracking bridge modules.

``c_src/xt_bridge.cpp`` is one translation unit compiled per flavor through
``ContextCpu.build_kernels`` (C++, linking ``xgtpsa``'s GTPSA core), yielding cffi
API-mode modules that export the track functions. The built ``.so`` is cached on disk
with a key based on the content of the bridge sources: a matching key means the sources have not changed,
so the module is reused without recompiling. Otherwise, the module is rebuilt and the cache updated.

Both those modules and ``xgtpsa``'s ``Descriptor``/``Tpsa`` link the same shared object
file, so the map's ``tpsa_t`` handles share mad's global descriptor state.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from typing import TYPE_CHECKING, Any, Callable

import cffi
import xgtpsa

if TYPE_CHECKING:
    # xobjects does not import context_cpu eagerly (it pulls in the compiler path),
    # so these are named for readers/checkers only.
    from xobjects.context import Kernel
    from xobjects.context_cpu import ContextCpu, KernelCpu

_BRIDGE_FLAVORS = ("num", "tpsa", "tpsa_param")
# Preprocessor defines per flavor. tpsa_param = tpsa + parametric knobs:
# XT_STRENGTH becomes mad::tpsa and strengths flow through the knob table.
_FLAVOR_DEFINES = {
    "num": ["-DXT_FLAVOR_NUM"],
    "tpsa": ["-DXT_FLAVOR_TPSA"],
    "tpsa_param": ["-DXT_FLAVOR_TPSA", "-DXT_KNOBS"],
}
_bridge_modules: dict[str, dict[str, KernelCpu]] = {}  # flavor -> kernels
_bridge_ctx: ContextCpu | None = None


def _src_dir() -> str:
    """The bridge C++ sources (xt_bridge.cpp + generated/), shipped with xtrack."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "c_src")


def _cache_dir() -> str:
    return os.environ.get("XTRACK_TPSA_CACHE") or os.path.join(
        _src_dir(), "_bridge_cache"
    )


def _xtrack_rev() -> str:
    try:
        import xtrack

        root = os.path.dirname(os.path.dirname(os.path.abspath(xtrack.__file__)))
        return (
            subprocess.check_output(
                ["git", "-C", root, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _bridge_entry_cdef(f: str) -> str:
    cdef = (
        f"void xt_bridge_track_element_{f}(int64_t type_id, void* el, void* p);\n"
        f"void xt_bridge_track_line_{f}(void* ref, int64_t ele_start, "
        f"int64_t num_elements, void* p, void* mon, int64_t flag_monitor, "
        f"const int64_t* observe);\n"
    )
    if f == "tpsa_param":  # the parametric flavor also exposes the knob-table setter
        cdef += (
            f"void xt_bridge_set_knob_table_{f}(const void** addrs, const void** tpsas, "
            f"const void* proto, int64_t n);\n"
        )
    return cdef


def _bridge_kernel_descs(f: str) -> dict[str, Kernel]:
    import xobjects as xo

    names = [f"xt_bridge_track_element_{f}", f"xt_bridge_track_line_{f}"]
    if f == "tpsa_param":
        names.append(f"xt_bridge_set_knob_table_{f}")
    return {n: xo.Kernel(args=[], c_name=n) for n in names}


def _bridge_sources() -> list[str]:
    """List glue code and generated files so their content can be hashed into the cache key.

    generated/ encodes the registry (dispatch switch, typeids) and the xtrack-derived
    struct/accessors, so hashing it replaces the old hand-kept ABI number. It is a build
    artifact: if it is missing, generate it now (pure Python, cheap).
    """
    here = _src_dir()
    gen = os.path.join(here, "generated")
    if not os.path.isdir(gen):
        from . import gen_bridge

        gen_bridge.main()
    files = [
        os.path.join(here, "xt_bridge.cpp"),
        os.path.join(here, "xt_local_particle.hpp"),
        os.path.join(here, "xt_knob.hpp"),
    ]
    files += sorted(
        os.path.join(gen, f)
        for f in os.listdir(gen)
        if f.endswith((".h", ".hpp", ".inc"))
    )
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise RuntimeError(
            f"missing bridge sources {missing}; run `python -m xtrack.tpsa.gen_bridge`."
        )
    return files


def _bridge_cache_key(flavor: str) -> str:
    """Hash with sha1 over (flavor, xtrack rev, core .so identity, bridge source contents).

    Content-addressed, so editing ``xt_bridge.cpp`` busts the cache by itself; there is no
    version constant to remember to bump. The xtrack rev covers the physics headers, which
    compile from the xtrack checkout rather than from c_src, and the core stat covers a
    rebuilt GTPSA engine.
    """
    core = xgtpsa.core_library()
    st = os.stat(core)
    h = hashlib.sha1()
    h.update(
        json.dumps(
            {
                "flavor": flavor,
                "xtrack": _xtrack_rev(),
                "core": [os.path.abspath(core), st.st_size, int(st.st_mtime)],
            },
            sort_keys=True,
        ).encode()
    )
    for path in _bridge_sources():
        with open(path, "rb") as fid:
            h.update(fid.read())
    return "bridge_" + flavor + "_" + h.hexdigest()[:12]


def bridge_lib(flavor: str, force: bool = False) -> dict[str, KernelCpu]:
    """Build (cached) and return the xobjects-compiled cffi module for one flavor.

    Returns ``{kernel_name: KernelCpu}``; each ``KernelCpu`` exposes ``.function`` (the
    callable) and ``.ffi_interface`` (the module's cffi, used to cast the void* args).
    """
    global _bridge_ctx
    if flavor not in _BRIDGE_FLAVORS:
        raise ValueError(f"flavor must be one of {_BRIDGE_FLAVORS}, got {flavor!r}")
    if flavor in _bridge_modules and not force:
        return _bridge_modules[flavor]

    from xobjects.context_cpu import ContextCpu, _so_for_module_name

    if _bridge_ctx is None:
        _bridge_ctx = ContextCpu()
    here = _src_dir()
    core = xgtpsa.core_library()
    cache_dir = _cache_dir()
    module_name = _bridge_cache_key(flavor)
    descs = _bridge_kernel_descs(flavor)
    so = _so_for_module_name(module_name, cache_dir)
    if so.exists() and not force:
        kernels = _bridge_ctx.kernels_from_file(
            module_name, descs, containing_dir=cache_dir
        )
    else:
        with open(os.path.join(here, "xt_bridge.cpp")) as fid:
            src = fid.read()
        os.makedirs(cache_dir, exist_ok=True)
        # Route B: knobs are a compile flavor (tpsa_param = -DXT_FLAVOR_TPSA -DXT_KNOBS),
        # so strengths become mad::tpsa across the whole line and the address-keyed knob
        # table (xt_knob.hpp, included directly into this one TU) supplies the parametric
        # values. No separate knob translation unit to link.
        kernels = _bridge_ctx.build_kernels(
            kernel_descriptions=descs,
            module_name=module_name,
            containing_dir=cache_dir,
            sources=[src],
            specialize=False,
            compiler_language="c++",
            extra_cdef=_bridge_entry_cdef(flavor),
            extra_compile_args=[
                *_FLAVOR_DEFINES[flavor],
                "-DXTRACK_MULTIPOLE_NO_SYNRAD",
                f"-I{here}",
                "-include", "complex",
            ],
            extra_source_files=(),
        )
    _bridge_modules[flavor] = kernels
    return kernels


def bridge_entry(flavor: str, fn_name: str) -> tuple[Callable[..., Any], cffi.FFI]:
    """Return ``(callable, ffi)`` for a bridge entry point.

    The callable is the xobjects API module's function.
    ``ffi`` is that module's own cffi.
    Callers cast their void* args with the returned ffi.
    """
    k = bridge_lib(flavor)[fn_name]
    return k.function, k.ffi_interface
