from __future__ import annotations

import weakref
from typing import Any

import numpy as np
import xgtpsa
from xgtpsa.paths import lib_dir
import xobjects as xo
from xobjects.context import sort_classes

from xtrack.tracker import _element_ref_data_class_from_element_classes
from xtrack.tracker import _float_or_tpsa_getter_block
from xtrack.track_flags import TrackFlags
from xtrack.base_element import _handle_per_particle_blocks

from .particles import (
    ParticlesTpsa,
    TpsaParticleData,
    _COORDS,
    _DERIVED_COORDS,
    _LOCAL_COORDS,
    _REF_VARS,
)

_TPSA_KERNELS: weakref.WeakKeyDictionary[Any, dict[tuple, Any]] = (
    weakref.WeakKeyDictionary()
)
_TPSA_REFDATA: weakref.WeakKeyDictionary[Any, dict[tuple, Any]] = (
    weakref.WeakKeyDictionary()
)


def _xobject_ptr(xobj, ffi):
    buf = np.frombuffer(xobj._buffer.buffer, dtype="int8")
    return ffi.cast("void*", buf.ctypes.data + xobj._offset)


def _tpsa_local_particle_source():
    from xtrack.particles.particles import (
        gen_local_particle_field_accessors,
        gen_local_particle_struct,
        pointer_struct_local_particle_flavor,
        tpsa_pointer_local_particle_flavor,
    )

    coords = list(_COORDS + _DERIVED_COORDS + _LOCAL_COORDS)
    accessors = "\n".join(
        gen_local_particle_field_accessors(tpsa_pointer_local_particle_flavor(coords))
    )

    tail = [
        "    TpsaParticleData tp;",
        "    double line_length;",
        "    int64_t ipart, endpart, _num_active_particles, _num_lost_particles;",
        "    uint64_t track_flags;",
        "    int8_t* io_buffer;",
    ]
    struct_source = gen_local_particle_struct(
        pointer_struct_local_particle_flavor(coords, tail)
    )

    ref_accessors = []
    for nm in _REF_VARS:
        ref_accessors.append(
            f"static inline double LocalParticle_get_{nm}(LocalParticle* p){{ "
            f"return TpsaParticleData_get_{nm}(p->tp); }}"
        )
    for nm in ("state", "at_element"):
        ref_accessors += [
            f"static inline int64_t LocalParticle_get_{nm}(LocalParticle* p){{ "
            f"return TpsaParticleData_get_{nm}(p->tp); }}",
            f"static inline void LocalParticle_set_{nm}(LocalParticle* p, int64_t v){{ "
            f"TpsaParticleData_set_{nm}(p->tp, v); }}",
            f"static inline void LocalParticle_add_to_{nm}(LocalParticle* p, int64_t v){{ "
            f"TpsaParticleData_set_{nm}(p->tp, TpsaParticleData_get_{nm}(p->tp) + v); }}",
        ]

    return f"""
#ifndef XTRACK_TPSA_LOCAL_PARTICLE_H
#define XTRACK_TPSA_LOCAL_PARTICLE_H

#include <cstdint>
#include <math.h>
#include "mad_tpsa.hpp"

using mad::tpsa;
using mad::tpsa_ref;
#define XT_FLAVOR_TPSA 1
#define XT_NUM mad::tpsa
#define XT_NUM_CONST_ARG const XT_NUM&
#define XT_KNOBS 1
#define XT_STRENGTH mad::tpsa
#define XT_STRENGTH_CONST_ARG const XT_STRENGTH&
#define XT_STRENGTH_ARG const XT_STRENGTH&
#define XT_STRENGTH_CONST(v) ((v)[0])
#define XT_STRENGTH_LIFT(v) xt_float_or_tpsa_lift(v)

#define XT_TPSA_REL(OP) \\
  template<class A> inline bool operator OP (const mad::tpsa_base<A>& a, double b){{ return a[0] OP b; }} \\
  template<class A> inline bool operator OP (double a, const mad::tpsa_base<A>& b){{ return a OP b[0]; }} \\
  template<class A, class B> inline bool operator OP (const mad::tpsa_base<A>& a, const mad::tpsa_base<B>& b){{ return a[0] OP b[0]; }}
XT_TPSA_REL(>) XT_TPSA_REL(<) XT_TPSA_REL(>=) XT_TPSA_REL(<=) XT_TPSA_REL(==) XT_TPSA_REL(!=)
#undef XT_TPSA_REL

typedef tpsa_t XT_COORD;
static tpsa_t* xt_tpsa_proto = nullptr;

static inline double xt_float_or_tpsa_bits_to_double(uint64_t bits){{
    union {{ uint64_t u; double d; }} value;
    value.u = bits;
    return value.d;
}}
#define XTRACK_FLOAT_OR_TPSA_BITS_TO_DOUBLE 1

static inline XT_NUM xt_float_or_tpsa_lift(double value){{
    return 0.0 * mad::tpsa_ref(xt_tpsa_proto) + value;
}}

static inline XT_NUM xt_float_or_tpsa_lift(uint64_t bits){{
    return xt_float_or_tpsa_lift(xt_float_or_tpsa_bits_to_double(bits));
}}

template<class ElementData>
static inline XT_NUM xt_float_or_tpsa_get(
        ElementData, uint64_t* slot, int64_t tpsa_enabled){{
    if (tpsa_enabled) {{
        return 1.0 * mad::tpsa_ref((tpsa_t*)(uintptr_t)(*slot));
    }}
    return xt_float_or_tpsa_lift(*slot);
}}

{struct_source}

static inline int64_t LocalParticle_get__num_active_particles(LocalParticle* p){{
    return p->_num_active_particles;
}}
static inline uint64_t LocalParticle_check_track_flag(LocalParticle* p, uint8_t index){{
    return (p->track_flags >> index) & 1;
}}
static inline void LocalParticle_exchange(LocalParticle*, int64_t, int64_t){{}}
static inline void LocalParticle_add_to_at_turn(LocalParticle*, int64_t){{}}
static inline int64_t LocalParticle_get_at_turn(LocalParticle*){{ return 0; }}
static inline int64_t LocalParticle_get_particle_id(LocalParticle*){{ return 0; }}
static inline int8_t* LocalParticle_get_io_buffer(LocalParticle* p){{ return p->io_buffer; }}

{accessors}

{chr(10).join(ref_accessors)}

static inline double LocalParticle_get_energy0(LocalParticle* part) {{
    double const p0c = LocalParticle_get_p0c(part);
    double const m0  = LocalParticle_get_mass0(part);
    return sqrt(p0c * p0c + m0 * m0);
}}

static inline void LocalParticle_update_delta(LocalParticle* part,
        const XT_NUM& new_delta_value) {{
    double const beta0 = LocalParticle_get_beta0(part);
    XT_NUM const irpp = new_delta_value + 1.0;
    XT_NUM const rpp = 1.0 / irpp;
    XT_NUM const ptau = (sqrt(1.0 + 2.0 * beta0 * new_delta_value
                              + new_delta_value * new_delta_value) - 1.0) / beta0;
    XT_NUM const rvv = rpp * (1.0 + beta0 * ptau);
    LocalParticle_set_delta(part, new_delta_value);
    LocalParticle_set_rvv(part, rvv);
    LocalParticle_set_rpp(part, rpp);
    LocalParticle_set_ptau(part, ptau);
}}

static inline void LocalParticle_update_delta(LocalParticle* part, double value) {{
    LocalParticle_update_delta(part, xt_float_or_tpsa_lift(value));
}}

static inline void LocalParticle_update_ptau(LocalParticle* part,
        const XT_NUM& new_ptau_value) {{
    double const beta0 = LocalParticle_get_beta0(part);
    XT_NUM const delta = sqrt(1.0 + 2.0 * beta0 * new_ptau_value
                              + new_ptau_value * new_ptau_value) - 1.0;
    LocalParticle_update_delta(part, delta);
}}

static inline void LocalParticle_update_ptau(LocalParticle* part, double value) {{
    LocalParticle_update_ptau(part, xt_float_or_tpsa_lift(value));
}}

static inline XT_NUM LocalParticle_get_pzeta(LocalParticle* part) {{
    XT_NUM const ptau = LocalParticle_get_ptau(part);
    double const beta0 = LocalParticle_get_beta0(part);
    return ptau / beta0;
}}

static inline void LocalParticle_update_pzeta(LocalParticle* part,
        const XT_NUM& new_pzeta_value) {{
    double const beta0 = LocalParticle_get_beta0(part);
    LocalParticle_update_ptau(part, beta0 * new_pzeta_value);
}}

static inline void LocalParticle_add_to_energy(LocalParticle* part,
        const XT_NUM& delta_energy, int pz_only) {{
    XT_NUM ptau = LocalParticle_get_ptau(part);
    double const p0c = LocalParticle_get_p0c(part);
    double const charge_ratio = LocalParticle_get_charge_ratio(part);
    double const chi = LocalParticle_get_chi(part);
    double const mass_ratio = charge_ratio / chi;

    ptau += delta_energy / p0c / mass_ratio;

    XT_NUM const old_rpp = LocalParticle_get_rpp(part);

    LocalParticle_update_ptau(part, ptau);

    if (!pz_only) {{
        XT_NUM const new_rpp = LocalParticle_get_rpp(part);
        XT_NUM const f = old_rpp / new_rpp;
        LocalParticle_scale_px(part, f);
        LocalParticle_scale_py(part, f);
    }}
}}

static inline void LocalParticle_add_to_energy(LocalParticle* part,
        double delta_energy, int pz_only) {{
    LocalParticle_add_to_energy(part, xt_float_or_tpsa_lift(delta_energy), pz_only);
}}

static inline void increment_at_element(LocalParticle* part, int64_t increment) {{
    LocalParticle_add_to_at_element(part, increment);
}}

static inline void LocalParticle_kill_particle(LocalParticle* part,
        int64_t kill_state) {{
    LocalParticle_set_x(part, 1e30);
    LocalParticle_set_px(part, 1e30);
    LocalParticle_set_y(part, 1e30);
    LocalParticle_set_py(part, 1e30);
    LocalParticle_set_zeta(part, 1e30);
    LocalParticle_update_delta(part, -1.0);
    LocalParticle_set_state(part, kill_state);
}}

static inline int64_t check_is_active(LocalParticle* part) {{
    return LocalParticle_get_state(part) > 0;
}}

#endif
"""


def _insert_tpsa_local_particle(source):
    marker = '#include "xtrack/headers/track.h"'
    idx = source.find(marker)
    block = _tpsa_local_particle_source()
    if idx < 0:
        marker = '#include "xtrack/beam_elements/'
        idx = source.find(marker)
    if idx < 0:
        return block + "\n" + source
    return source[:idx] + block + "\n" + source[idx:]


def _element_classes_from_line(line):
    if not line._has_valid_tracker():
        line.build_tracker()
    classes = []
    for name in line.element_names:
        cls = line.element_dict[name]._XoStruct
        if cls not in classes:
            classes.append(cls)
    return sorted(classes, key=lambda cls: cls._DressingClass.__name__)


def _refdata_for_line(line, kernel_element_classes):
    names = tuple(line.element_names)
    key = (names, tuple(cls.__name__ for cls in kernel_element_classes))
    cache = _TPSA_REFDATA.setdefault(line, {})
    if key in cache:
        return cache[key]
    RefData = _element_ref_data_class_from_element_classes(kernel_element_classes)
    td = line.tracker._tracker_data_base
    erd = RefData(elements=len(names), names=list(names), _buffer=td._buffer)
    erd.elements = [line.element_dict[nn]._xobject for nn in names]
    cache[key] = erd
    return erd


def _build_tpsa_kernel(line):
    import xtrack as xt

    context = line.tracker._context
    if not isinstance(context, xo.ContextCpu):
        raise NotImplementedError("TPSA tracking is implemented only for ContextCpu")

    kernel_element_classes = _element_classes_from_line(line)
    src = ["""
extern "C" void tpsa_track_line(
        void* elem_ref_data_,
        void* particles_,
        int64_t ele_start,
        int64_t num_elements,
        double line_length,
        void* io_buffer_,
        uint64_t track_flags){

    ElementRefData elem_ref_data = (ElementRefData) elem_ref_data_;
    TpsaParticleData particles = (TpsaParticleData) particles_;
    int8_t* io_buffer = (int8_t*) io_buffer_;

    LocalParticle lpart;
    lpart.tp = particles;
    lpart.x = (tpsa_t*)(uintptr_t)TpsaParticleData_get_x(particles);
    lpart.px = (tpsa_t*)(uintptr_t)TpsaParticleData_get_px(particles);
    lpart.y = (tpsa_t*)(uintptr_t)TpsaParticleData_get_y(particles);
    lpart.py = (tpsa_t*)(uintptr_t)TpsaParticleData_get_py(particles);
    lpart.zeta = (tpsa_t*)(uintptr_t)TpsaParticleData_get_zeta(particles);
    lpart.delta = (tpsa_t*)(uintptr_t)TpsaParticleData_get_delta(particles);
    lpart.ptau = (tpsa_t*)(uintptr_t)TpsaParticleData_get_ptau(particles);
    lpart.rvv = (tpsa_t*)(uintptr_t)TpsaParticleData_get_rvv(particles);
    lpart.rpp = (tpsa_t*)(uintptr_t)TpsaParticleData_get_rpp(particles);
    lpart.s = (tpsa_t*)(uintptr_t)TpsaParticleData_get_s(particles);
    lpart.ax = (tpsa_t*)(uintptr_t)TpsaParticleData_get_ax(particles);
    lpart.ay = (tpsa_t*)(uintptr_t)TpsaParticleData_get_ay(particles);
    lpart.line_length = line_length;
    lpart.ipart = 0;
    lpart.endpart = 1;
    lpart._num_active_particles = 1;
    lpart._num_lost_particles = 0;
    lpart.track_flags = track_flags;
    lpart.io_buffer = io_buffer;
    xt_tpsa_proto = lpart.x;

    TpsaParticleData_set_state(particles, 1);
    TpsaParticleData_set_at_element(particles, ele_start);
    TpsaParticleData_set_track_flags(particles, track_flags);
    TpsaParticleData_set_line_length(particles, line_length);
    LocalParticle_update_delta(&lpart, LocalParticle_get_delta(&lpart));
    LocalParticle_set_s(&lpart, 0.0);
    LocalParticle_set_ax(&lpart, 0.0);
    LocalParticle_set_ay(&lpart, 0.0);

    int64_t ele_stop = ele_start + num_elements;
    for (int64_t elem_idx = ele_start; elem_idx < ele_stop; elem_idx++){
        void* el = ElementRefData_member_elements(elem_ref_data, elem_idx);
        int64_t elem_type = ElementRefData_typeid_elements(elem_ref_data, elem_idx);
        switch(elem_type){
"""]

    for ii, cls in enumerate(kernel_element_classes):
        base = cls.__name__.replace("Data", "")
        src.append(f"""
        case {ii}:
            {base}_track_local_particle_with_transformations(({base}Data) el, &lpart);
            break;
""")

    src.append("""
        default:
            break;
        }
        if (LocalParticle_get_state(&lpart) <= 0){
            break;
        }
        increment_at_element(&lpart, 1);
    }
}
""")

    RefData = _element_ref_data_class_from_element_classes(kernel_element_classes)
    saved_depends = {}
    for cls in [RefData, *kernel_element_classes]:
        if hasattr(cls, "_depends_on"):
            saved_depends[cls] = cls._depends_on
            cls._depends_on = [
                dep for dep in cls._depends_on
                if dep.__name__ != "ParticlesData"
                and not dep.__name__.startswith("Random")
            ]
    try:
        source_classes = sort_classes([
            RefData, TpsaParticleData, *kernel_element_classes
        ])
        source_classes = [
            cls for cls in source_classes if cls.__name__ != "ParticlesData"
        ]
        c_api_sources = []
        extra_sources = []
        for cls in source_classes:
            c_api_sources.append(cls._gen_c_api())
            for extra_source in getattr(cls, "_extra_c_sources", ()):
                if hasattr(extra_source, "read"):
                    extra_text = extra_source.read()
                elif hasattr(extra_source, "read_text"):
                    extra_text = extra_source.read_text()
                else:
                    extra_text = extra_source
                if "ParticlesData particles" in extra_text:
                    continue
                extra_sources.append(extra_text)
        kernels = context.build_kernels(
            sources=[
                *c_api_sources,
                _float_or_tpsa_getter_block(source_classes),
                *extra_sources,
                "\n".join(src),
            ],
            kernel_descriptions={
                "tpsa_track_line": xo.Kernel(
                    c_name="tpsa_track_line",
                    args=[],
                )
            },
            extra_headers=[
                "#define restrict __restrict",
                xt._pkg_root.joinpath("headers/constants.h"),
                TrackFlags.c_header_flag_mapping,
            ],
            extra_classes=[],
            apply_to_source=[
                _insert_tpsa_local_particle,
                _handle_per_particle_blocks,
            ],
            specialize=True,
            compiler_language="c++",
            extra_compile_args=[
                "-DXTRACK_MULTIPOLE_NO_SYNRAD",
                "-include", "complex",
                f"-I{xgtpsa.include_dir()}",
            ],
            extra_link_args=[
                f"-L{lib_dir()}",
                "-lmadng_tpsa",
            ],
            extra_cdef=(
                "void tpsa_track_line(void*, void*, int64_t, int64_t, "
                "double, void*, uint64_t);"
            ),
        )
    finally:
        for cls, depends_on in saved_depends.items():
            cls._depends_on = depends_on
    return kernels["tpsa_track_line"]


def _kernel_for_line(line):
    cache = _TPSA_KERNELS.setdefault(line, {})
    classes = tuple(cls.__name__ for cls in _element_classes_from_line(line))
    config = tuple(sorted(line.tracker.config.items()))
    key = (classes, config)
    if key not in cache:
        cache[key] = _build_tpsa_kernel(line)
    return cache[key]


class IntegratedTpsaBackend:
    def track_element(self, element, particles: ParticlesTpsa):
        import xtrack as xt

        line = xt.Line(elements=[element], element_names=["__tpsa_element__"])
        line.particle_ref = particles._ref_particle.copy()
        line.build_tracker(use_prebuilt_kernels=False)
        return self.track_line(line, particles)

    def track_line(
        self,
        line,
        particles: ParticlesTpsa,
        ele_start=0,
        ele_stop=None,
        num_elements=None,
        num_turns=None,
        turn_by_turn_monitor=None,
        multi_element_monitor_at=None,
    ):
        if turn_by_turn_monitor not in (None, False):
            raise NotImplementedError("TPSA monitors are not wired in the integrated kernel yet")
        if multi_element_monitor_at is not None:
            raise NotImplementedError("TPSA multi-element monitors are not wired yet")
        if num_turns not in (None, 1):
            raise NotImplementedError("TPSA tracking currently supports one turn")

        if isinstance(ele_start, str):
            ele_start = line.element_names.index(ele_start)
        ele_start = ele_start or 0
        if num_elements is None:
            if isinstance(ele_stop, str):
                ele_stop = line.element_names.index(ele_stop)
            if ele_stop is None or ele_stop == 0:
                ele_stop = len(line.element_names)
            if ele_stop < ele_start:
                raise NotImplementedError("TPSA wrap-around ranges are not wired yet")
            num_elements = ele_stop - ele_start

        if not line._has_valid_tracker():
            line.build_tracker()
        line.tracker.config.XTRACK_NO_TPSA_TRACK = False
        kernel = _kernel_for_line(line)
        kernel.description.n_threads = 1
        td = line.tracker._tracker_data_base
        kernel_element_classes = _element_classes_from_line(line)
        refdata = _refdata_for_line(line, kernel_element_classes)
        p = particles._xobject
        p.state = 1
        p.at_element = ele_start
        p.track_flags = 0
        p.line_length = float(td.line_length)
        ffi = kernel.ffi_interface
        kernel.function(
            _xobject_ptr(refdata, ffi),
            _xobject_ptr(p, ffi),
            int(ele_start),
            int(num_elements),
            float(td.line_length),
            ffi.cast("void*", np.frombuffer(
                line.tracker.io_buffer.buffer, dtype="int8"
            ).ctypes.data),
            0,
        )
        if p.state <= 0:
            at = p.at_element
            name = line.element_names[at] if at < len(line.element_names) else "?"
            raise RuntimeError(
                f"TPSA map lost at element index {at} ('{name}')"
            )
        return particles
