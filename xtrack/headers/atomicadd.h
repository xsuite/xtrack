// copyright ################################# //
// This file is part of the Xtrack Package.   //
// Copyright (c) CERN, 2025.                   //
// ########################################### //

#ifndef _ATOMICADD_H_
#define _ATOMICADD_H_

#include <stdint.h>

#if defined XO_CONTEXT_CPU || defined XO_CONTEXT_CL
#ifdef atomicAdd
    #warning "Warning: atomicAdd macro already defined, undefining it"
    #undef atomicAdd
#endif
#endif


#ifdef XO_CONTEXT_CPU
#ifdef XO_CONTEXT_CPU_OPENMP
    #ifndef _OPENMP
    #error "XO_CONTEXT_CPU_OPENMP set, but compiled without -fopenmp"
    #endif
    /* OpenMP atomic capture gives us read+write atomically */
    #define OMP_ATOMIC_CAPTURE  _Pragma("omp atomic capture")
#else
    #define OMP_ATOMIC_CAPTURE  /* no OpenMP: non-atomic fallback */
#endif // XO_CONTEXT_CPU_OPENMP

#define DEF_ATOMIC_ADD(T, SUF)                                  \
static inline T atomicAdd_##SUF(T *addr, T val) {               \
    T old;                                                      \
    const T inc = (T)val;                                       \
    OMP_ATOMIC_CAPTURE                                          \
    { old = *addr; *addr = *addr + inc; }                       \
    return old;                                                 \
}

DEF_ATOMIC_ADD( int8_t ,  i8)
DEF_ATOMIC_ADD( int16_t, i16)
DEF_ATOMIC_ADD( int32_t, i32)
DEF_ATOMIC_ADD( int64_t, i64)
DEF_ATOMIC_ADD(uint8_t ,  u8)
DEF_ATOMIC_ADD(uint16_t, u16)
DEF_ATOMIC_ADD(uint32_t, u32)
DEF_ATOMIC_ADD(uint64_t, u64)
DEF_ATOMIC_ADD(float   , f32)
DEF_ATOMIC_ADD(double  , f64)

#define atomicAdd(addr, val) _Generic((addr),        \
    int8_t*:   atomicAdd_i8,                         \
    int16_t*:  atomicAdd_i16,                        \
    int32_t*:  atomicAdd_i32,                        \
    int64_t*:  atomicAdd_i64,                        \
    uint8_t*:  atomicAdd_u8,                         \
    uint16_t*: atomicAdd_u16,                        \
    uint32_t*: atomicAdd_u32,                        \
    uint64_t*: atomicAdd_u64,                        \
    float*:    atomicAdd_f32,                        \
    double*:   atomicAdd_f64                         \
)(addr, (val))
#endif // XO_CONTEXT_CPU


// CUDA provides atomicAdd() natively


#ifdef XO_CONTEXT_CL
// atomic_add_compat.cl (portable)
#pragma OPENCL EXTENSION cl_khr_fp64                   : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics     : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
// Map 1.0 "atom_*" names to 1.1+ "atomic_*" if needed.
#if !defined(__OPENCL_C_VERSION__) || (__OPENCL_C_VERSION__ < 110)
  #define atomic_add     atom_add
  #define atomic_cmpxchg atom_cmpxchg
#endif

#define OCL_OVERLOAD __attribute__((overloadable))

// -------------------- 32-bit integers (core) --------------------
inline int  OCL_OVERLOAD atomicAdd(volatile __global int  *p, int  v) { return atomic_add(p, v); }
inline uint OCL_OVERLOAD atomicAdd(volatile __global uint *p, uint v) { return atomic_add(p, v); }
inline int  OCL_OVERLOAD atomicAdd(volatile __local  int  *p, int  v) { return atomic_add(p, v); }
inline uint OCL_OVERLOAD atomicAdd(volatile __local  uint *p, uint v) { return atomic_add(p, v); }

// -------------------- 64-bit integers (needs cl_khr_int64_* ) ---
#ifdef cl_khr_int64_base_atomics
inline long  OCL_OVERLOAD atomicAdd(volatile __global long  *p, long  v) { return atom_add(p, v); }
inline ulong OCL_OVERLOAD atomicAdd(volatile __global ulong *p, ulong v) { return atom_add(p, v); }
inline long  OCL_OVERLOAD atomicAdd(volatile __local  long  *p, long  v) { return atom_add(p, v); }
inline ulong OCL_OVERLOAD atomicAdd(volatile __local  ulong *p, ulong v) { return atom_add(p, v); }
#endif // cl_khr_int64_base_atomics

// -------------------- 32-bit float via CAS ----------------------
inline float OCL_OVERLOAD atomicAdd(volatile __global float *p, float v){
    uint old_bits, new_bits;
    do {
        old_bits = as_uint(*p);
        new_bits = as_uint(as_float(old_bits) + v);
    } while (atomic_cmpxchg((volatile __global uint*)p, old_bits, new_bits) != old_bits);
    return as_float(old_bits);  // return previous value (fetch-add)
}

inline float OCL_OVERLOAD atomicAdd(volatile __local float *p, float v){
    uint old_bits, new_bits;
    do {
        old_bits = as_uint(*p);
        new_bits = as_uint(as_float(old_bits) + v);
    } while (atomic_cmpxchg((volatile __local uint*)p, old_bits, new_bits) != old_bits);
    return as_float(old_bits);
}

// -------------------- 64-bit double via CAS ---------------------
#if defined(cl_khr_fp64) && defined(cl_khr_int64_base_atomics)
inline double OCL_OVERLOAD atomicAdd(volatile __global double *p, double v){
    ulong old_bits, new_bits;
    do {
        old_bits = as_ulong(*p);
        new_bits = as_ulong(as_double(old_bits) + v);
    } while (atom_cmpxchg((volatile __global ulong*)p, old_bits, new_bits) != old_bits);
    return as_double(old_bits);
}
inline double OCL_OVERLOAD atomicAdd(volatile __local double *p, double v){
    ulong old_bits, new_bits;
    do {
        old_bits = as_ulong(*p);
        new_bits = as_ulong(as_double(old_bits) + v);
    } while (atom_cmpxchg((volatile __local ulong*)p, old_bits, new_bits) != old_bits);
    return as_double(old_bits);
}
#endif // cl_khr_fp64 && cl_khr_int64_base_atomics

#endif // XO_CONTEXT_CL

#endif //_ATOMICADD_H_
