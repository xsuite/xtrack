// copyright ################################# //
// This file is part of the Xtrack Package.    //
// Copyright (c) CERN, 2025.                   //
// ########################################### //

#ifndef _ATOMICADD_H_
#define _ATOMICADD_H_

#include <headers/track.h>


// Ensure no conflicting atomicAdd macro is defined earlier for CPU or OpenCL
#ifndef XO_CONTEXT_CUDA
#ifdef atomicAdd
    #warning "Warning: atomicAdd macro already defined, undefining it."
    #undef atomicAdd
#endif // atomicAdd
#endif // XO_CONTEXT_CUDA


// ################################# //
// ###        CPU Contexts       ### //
// ################################# //

#ifdef XO_CONTEXT_CPU
#include <stdint.h>

#ifdef XO_CONTEXT_CPU_OPENMP
    #ifndef _OPENMP
    #error "XO_CONTEXT_CPU_OPENMP set, but compiled without -fopenmp"
    #endif
    // OpenMP atomic capture gives us read+write atomically
    #define OMP_ATOMIC_CAPTURE  _Pragma("omp atomic capture")
#else
    #define OMP_ATOMIC_CAPTURE  // No OpenMP: non-atomic fallback
#endif // XO_CONTEXT_CPU_OPENMP

// Macro to define atomicAdd for different types, will be overloaded via _Generic.
#define DEF_ATOMIC_ADD(T, SUF)                                  \
GPUFUN T atomicAdd_##SUF(GPUGLMEM T *addr, T val) {               \
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

// Using _Generic to select the right function based on type (since C11).
// See https://en.cppreference.com/w/c/language/generic.html
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


// ################################# //
// ###        CUDA Context       ### //
// ################################# //

#ifdef XO_CONTEXT_CUDA
// CUDA provides atomicAdd() natively, but only for floats, doubles, 32/64-bit
// unsigned integers, and 32-bit signed integers. We will add the other integer
// types (8/16-bit signed/unsigned and 64-bit signed) via CAS using templates.

// CUDA compiler may not have <stdint.h>, so define the types if needed.
#ifdef __CUDACC_RTC__
    // NVRTC (CuPy RawModule default) can’t see <stdint.h>, so detect it via __CUDACC_RTC__
    typedef signed char        int8_t;
    typedef short              int16_t;
    typedef int                int32_t;
    typedef long long          int64_t;
    typedef unsigned char      uint8_t;
    typedef unsigned short     uint16_t;
    typedef unsigned int       uint32_t;
    typedef unsigned long long uint64_t;
#else
    // Alternatively, NVCC path is fine with host headers
    #include <stdint.h>
#endif // __CUDACC_RTC__

/* -------------- Design notes ------------------------------------------------
 * We implement 8/16-bit atomic add by atomically CAS'ing the *containing*
 * 32-bit word. Only the target byte/halfword lane is modified; the neighbor
 * lane is preserved. This is linearisable: each successful CAS is one atomic
 * RMW on the 32-bit word.
 * Assumptions: little-endian lane layout (true on NVIDIA) and natural alignment
 * of the 16-bit addresses (addr % 2 == 0). 8-bit has no alignment requirement.
 * Return value matches CUDA semantics: the **old** value at *addr* (fetch-add).
 * ---------------------------------------------------------------------------*/

// Helper: compute (base 32-bit word pointer, shift, mask) for a byte in that word.
GPUFUN void __xt_lane8(const void* addr, uint32_t*& word, uint32_t& shift, uint32_t& mask){
    uint64_t a = (uint64_t)addr;
    word  = (uint32_t*)(a & ~3ULL);        // align down to 4-byte boundary
    shift = (uint32_t)((a & 3ULL) * 8ULL); // 0,8,16,24 depending on byte lane
    mask  = 0xFFu << shift;
}

// Helper: same for a halfword (16-bit) in the containing 32-bit word.
GPUFUN void __xt_lane16(const void* addr, uint32_t*& word, uint32_t& shift, uint32_t& mask){
    uint64_t a = (uint64_t)addr;
    word  = (uint32_t*)(a & ~3ULL);            // align down to 4-byte boundary
    shift = (uint32_t)((a & 2ULL) ? 16U : 0U); // 0 or 16 depending on halfword
    mask  = 0xFFFFu << shift;
}

// ---------------- 8-bit: int8_t / uint8_t (CAS on 32-bit word) --------------
GPUFUN int8_t xt_atomicAdd_i8(int8_t* addr, int8_t val){
    uint32_t *w, sh, mask;
    __xt_lane8(addr, w, sh, mask);
    uint32_t old = *w, assumed, byte, newbyte, nw;
    do {
        assumed = old;
        byte    = (assumed & mask) >> sh;   // Extract current 8-bit lane
        newbyte = (uint32_t)((uint8_t)byte + (uint8_t)val);  // Add in modulo-256 (two's complement)
        nw      = (assumed & ~mask) | ((newbyte & 0xFFu) << sh);  // Merge back updated lane; leave neighbor lanes intact
        // Try to publish; if someone raced us, retry with their value
        old     = atomicCAS(w, assumed, nw);
    } while (old != assumed);
    return (int8_t)((assumed & mask) >> sh);
}

GPUFUN uint8_t xt_atomicAdd_u8(uint8_t* addr, uint8_t val){
    uint32_t *w, sh, mask;
    __xt_lane8(addr, w, sh, mask);
    uint32_t old = *w, assumed, byte, newbyte, nw;
    do {
        assumed = old;
        byte    = (assumed & mask) >> sh;
        newbyte = (uint32_t)(byte + val);       // modulo-256
        nw      = (assumed & ~mask) | ((newbyte & 0xFFu) << sh);
        old     = atomicCAS(w, assumed, nw);
    } while (old != assumed);
    return (uint8_t)((assumed & mask) >> sh);
}

// ---------------- 16-bit: int16_t / uint16_t (CAS on 32-bit word) -----------
GPUFUN int16_t xt_atomicAdd_i16(int16_t* addr, int16_t val){
    uint32_t *w, sh, mask;
    __xt_lane16(addr, w, sh, mask);
    uint32_t old = *w, assumed, half, newhalf, nw;
    do {
        assumed = old;
        half    = (assumed & mask) >> sh;                     // current 16-bit lane
        newhalf = (uint32_t)((uint16_t)half + (uint16_t)val); // modulo-65536
        nw      = (assumed & ~mask) | ((newhalf & 0xFFFFu) << sh);
        old     = atomicCAS(w, assumed, nw);
    } while (old != assumed);
    return (int16_t)((assumed & mask) >> sh);
}

GPUFUN uint16_t xt_atomicAdd_u16(uint16_t* addr, uint16_t val){
    uint32_t *w, sh, mask;
    __xt_lane16(addr, w, sh, mask);
    uint32_t old = *w, assumed, half, newhalf, nw;
    do {
        assumed = old;
        half    = (assumed & mask) >> sh;
        newhalf = (uint32_t)(half + val);                   // modulo-65536
        nw      = (assumed & ~mask) | ((newhalf & 0xFFFFu) << sh);
        old     = atomicCAS(w, assumed, nw);
    } while (old != assumed);
    return (uint16_t)((assumed & mask) >> sh);

// ---------------- 64-bit: double (built-in or CAS on 64-bit word) -----------
#if not defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 600)
// Old hardware does not have the built-in atomicAdd for doubles, so we use
// a bitwise CAS loop on the 64-bit value.
GPUFUN  double   xt_atomicAdd_f64(double* p, double v) {
    uint64_t* w = (uint64_t*)p;
    uint64_t old = *w, assumed, nw;
    do {
        assumed = old;
        double cur = __longlong_as_double(assumed);
        double sum = cur + v;
        nw = __double_as_longlong(sum);
        old = atomicCAS(w, assumed, nw);
    } while (old != assumed);
    return __longlong_as_double(assumed);
#endif
}

// NVRTC (CuPy RawModule) usually compiles under extern "C".
// Overloads are C++ only; force C++ linkage just for this block.
#ifdef __cplusplus
extern "C++" {

GPUFUN int8_t   xt_atomicAdd(int8_t*  p, int8_t  v)  { return xt_atomicAdd_i8 (p, v); }
GPUFUN uint8_t  xt_atomicAdd(uint8_t* p, uint8_t v)  { return xt_atomicAdd_u8 (p, v); }
GPUFUN int16_t  xt_atomicAdd(int16_t* p, int16_t v)  { return xt_atomicAdd_i16(p, v); }
GPUFUN uint16_t xt_atomicAdd(uint16_t*p, uint16_t v) { return xt_atomicAdd_u16(p, v); }

// wide types: forward to CUDA built-ins
GPUFUN int32_t  xt_atomicAdd(int32_t* p, int32_t v)   { return ::atomicAdd(p, v); }
GPUFUN uint32_t xt_atomicAdd(uint32_t* p, uint32_t v) { return ::atomicAdd(p, v); }
GPUFUN uint64_t xt_atomicAdd(uint64_t* p, uint64_t v) { return ::atomicAdd(p, v); }
GPUFUN float    xt_atomicAdd(float* p, float v)       { return ::atomicAdd(p, v); }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
GPUFUN double   xt_atomicAdd(double* p, double v)     { return ::atomicAdd(p, v); }
#else
GPUFUN double   xt_atomicAdd(double* p, double v)     { return xt_atomicAdd_f64(p, v); }
#endif

}
#endif // __cplusplus

// ---------- Global remap of the public name on device code ----------
// Define AFTER the wrappers so we don't macro-rewrite our own calls.
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#  ifdef atomicAdd
#    undef atomicAdd
#  endif
#  define atomicAdd(ptr, val) xt_atomicAdd((ptr), (val))
#endif

#endif /* XO_CONTEXT_CUDA */


// ################################# //
// ###       OpenCL Context      ### //
// ################################# //

#ifdef XO_CONTEXT_CL
// Note that the OpenCL context already has the types from <stdint.h> defined.

// atomic_add_compat.cl (portable)
#pragma OPENCL EXTENSION cl_khr_fp64                   : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics     : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
// Map 1.0 "atom_*" names to 1.1+ "atomic_*" if needed.
#if !defined(__OPENCL_C_VERSION__) || (__OPENCL_C_VERSION__ < 110)
  #define atomic_add     atom_add
  #define atomic_cmpxchg atom_cmpxchg
#endif

// #define OCL_OVERLOAD __attribute__((overloadable))

// // -------- helpers: shift/mask for subword updates (little-endian) -------- 
// inline uint _shift8(GPUGLMEM const void* p)  { return (uint)((size_t)p & 3u) * 8u; }
// inline uint _shift16(GPUGLMEM const void* p) { return (uint)(((size_t)p & 2u) ? 16u : 0u); }

// // -------------------- 8-bit integers --------------------
// inline char  atomicAdd(volatile GPUGLMEM char  *p, char  v) {
//     volatile GPUGLMEM uint *w = (volatile GPUGLMEM uint*)((size_t)p & ~3u);
//     uint sh = _shift8((GPUGLMEM const void*)p), mask = 0xFFu << sh;
//     uint old = *w, assumed, byte, newbyte, nw;
//     do {
//         assumed = old;
//         byte    = (assumed & mask) >> sh;
//         newbyte = (uint)((uchar)byte + (uchar)v);
//         nw      = (assumed & ~mask) | ((newbyte & 0xFFu) << sh);
//         old     = atomic_cmpxchg(w, assumed, nw);
//     } while (old != assumed);
//     return (char)((assumed & mask) >> sh);
// }
// inline uchar atomicAdd(volatile GPUGLMEM uchar *p, uchar v) {
//     volatile GPUGLMEM uint *w = (volatile GPUGLMEM uint*)((size_t)p & ~3u);
//     uint sh = _shift8((GPUGLMEM const void*)p), mask = 0xFFu << sh;
//     uint old = *w, assumed, byte, newbyte, nw;
//     do {
//         assumed = old;
//         byte    = (assumed & mask) >> sh;
//         newbyte = (uint)(byte + v);
//         nw      = (assumed & ~mask) | ((newbyte & 0xFFu) << sh);
//         old     = atomic_cmpxchg(w, assumed, nw);
//     } while (old != assumed);
//     return (uchar)((assumed & mask) >> sh);
// }

// // -------------------- 16-bit integers --------------------
// inline short  atomicAdd(volatile GPUGLMEM short  *p, short  v) {
//     volatile GPUGLMEM uint *w = (volatile GPUGLMEM uint*)((size_t)p & ~3u);
//     uint sh = _shift16((GPUGLMEM const void*)p), mask = 0xFFFFu << sh;
//     uint old = *w, assumed, half, newhalf, nw;
//     do {
//         assumed = old;
//         half    = (assumed & mask) >> sh;
//         newhalf = (uint)((ushort)half + (ushort)v);
//         nw      = (assumed & ~mask) | ((newhalf & 0xFFFFu) << sh);
//         old     = atomic_cmpxchg(w, assumed, nw);
//     } while (old != assumed);
//     return (short)((assumed & mask) >> sh);
// }
// inline ushort atomicAdd(volatile GPUGLMEM ushort *p, ushort v) {
//     volatile GPUGLMEM uint *w = (volatile GPUGLMEM uint*)((size_t)p & ~3u);
//     uint sh = _shift16((GPUGLMEM const void*)p), mask = 0xFFFFu << sh;
//     uint old = *w, assumed, half, newhalf, nw;
//     do {
//         assumed = old;
//         half    = (assumed & mask) >> sh;
//         newhalf = (uint)(half + v);
//         nw      = (assumed & ~mask) | ((newhalf & 0xFFFFu) << sh);
//         old     = atomic_cmpxchg(w, assumed, nw);
//     } while (old != assumed);
//     return (ushort)((assumed & mask) >> sh);
// }


// // -------------------- 32-bit integers (core) --------------------
// // inline int  OCL_OVERLOAD atomicAdd(volatile __global int  *p, int  v) { return atomic_add(p, v); }
// // inline uint OCL_OVERLOAD atomicAdd(volatile __global uint *p, uint v) { return atomic_add(p, v); }
// GPUFUN int  OCL_OVERLOAD atomicAdd(volatile GPUGLMEM int  *p, int  v) { return atomic_add(p, v); }
// GPUFUN uint OCL_OVERLOAD atomicAdd(volatile GPUGLMEM uint *p, uint v) { return atomic_add(p, v); }
// // inline int  OCL_OVERLOAD atomicAdd(volatile __local  int  *p, int  v) { return atomic_add(p, v); }
// // inline uint OCL_OVERLOAD atomicAdd(volatile __local  uint *p, uint v) { return atomic_add(p, v); }

// // -------------------- 64-bit integers (needs cl_khr_int64_* ) ---
// #ifdef cl_khr_int64_base_atomics
// // inline long  OCL_OVERLOAD atomicAdd(volatile __global long  *p, long  v) { return atom_add(p, v); }
// // inline ulong OCL_OVERLOAD atomicAdd(volatile __global ulong *p, ulong v) { return atom_add(p, v); }
// GPUFUN long  OCL_OVERLOAD atomicAdd(volatile GPUGLMEM long  *p, long  v) { return atom_add(p, v); }
// GPUFUN ulong OCL_OVERLOAD atomicAdd(volatile GPUGLMEM ulong *p, ulong v) { return atom_add(p, v); }
// // inline long  OCL_OVERLOAD atomicAdd(volatile __local  long  *p, long  v) { return atom_add(p, v); }
// // inline ulong OCL_OVERLOAD atomicAdd(volatile __local  ulong *p, ulong v) { return atom_add(p, v); }
// #endif // cl_khr_int64_base_atomics

// // -------------------- 32-bit float via CAS ----------------------
// // inline float OCL_OVERLOAD atomicAdd(volatile __global float *p, float v){
// //     uint old_bits, new_bits;
// //     do {
// //         old_bits = as_uint(*p);
// //         new_bits = as_uint(as_float(old_bits) + v);
// //     } while (atomic_cmpxchg((volatile __global uint*)p, old_bits, new_bits) != old_bits);
// //     return as_float(old_bits);  // return previous value (fetch-add)
// // }
// GPUFUN float OCL_OVERLOAD atomicAdd(volatile GPUGLMEM float *p, float v){
//     uint old_bits, new_bits;
//     do {
//         old_bits = as_uint(*p);
//         new_bits = as_uint(as_float(old_bits) + v);
//     } while (atomic_cmpxchg((volatile GPUGLMEM uint*)p, old_bits, new_bits) != old_bits);
//     return as_float(old_bits);  // return previous value (fetch-add)
// }

// // inline float OCL_OVERLOAD atomicAdd(volatile __local float *p, float v){
// //     uint old_bits, new_bits;
// //     do {
// //         old_bits = as_uint(*p);
// //         new_bits = as_uint(as_float(old_bits) + v);
// //     } while (atomic_cmpxchg((volatile __local uint*)p, old_bits, new_bits) != old_bits);
// //     return as_float(old_bits);
// // }

// // -------------------- 64-bit double via CAS ---------------------
// // #if defined(cl_khr_fp64) && defined(cl_khr_int64_base_atomics)
// // inline double OCL_OVERLOAD atomicAdd(volatile __global double *p, double v){
// //     ulong old_bits, new_bits;
// //     do {
// //         old_bits = as_ulong(*p);
// //         new_bits = as_ulong(as_double(old_bits) + v);
// //     } while (atom_cmpxchg((volatile __global ulong*)p, old_bits, new_bits) != old_bits);
// //     return as_double(old_bits);
// // }
// #if defined(cl_khr_fp64) && defined(cl_khr_int64_base_atomics)
// GPUFUN double OCL_OVERLOAD atomicAdd(volatile GPUGLMEM double *p, double v){
//     ulong old_bits, new_bits;
//     do {
//         old_bits = as_ulong(*p);
//         new_bits = as_ulong(as_double(old_bits) + v);
//     } while (atom_cmpxchg((volatile GPUGLMEM ulong*)p, old_bits, new_bits) != old_bits);
//     return as_double(old_bits);
// }
// // inline double OCL_OVERLOAD atomicAdd(volatile __local double *p, double v){
// //     ulong old_bits, new_bits;
// //     do {
// //         old_bits = as_ulong(*p);
// //         new_bits = as_ulong(as_double(old_bits) + v);
// //     } while (atom_cmpxchg((volatile __local ulong*)p, old_bits, new_bits) != old_bits);
// //     return as_double(old_bits);
// // }
// #endif // cl_khr_fp64 && cl_khr_int64_base_atomics


/* ---------- 32-bit ints (built-in) ---------- */
static int  atomicAdd(volatile __global int  *p, int  v)  { return atomic_add(p, v); }
static uint atomicAdd(volatile __global uint *p, uint v)  { return atomic_add(p, v); }

/* ---------- 8/16-bit via 32-bit CAS (little-endian) ---------- */
static char  atomicAdd(volatile __global char  *p, char  v) {
    volatile __global uint *w = (volatile __global uint*)((size_t)p & ~3u);
    uint sh = (uint)((size_t)p & 3u) * 8u, mask = 0xFFu << sh;
    uint old = *w, assumed, byte, newbyte, nw;
    do {
        assumed = old;
        byte    = (assumed & mask) >> sh;
        newbyte = (uint)((uchar)byte + (uchar)v);
        nw      = (assumed & ~mask) | ((newbyte & 0xFFu) << sh);
        old     = atomic_cmpxchg(w, assumed, nw);
    } while (old != assumed);
    return (char)((assumed & mask) >> sh);
}
static uchar atomicAdd(volatile __global uchar *p, uchar v) {
    volatile __global uint *w = (volatile __global uint*)((size_t)p & ~3u);
    uint sh = (uint)((size_t)p & 3u) * 8u, mask = 0xFFu << sh;
    uint old = *w, assumed, byte, newbyte, nw;
    do {
        assumed = old;
        byte    = (assumed & mask) >> sh;
        newbyte = (uint)(byte + v);
        nw      = (assumed & ~mask) | ((newbyte & 0xFFu) << sh);
        old     = atomic_cmpxchg(w, assumed, nw);
    } while (old != assumed);
    return (uchar)((assumed & mask) >> sh);
}

static short  atomicAdd(volatile __global short  *p, short  v) {
    volatile __global uint *w = (volatile __global uint*)((size_t)p & ~3u);
    uint sh = (uint)(((size_t)p & 2u) ? 16u : 0u), mask = 0xFFFFu << sh;
    uint old = *w, assumed, half, newhalf, nw;
    do {
        assumed = old;
        half    = (assumed & mask) >> sh;
        newhalf = (uint)((ushort)half + (ushort)v);
        nw      = (assumed & ~mask) | ((newhalf & 0xFFFFu) << sh);
        old     = atomic_cmpxchg(w, assumed, nw);
    } while (old != assumed);
    return (short)((assumed & mask) >> sh);
}
static ushort atomicAdd(volatile __global ushort *p, ushort v) {
    volatile __global uint *w = (volatile __global uint*)((size_t)p & ~3u);
    uint sh = (uint)(((size_t)p & 2u) ? 16u : 0u), mask = 0xFFFFu << sh;
    uint old = *w, assumed, half, newhalf, nw;
    do {
        assumed = old;
        half    = (assumed & mask) >> sh;
        newhalf = (uint)(half + v);
        nw      = (assumed & ~mask) | ((newhalf & 0xFFFFu) << sh);
        old     = atomic_cmpxchg(w, assumed, nw);
    } while (old != assumed);
    return (ushort)((assumed & mask) >> sh);
}

/* ---------- float via 32-bit CAS ---------- */
static float atomicAdd(volatile __global float *p, float v) {
    uint oldb, newb;
    do {
        oldb = as_uint(*p);
        newb = as_uint(as_float(oldb) + v);
    } while (atomic_cmpxchg((volatile __global uint*)p, oldb, newb) != oldb);
    return as_float(oldb);
}

/* ---------- 64-bit (ints/double) need extensions ---------- */
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#ifdef cl_khr_int64_base_atomics
static ulong atomicAdd(volatile __global ulong *p, ulong v) { return atom_add(p, v); }
static long  atomicAdd(volatile __global long  *p, long  v) { return atom_add(p, v); }
#endif

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#if defined(cl_khr_fp64) && defined(cl_khr_int64_base_atomics)
static double atomicAdd(volatile __global double *p, double v) {
    ulong oldb, newb;
    do {
        oldb = as_ulong(*p);
        newb = as_ulong(as_double(oldb) + v);
    } while (atom_cmpxchg((volatile __global ulong*)p, oldb, newb) != oldb);
    return as_double(oldb);
}
#endif


#endif // XO_CONTEXT_CL

#endif //_ATOMICADD_H_
