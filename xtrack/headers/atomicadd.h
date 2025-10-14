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

DEF_ATOMIC_ADD(int8_t ,   i8)
DEF_ATOMIC_ADD(int16_t,  i16)
DEF_ATOMIC_ADD(int32_t,  i32)
DEF_ATOMIC_ADD(int64_t,  i64)
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
// ###        GPU Contexts       ### //
// ################################# //

/* -------------- Design notes for GPU atomics ---------------------------------
* We will go for a unified approach for CUDA and OpenCL, with macros where needed.
* We implement 8/16-bit atomic add by atomically CAS'ing the *containing*
 * 32-bit word. Only the target byte/halfword lane is modified; the neighbor
 * lane is preserved. This is linearisable: each successful CAS is one atomic
 * RMW on the 32-bit word.
 * Assumptions: little-endian lane layout (true on NVIDIA) and natural alignment
 * of the 16-bit addresses (addr % 2 == 0). 8-bit has no alignment requirement.
 * Return value matches CUDA semantics: the **old** value at *addr* (fetch-add).
 * ---------------------------------------------------------------------------*/

// Info: a CAS function (Compare-And-Swap) takes three arguments: a pointer to
// the value to be modified, the expected old value, and the new value. The
// second argument helps to recognise if another thread has modified the value
// in the meantime. The function compares the value at the address with the
// expected values, and if they are the same, it writes the new value at the
// address. In any case, the function returns the actual old value at the
// address. If the returned value is different from the expected value, it
// means that another thread has modified the value in the meantime.

// Define types and macros for CUDA and OpenCL
// -------------------------------------------
#if defined(XO_CONTEXT_CUDA)
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
    #define GPUVOLATILE    GPUGLMEM
    #define __XT_CAS_U32(ptr, exp, val) atomicCAS((GPUVOLATILE uint32_t*)(ptr), (uint32_t)(exp), (uint32_t)(val))
    #define __XT_CAS_U64(ptr, exp, val) atomicCAS((GPUVOLATILE uint64_t*)(ptr), (uint64_t)(exp), (uint64_t)(val))
    #define __XT_AS_U32_FROM_F32(x) __float_as_uint(x)
    #define __XT_AS_F32_FROM_U32(x) __uint_as_float(x)
    #define __XT_AS_U64_FROM_F64(x) __double_as_longlong(x)
    #define __XT_AS_F64_FROM_U64(x) __longlong_as_double(x)
#elif defined(XO_CONTEXT_CL)
    // It seems OpenCL already has the types from <stdint.h> defined.
    // typedef char           int8_t;
    // typedef short          int16_t;
    // typedef int            int32_t;
    // typedef long           int64_t;
    // typedef unsigned char  uint8_t;
    // typedef unsigned short uint16_t;
    // typedef unsigned int   uint32_t;
    // typedef unsigned long  uint64_t;
    #define GPUVOLATILE    GPUGLMEM volatile
    #if __OPENCL_C_VERSION__ < 110
        // Map 1.0 "atom_*" names to 1.1+ "atomic_*"  
        #pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
        #define atomic_add     atom_add
        #define atomic_cmpxchg atom_cmpxchg
    #endif
    #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #define __XT_CAS_U32(ptr, exp, val) atomic_cmpxchg((GPUVOLATILE uint32_t*)(ptr), (uint32_t)(exp), (uint32_t)(val))
    #define __XT_CAS_U64(ptr, exp, val) atom_cmpxchg((GPUVOLATILE uint64_t*)(ptr), (uint64_t)(exp), (uint64_t)(val))
    #define __XT_AS_U32_FROM_F32(x) as_uint(x)
    #define __XT_AS_F32_FROM_U32(x) as_float(x)
    #define __XT_AS_U64_FROM_F64(x) as_ulong(x)
    #define __XT_AS_F64_FROM_U64(x) as_double(x)
#endif // XO_CONTEXT_CUDA / XO_CONTEXT_CL

// Define atomic functions per type
// --------------------------------
#if defined(XO_CONTEXT_CUDA) || defined(XO_CONTEXT_CL)
    // Helper: compute (base 32-bit word pointer, shift, mask) for a byte in that word.
    GPUFUN inline void __xt_lane8(GPUVOLATILE void* addr, GPUVOLATILE uint32_t** w, uint32_t* sh, uint32_t* mask){
        size_t a = (size_t)addr;
        *w    = (GPUVOLATILE uint32_t*)(a & ~(size_t)3); // word: align down to 4-byte boundary
        *sh   = (uint32_t)((a & (size_t)3) * 8U);        // shift 0,8,16,24 depending on byte lane
        *mask = (uint32_t)(0xFFu << *sh);
    }
    // Helper: same for a halfword (16-bit) in the containing 32-bit word.
    GPUFUN inline void __xt_lane16(GPUVOLATILE void* addr, GPUVOLATILE uint32_t** w, uint32_t* sh, uint32_t* mask){
        size_t a = (size_t)addr;
        *w  = (GPUVOLATILE uint32_t*)(a & ~(size_t)3);  // word: align down to 4-byte boundary
        *sh = (uint32_t)((a & (size_t)2) ? 16U : 0U);   // shift0 or 16 depending on halfword
        *mask  = (uint32_t)(0xFFFFU << *sh);
    }

    // ---------------- 8-bit: int8_t / uint8_t (CAS on 32-bit word) --------------
    GPUFUN int8_t atomicAdd_i8(GPUVOLATILE int8_t* addr, int8_t val){
        GPUVOLATILE uint32_t *w;
        uint32_t sh, mask;
        __xt_lane8(addr, &w, &sh, &mask);
        uint32_t old = *w, assumed, b, nb, nw;  // byte, newbyte, newword
        do {
            assumed = old;
            b  = (assumed & mask) >> sh;                    // Extract current 8-bit lane
            nb = (uint32_t)((uint8_t)b + (uint8_t)val);     // Add in modulo-256 (two's complement)
            nw = (assumed & ~mask) | ((nb & 0xFFU) << sh);  // Merge back updated lane; leave neighbor lanes intact
            // Try to publish; if someone raced us, retry with their value
            old = __XT_CAS_U32(w, assumed, nw);
        } while (old != assumed);
        return (int8_t)((assumed & mask) >> sh);
    }
    GPUFUN uint8_t atomicAdd_u8(GPUVOLATILE uint8_t* addr, uint8_t val){
        GPUVOLATILE uint32_t *w;
        uint32_t sh, mask;
        __xt_lane8(addr, &w, &sh, &mask);
        uint32_t old = *w, assumed, b, nb, nw;
        do {
            assumed = old;
            b  = (assumed & mask) >> sh;
            nb = (uint32_t)(b + val);                             /* mod 256 */
            nw = (assumed & ~mask) | ((nb & 0xFFU) << sh);
            old = __XT_CAS_U32(w, assumed, nw);
        } while (old != assumed);
        return (uint8_t)((assumed & mask) >> sh);
    }

    // ---------------- 16-bit: int16_t / uint16_t (CAS on 32-bit word) -----------
    GPUFUN int16_t atomicAdd_i16(GPUVOLATILE int16_t* addr, int16_t val){
        GPUVOLATILE uint32_t* w;
        uint32_t sh, mask;
        __xt_lane16(addr, &w, &sh, &mask);
        uint32_t old = *w, assumed, b, nb, nw;
        do {
            assumed = old;
            b  = (assumed & mask) >> sh;
            nb = (uint32_t)((uint16_t)b + (uint16_t)val);
            nw = (assumed & ~mask) | ((nb & 0xFFFFU) << sh);
            old = __XT_CAS_U32(w, assumed, nw);
        } while (old != assumed);
        return (int16_t)((assumed & mask) >> sh);
    }
    GPUFUN uint16_t atomicAdd_u16(GPUVOLATILE uint16_t* addr, uint16_t val){
        GPUVOLATILE uint32_t* w;
        uint32_t sh, mask;
        __xt_lane16(addr, &w, &sh, &mask);
        uint32_t old = *w, assumed, b, nb, nw;
        do {
            assumed = old;
            b  = (assumed & mask) >> sh;
            nb = (uint32_t)(b + val);
            nw = (assumed & ~mask) | ((nb & 0xFFFFU) << sh);
            old = __XT_CAS_U32(w, assumed, nw);
        } while (old != assumed);
        return (uint16_t)((assumed & mask) >> sh);
    }

    // ---------------- 32-bit: int32_t / uint32_t (built-in) -----------
    GPUFUN int32_t atomicAdd_i32(GPUVOLATILE int32_t* addr, int32_t val){
    #ifdef XO_CONTEXT_CUDA
        return atomicAdd(addr, val);
    #else // XO_CONTEXT_CL
        return atomic_add(addr, val);
    #endif // XO_CONTEXT_CUDA / XO_CONTEXT_CL
    }
    GPUFUN uint32_t atomicAdd_u32(GPUVOLATILE uint32_t* addr, uint32_t val){
    #ifdef XO_CONTEXT_CUDA
        return atomicAdd(addr, val);
    #else // XO_CONTEXT_CL
        return atomic_add(addr, val);
    #endif // XO_CONTEXT_CUDA / XO_CONTEXT_CL
    }

    // ---------------- 64-bit: int64_t / uint64_t (built-in or CAS on 64-bit word) -----------
    GPUFUN int64_t atomicAdd_i64(GPUVOLATILE int64_t* addr, int64_t val){
        uint64_t old, nw;
        do {
            old = *addr;
            nw = old + val;
        } while (__XT_CAS_U64((GPUVOLATILE uint64_t*)addr, old, nw) != old);
        return old;
    }
    GPUFUN uint64_t atomicAdd_u64(GPUVOLATILE uint64_t* addr, uint64_t val){
    #ifdef XO_CONTEXT_CUDA
        return atomicAdd(addr, val);
    #else // XO_CONTEXT_CL
        return atom_add(addr, val);
    #endif // XO_CONTEXT_CUDA / XO_CONTEXT_CL
    }

    // ---------------- 32-bit: float (built-in or CAS on 32-bit word) -----------
    GPUFUN float atomicAdd_f32(GPUVOLATILE float* addr, float val){
    #ifdef XO_CONTEXT_CUDA
        return atomicAdd(addr, val);
    #else // XO_CONTEXT_CL
        uint32_t old, nw;
        do {
            old = __XT_AS_U32_FROM_F32(*addr);
            nw = __XT_AS_U32_FROM_F32(__XT_AS_F32_FROM_U32(old) + val);
        } while (__XT_CAS_U32((GPUVOLATILE uint32_t*)addr, old, nw) != old);
        return __XT_AS_F32_FROM_U32(old);
    #endif // XO_CONTEXT_CUDA / XO_CONTEXT_CL
    }

    // ---------------- 64-bit: float (built-in or CAS on 64-bit word) -----------
    GPUFUN double atomicAdd_f64(GPUVOLATILE double* addr, double val){
    #if __CUDA_ARCH__ >= 600
        return atomicAdd(addr, val);
    #else // XO_CONTEXT_CL || __CUDA_ARCH__ < 600
        uint64_t old, nw;
        do {
            old = __XT_AS_U64_FROM_F64(*addr);
            nw = __XT_AS_U64_FROM_F64(__XT_AS_F64_FROM_U64(old) + val);
        } while (__XT_CAS_U64((GPUVOLATILE uint64_t*)addr, old, nw) != old);
        return __XT_AS_F64_FROM_U64(old);
    #endif // __CUDA_ARCH__ >= 600 / XO_CONTEXT_CL
    }
#endif // defined(XO_CONTEXT_CUDA) || defined(XO_CONTEXT_CL)

// Define the overloaded function
// ------------------------------
#ifdef XO_CONTEXT_CUDA
    // NVRTC (CuPy RawModule) usually compiles under extern "C".
    // In C, function overloading is not possible, but we can cheat by doing it in
    // C++ (with a different name to avoid clashes with the built-in atomicAdd).
    // This function will then be remapped to atomicAdd() via a macro in C.
    #ifdef __cplusplus
    extern "C++" {

    GPUFUN int8_t   xt_atomicAdd(GPUVOLATILE int8_t*  p, int8_t  v)  { return atomicAdd_i8 (p, v); }
    GPUFUN uint8_t  xt_atomicAdd(GPUVOLATILE uint8_t* p, uint8_t v)  { return atomicAdd_u8 (p, v); }
    GPUFUN int16_t  xt_atomicAdd(GPUVOLATILE int16_t* p, int16_t v)  { return atomicAdd_i16(p, v); }
    GPUFUN uint16_t xt_atomicAdd(GPUVOLATILE uint16_t*p, uint16_t v) { return atomicAdd_u16(p, v); }
    GPUFUN int64_t  xt_atomicAdd(GPUVOLATILE int64_t*p, int64_t v)   { return atomicAdd_i64(p, v); }

    // Existing type definitions: forward to CUDA built-ins
    GPUFUN int32_t  xt_atomicAdd(GPUVOLATILE int32_t* p, int32_t v)   { return ::atomicAdd(p, v); }
    GPUFUN uint32_t xt_atomicAdd(GPUVOLATILE uint32_t* p, uint32_t v) { return ::atomicAdd(p, v); }
    GPUFUN uint64_t xt_atomicAdd(GPUVOLATILE uint64_t* p, uint64_t v) { return ::atomicAdd(p, v); }
    GPUFUN float    xt_atomicAdd(GPUVOLATILE float* p, float v)       { return ::atomicAdd(p, v); }
    #if __CUDA_ARCH__ >= 600
    GPUFUN double   xt_atomicAdd(GPUVOLATILE double* p, double v)     { return ::atomicAdd(p, v); }
    #else
    GPUFUN double   xt_atomicAdd(GPUVOLATILE double* p, double v)     { return atomicAdd_f64(p, v); }
    #endif

    }
    #endif // __cplusplus

    // ---------- Global remap of the public name on device code ----------
    // Define AFTER the wrappers so we don't macro-rewrite our own calls.
    #ifdef atomicAdd
    #undef atomicAdd
    #endif
    #define atomicAdd(ptr, val) xt_atomicAdd((ptr), (val))
#endif /* XO_CONTEXT_CUDA */

#ifdef XO_CONTEXT_CL
    #if !__has_attribute(overloadable)
        #error "The current OpenCL compiler/architecture does not support __attribute__((overloadable))"
    #endif
    #define OCL_OVERLOAD __attribute__((overloadable))
    GPUFUN int8_t   OCL_OVERLOAD atomicAdd(GPUVOLATILE int8_t*  p, int8_t  v)  { return atomicAdd_i8 (p, v); }
    GPUFUN uint8_t  OCL_OVERLOAD atomicAdd(GPUVOLATILE uint8_t* p, uint8_t v)  { return atomicAdd_u8 (p, v); }
    GPUFUN int16_t  OCL_OVERLOAD atomicAdd(GPUVOLATILE int16_t* p, int16_t v)  { return atomicAdd_i16(p, v); }
    GPUFUN uint16_t OCL_OVERLOAD atomicAdd(GPUVOLATILE uint16_t*p, uint16_t v) { return atomicAdd_u16(p, v); }
    GPUFUN int64_t  OCL_OVERLOAD atomicAdd(GPUVOLATILE int64_t*p, int64_t v)   { return atomicAdd_i64(p, v); }
    GPUFUN float    OCL_OVERLOAD atomicAdd(GPUVOLATILE float* p, float v)      { return atomicAdd_f32(p, v); }
    GPUFUN double   OCL_OVERLOAD atomicAdd(GPUVOLATILE double* p, double v)    { return atomicAdd_f64(p, v); }

    // Existing type definitions: forward to OpenCL built-ins
    GPUFUN int32_t  OCL_OVERLOAD atomicAdd(GPUVOLATILE int32_t* p, int32_t v)   { return atomic_add(p, v); }
    GPUFUN uint32_t OCL_OVERLOAD atomicAdd(GPUVOLATILE uint32_t* p, uint32_t v) { return atomic_add(p, v); }
    GPUFUN uint64_t OCL_OVERLOAD atomicAdd(GPUVOLATILE uint64_t* p, uint64_t v) { return atom_add(p, v); }
#endif // XO_CONTEXT_CL

#endif //_ATOMICADD_H_
