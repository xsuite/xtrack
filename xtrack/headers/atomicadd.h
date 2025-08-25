// copyright ################################# //
// This file is part of the Xtrack Package.   //
// Copyright (c) CERN, 2025.                   //
// ########################################### //

#ifndef _ATOMICADD_H_
#define _ATOMICADD_H_


#ifdef XO_CONTEXT_CPU
#ifdef XO_CONTEXT_CPU_OPENMP
	/* OpenMP atomic capture gives us read+write atomically */
	#define OMP_ATOMIC_CAPTURE  _Pragma("omp atomic capture")
#else
	#define OMP_ATOMIC_CAPTURE
#endif // XO_CONTEXT_CPU_OPENMP

#define DEF_ATOMIC_ADD(T, SUF)                                  \
static inline T atomicAdd_##SUF(T *addr, T val) {               \
	T old;                                                      \
    OMP_ATOMIC_CAPTURE                               			\
    { old = *addr; *addr = *addr + val; }                       \
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
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
inline void atomicAdd(volatile __global double *addr, double val)
{
	union {
		long u64;
		double f64;
	} next, expected, current;
	current.f64 = *addr;
	do {
		expected.f64 = current.f64;
		next.f64 = expected.f64 + val;
		current.u64 = atom_cmpxchg(
			(volatile __global long *)addr,
		        (long) expected.u64,
			(long) next.u64);
	} while( current.u64 != expected.u64 );
}
#endif // XO_CONTEXT_CL

#endif //_ATOMICADD_H_
