// copyright ################################# //
// This file is part of the Xtrack Package.   //
// Copyright (c) CERN, 2025.                   //
// ########################################### //

#ifndef _ATOMICADD_H_
#define _ATOMICADD_H_


#if defined XO_CONTEXT_CPU_SERIAL || defined XO_CONTEXT_CPU_OPENMP
inline void atomicAdd(double *addr, double val){
#ifdef XO_CONTEXT_CPU_OPENMP
   #pragma omp atomic
#endif
   *addr = *addr + val;
}
#endif // XO_CONTEXT_CPU_SERIAL || XO_CONTEXT_CPU_OPENMP


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
