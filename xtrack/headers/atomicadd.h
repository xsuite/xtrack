// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef _ATOMICADD_H_
#define _ATOMICADD_H_

#define ATOMICADD_CPU  //only_for_context cpu_serial cpu_openmp
#define ATOMICADD_OPENCL //only_for_context opencl
// CUDA provides atomicAdd() natively

#ifdef ATOMICADD_CPU
inline void atomicAdd(double *addr, double val)
{
   #pragma omp atomic //only_for_context cpu_openmp
   *addr = *addr + val;
}
#endif // ATOMICADD_CPU

#ifdef ATOMICADD_OPENCL
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
#endif // ATOMICADD_OPENCL

#endif
