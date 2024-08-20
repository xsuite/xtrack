// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2024.                 //
// ######################################### //

#ifndef XTRACK_BASE_RNG_H
#define XTRACK_BASE_RNG_H

#include <stdint.h> //only_for_context none

// Combined LCG-Thausworthe generator from (example 37-4):
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application
#define MASK 0xffffffffUL
#define TAUSWORTHE(s,a,b,c,d) ((((s) &c) <<d) &MASK) ^ (((((s) <<a) &MASK)^(s)) >>b)
#define LCG(s,A,C) ((((A*(s)) &MASK) + C) &MASK)

/*gpufun*/
uint32_t rng_get_int32 (uint32_t *s1, uint32_t *s2, uint32_t *s3, uint32_t *s4 ){
  *s1 = TAUSWORTHE (*s1, 13, 19, 4294967294UL, 12);  // p1=2^31-1
  *s2 = TAUSWORTHE (*s2, 2, 25, 4294967288UL, 4);    // p2=2^30-1
  *s3 = TAUSWORTHE (*s3, 3, 11, 4294967280UL, 17);   // p3=2^28-1
  *s4 = LCG(*s4, 1664525, 1013904223UL);             // p4=2^32

  // Combined period is lcm(p1,p2,p3,p4) ~ 2^121
  return ((*s1) ^ (*s2) ^ (*s3) ^ (*s4));
}

#ifndef TWO_TO_32
#define TWO_TO_32 4294967296.0
#endif

/*gpufun*/
double rng_get (uint32_t *s1, uint32_t *s2, uint32_t *s3, uint32_t *s4 ){

  return rng_get_int32(s1, s2, s3, s4) / TWO_TO_32; // uniform in [0, 1) 1e10 resolution

}

/*gpufun*/
void rng_set (uint32_t *s1, uint32_t *s2, uint32_t *s3, uint32_t *s4, uint32_t s ){
  *s1 = LCG (s, 69069, 0);
  if (*s1 < 2) *s1 += 2UL;
  *s2 = LCG (*s1, 69069, 0);
  if (*s2 < 8) *s2 += 8UL;
  *s3 = LCG (*s2, 69069, 0);
  if (*s3 < 16) *s3 += 16UL;
  *s4 = LCG (*s3, 69069, 0);

  /* "warm it up" */
  rng_get (s1, s2, s3, s4);
  rng_get (s1, s2, s3, s4);
  rng_get (s1, s2, s3, s4);
  rng_get (s1, s2, s3, s4);
  rng_get (s1, s2, s3, s4);
  rng_get (s1, s2, s3, s4);
  return;
}

#endif /* XTRACK_BASE_RNG_H */
