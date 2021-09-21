#ifndef XTRACK_RNG_H
#define XTRACK_RNG_H

#include <stdint.h> //only_for_context none

/*gpufun*/
double rng_get (uint32_t *s1, uint32_t *s2, uint32_t *s3 )
{
#define MASK 0xffffffffUL
#define TAUSWORTHE(s,a,b,c,d) ((((s) &c) <<d) &MASK) ^ (((((s) <<a) &MASK)^(s)) >>b)

  *s1 = TAUSWORTHE (*s1, 13, 19, 4294967294UL, 12);
  *s2 = TAUSWORTHE (*s2, 2, 25, 4294967288UL, 4);
  *s3 = TAUSWORTHE (*s3, 3, 11, 4294967280UL, 17);

  return ((*s1) ^ (*s2) ^ (*s3)) / 4294967296.0 ;
}

/*gpufun*/
void rng_set (uint32_t *s1, uint32_t *s2, uint32_t *s3, uint32_t s )
{
  if (s == 0)
    s = 1;      /* default seed is 1 */

#define LCG(n) ((69069 * (n)) & 0xffffffffUL)
  *s1 = LCG (s);
  if (*s1 < 2) *s1 += 2UL;
  *s2 = LCG (*s1);
  if (*s2 < 8) *s2 += 8UL;
  *s3 = LCG (*s2);
  if (*s3 < 16) *s3 += 16UL;

  /* "warm it up" */
  rng_get (s1, s2, s3);
  rng_get (s1, s2, s3);
  rng_get (s1, s2, s3);
  rng_get (s1, s2, s3);
  rng_get (s1, s2, s3);
  rng_get (s1, s2, s3);
  return;
}

#endif /* XTRACK_RNG_H */
