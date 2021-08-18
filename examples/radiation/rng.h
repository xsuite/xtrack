#ifndef XTRACK_RNG_H
#define XTRACK_RNG_H

#include <stdint.h> //only_for_context none

typedef struct
  {
    uint64_t s1, s2, s3;
  }
rng_state_t;

static double
rng_get (rng_state_t *state )
{
#define MASK 0xffffffffUL
#define TAUSWORTHE(s,a,b,c,d) (((s &c) <<d) &MASK) ^ ((((s <<a) &MASK)^s) >>b)

  state->s1 = TAUSWORTHE (state->s1, 13, 19, 4294967294UL, 12);
  state->s2 = TAUSWORTHE (state->s2, 2, 25, 4294967288UL, 4);
  state->s3 = TAUSWORTHE (state->s3, 3, 11, 4294967280UL, 17);

  return (state->s1 ^ state->s2 ^ state->s3) / 4294967296.0 ;
}

static void
rng_set (rng_state_t *state, uint64_t s )
{
  if (s == 0)
    s = 1;      /* default seed is 1 */

#define LCG(n) ((69069 * n) & 0xffffffffUL)
  state->s1 = LCG (s);
  if (state->s1 < 2) state->s1 += 2UL;
  state->s2 = LCG (state->s1);
  if (state->s2 < 8) state->s2 += 8UL;
  state->s3 = LCG (state->s2);
  if (state->s3 < 16) state->s3 += 16UL;

  /* "warm it up" */
  rng_get (state);
  rng_get (state);
  rng_get (state);
  rng_get (state);
  rng_get (state);
  rng_get (state);
  return;
}

#endif /* XTRACK_RNG_H */
