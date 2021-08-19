#ifndef LOCALPARTICE_RNG_H
#define LOCALPARTICE_RNG_H

/*gpufun*/
double LocalParticle_generate_random_double(LocalParticle* part){

    uint32_t s1 = LocalParticle_get___rng_s1(part);
    uint32_t s2 = LocalParticle_get___rng_s2(part);
    uint32_t s3 = LocalParticle_get___rng_s3(part);
    
    double r = rng_get(&s1, &s2, &s3);
	    
    LocalParticle_set___rng_s1(part, s1);
    LocalParticle_set___rng_s2(part, s2);
    LocalParticle_set___rng_s3(part, s3);

    return r;

}

#endif
