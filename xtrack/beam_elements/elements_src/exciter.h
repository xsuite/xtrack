// ##################################
// Exciter element
//
// Author: Philipp Niedermayer
// Date: 11.11.2022
// ##################################

#ifndef XTRACK_EXCITER_H
#define XTRACK_EXCITER_H

#if !defined( C_LIGHT )
    #define   C_LIGHT ( 299792458.0 )
#endif /* !defined( C_LIGHT ) */

/*gpufun*/
void Exciter_track_local_particle(ExciterData el, LocalParticle* part0){

    // get parameters
    int64_t const order = ExciterData_get_order(el);
    /*gpuglmem*/ double const* knl = ExciterData_getp1_knl(el, 0);
    /*gpuglmem*/ double const* ksl = ExciterData_getp1_ksl(el, 0);
    /*gpuglmem*/ double const* samples = ExciterData_getp1_samples(el, 0);
    int64_t const nsamples = ExciterData_get_nsamples(el);
	int64_t const nduration = ExciterData_get_nduration(el);
    double const sampling_frequency = ExciterData_get_sampling_frequency(el);
    double const frev = ExciterData_get_frev(el);
    int64_t const start_turn = ExciterData_get_start_turn(el);

    //start_per_particle_block (part0->part)

        // zeta is the absolute path length deviation from the reference particle: zeta = (s - beta0*c*t)
        // but without limits, i.e. it can exceed the circumference (for coasting beams)
        // as the particle falls behind or overtakes the reference particle
        double const zeta = LocalParticle_get_zeta(part);
        double const at_turn = LocalParticle_get_at_turn(part);
        double const beta0 = LocalParticle_get_beta0(part);

        // compute excitation sample index
        int64_t i = sampling_frequency * ( ( at_turn - start_turn ) / frev - zeta / beta0 / C_LIGHT );

        if (i >= 0 && i < nduration){
			if (i >= nsamples){
				i = i % nsamples;
			}

            // compute normal and skew multipole components
            double const x = LocalParticle_get_x(part);
            double const y = LocalParticle_get_y(part);
            double dpx = 0.0;
            double dpy = 0.0;
            double zre = 1.0;
            double zim = 0.0;
            double factorial = 1.0;
            for (int64_t kk = 0; kk <= order; kk++){
                if (kk>0){
                    factorial *= kk;
                }

                dpx += (knl[kk] * zre - ksl[kk] * zim) / factorial;
                dpy += (knl[kk] * zim + ksl[kk] * zre) / factorial;

                double const zret = zre * x - zim * y;
                zim = zim * x + zre * y;
                zre = zret;
            }

            // scale by excitation strength
            dpx *= samples[i];
            dpy *= samples[i];

            // apply kick
            double const chi = LocalParticle_get_chi(part);
            LocalParticle_add_to_px(part, - chi * dpx);
            LocalParticle_add_to_py(part, + chi * dpy);

        }
 

    //end_per_particle_block

}

#endif
