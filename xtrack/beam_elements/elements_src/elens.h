// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_ELENS_H
#define XTRACK_ELENS_H

/*gpufun*/
void Elens_track_local_particle(ElensData el, LocalParticle* part0){

    double elens_length = ElensData_get_elens_length(el);

    #ifdef XSUITE_BACKTRACK
        elens_length = -elens_length;
    #endif

    double const inner_radius = ElensData_get_inner_radius(el);
    double const outer_radius = ElensData_get_outer_radius(el);
    double const current = ElensData_get_current(el);
    double const voltage = ElensData_get_voltage(el);
    double const residual_kick_x = ElensData_get_residual_kick_x(el);
    double const residual_kick_y = ElensData_get_residual_kick_y(el);

    int const polynomial_order = ElensData_get_polynomial_order(el);
    int const len_noise = ElensData_get_len_noise(el);

    /*gpuglmem*/ double const* coefficients_polynomial =
                                ElensData_getp1_coefficients_polynomial(el, 0);

    //start_per_particle_block (part0->part)
        int at_turn = LocalParticle_get_at_turn(part);
        const int noise_mode = ElensData_get_noise_mode(el);
        double noise;
        if (noise_mode == 0)
        {
            at_turn = (at_turn % len_noise + len_noise) % len_noise; // Handle wrapping for negative values
            noise = ElensData_get_noise(el, at_turn);
        }
        else if (noise_mode == 1 || noise_mode == 2 || noise_mode == 3)
        {
            if (at_turn >= len_noise)
                noise = (noise_mode == 1)   ? ElensData_get_noise(el, len_noise - 1)
                        : (noise_mode == 2) ? 0.0
                                            : 1.0; // noise_mode == 3
            else
                noise = ElensData_get_noise(el, at_turn);
        }
        else
        {
            // Error case: default to noise = 1.0
            noise = 1.0;
        }

        // electron mass
        double const EMASS  = 510998.928;
        // speed of light

        #if !defined( C_LIGHT )
            #define   C_LIGHT ( 299792458.0 )
        #endif /* !defined( C_LIGHT ) */

        #if !defined( EPSILON_0 )
            #define   EPSILON_0 (8.854187817620e-12)
        #endif /* !defined( EPSILON_0 ) */

        #if !defined( PI )
            #define PI (3.1415926535897932384626433832795028841971693993751)
        #endif /* !defined( PI ) */

        double x      = LocalParticle_get_x(part);
        double y      = LocalParticle_get_y(part);

        // delta
        // double delta  = LocalParticle_get_delta(part);
        // charge ratio: q/q0
        // double qratio = LocalParticle_get_charge_ratio(part);
        // chi = q/q0 * m0/m
        double const chi    = LocalParticle_get_chi(part);
        // reference particle momentum
        double const p0c    = LocalParticle_get_p0c(part);
        // particle momentum
        // double pc     = (1+delta)*(chi/qratio)*(p0c);
        // reference particle charge
        double const q0     = LocalParticle_get_q0(part);

        // rpp = P0/P
        double const rpp     = LocalParticle_get_rpp(part);


        // transverse radius
        double r      = sqrt(x*x + y*y);

        double rvv    = LocalParticle_get_rvv(part);
        double beta0  = LocalParticle_get_beta0(part);

        // # magnetic rigidity
        double const Brho0  = p0c/(q0*C_LIGHT);

        // # Electron properties
        // total electron energy
        double const etot_e       = voltage + EMASS;
        // // electron momentum
        double const p_e          = sqrt(etot_e*etot_e - EMASS*EMASS);
        // // relativistic beta of electron
        double const beta_e       = p_e/etot_e;
        //
        // // # relativistic beta  of protons
        double beta_p = rvv*beta0;

        // keep the formulas more compact
        double const r1 = inner_radius;
        double const r2 = outer_radius;

        // # geometric factor frr uniform distribution
        double frr = 0.;

        //
        if( r < r1 )
        {
          frr = 0.;
        }
        else if ( r > r2 )
        {
          frr = 1.;
        }
        else
        {
          // frr = ((r*r - r1*r1)/(r2*r2 - r1*r1));
          if (polynomial_order ==0)
            {
              frr = ((r*r - r1*r1)/(r2*r2 - r1*r1));
            }
          else
            {
              frr = 0;
              for(int i=0; i<(polynomial_order+1); ++i){
                frr += coefficients_polynomial[i]*pow((double)(r*1e3), (double)(polynomial_order-i));
            }

            }
        }

        // # calculate the kick at r2 (maximum kick)
        double theta_max = ((1.0/(4.0*PI*EPSILON_0)));

        double actual_current = current*noise;
        theta_max = theta_max * (2 * elens_length * actual_current);

        // for the moment: e-beam in the opposite direction from proton beam
        // generalize later
        theta_max = theta_max*(1+beta_e*beta_p);

        theta_max = theta_max/(outer_radius*Brho0*beta_e*beta_p);
        theta_max = theta_max/(C_LIGHT*C_LIGHT);
        // theta max is now completed
        // theta_max = (-1)*theta_max/(rpp*chi);


        // now the actual kick the particle receives

        double theta_pxpy = 0.;
        double dpx = 0.;
        double dpy = 0.;
        //


        if ( r > r1 )
        {
          theta_pxpy = (-1)*frr*theta_max*(outer_radius/r)*(1/(rpp*chi));
          dpx        = x*theta_pxpy/r;
          dpy        = y*theta_pxpy/r;
        }
        else
        {
          // if the particle is not inside the e-beam, it will only
          // be subject to the residual kick
          dpx = residual_kick_x;
          dpy = residual_kick_y;
        }


        LocalParticle_add_to_px(part, dpx );
        LocalParticle_add_to_py(part, dpy );

        // we can update the particle properties or add to the particle properties
        // LocalParticle_add_to_px(part, dpx);
        // LocalParticle_add_to_py(part, dpy);

        // LocalParticle_set_py(part, py_hat);
    //end_per_particle_block
}

#endif